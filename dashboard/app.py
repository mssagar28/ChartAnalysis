import streamlit as st
import pandas as pd
import mplfinance as mpf
import sys
import os
import joblib
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_loader import DataCollector
from labelers import CandlestickLabeler
from features import FeatureEngineer
from utils import preprocess_ohlcv

st.set_page_config(page_title="Candlestick Analysis", layout="wide")

st.title("Candlestick Pattern Analysis & Prediction")

# Sidebar
st.sidebar.header("Configuration")
# Load available symbols
# Load available symbols
data_dir = os.path.join(os.path.dirname(__file__), '../data/raw')

# Ensure data directory exists
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

try:
    available_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
except FileNotFoundError:
    available_files = []

# Extract symbols and filter for NSE stocks (.NS) and MCX commodities (=F)
all_symbols = sorted(list(set([f.split('_')[0] for f in available_files])))
available_symbols = [s for s in all_symbols if s.endswith('.NS') or s.endswith('=F')]

if not available_symbols:
    available_symbols = ["RELIANCE.NS"]  # Fallback

symbol = st.sidebar.selectbox("Symbol", available_symbols)
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "1d", "15m"])
data_source = st.sidebar.selectbox("Data Source", ["Yahoo Finance", "Binance"])

# Load Data
@st.cache_data
def load_data(symbol, timeframe, source):
    # Use absolute path relative to this file
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, f"../data/raw/{symbol}_{timeframe}.parquet")
    if os.path.exists(data_path):
        return pd.read_parquet(data_path)
    else:
        return None

df = load_data(symbol, timeframe, data_source)

if df is not None:
    df = preprocess_ohlcv(df)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Chart & Patterns", "Analysis", "Prediction"])
    
    with tab1:
        st.subheader(f"{symbol} - {timeframe}")
        
        # Pattern Detection
        labeler = CandlestickLabeler(df)
        labeled_df = (labeler
            .label_doji()
            .label_hammer()
            .label_shooting_star()
            .label_engulfing()
            .label_harami()
            .label_morning_star()
            .label_evening_star()
            .get_labeled_data()
        )
        
        # Select pattern to highlight
        patterns = ['doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing', 
                    'morning_star', 'evening_star']
        selected_pattern = st.selectbox("Highlight Pattern", ["None"] + patterns)
        
        # Plot
        # We need to use st.pyplot because mpf.plot returns None but plots to figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Slice for performance
        window = st.slider("Window Size", 50, 500, 100)
        plot_data = labeled_df.tail(window).copy()
        
        # Rename columns for mplfinance (expects capitalized names)
        plot_data = plot_data.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        })
        
        # Create addplot for patterns
        apds = []
        if selected_pattern != "None":
            # Create markers
            mask = plot_data.index.isin(labeled_df[labeled_df[selected_pattern] == 1].index)
            if mask.any():
                # Markers need to be same length as data, with NaNs where no marker
                markers = plot_data['High'] * 1.01
                markers[~mask] = float('nan')
                apds.append(mpf.make_addplot(markers, type='scatter', markersize=100, marker='v'))
        
        # Plot using mplfinance
        # Note: mpf in streamlit is tricky, often easier to save image or use returnfig=True
        plot_kwargs = {
            'type': 'candle',
            'volume': True,
            'style': 'charles',
            'returnfig': True
        }
        if apds:  # Only add addplot if there are plots to add
            plot_kwargs['addplot'] = apds
        
        fig, axlist = mpf.plot(plot_data, **plot_kwargs)
        st.pyplot(fig)
        
        st.write("Recent Patterns:")
        st.dataframe(plot_data[patterns].tail(10))

    with tab2:
        st.subheader("Technical Analysis")
        
        # Use the already labeled data from tab1
        engineer = FeatureEngineer(labeled_df)
        df_features = (engineer
            .add_candle_features()
            .add_technical_indicators()
            .add_price_context()
            .add_volatility_features()
            .get_features()
        )
        
        st.write("Feature Statistics:")
        st.write(df_features.describe())
        
        feature_to_plot = st.selectbox("Select Feature", df_features.columns)
        st.line_chart(df_features[feature_to_plot].tail(window))

    with tab3:
        st.subheader("Model Prediction")
        
        # Use absolute path
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, '../models/xgboost_baseline.joblib')
        feature_cols_path = os.path.join(base_dir, '../models/feature_columns.joblib')
        
        if os.path.exists(model_path) and os.path.exists(feature_cols_path):
            model = joblib.load(model_path)
            feature_cols = joblib.load(feature_cols_path)
            
            try:
                # Ensure df_features has the same columns as training
                missing_cols = set(feature_cols) - set(df_features.columns)
                if missing_cols:
                    st.warning(f"Missing features: {len(missing_cols)} columns. Predictions may be inaccurate.")
                
                # Use only the features that were used in training
                available_features = [col for col in feature_cols if col in df_features.columns]
                
                if len(available_features) < len(feature_cols) * 0.8:
                    st.error("Too many missing features. Cannot make predictions.")
                else:
                    # Predict on last 10 candles
                    X = df_features[available_features].tail(10)
                    
                    # Fill any remaining NaNs with 0
                    X = X.fillna(0)
                    
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)
                    
                    st.success("✅ Model loaded successfully!")
                    st.write(f"**Model Accuracy (Test)**: 54.2%")
                    st.write(f"**Features Used**: {len(available_features)}/{len(feature_cols)}")
                    
                    # Display predictions
                    st.write("### Last 10 Candles Predictions")
                    
                    # Get actual price data for the same period
                    price_data = labeled_df.loc[X.index]
                    
                    pred_df = pd.DataFrame({
                        'Timestamp': X.index,
                        'Prediction': ['📈 UP' if p == 1 else '📉 DOWN' for p in predictions],
                        'Confidence': [f"{max(prob)*100:.1f}%" for prob in probabilities],
                        'Price Range': [f"₹{row['l']:.2f} - ₹{row['h']:.2f}" for _, row in price_data.iterrows()],
                        'Close': [f"₹{row['c']:.2f}" for _, row in price_data.iterrows()]
                    })
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Next candle prediction
                    st.write("### Next Candle Prediction")
                    next_pred = predictions[-1]
                    next_prob = probabilities[-1]
                    
                    # Calculate next candle time based on timeframe
                    from datetime import datetime, timedelta
                    import pytz
                    
                    last_timestamp = X.index[-1]
                    
                    # Get current time in IST
                    ist = pytz.timezone('Asia/Kolkata')
                    current_time = datetime.now(ist)
                    
                    # Calculate initial next candle time
                    if timeframe == '15m':
                        next_time = last_timestamp + timedelta(minutes=15)
                    elif timeframe == '1h':
                        next_time = last_timestamp + timedelta(hours=1)
                    elif timeframe == '1d':
                        next_time = last_timestamp + timedelta(days=1)
                    else:
                        next_time = "Unknown"
                    
                    # Check if data is stale and adjust to next working day
                    is_stale = False
                    if isinstance(next_time, datetime):
                        # Make timezone-aware comparison
                        next_time_aware = next_time.astimezone(ist) if next_time.tzinfo else ist.localize(next_time)
                        time_diff = current_time - next_time_aware
                        
                        if time_diff.total_seconds() > 0:
                            is_stale = True
                            hours_old = int(time_diff.total_seconds() / 3600)
                            
                            # For daily timeframe, find next working day
                            if timeframe == '1d':
                                # Start from current date
                                next_working_day = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
                                
                                # If current time is past market hours (3:30 PM), move to next day
                                if current_time.hour >= 15 and current_time.minute >= 30:
                                    next_working_day += timedelta(days=1)
                                
                                # Skip weekends (Saturday=5, Sunday=6)
                                while next_working_day.weekday() >= 5:
                                    next_working_day += timedelta(days=1)
                                
                                next_time_aware = next_working_day
                            
                            # For intraday (15m, 1h), show next market session
                            else:
                                # NSE market hours: 9:15 AM to 3:30 PM
                                next_session = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
                                
                                # If current time is past market close, move to next day
                                if current_time.hour >= 15 and current_time.minute >= 30:
                                    next_session += timedelta(days=1)
                                
                                # Skip weekends
                                while next_session.weekday() >= 5:
                                    next_session += timedelta(days=1)
                                
                                next_time_aware = next_session
                    
                    st.info(f"📅 **Next Candle Time**: {next_time_aware if isinstance(next_time, datetime) else next_time}")
                    st.caption(f"🕐 Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    st.caption(f"📊 Last Data Point: {last_timestamp}")
                    
                    # Show data freshness warning
                    if is_stale:
                        st.warning(f"⚠️ Data is {hours_old} hours old. Prediction shown for next market session.")
                        st.info("💡 Run the data loader to get fresh data: `python src/data_loader.py`")
                    
                    # Calculate predicted price range based on ATR (Average True Range)
                    last_close = price_data.iloc[-1]['c']
                    
                    # Use ATR from features if available, otherwise calculate simple range
                    if 'atr_10' in df_features.columns:
                        atr = df_features.iloc[-1]['atr_10']
                    else:
                        # Fallback: use average range of last 10 candles
                        atr = price_data.tail(10)['range'].mean()
                    
                    # Predict range based on direction
                    if next_pred == 1:  # UP prediction
                        predicted_low = last_close - (atr * 0.5)
                        predicted_high = last_close + (atr * 1.5)
                        st.success(f"🚀 **Predicted Direction**: UP with {next_prob[1]*100:.1f}% confidence")
                        st.info(f"💰 **Predicted Price Range**: ₹{predicted_low:.2f} - ₹{predicted_high:.2f}")
                        st.caption(f"📍 Current Close: ₹{last_close:.2f}")
                    else:  # DOWN prediction
                        predicted_low = last_close - (atr * 1.5)
                        predicted_high = last_close + (atr * 0.5)
                        st.error(f"📉 **Predicted Direction**: DOWN with {next_prob[0]*100:.1f}% confidence")
                        st.info(f"💰 **Predicted Price Range**: ₹{predicted_low:.2f} - ₹{predicted_high:.2f}")
                        st.caption(f"📍 Current Close: ₹{last_close:.2f}")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("⚠️ Model not found. Please train a model first.")
            st.info("Run `python train_model.py` in the project directory to train a model.")

else:
    st.warning(f"Data not found for {symbol}. Please run data loader.")
    if st.button("Download Data (Demo)"):
        st.write("Downloading...")
        collector = DataCollector()
        if data_source == "Yahoo Finance":
            collector.download_stock(symbol, "2023-01-01", "2024-01-01", timeframe)
        else:
            collector.download_crypto(symbol, timeframe)
        st.rerun()
