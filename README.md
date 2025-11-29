# Candlestick Pattern Analysis & Next-Candle Prediction

A comprehensive stock market analysis tool featuring candlestick pattern detection, technical indicators, and ML-powered price predictions for NSE stocks and MCX commodities.

## 🌟 Features

- **📊 Real-time Charts**: Interactive candlestick charts with pattern highlighting
- **🔍 Pattern Detection**: Automatic detection of 12+ candlestick patterns
- **📈 Price Predictions**: ML-powered next-candle direction and price range predictions
- **💹 Multi-Asset Support**: 
  - 31 NSE stocks (Nifty 50 + MCX stock)
  - 5 MCX commodities (Gold, Silver, Crude Oil, Natural Gas, Copper)
- **⏰ Multiple Timeframes**: 15-minute, 1-hour, and daily charts
- **🤖 Machine Learning**: XGBoost model with 54%+ accuracy

## 🚀 Quick Start

### Local Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data**:
   ```bash
   python src/data_loader.py
   ```

3. **Train Model** (Optional):
   ```bash
   python train_model.py
   ```

4. **Run Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
   Or double-click `run_dashboard.bat`

### 📱 Access from Phone/Tablet

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for instructions to deploy to Streamlit Cloud and access from any device!

## 📂 Project Structure

```
candlestick_project/
├── dashboard/          # Streamlit web app
├── src/               # Core logic (data, patterns, features, models)
├── notebooks/         # Jupyter notebooks for analysis
├── data/              # Market data (raw, processed, features)
├── models/            # Trained ML models
├── results/           # Backtest results
└── tests/             # Unit tests
```

## 🎯 Supported Assets

### NSE Stocks (31)
RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY, SBIN, BHARTIARTL, ITC, KOTAKBANK, HINDUNILVR, LT, AXISBANK, BAJFINANCE, ASIANPAINT, MARUTI, HCLTECH, SUNPHARMA, TITAN, ULTRACEMCO, NESTLEIND, WIPRO, POWERGRID, NTPC, TATAMOTORS, M&M, TECHM, ONGC, TATASTEEL, ADANIPORTS, BAJAJFINSV, MCX

### MCX Commodities (5)
Gold (GC=F), Silver (SI=F), Crude Oil (CL=F), Natural Gas (NG=F), Copper (HG=F)

## 🔮 Prediction Features

- **Direction Prediction**: UP/DOWN with confidence score
- **Price Range**: Predicted Low-High range based on ATR
- **Next Candle Time**: Smart calculation accounting for market hours and weekends
- **Historical Performance**: Last 10 candles with actual vs predicted

## 📊 Candlestick Patterns Detected

- Doji
- Hammer / Hanging Man
- Shooting Star / Inverted Hammer
- Bullish/Bearish Engulfing
- Bullish/Bearish Harami
- Morning/Evening Star
- Three White Soldiers / Three Black Crows

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data**: yfinance, ccxt
- **ML**: XGBoost, TensorFlow/Keras
- **Analysis**: pandas, numpy, ta (technical analysis)
- **Visualization**: mplfinance, matplotlib, plotly

## 📈 Model Performance

- **Algorithm**: XGBoost Classifier
- **Test Accuracy**: 54.2%
- **Features**: 129 technical indicators + candlestick patterns
- **Training Data**: Multiple NSE stocks, 2020-2024

## ⚙️ Configuration

### Data Timeframes
- **Daily (1d)**: 2020 to present
- **Hourly (1h)**: Last 720 days
- **15-minute (15m)**: Last 60 days (Yahoo Finance limitation)

### Market Hours
- NSE: 9:15 AM - 3:30 PM IST
- Weekends: Automatically skipped in predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Stock market predictions are inherently uncertain. Always do your own research and consult with financial advisors before making investment decisions.

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment help

---

**Built with ❤️ for traders and data enthusiasts**
"# ChartAnalysis" 
