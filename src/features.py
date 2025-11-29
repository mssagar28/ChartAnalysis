import pandas as pd
import numpy as np
from ta import add_all_ta_features

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
    
    def add_candle_features(self):
        """Add geometric candle features"""
        # Ensure basic columns exist
        if 'body' not in self.df.columns:
            self.df['body'] = abs(self.df['c'] - self.df['o'])
        
        self.df['range'] = self.df['h'] - self.df['l']
        self.df['lower_wick'] = self.df[['o', 'c']].min(axis=1) - self.df['l']
        self.df['upper_wick'] = self.df['h'] - self.df[['o', 'c']].max(axis=1)
        
        # Ratios
        self.df['body_ratio'] = self.df['body'] / (self.df['range'] + 1e-9)
        self.df['upper_wick_ratio'] = self.df['upper_wick'] / (self.df['range'] + 1e-9)
        self.df['lower_wick_ratio'] = self.df['lower_wick'] / (self.df['range'] + 1e-9)
        
        # Direction
        self.df['is_bullish'] = (self.df['c'] > self.df['o']).astype(int)
        
        # Typical price
        self.df['typical_price'] = (self.df['h'] + self.df['l'] + self.df['c']) / 3
        
        # Returns
        self.df['return_oc'] = (self.df['c'] / self.df['o']) - 1
        self.df['return_cc'] = self.df['c'].pct_change()
        
        # Volume features
        if 'v' in self.df.columns:
            self.df['volume_change'] = self.df['v'].pct_change()
            self.df['volume_ma_ratio'] = self.df['v'] / (self.df['v'].rolling(20).mean() + 1e-9)
        
        return self
    
    def add_technical_indicators(self):
        """Add technical analysis indicators using ta library"""
        # This adds ~80 indicators automatically
        try:
            self.df = add_all_ta_features(
                self.df, open="o", high="h", low="l", 
                close="c", volume="v", fillna=True
            )
        except Exception as e:
            print(f"Error adding TA features: {e}")
        return self
    
    def add_price_context(self, windows=[5, 10, 20, 50]):
        """Add price position relative to moving averages"""
        for window in windows:
            ma = self.df['c'].rolling(window).mean()
            self.df[f'price_to_ma{window}'] = (self.df['c'] / (ma + 1e-9)) - 1
            
            # Distance from high/low
            rolling_high = self.df['h'].rolling(window).max()
            rolling_low = self.df['l'].rolling(window).min()
            self.df[f'pct_from_high_{window}'] = (self.df['c'] - rolling_high) / (rolling_high + 1e-9)
            self.df[f'pct_from_low_{window}'] = (self.df['c'] - rolling_low) / (rolling_low + 1e-9)
        
        return self
    
    def add_volatility_features(self, windows=[10, 20, 50]):
        """Add volatility measurements"""
        for window in windows:
            # Standard deviation of returns
            self.df[f'volatility_{window}'] = self.df['return_cc'].rolling(window).std()
            
            # Average True Range (ATR)
            tr1 = self.df['h'] - self.df['l']
            tr2 = abs(self.df['h'] - self.df['c'].shift())
            tr3 = abs(self.df['l'] - self.df['c'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.df[f'atr_{window}'] = tr.rolling(window).mean()
        
        return self
    
    def create_sequence_features(self, lookback=30):
        """Create sequences for LSTM/CNN models"""
        feature_cols = [col for col in self.df.columns 
                       if col not in ['o', 'h', 'l', 'c', 'v', 'symbol', 'timestamp', 'datetime']]
        
        # Ensure numeric
        feature_df = self.df[feature_cols].select_dtypes(include=[np.number])
        
        sequences = []
        for i in range(lookback, len(self.df)):
            sequence = feature_df.iloc[i-lookback:i].values
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def get_features(self):
        """Return engineered features DataFrame"""
        return self.df

def create_targets(df, horizon=1, threshold=0.001):
    """
    Create target labels for next-candle prediction
    
    horizon: how many candles ahead to predict
    threshold: minimum move to classify as UP/DOWN
    """
    df = df.copy()
    
    # Future close
    future_close = df['c'].shift(-horizon)
    
    # Return
    future_return = (future_close / df['c']) - 1
    
    # Classification target (UP/DOWN/NEUTRAL)
    df['target_return'] = future_return
    df['target_direction'] = 0  # NEUTRAL
    df.loc[future_return > threshold, 'target_direction'] = 1  # UP
    df.loc[future_return < -threshold, 'target_direction'] = -1  # DOWN
    
    # Binary target (UP/DOWN only)
    df['target_binary'] = (future_return > 0).astype(int)
    
    # Regression target (actual return)
    df['target_next_close'] = future_close
    
    return df
