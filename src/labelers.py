import pandas as pd
import numpy as np

class CandlestickLabeler:
    def __init__(self, df):
        """
        df: DataFrame with columns ['o', 'h', 'l', 'c', 'v']
        """
        self.df = df.copy()
        self._calculate_geometry()
    
    def _calculate_geometry(self):
        """Calculate basic candlestick geometry"""
        self.df['body'] = abs(self.df['c'] - self.df['o'])
        self.df['range'] = self.df['h'] - self.df['l']
        self.df['lower_wick'] = self.df[['o', 'c']].min(axis=1) - self.df['l']
        self.df['upper_wick'] = self.df['h'] - self.df[['o', 'c']].max(axis=1)
        
        # Avoid division by zero
        self.df['body_ratio'] = self.df['body'] / (self.df['range'] + 1e-9)
        self.df['is_bullish'] = (self.df['c'] > self.df['o']).astype(int)
    
    def label_doji(self, body_threshold=0.1):
        """
        Doji: Very small body relative to range
        Indicates indecision
        """
        self.df['doji'] = (self.df['body_ratio'] < body_threshold).astype(int)
        return self
    
    def label_hammer(self, body_threshold=0.3, wick_ratio=2.0):
        """
        Hammer: Small body at top, long lower wick
        Bullish reversal at bottom of downtrend
        """
        condition = (
            (self.df['body_ratio'] < body_threshold) &
            (self.df['lower_wick'] / (self.df['body'] + 1e-9) > wick_ratio) &
            (self.df['upper_wick'] / (self.df['body'] + 1e-9) < 0.5) &
            (self.df['is_bullish'] == 1)
        )
        self.df['hammer'] = condition.astype(int)
        return self
    
    def label_shooting_star(self, body_threshold=0.3, wick_ratio=2.0):
        """
        Shooting Star: Small body at bottom, long upper wick
        Bearish reversal at top of uptrend
        """
        condition = (
            (self.df['body_ratio'] < body_threshold) &
            (self.df['upper_wick'] / (self.df['body'] + 1e-9) > wick_ratio) &
            (self.df['lower_wick'] / (self.df['body'] + 1e-9) < 0.5) &
            (self.df['is_bullish'] == 0)
        )
        self.df['shooting_star'] = condition.astype(int)
        return self
    
    def label_marubozu(self, wick_threshold=0.05):
        """
        Marubozu: Little to no wicks, strong body
        Indicates strong momentum
        """
        condition = (
            (self.df['upper_wick'] / (self.df['range'] + 1e-9) < wick_threshold) &
            (self.df['lower_wick'] / (self.df['range'] + 1e-9) < wick_threshold) &
            (self.df['body_ratio'] > 0.9)
        )
        self.df['marubozu'] = condition.astype(int)
        return self
    
    def label_inverted_hammer(self, body_threshold=0.3, wick_ratio=2.0):
        """
        Inverted Hammer: Small body at bottom, long upper wick
        Potential bullish reversal
        """
        condition = (
            (self.df['body_ratio'] < body_threshold) &
            (self.df['upper_wick'] / (self.df['body'] + 1e-9) > wick_ratio) &
            (self.df['lower_wick'] / (self.df['body'] + 1e-9) < 0.5)
        )
        self.df['inverted_hammer'] = condition.astype(int)
        return self
    
    def label_hanging_man(self, body_threshold=0.3, wick_ratio=2.0):
        """
        Hanging Man: Like hammer but appears at top of uptrend
        Bearish reversal signal
        """
        condition = (
            (self.df['body_ratio'] < body_threshold) &
            (self.df['lower_wick'] / (self.df['body'] + 1e-9) > wick_ratio) &
            (self.df['upper_wick'] / (self.df['body'] + 1e-9) < 0.5) &
            (self.df['is_bullish'] == 0)
        )
        self.df['hanging_man'] = condition.astype(int)
        return self

    def label_engulfing(self):
        """
        Bullish Engulfing: Bearish candle followed by larger bullish candle
        Bearish Engulfing: Bullish candle followed by larger bearish candle
        """
        # Shift to compare with previous candle
        prev_o = self.df['o'].shift(1)
        prev_c = self.df['c'].shift(1)
        prev_body = self.df['body'].shift(1)
        
        # Bullish engulfing
        bullish_condition = (
            (prev_c < prev_o) &  # Previous was bearish
            (self.df['c'] > self.df['o']) &  # Current is bullish
            (self.df['o'] <= prev_c) &  # Opens at/below prev close
            (self.df['c'] >= prev_o) &  # Closes at/above prev open
            (self.df['body'] > prev_body)  # Larger body
        )
        self.df['bullish_engulfing'] = bullish_condition.astype(int)
        
        # Bearish engulfing
        bearish_condition = (
            (prev_c > prev_o) &  # Previous was bullish
            (self.df['c'] < self.df['o']) &  # Current is bearish
            (self.df['o'] >= prev_c) &  # Opens at/above prev close
            (self.df['c'] <= prev_o) &  # Closes at/below prev open
            (self.df['body'] > prev_body)  # Larger body
        )
        self.df['bearish_engulfing'] = bearish_condition.astype(int)
        
        return self
    
    def label_harami(self):
        """
        Harami: Small candle contained within previous large candle
        Indicates potential reversal
        """
        prev_o = self.df['o'].shift(1)
        prev_c = self.df['c'].shift(1)
        prev_body = self.df['body'].shift(1)
        
        # Bullish harami
        bullish_condition = (
            (prev_c < prev_o) &  # Previous was bearish
            (self.df['c'] > self.df['o']) &  # Current is bullish
            (self.df['o'] > prev_c) &  # Opens above prev close
            (self.df['c'] < prev_o) &  # Closes below prev open
            (self.df['body'] < prev_body * 0.5)  # Much smaller body
        )
        self.df['bullish_harami'] = bullish_condition.astype(int)
        
        # Bearish harami
        bearish_condition = (
            (prev_c > prev_o) &  # Previous was bullish
            (self.df['c'] < self.df['o']) &  # Current is bearish
            (self.df['o'] < prev_c) &  # Opens below prev close
            (self.df['c'] > prev_o) &  # Closes above prev open
            (self.df['body'] < prev_body * 0.5)  # Much smaller body
        )
        self.df['bearish_harami'] = bearish_condition.astype(int)
        
        return self

    def label_morning_star(self):
        """
        Morning Star: Bearish, small body, bullish (reversal pattern)
        Bullish reversal at bottom of downtrend
        """
        # Get three consecutive candles
        c1_o, c1_c = self.df['o'].shift(2), self.df['c'].shift(2)
        c2_body = self.df['body'].shift(1)
        c3_o, c3_c = self.df['o'], self.df['c']
        
        condition = (
            (c1_c < c1_o) &  # First candle bearish
            (c2_body < (self.df['body'].shift(2) * 0.3)) &  # Middle small
            (c3_c > c3_o) &  # Third candle bullish
            (c3_c > ((c1_o + c1_c) / 2))  # Third closes above midpoint of first
        )
        self.df['morning_star'] = condition.astype(int)
        return self
    
    def label_evening_star(self):
        """
        Evening Star: Bullish, small body, bearish (reversal pattern)
        Bearish reversal at top of uptrend
        """
        c1_o, c1_c = self.df['o'].shift(2), self.df['c'].shift(2)
        c2_body = self.df['body'].shift(1)
        c3_o, c3_c = self.df['o'], self.df['c']
        
        condition = (
            (c1_c > c1_o) &  # First candle bullish
            (c2_body < (self.df['body'].shift(2) * 0.3)) &  # Middle small
            (c3_c < c3_o) &  # Third candle bearish
            (c3_c < ((c1_o + c1_c) / 2))  # Third closes below midpoint
        )
        self.df['evening_star'] = condition.astype(int)
        return self
    
    def label_three_white_soldiers(self):
        """
        Three consecutive bullish candles with higher closes
        Strong bullish continuation
        """
        c1_c, c1_o = self.df['c'].shift(2), self.df['o'].shift(2)
        c2_c, c2_o = self.df['c'].shift(1), self.df['o'].shift(1)
        c3_c, c3_o = self.df['c'], self.df['o']
        
        condition = (
            (c1_c > c1_o) & (c2_c > c2_o) & (c3_c > c3_o) &  # All bullish
            (c2_c > c1_c) & (c3_c > c2_c) &  # Higher closes
            (c2_o > c1_o) & (c3_o > c2_o)  # Higher opens
        )
        self.df['three_white_soldiers'] = condition.astype(int)
        return self
    
    def label_three_black_crows(self):
        """
        Three consecutive bearish candles with lower closes
        Strong bearish continuation
        """
        c1_c, c1_o = self.df['c'].shift(2), self.df['o'].shift(2)
        c2_c, c2_o = self.df['c'].shift(1), self.df['o'].shift(1)
        c3_c, c3_o = self.df['c'], self.df['o']
        
        condition = (
            (c1_c < c1_o) & (c2_c < c2_o) & (c3_c < c3_o) &  # All bearish
            (c2_c < c1_c) & (c3_c < c2_c) &  # Lower closes
            (c2_o < c1_o) & (c3_o < c2_o)  # Lower opens
        )
        self.df['three_black_crows'] = condition.astype(int)
        return self
    
    def get_labeled_data(self):
        """Return DataFrame with all labels"""
        return self.df
