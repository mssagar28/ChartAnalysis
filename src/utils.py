import pandas as pd
import numpy as np

def preprocess_ohlcv(df):
    """Clean and prepare OHLCV data"""
    if df is None or df.empty:
        return df
        
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Forward fill missing values (max 3 periods)
    df = df.fillna(method='ffill', limit=3)
    
    # Drop remaining NaNs
    df = df.dropna()
    
    # Ensure OHLC consistency
    # Sometimes data sources have slight inconsistencies
    if 'o' in df.columns and 'h' in df.columns and 'c' in df.columns and 'l' in df.columns:
        df['h'] = df[['o', 'h', 'c']].max(axis=1)
        df['l'] = df[['o', 'l', 'c']].min(axis=1)
    
    # Remove zero volume candles (optional, but good for active markets)
    if 'v' in df.columns:
        df = df[df['v'] > 0]
    
    # Sort by datetime
    df = df.sort_index()
    
    return df
