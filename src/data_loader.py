import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import os

class DataCollector:
    def __init__(self, save_path='data/raw'):
        # Ensure save_path is absolute or relative to CWD correctly
        # If running from project root, 'data/raw' is fine.
        # But let's make it absolute to be safe
        if not os.path.isabs(save_path):
            save_path = os.path.abspath(save_path)
            
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        
    def download_stock(self, symbol, start_date, end_date, interval='1d'):
        """Download stock OHLCV from Yahoo Finance"""
        print(f"Downloading {symbol} from {start_date} to {end_date}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                print(f"No data found for {symbol}")
                return None
                
            df = df.rename(columns={
                'Open': 'o', 'High': 'h', 'Low': 'l', 
                'Close': 'c', 'Volume': 'v'
            })
            df['symbol'] = symbol
            # Keep only relevant columns
            cols = [c for c in ['o', 'h', 'l', 'c', 'v'] if c in df.columns]
            df = df[['symbol'] + cols]
            
            # Save to parquet
            filename = os.path.join(self.save_path, f"{symbol}_{interval}.parquet")
            df.to_parquet(filename)
            print(f"Saved {symbol} data to {filename}")
            return df
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
    
    def download_crypto(self, symbol, timeframe='1h', limit=1000):
        """Download crypto OHLCV from Binance"""
        print(f"Downloading {symbol} ({timeframe})...")
        exchange = ccxt.binance()
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
        
        if not ohlcv:
            print(f"No data found for {symbol}")
            return None

        df = pd.DataFrame(
            ohlcv, 
            columns=['timestamp', 'o', 'h', 'l', 'c', 'v']
        )
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime')
        df['symbol'] = symbol
        
        filename = os.path.join(self.save_path, f"{symbol.replace('/', '_')}_{timeframe}.parquet")
        df.to_parquet(filename)
        print(f"Saved {symbol} data to {filename}")
        return df

if __name__ == "__main__":
    # Use absolute path for data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    
    collector = DataCollector(save_path=data_dir)
    
    # Yahoo Finance 1h data is limited to last 730 days
    start_date = (datetime.now() - timedelta(days=720)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # NSE Stocks List (Top 30 from Nifty 50 + MCX)
    nse_symbols = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
        'INFY.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 
        'KOTAKBANK.NS', 'HINDUNILVR.NS', 'LT.NS', 'AXISBANK.NS',
        'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
        'WIPRO.NS', 'POWERGRID.NS', 'NTPC.NS', 'TATAMOTORS.NS',
        'M&M.NS', 'TECHM.NS', 'ONGC.NS', 'TATASTEEL.NS',
        'ADANIPORTS.NS', 'BAJAJFINSV.NS', 'MCX.NS'
    ]
    
    # MCX Commodities (using international symbols as MCX direct data not available via yfinance)
    mcx_symbols = [
        'GC=F',      # Gold Futures
        'SI=F',      # Silver Futures
        'CL=F',      # Crude Oil WTI Futures
        'NG=F',      # Natural Gas Futures
        'HG=F'       # Copper Futures
    ]
    
    # Combine all symbols
    symbols = nse_symbols + mcx_symbols
    
    print("Starting download for NSE stocks...")
    for symbol in symbols:
        collector.download_stock(
            symbol, 
            start_date=start_date, 
            end_date=end_date,
            interval='1h'
        )
    
    # Download 15m data (Yahoo Finance limits to last 60 days)
    print("\nDownloading 15-minute data (last 60 days only)...")
    start_date_15m = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')
    for symbol in symbols:
        collector.download_stock(
            symbol, 
            start_date=start_date_15m, 
            end_date=end_date,
            interval='15m'
        )
    
    # Also download daily data for longer history
    start_date_daily = '2020-01-01'
    print("\nDownloading daily data...")
    for symbol in symbols:
        collector.download_stock(
            symbol, 
            start_date=start_date_daily, 
            end_date=end_date,
            interval='1d'
        )
