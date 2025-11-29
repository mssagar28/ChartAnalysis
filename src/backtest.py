import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, initial_capital=10000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run_backtest(self, df, signals, price_col='c'):
        """
        Run backtest based on signals
        
        df: DataFrame with price data
        signals: Series/Array of signals (1: Buy, -1: Sell, 0: Hold)
        """
        capital = self.initial_capital
        position = 0  # Current position size
        portfolio_value = []
        trades = []
        
        prices = df[price_col].values
        dates = df.index
        
        for i in range(len(prices)):
            price = prices[i]
            signal = signals[i]
            date = dates[i]
            
            # Execute trades
            if signal == 1 and position <= 0:  # Buy signal
                # Close short if any
                if position < 0:
                    capital += abs(position) * price * (1 - self.commission)
                    trades.append({'date': date, 'type': 'cover', 'price': price, 'capital': capital})
                    position = 0
                
                # Open long
                shares = capital / (price * (1 + self.commission))
                capital = 0
                position = shares
                trades.append({'date': date, 'type': 'buy', 'price': price, 'shares': shares})
                
            elif signal == -1 and position >= 0:  # Sell signal
                # Close long if any
                if position > 0:
                    capital += position * price * (1 - self.commission)
                    trades.append({'date': date, 'type': 'sell', 'price': price, 'capital': capital})
                    position = 0
                
                # Open short (simplified: assume we can short with full capital)
                # For simplicity in this basic version, we might just go to cash
                # To implement shorting:
                # shares = capital / (price * (1 + commission))
                # position = -shares
                pass # Keeping it simple: Long-only or Long-Cash for now unless specified
            
            # Calculate current portfolio value
            current_val = capital
            if position != 0:
                current_val += position * price
            
            portfolio_value.append(current_val)
            
        # Create results DataFrame
        results = pd.DataFrame({
            'date': dates,
            'price': prices,
            'signal': signals,
            'portfolio_value': portfolio_value
        }).set_index('date')
        
        # Calculate metrics
        returns = results['portfolio_value'].pct_change()
        total_return = (results['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252*24) # Hourly assumption
        max_drawdown = (results['portfolio_value'] / results['portfolio_value'].cummax() - 1).min()
        
        metrics = {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Final Value': results['portfolio_value'].iloc[-1],
            'Trades': len(trades)
        }
        
        return results, metrics, pd.DataFrame(trades)

    def plot_results(self, results, trades_df=None):
        """Plot backtest results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Portfolio Value
        ax1.plot(results.index, results['portfolio_value'], label='Portfolio')
        ax1.set_title('Portfolio Value')
        ax1.grid(True)
        
        # Price and Signals
        ax2.plot(results.index, results['price'], label='Price', alpha=0.5)
        
        # Buy signals
        buys = results[results['signal'] == 1]
        ax2.scatter(buys.index, buys['price'], marker='^', color='g', label='Buy Signal')
        
        # Sell signals
        sells = results[results['signal'] == -1]
        ax2.scatter(sells.index, sells['price'], marker='v', color='r', label='Sell Signal')
        
        ax2.set_title('Price & Signals')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
