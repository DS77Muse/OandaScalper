"""
Simple Backtesting Test

A simplified test to verify backtesting functionality works before running the full system.
"""

import pandas as pd
from backtesting import Backtest, Strategy
from oanda_handler import get_api_client, get_historical_data

class SimpleStrategy(Strategy):
    """
    Simple moving average crossover strategy for testing.
    """
    
    def init(self):
        # Simple 10 and 20 period moving averages using SMA indicator
        from backtesting.lib import crossover
        
        # Calculate moving averages properly for backtesting.py
        close_prices = pd.Series(self.data.Close, index=self.data.index)
        self.ma10 = close_prices.rolling(10).mean()
        self.ma20 = close_prices.rolling(20).mean()
    
    def next(self):
        # Simple crossover strategy
        if self.ma10[-1] > self.ma20[-1] and self.ma10[-2] <= self.ma20[-2]:
            if not self.position:
                self.buy()
        elif self.ma10[-1] < self.ma20[-1] and self.ma10[-2] >= self.ma20[-2]:
            if self.position:
                self.sell()

def test_simple_backtest():
    try:
        print("🧪 Simple Backtesting Test")
        print("=" * 30)
        
        # Fetch data
        print("📥 Fetching test data...")
        client = get_api_client()
        df = get_historical_data(client, 'EUR_USD', count=100, granularity='M5')
        
        # Format for backtesting
        df_bt = df.copy()
        df_bt.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_bt = df_bt.dropna()
        
        print(f"✓ Data ready: {len(df_bt)} candles")
        
        # Run simple backtest
        print("🚀 Running simple backtest...")
        bt = Backtest(df_bt, SimpleStrategy, cash=10000, commission=0.002)
        stats = bt.run()
        
        print(f"\n📊 Results:")
        print(f"Return: {stats['Return [%]']:.2f}%")
        print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"# Trades: {stats['# Trades']}")
        
        if stats['# Trades'] > 0:
            print(f"Win Rate: {stats['Win Rate [%]']:.1f}%")
        
        print("✅ Simple backtest completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple backtest failed: {e}")
        return False

if __name__ == "__main__":
    test_simple_backtest()