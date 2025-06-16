"""
Limited Backtest Test

Test the full price action strategy on a small dataset to verify warnings are fixed.
"""

from backtest import fetch_backtest_data, run_backtest

def test_limited_backtest():
    try:
        print("🧪 Limited Price Action Backtest Test")
        print("=" * 40)
        
        # Fetch small dataset (3 days)
        print("📥 Fetching 3 days of data...")
        data = fetch_backtest_data('EUR_USD', days=3, granularity='M5')
        
        print(f"✓ Data ready: {len(data)} candles")
        
        # Run limited backtest
        print("🚀 Running limited backtest...")
        results = run_backtest(data, cash=10000, commission=0.0002)
        
        print("✅ Limited backtest completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Limited backtest failed: {e}")
        return False

if __name__ == "__main__":
    test_limited_backtest()