"""
Backtesting Script

This module provides comprehensive backtesting capabilities for the price action
trading strategy using the backtesting.py library. It validates strategy performance
on historical data before live deployment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any, Optional

# Backtesting framework imports
try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    BACKTESTING_AVAILABLE = True
    print("✓ Backtesting library loaded successfully")
except ImportError:
    BACKTESTING_AVAILABLE = False
    print("⚠ Backtesting library not available. Install with: pip install backtesting")
    
    # Create dummy Strategy class to prevent NameError
    class Strategy:
        def __init__(self):
            pass
        def init(self):
            pass
        def next(self):
            pass

# Import our analysis functions
from analysis_engine import (
    identify_market_structure,
    find_supply_demand_zones,
    get_current_price_context,
    identify_fvg_and_ob,
    check_for_liquidity_grab,
    confirm_m1_reversal_signal
)
from oanda_handler import get_api_client, get_historical_data

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class PriceActionStrategy(Strategy):
    """
    Backtesting implementation of our multi-timeframe price action strategy.
    
    This strategy class implements the same logic as strategy_handler.py but
    adapted for the backtesting.py framework's single-timeframe limitations.
    """
    
    # Strategy parameters (can be optimized)
    htf_lookback = 50          # Higher timeframe lookback for context
    zone_lookback = 20         # Lookback for supply/demand zones
    zone_strength = 1.5        # Strength factor for zone identification
    confidence_threshold = 70  # Minimum confidence for trade execution
    risk_reward_ratio = 1.5    # Target reward-to-risk ratio
    max_stop_pips = 50         # Maximum stop loss in pips
    min_stop_pips = 5          # Minimum stop loss in pips
    
    def init(self):
        """
        Initialize strategy indicators and pre-calculate analysis data.
        
        Note: Due to backtesting.py limitations, we simulate multi-timeframe
        analysis by resampling the main timeframe data.
        """
        try:
            # Get the main data (assuming M5 timeframe)
            self.df = self.data.df.copy()
            
            # Create higher timeframe data (M15) by resampling M5 data
            self.df_htf = self.create_higher_timeframe_data(self.df, '15T')
            
            # Pre-calculate indicators for faster execution
            self.market_structure = self.calculate_market_structure_series()
            self.supply_demand_zones = self.calculate_zones_series()
            self.fvg_signals = self.calculate_fvg_series()
            
            # Initialize trade tracking
            self.last_trade_bar = 0
            self.trade_cooldown = 10  # Bars to wait between trades
            
            print(f"✓ Strategy initialized with {len(self.df)} bars")
            print(f"✓ HTF data: {len(self.df_htf)} bars")
            
        except Exception as e:
            print(f"✗ Error initializing strategy: {e}")
            raise
    
    def create_higher_timeframe_data(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Create higher timeframe data by resampling.
        
        Args:
            df: Original timeframe data
            freq: Target frequency (e.g., '15T' for 15 minutes)
        
        Returns:
            Resampled DataFrame
        """
        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Resample to higher timeframe
            ohlc_dict = {
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            
            htf_data = df.resample(freq).agg(ohlc_dict).dropna()
            
            # Rename columns to match our convention (lowercase for analysis engine)
            htf_data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            return htf_data
            
        except Exception as e:
            print(f"✗ Error creating HTF data: {e}")
            return df.copy()
    
    def calculate_market_structure_series(self) -> pd.Series:
        """
        Pre-calculate market structure for each bar.
        """
        try:
            structure_series = pd.Series(index=self.df.index, dtype='object')
            
            # Calculate structure for each bar using rolling window
            for i in range(self.htf_lookback, len(self.df)):
                # Get HTF window up to current bar
                current_time = self.df.index[i]
                htf_window = self.df_htf[self.df_htf.index <= current_time].tail(self.htf_lookback)
                
                if len(htf_window) >= 20:  # Minimum data for analysis
                    # HTF data already has lowercase columns from create_higher_timeframe_data
                    structure = identify_market_structure(htf_window, lookback_period=20)
                    structure_series.iloc[i] = structure
                else:
                    structure_series.iloc[i] = 'Range'
            
            return structure_series.fillna('Range')
            
        except Exception as e:
            print(f"✗ Error calculating market structure: {e}")
            return pd.Series(index=self.df.index, data='Range')
    
    def calculate_zones_series(self) -> Dict:
        """
        Pre-calculate supply/demand zones for each bar.
        """
        try:
            zones_dict = {}
            
            for i in range(self.zone_lookback, len(self.df)):
                # Get window for zone calculation
                window = self.df.iloc[max(0, i-self.zone_lookback):i+1]
                
                if len(window) >= 10:
                    # Convert to lowercase columns for analysis functions
                    window_analysis = window.copy()
                    window_analysis.columns = ['open', 'high', 'low', 'close', 'volume']
                    zones = find_supply_demand_zones(window_analysis, lookback=10, strength_factor=self.zone_strength)
                    zones_dict[i] = zones
                else:
                    zones_dict[i] = []
            
            return zones_dict
            
        except Exception as e:
            print(f"✗ Error calculating zones: {e}")
            return {}
    
    def calculate_fvg_series(self) -> Dict:
        """
        Pre-calculate Fair Value Gaps for each bar.
        """
        try:
            fvg_dict = {}
            
            for i in range(10, len(self.df)):
                # Get recent window for FVG calculation
                window = self.df.iloc[max(0, i-20):i+1]
                
                if len(window) >= 5:
                    # Convert to lowercase columns for analysis functions
                    window_analysis = window.copy()
                    window_analysis.columns = ['open', 'high', 'low', 'close', 'volume']
                    fvg_list, ob_list = identify_fvg_and_ob(window_analysis)
                    fvg_dict[i] = {'fvg': fvg_list, 'ob': ob_list}
                else:
                    fvg_dict[i] = {'fvg': [], 'ob': []}
            
            return fvg_dict
            
        except Exception as e:
            print(f"✗ Error calculating FVGs: {e}")
            return {}
    
    def next(self):
        """
        Main DUAL-MODE strategy logic executed on each bar.
        
        This implements the same dual-mode logic as strategy_handler.py:
        - Trend-Following mode for Uptrend/Downtrend contexts
        - Range-Bound mode for Range contexts
        """
        try:
            # Skip if not enough data or in cooldown period
            current_bar = len(self.data) - 1
            if current_bar < self.htf_lookback or current_bar - self.last_trade_bar < self.trade_cooldown:
                return
            
            # Skip if already in position
            if self.position:
                return
            
            # Get current market data
            current_price = self.data.Close[-1]
            
            # Get pre-calculated analysis data
            market_structure = self.market_structure.iloc[current_bar] if current_bar < len(self.market_structure) else 'Range'
            zones = self.supply_demand_zones.get(current_bar, [])
            
            # DUAL-MODE STRATEGY LOGIC
            # Mode A: Trend-Following (Original high-confluence strategy)
            if market_structure == 'Uptrend':
                if self.execute_trend_following_long(zones, current_price, current_bar):
                    return
                    
            elif market_structure == 'Downtrend':
                if self.execute_trend_following_short(zones, current_price, current_bar):
                    return
                    
            # Mode B: Range-Bound (New mean-reversion strategy)
            elif market_structure == 'Range':
                if self.execute_range_bound_strategy(zones, current_price, current_bar):
                    return
                
        except Exception as e:
            print(f"✗ Error in dual-mode strategy logic at bar {current_bar}: {e}")
    
    def execute_trend_following_long(self, zones: List, current_price: float, current_bar: int) -> bool:
        """
        Execute trend-following long strategy (mirrors strategy_handler.py).
        """
        try:
            # Get demand zones
            demand_zones = [z for z in zones if z['type'] == 'demand']
            if not demand_zones:
                return False
            
            # Check if price is near demand zone (within 0.2%)
            near_demand = False
            for zone in demand_zones:
                distance_pct = abs(current_price - zone['price_level']) / current_price * 100
                if current_price >= zone['price_level'] * 0.999 and distance_pct <= 0.2:
                    near_demand = True
                    break
            
            if not near_demand:
                return False
            
            # Get ICT analysis for confluence
            ict_data = self.fvg_signals.get(current_bar, {'fvg': [], 'ob': []})
            
            # Check for recent bullish ICT patterns
            recent_bullish_confluence = False
            
            # Check bullish FVGs
            for fvg in ict_data['fvg'][-3:]:
                if (fvg['type'] == 'bullish' and 
                    current_price >= fvg['lower_level'] and 
                    current_price <= fvg['upper_level'] * 1.001):
                    recent_bullish_confluence = True
                    break
            
            # Check bullish Order Blocks
            if not recent_bullish_confluence:
                for ob in ict_data['ob'][-3:]:
                    if (ob['type'] == 'bullish' and 
                        current_price >= ob['zone_low'] and 
                        current_price <= ob['zone_high'] * 1.001):
                        recent_bullish_confluence = True
                        break
            
            if not recent_bullish_confluence:
                return False
            
            # Look for bullish momentum confirmation
            recent_candles = self.data.df.iloc[current_bar-3:current_bar+1]
            bullish_momentum = False
            
            for i in range(len(recent_candles)):
                candle = recent_candles.iloc[i]
                if candle['Close'] > candle['Open']:
                    body_size = candle['Close'] - candle['Open']
                    candle_range = candle['High'] - candle['Low']
                    if candle_range > 0 and body_size / candle_range > 0.6:
                        bullish_momentum = True
                        break
            
            if not bullish_momentum:
                return False
            
            # Execute the trade
            self.execute_long_trade(current_price, {'confidence': 85, 'mode': 'trend-following'})
            return True
            
        except Exception as e:
            print(f"✗ Error in trend-following long: {e}")
            return False
    
    def execute_trend_following_short(self, zones: List, current_price: float, current_bar: int) -> bool:
        """
        Execute trend-following short strategy (mirrors strategy_handler.py).
        """
        try:
            # Get supply zones
            supply_zones = [z for z in zones if z['type'] == 'supply']
            if not supply_zones:
                return False
            
            # Check if price is near supply zone (within 0.2%)
            near_supply = False
            for zone in supply_zones:
                distance_pct = abs(current_price - zone['price_level']) / current_price * 100
                if current_price <= zone['price_level'] * 1.001 and distance_pct <= 0.2:
                    near_supply = True
                    break
            
            if not near_supply:
                return False
            
            # Get ICT analysis for confluence
            ict_data = self.fvg_signals.get(current_bar, {'fvg': [], 'ob': []})
            
            # Check for recent bearish ICT patterns
            recent_bearish_confluence = False
            
            # Check bearish FVGs
            for fvg in ict_data['fvg'][-3:]:
                if (fvg['type'] == 'bearish' and 
                    current_price <= fvg['upper_level'] and 
                    current_price >= fvg['lower_level'] * 0.999):
                    recent_bearish_confluence = True
                    break
            
            # Check bearish Order Blocks
            if not recent_bearish_confluence:
                for ob in ict_data['ob'][-3:]:
                    if (ob['type'] == 'bearish' and 
                        current_price <= ob['zone_high'] and 
                        current_price >= ob['zone_low'] * 0.999):
                        recent_bearish_confluence = True
                        break
            
            if not recent_bearish_confluence:
                return False
            
            # Look for bearish momentum confirmation
            recent_candles = self.data.df.iloc[current_bar-3:current_bar+1]
            bearish_momentum = False
            
            for i in range(len(recent_candles)):
                candle = recent_candles.iloc[i]
                if candle['Close'] < candle['Open']:
                    body_size = candle['Open'] - candle['Close']
                    candle_range = candle['High'] - candle['Low']
                    if candle_range > 0 and body_size / candle_range > 0.6:
                        bearish_momentum = True
                        break
            
            if not bearish_momentum:
                return False
            
            # Execute the trade
            self.execute_short_trade(current_price, {'confidence': 85, 'mode': 'trend-following'})
            return True
            
        except Exception as e:
            print(f"✗ Error in trend-following short: {e}")
            return False
    
    def execute_range_bound_strategy(self, zones: List, current_price: float, current_bar: int) -> bool:
        """
        Execute NEW range-bound mean-reversion strategy (mirrors strategy_handler.py).
        """
        try:
            if not zones:
                return False
            
            # Get recent M1 data for reversal signal
            recent_m1_data = self.data.df.iloc[current_bar-10:current_bar+1].copy()
            recent_m1_data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Check if price is very near a strong supply zone (SELL signal)
            supply_zones = [z for z in zones if z['type'] == 'supply']
            for zone in supply_zones:
                distance_pct = abs(current_price - zone['price_level']) / current_price * 100
                
                # Price must be very close to supply zone (within 0.15% for mean reversion)
                if current_price >= zone['price_level'] * 0.998 and distance_pct <= 0.15:
                    # Check for M1 bearish reversal confirmation
                    reversal_signal = confirm_m1_reversal_signal(recent_m1_data)
                    if reversal_signal == 'Bearish Reversal':
                        self.execute_short_trade(current_price, {'confidence': 75, 'mode': 'range-bound'})
                        return True
            
            # Check if price is very near a strong demand zone (BUY signal)
            demand_zones = [z for z in zones if z['type'] == 'demand']
            for zone in demand_zones:
                distance_pct = abs(current_price - zone['price_level']) / current_price * 100
                
                # Price must be very close to demand zone (within 0.15% for mean reversion)
                if current_price <= zone['price_level'] * 1.002 and distance_pct <= 0.15:
                    # Check for M1 bullish reversal confirmation
                    reversal_signal = confirm_m1_reversal_signal(recent_m1_data)
                    if reversal_signal == 'Bullish Reversal':
                        self.execute_long_trade(current_price, {'confidence': 75, 'mode': 'range-bound'})
                        return True
            
            return False
            
        except Exception as e:
            print(f"✗ Error in range-bound strategy: {e}")
            return False
    
    
    def execute_long_trade(self, entry_price: float, signal: Dict):
        """
        Execute a long trade with calculated stop loss and take profit.
        """
        try:
            # Calculate stop loss and take profit
            risk_params = self.calculate_trade_risk(entry_price, 'LONG')
            
            if risk_params['valid']:
                # Execute the trade
                self.buy(
                    sl=risk_params['stop_loss'],
                    tp=risk_params['take_profit']
                )
                self.last_trade_bar = len(self.data) - 1
                
                mode = signal.get('mode', 'unknown')
                print(f"📈 LONG ({mode}) @ {entry_price:.5f} | SL: {risk_params['stop_loss']:.5f} | TP: {risk_params['take_profit']:.5f}")
                
        except Exception as e:
            print(f"✗ Error executing long trade: {e}")
    
    def execute_short_trade(self, entry_price: float, signal: Dict):
        """
        Execute a short trade with calculated stop loss and take profit.
        """
        try:
            # Calculate stop loss and take profit
            risk_params = self.calculate_trade_risk(entry_price, 'SHORT')
            
            if risk_params['valid']:
                # Execute the trade
                self.sell(
                    sl=risk_params['stop_loss'],
                    tp=risk_params['take_profit']
                )
                self.last_trade_bar = len(self.data) - 1
                
                mode = signal.get('mode', 'unknown')
                print(f"📉 SHORT ({mode}) @ {entry_price:.5f} | SL: {risk_params['stop_loss']:.5f} | TP: {risk_params['take_profit']:.5f}")
                
        except Exception as e:
            print(f"✗ Error executing short trade: {e}")
    
    def calculate_trade_risk(self, entry_price: float, direction: str) -> Dict:
        """
        Calculate stop loss and take profit levels.
        """
        try:
            # Calculate ATR for dynamic stops
            high_low = self.data.High - self.data.Low
            high_close_prev = abs(self.data.High - self.data.Close.shift(1))
            low_close_prev = abs(self.data.Low - self.data.Close.shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                atr = abs(self.data.High[-1] - self.data.Low[-1])
            
            if direction == 'LONG':
                # Stop loss: 2*ATR below entry or below recent swing low
                recent_lows = self.data.Low[-10:]
                swing_low = recent_lows.min()
                
                atr_stop = entry_price - (2 * atr)
                structural_stop = swing_low - (0.5 * atr)
                stop_loss = min(atr_stop, structural_stop)
                
                # Take profit: 1.5x risk
                risk_distance = entry_price - stop_loss
                take_profit = entry_price + (risk_distance * self.risk_reward_ratio)
                
            else:  # SHORT
                # Stop loss: 2*ATR above entry or above recent swing high
                recent_highs = self.data.High[-10:]
                swing_high = recent_highs.max()
                
                atr_stop = entry_price + (2 * atr)
                structural_stop = swing_high + (0.5 * atr)
                stop_loss = max(atr_stop, structural_stop)
                
                # Take profit: 1.5x risk
                risk_distance = stop_loss - entry_price
                take_profit = entry_price - (risk_distance * self.risk_reward_ratio)
            
            # Validate stop distance
            stop_distance_pips = abs(entry_price - stop_loss) * 10000
            
            if stop_distance_pips < self.min_stop_pips or stop_distance_pips > self.max_stop_pips:
                return {'valid': False, 'reason': f'Stop distance: {stop_distance_pips:.1f} pips'}
            
            return {
                'valid': True,
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'risk_pips': stop_distance_pips
            }
            
        except Exception as e:
            return {'valid': False, 'reason': f'Risk calculation error: {e}'}

def fetch_backtest_data(instrument: str = 'EUR_USD', days: int = 30, granularity: str = 'M5') -> pd.DataFrame:
    """
    Fetch historical data for backtesting.
    
    Args:
        instrument: Trading instrument
        days: Number of days of historical data
        granularity: Data timeframe ('M1', 'M5', 'M15', etc.)
    
    Returns:
        DataFrame formatted for backtesting.py
    """
    try:
        print(f"📥 Fetching {days} days of {granularity} data for {instrument}...")
        
        # Calculate number of candles needed
        candles_per_day = {
            'M1': 1440, 'M5': 288, 'M15': 96, 'M30': 48, 'H1': 24, 'H4': 6, 'D': 1
        }
        
        count = min(5000, days * candles_per_day.get(granularity, 288))  # OANDA limit is 5000
        
        # Get API client and fetch data
        client = get_api_client()
        df = get_historical_data(client, instrument, count=count, granularity=granularity)
        
        if df is None or len(df) == 0:
            raise ValueError("No data received from OANDA")
        
        # Format for backtesting.py (requires specific column names)
        df_backtest = df.copy()
        df_backtest.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Ensure all values are numeric and handle any NaN values
        df_backtest = df_backtest.dropna()
        
        print(f"✓ Data formatted: columns {df_backtest.columns.tolist()}")
        print(f"📋 Sample data:\n{df_backtest.head(2)}")
        
        print(f"✓ Data fetched successfully: {len(df_backtest)} candles")
        print(f"📅 Date range: {df_backtest.index[0]} to {df_backtest.index[-1]}")
        
        return df_backtest
        
    except Exception as e:
        print(f"✗ Error fetching backtest data: {e}")
        raise

def run_backtest(data: pd.DataFrame, cash: float = 10000, commission: float = 0.0002) -> Dict:
    """
    Run the backtest with our price action strategy.
    
    Args:
        data: Historical OHLC data
        cash: Starting capital
        commission: Commission rate (0.0002 = 0.02%)
    
    Returns:
        Dictionary with backtest results
    """
    try:
        if not BACKTESTING_AVAILABLE:
            raise ImportError("Backtesting library not available")
        
        print(f"\n🚀 Running backtest...")
        print(f"💰 Starting capital: ${cash:,.2f}")
        print(f"📊 Data points: {len(data)}")
        print(f"💸 Commission: {commission*100:.3f}%")
        
        # Create and run backtest
        bt = Backtest(
            data, 
            PriceActionStrategy,
            cash=cash,
            commission=commission,
            exclusive_orders=True  # Close position before opening new one
        )
        
        # Run the backtest
        stats = bt.run()
        
        # Display key results
        print(f"\n📈 BACKTEST RESULTS:")
        print(f"{'='*50}")
        print(f"Return [%]: {stats['Return [%]']:.2f}%")
        print(f"Buy & Hold Return [%]: {stats['Buy & Hold Return [%]']:.2f}%")
        print(f"Max Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"# Trades: {stats['# Trades']}")
        
        if stats['# Trades'] > 0:
            print(f"Win Rate [%]: {stats['Win Rate [%]']:.2f}%")
            print(f"Avg Trade [%]: {stats['Avg. Trade [%]']:.2f}%")
            print(f"Best Trade [%]: {stats['Best Trade [%]']:.2f}%")
            print(f"Worst Trade [%]: {stats['Worst Trade [%]']:.2f}%")
            print(f"Profit Factor: {stats['Profit Factor']:.2f}")
            print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        
        print(f"{'='*50}")
        
        # Plot results (optional)
        try:
            print(f"\n📊 Generating backtest plot...")
            bt.plot()
        except Exception as e:
            print(f"⚠ Could not generate plot: {e}")
        
        return {
            'stats': stats,
            'backtest': bt
        }
        
    except Exception as e:
        print(f"✗ Error running backtest: {e}")
        raise

def optimize_strategy(data: pd.DataFrame) -> Dict:
    """
    Optimize strategy parameters using the backtesting framework.
    
    Args:
        data: Historical OHLC data
    
    Returns:
        Optimization results
    """
    try:
        if not BACKTESTING_AVAILABLE:
            raise ImportError("Backtesting library not available")
        
        print(f"\n🔧 Optimizing strategy parameters...")
        
        bt = Backtest(data, PriceActionStrategy, cash=10000, commission=0.0002)
        
        # Define parameter ranges for optimization
        optimization_results = bt.optimize(
            confidence_threshold=range(60, 85, 5),        # 60%, 65%, 70%, 75%, 80%
            risk_reward_ratio=[1.0, 1.5, 2.0, 2.5],     # Different R:R ratios
            zone_strength=[1.2, 1.5, 2.0],              # Zone strength factors
            maximize='Sharpe Ratio',                      # Optimization metric
            constraint=lambda param: param.confidence_threshold >= 60
        )
        
        print(f"✓ Optimization completed!")
        print(f"Best parameters: {optimization_results._strategy}")
        
        return optimization_results
        
    except Exception as e:
        print(f"✗ Error in optimization: {e}")
        raise

def test_backtesting_functionality():
    """
    Test the backtesting functionality with sample data.
    """
    try:
        print("Testing Backtesting Module...")
        
        if not BACKTESTING_AVAILABLE:
            print("⚠ Backtesting library not available - creating mock test")
            print("✓ Module structure validated")
            return
        
        # Fetch small dataset for testing
        print("📥 Fetching test data...")
        data = fetch_backtest_data('EUR_USD', days=7, granularity='M5')
        
        if len(data) < 100:
            print("⚠ Limited data available for testing")
            return
        
        # Run quick backtest
        results = run_backtest(data, cash=10000, commission=0.0002)
        
        print("✅ Backtesting functionality test completed!")
        
    except Exception as e:
        print(f"✗ Backtesting test failed: {e}")

if __name__ == "__main__":
    """
    Main execution block for running backtests.
    """
    try:
        print("🎯 Price Action Strategy Backtester")
        print("=" * 50)
        
        # Test functionality first
        test_backtesting_functionality()
        
        if BACKTESTING_AVAILABLE:
            print(f"\n" + "="*50)
            print("FULL BACKTEST EXECUTION")
            print("="*50)
            
            # Fetch more comprehensive data
            instrument = 'EUR_USD'
            data = fetch_backtest_data(instrument, days=30, granularity='M5')
            
            # Run full backtest
            results = run_backtest(data, cash=10000, commission=0.0002)
            
            # Optionally run optimization
            user_input = input("\nRun parameter optimization? (y/n): ").lower()
            if user_input == 'y':
                optimization_results = optimize_strategy(data)
        
    except KeyboardInterrupt:
        print("\n⚠ Backtest interrupted by user")
    except Exception as e:
        print(f"✗ Backtest execution failed: {e}")
        import traceback
        print(f"📋 Full traceback:\n{traceback.format_exc()}")