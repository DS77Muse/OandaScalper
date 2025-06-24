"""
MeanReversionSR Strategy Handler Module

This module implements the MeanReversionSR strategy - a mean reversion strategy
that looks for bullish reversal patterns at oversold support levels.

Strategy Parameters:
- Type: Mean reversion at support levels
- Entry: Bullish reversal at oversold support
- Timeframe: M5 (5-minute candles)
- RSI Threshold: < 35 (oversold)
- Take Profit: Entry + 3×ATR(14) (fallback: 0.5% above entry)
- Stop Loss: 0.2% below support candle low (fallback: 0.5% below entry)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import traceback

# Import our custom modules
from oanda_handler import get_api_client, get_historical_data, place_market_order, get_account_summary, get_open_trades_from_oanda
from journal import log_new_trade, get_open_trades

# Import for logging
from loguru import logger

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index)
    
    Args:
        data: Price data series (typically close prices)
        period: RSI period (default 14)
    
    Returns:
        RSI values as pandas Series
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ATR (Average True Range)
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period (default 14)
    
    Returns:
        ATR values as pandas Series
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def identify_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Identify support and resistance levels using rolling window approach
    
    Args:
        df: DataFrame with OHLC data
        window: Window size for identifying pivots
    
    Returns:
        Tuple of (support_levels, resistance_levels) as boolean Series
    """
    # Rolling window to identify local minima (support) and maxima (resistance)
    support = df['low'] == df['low'].rolling(window=window, center=True).min()
    resistance = df['high'] == df['high'].rolling(window=window, center=True).max()
    
    # Fill NaN values with False
    support = support.fillna(False)
    resistance = resistance.fillna(False)
    
    return support, resistance

def detect_hammer_pattern(df: pd.DataFrame) -> pd.Series:
    """
    Detect hammer candlestick pattern
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Boolean Series indicating hammer patterns
    """
    body_size = np.abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    lower_shadow = np.minimum(df['open'], df['close']) - df['low']
    upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
    
    # Hammer: long lower shadow (2x body), small upper shadow
    hammer = (
        (lower_shadow > 2 * body_size) & 
        (upper_shadow < body_size) & 
        (candle_range > 0)
    )
    
    return hammer

def detect_bullish_pattern(df: pd.DataFrame) -> pd.Series:
    """
    Detect basic bullish candlestick patterns
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Boolean Series indicating bullish patterns
    """
    # Simple bullish candle: close > open
    bullish = df['close'] > df['open']
    
    # Additional criteria: body size > 50% of candle range
    body_size = np.abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    
    strong_bullish = bullish & (body_size > 0.5 * candle_range)
    
    return strong_bullish

def run_strategy_check(client, instrument: str) -> bool:
    """
    Main MeanReversionSR strategy function.
    
    Entry Conditions:
    1. Support Level: Must be at identified support level
    2. RSI Oversold: RSI(14) < 35
    3. Reversal Pattern: Hammer OR bullish candlestick pattern
    
    Risk Management:
    - Stop Loss: 0.2% below support candle low (fallback: 0.5% below entry)
    - Take Profit: Entry + 3×ATR(14) (fallback: 0.5% above entry)
    - Minimum Spread: 0.1% validation
    
    Args:
        client: Authenticated OANDA API client
        instrument (str): Trading instrument (e.g., 'EUR_USD')
    
    Returns:
        bool: True if trade was executed, False otherwise
    """
    
    try:
        logger.info(f"MeanReversionSR strategy check initiated for {instrument}")
        
        print(f"\n{'='*60}")
        print(f"MEANREVERSIONSR STRATEGY CHECK: {instrument}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # ==================================================================
        # STEP 1: FETCH M5 DATA (5-minute candles)
        # ==================================================================
        print("\n📊 STEP 1: Fetching M5 data...")
        
        # Fetch M5 data (200 candles for proper indicator calculation)
        df_m5 = get_historical_data(client, instrument, count=200, granularity='M5')
        if df_m5 is None or len(df_m5) < 50:
            logger.error(f"Insufficient M5 data for {instrument}")
            print("✗ Insufficient M5 data for analysis")
            return False
        
        print(f"✓ M5 data fetched successfully: {len(df_m5)} candles")
        
        # ==================================================================
        # STEP 2: CHECK FOR EXISTING POSITIONS
        # ==================================================================
        print("\n🔍 STEP 2: Checking for existing positions...")
        
        try:
            oanda_open_trades = get_open_trades_from_oanda(client)
            instrument_open_trades = [t for t in oanda_open_trades if t['instrument'] == instrument]
            
            if instrument_open_trades:
                logger.info(f"Skipping {instrument} - existing trades found")
                print(f"⚠ Skipping {instrument} - {len(instrument_open_trades)} open trade(s) already exist")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to check OANDA positions: {e}")
            print(f"⚠ Warning: Could not verify OANDA positions")
        
        # ==================================================================
        # STEP 3: CALCULATE TECHNICAL INDICATORS
        # ==================================================================
        print("\n📈 STEP 3: Calculating technical indicators...")
        
        # Calculate RSI(14)
        df_m5['rsi'] = calculate_rsi(df_m5['close'], period=14)
        
        # Calculate ATR(14)
        df_m5['atr'] = calculate_atr(df_m5, period=14)
        
        # Identify support and resistance levels
        support_levels, resistance_levels = identify_support_resistance(df_m5, window=20)
        df_m5['is_support'] = support_levels
        df_m5['is_resistance'] = resistance_levels
        
        # Detect candlestick patterns
        df_m5['is_hammer'] = detect_hammer_pattern(df_m5)
        df_m5['is_bullish'] = detect_bullish_pattern(df_m5)
        
        print("✓ Technical indicators calculated successfully")
        
        # ==================================================================
        # STEP 4: CHECK ENTRY CONDITIONS
        # ==================================================================
        print("\n🎯 STEP 4: Checking entry conditions...")
        
        # Get current data (latest candle)
        current_idx = len(df_m5) - 1
        current_candle = df_m5.iloc[current_idx]
        current_price = current_candle['close']
        
        # Check if we have valid indicator values
        if pd.isna(current_candle['rsi']) or pd.isna(current_candle['atr']):
            print("❌ Invalid indicator values (RSI or ATR is NaN)")
            return False
        
        print(f"Current price: {current_price:.5f}")
        print(f"Current RSI: {current_candle['rsi']:.2f}")
        print(f"Current ATR: {current_candle['atr']:.5f}")
        
        # Entry Condition 1: Must be at support level
        if not current_candle['is_support']:
            print("❌ Not at support level")
            return False
        
        print("✓ Condition 1: At support level")
        
        # Entry Condition 2: RSI < 35 (oversold)
        if current_candle['rsi'] >= 35:
            print(f"❌ RSI not oversold ({current_candle['rsi']:.2f} >= 35)")
            return False
        
        print(f"✓ Condition 2: RSI oversold ({current_candle['rsi']:.2f} < 35)")
        
        # Entry Condition 3: Hammer OR bullish pattern
        has_reversal_pattern = current_candle['is_hammer'] or current_candle['is_bullish']
        if not has_reversal_pattern:
            print("❌ No reversal pattern (hammer or bullish)")
            return False
        
        pattern_type = "Hammer" if current_candle['is_hammer'] else "Bullish"
        print(f"✓ Condition 3: {pattern_type} reversal pattern detected")
        
        # ==================================================================
        # STEP 5: CALCULATE RISK MANAGEMENT
        # ==================================================================
        print("\n💼 STEP 5: Calculating risk management...")
        
        entry_price = current_price
        
        # Stop Loss: 0.2% below support candle low
        support_low = current_candle['low']
        sl_percent = support_low * 0.998  # 0.2% below support low
        sl_fallback = entry_price * 0.995  # 0.5% below entry (fallback)
        
        stop_loss = min(sl_percent, sl_fallback)  # Use more conservative stop
        
        # Take Profit: Entry + 3×ATR(14)
        tp_atr = entry_price + (3 * current_candle['atr'])
        tp_fallback = entry_price * 1.005  # 0.5% above entry (fallback)
        
        # Use ATR-based TP if it's reasonable, otherwise use fallback
        min_spread = entry_price * 0.001  # 0.1% minimum spread
        if tp_atr - entry_price >= min_spread:
            take_profit = tp_atr
        else:
            take_profit = tp_fallback
        
        # Validate minimum spread (0.1%)
        if take_profit - entry_price < min_spread:
            print(f"❌ Insufficient spread ({((take_profit - entry_price) / entry_price * 100):.3f}% < 0.1%)")
            return False
        
        # Validate proper risk/reward structure
        if not (stop_loss < entry_price < take_profit):
            print(f"❌ Invalid risk/reward structure: SL={stop_loss:.5f}, Entry={entry_price:.5f}, TP={take_profit:.5f}")
            return False
        
        risk_pips = (entry_price - stop_loss) * 10000
        reward_pips = (take_profit - entry_price) * 10000
        risk_reward_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
        
        print(f"✓ Risk management calculated:")
        print(f"  Entry: {entry_price:.5f}")
        print(f"  Stop Loss: {stop_loss:.5f} ({risk_pips:.1f} pips)")
        print(f"  Take Profit: {take_profit:.5f} ({reward_pips:.1f} pips)")
        print(f"  Risk/Reward Ratio: 1:{risk_reward_ratio:.2f}")
        
        # ==================================================================
        # STEP 6: POSITION SIZING
        # ==================================================================
        print("\n💰 STEP 6: Calculating position size...")
        
        # Get account balance
        account_summary = get_account_summary(client)
        account_balance = float(account_summary.get('balance', 10000))
        
        # Risk 0.5% of account per trade
        risk_amount = account_balance * 0.005
        
        # Calculate position size based on risk
        if risk_pips > 0:
            # Position size calculation for forex
            position_size = int((risk_amount / risk_pips) * 10000)
            
            # Apply position size limits
            max_position = 50000  # Maximum position size
            min_position = 1000   # Minimum position size
            
            position_size = max(min_position, min(position_size, max_position))
        else:
            position_size = 10000  # Default position size
        
        print(f"✓ Position sizing:")
        print(f"  Account Balance: ${account_balance:,.2f}")
        print(f"  Risk Amount: ${risk_amount:.2f}")
        print(f"  Position Size: {position_size:,} units")
        
        # ==================================================================
        # STEP 7: EXECUTE TRADE
        # ==================================================================
        print("\n🚀 STEP 7: Executing trade...")
        
        units = position_size  # Long position
        
        # Execute the trade
        order_response = place_market_order(
            client=client,
            instrument=instrument,
            units=units,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit
        )
        
        if order_response and 'orderFillTransaction' in order_response:
            fill_transaction = order_response['orderFillTransaction']
            trade_id = fill_transaction.get('tradeOpened', {}).get('tradeID', 'N/A')
            fill_price = float(fill_transaction.get('price', entry_price))
            
            entry_reason = f"MeanReversionSR: Support level + RSI oversold ({current_candle['rsi']:.1f}) + {pattern_type} pattern"
            
            # Log to journal
            log_success = log_new_trade(
                db_name='trading_journal.db',
                trade_id=int(trade_id) if trade_id != 'N/A' else 0,
                instrument=instrument,
                units=units,
                direction='LONG',
                entry_price=fill_price,
                sl_price=stop_loss,
                tp_price=take_profit,
                reason=entry_reason
            )
            
            if log_success:
                logger.info(f"MeanReversionSR trade executed successfully for {instrument}")
                print(f"✅ Trade executed and logged successfully!")
                print(f"📋 Trade ID: {trade_id}")
                print(f"📋 Fill Price: {fill_price:.5f}")
                print(f"📋 Entry Reason: {entry_reason}")
                return True
            else:
                logger.warning(f"Trade executed but logging failed for {instrument}")
                print(f"⚠ Trade executed but logging failed")
                return True
        else:
            logger.error(f"Trade execution failed for {instrument}")
            print(f"❌ Trade execution failed")
            return False
            
    except Exception as e:
        logger.error(f"Error in MeanReversionSR strategy for {instrument}: {e}")
        print(f"❌ Error in strategy check for {instrument}: {e}")
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return False

def test_strategy_handler():
    """
    Test the MeanReversionSR strategy handler with a demo instrument.
    """
    try:
        print("Testing MeanReversionSR Strategy Handler...")
        
        # Get API client
        client = get_api_client()
        
        # Test with EUR_USD
        result = run_strategy_check(client, 'EUR_USD')
        
        print(f"\n✅ MeanReversionSR strategy test completed. Trade executed: {result}")
        
    except Exception as e:
        print(f"❌ MeanReversionSR strategy test failed: {e}")

if __name__ == "__main__":
    # Run test when script is executed directly
    test_strategy_handler()