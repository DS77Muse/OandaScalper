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
        # Compact status tracking
        status_parts = []
        
        # STEP 1: Fetch data
        df_m5 = get_historical_data(client, instrument, count=200, granularity='M5')
        if df_m5 is None or len(df_m5) < 50:
            print(f"{instrument:8} | Data:❌ Insufficient M5 data")
            return False
        status_parts.append("Data:✓")
        
        # STEP 2: Check existing positions
        try:
            oanda_open_trades = get_open_trades_from_oanda(client)
            instrument_open_trades = [t for t in oanda_open_trades if t['instrument'] == instrument]
            
            if instrument_open_trades:
                print(f"{instrument:8} | Data:✓ | Pos:Skip ({len(instrument_open_trades)} open)")
                return False
            status_parts.append("Pos:✓")
        except Exception as e:
            status_parts.append("Pos:⚠")
        
        # STEP 3: Calculate indicators
        df_m5['rsi'] = calculate_rsi(df_m5['close'], period=14)
        df_m5['atr'] = calculate_atr(df_m5, period=14)
        support_levels, resistance_levels = identify_support_resistance(df_m5, window=20)
        df_m5['is_support'] = support_levels
        df_m5['is_resistance'] = resistance_levels
        df_m5['is_hammer'] = detect_hammer_pattern(df_m5)
        df_m5['is_bullish'] = detect_bullish_pattern(df_m5)
        status_parts.append("Calc:✓")
        
        # Get current data
        current_idx = len(df_m5) - 1
        current_candle = df_m5.iloc[current_idx]
        current_price = current_candle['close']
        
        if pd.isna(current_candle['rsi']) or pd.isna(current_candle['atr']):
            print(f"{instrument:8} | {' | '.join(status_parts)} | Signal:❌ Invalid indicators")
            return False
        
        # STEP 4: Check entry conditions
        rsi_val = current_candle['rsi']
        at_support = current_candle['is_support']
        has_pattern = current_candle['is_hammer'] or current_candle['is_bullish']
        pattern_type = "H" if current_candle['is_hammer'] else "B" if current_candle['is_bullish'] else "N"
        
        # Check all conditions
        conditions_met = at_support and (rsi_val < 35) and has_pattern
        
        if not conditions_met:
            reasons = []
            if not at_support: reasons.append("NoSupport")
            if rsi_val >= 35: reasons.append(f"RSI{rsi_val:.0f}")
            if not has_pattern: reasons.append("NoPattern")
            
            print(f"{instrument:8} | {' | '.join(status_parts)} | Signal:❌ {','.join(reasons)} | RSI:{rsi_val:.1f} Price:{current_price:.5f}")
            return False
        
        # All conditions met - proceed with trade setup
        status_parts.append(f"Signal:✓({pattern_type},RSI{rsi_val:.0f})")
        
        # STEP 5: Calculate risk management
        entry_price = current_price
        support_low = current_candle['low']
        sl_percent = support_low * 0.998
        sl_fallback = entry_price * 0.995
        stop_loss = min(sl_percent, sl_fallback)
        
        tp_atr = entry_price + (3 * current_candle['atr'])
        tp_fallback = entry_price * 1.005
        min_spread = entry_price * 0.001
        
        if tp_atr - entry_price >= min_spread:
            take_profit = tp_atr
        else:
            take_profit = tp_fallback
        
        # Validate
        if take_profit - entry_price < min_spread or not (stop_loss < entry_price < take_profit):
            print(f"{instrument:8} | {' | '.join(status_parts)} | Risk:❌ Invalid spread/structure")
            return False
        
        risk_pips = (entry_price - stop_loss) * 10000
        reward_pips = (take_profit - entry_price) * 10000
        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
        
        status_parts.append(f"Risk:✓({risk_pips:.0f}p,RR{rr_ratio:.1f})")
        
        # STEP 6: Position sizing
        account_summary = get_account_summary(client)
        account_balance = float(account_summary.get('balance', 10000))
        risk_amount = account_balance * 0.005
        
        if risk_pips > 0:
            position_size = int((risk_amount / risk_pips) * 10000)
            position_size = max(1000, min(position_size, 50000))
        else:
            position_size = 10000
        
        status_parts.append(f"Size:✓({position_size:,})")
        
        # STEP 7: Execute trade
        units = position_size
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
            
            entry_reason = f"MeanReversionSR: Support+RSI{rsi_val:.0f}+{pattern_type}"
            
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
            
            # Trade executed successfully - new line with details
            print(f"{instrument:8} | {' | '.join(status_parts)} | Exec:✓")
            print(f"🎯 TRADE EXECUTED: {instrument} LONG {units:,} units @ {fill_price:.5f} | SL:{stop_loss:.5f} TP:{take_profit:.5f} | ID:{trade_id} | {entry_reason}")
            
            logger.info(f"MeanReversionSR trade executed successfully for {instrument}")
            return True
        else:
            print(f"{instrument:8} | {' | '.join(status_parts)} | Exec:❌ Order failed")
            return False
            
    except Exception as e:
        logger.error(f"Error in MeanReversionSR strategy for {instrument}: {e}")
        print(f"{instrument:8} | Error:❌ {str(e)[:50]}...")
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