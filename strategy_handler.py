"""
Strategy Handler Module

This module contains the main trading strategy logic that combines multi-timeframe
analysis, price action patterns, and ICT concepts to generate high-probability
trading signals. It integrates all analysis components into a coherent strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import traceback

# Import our custom modules
from oanda_handler import get_api_client, get_historical_data, place_market_order, get_account_summary
from analysis_engine import (
    identify_market_structure, 
    find_supply_demand_zones, 
    get_current_price_context,
    identify_fvg_and_ob,
    check_for_liquidity_grab,
    confirm_m1_reversal_signal
)
from journal import log_new_trade, get_open_trades

def run_strategy_check(client, instrument: str) -> bool:
    """
    Main strategy function with DUAL-MODE logic for maximum trade frequency.
    
    DUAL-MODE STRATEGY:
    MODE A (Trend-Following): When M15 is trending, seeks high-confluence pullback entries
    MODE B (Range-Bound): When M15 is ranging, seeks mean-reversion at strong M5 zones
    
    This approach ensures the bot trades during BOTH trending and consolidation periods,
    dramatically increasing trade frequency while maintaining quality signals.
    
    Args:
        client: Authenticated OANDA API client
        instrument (str): Trading instrument (e.g., 'EUR_USD')
    
    Returns:
        bool: True if trade was executed, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"DUAL-MODE STRATEGY CHECK: {instrument}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # ==================================================================
        # STEP 1: FETCH MULTI-TIMEFRAME DATA
        # ==================================================================
        print("\n📊 STEP 1: Fetching multi-timeframe data...")
        
        # Fetch M15 data for context (100 candles = ~25 hours)
        df_m15 = get_historical_data(client, instrument, count=100, granularity='M15')
        if df_m15 is None or len(df_m15) < 50:
            print("✗ Insufficient M15 data for analysis")
            return False
        
        # Fetch M5 data for zones (100 candles = ~8 hours)  
        df_m5 = get_historical_data(client, instrument, count=100, granularity='M5')
        if df_m5 is None or len(df_m5) < 50:
            print("✗ Insufficient M5 data for analysis")
            return False
        
        # Fetch M1 data for entry signals (50 candles = ~50 minutes)
        df_m1 = get_historical_data(client, instrument, count=50, granularity='M1')
        if df_m1 is None or len(df_m1) < 20:
            print("✗ Insufficient M1 data for analysis")
            return False
        
        print(f"✓ Data fetched successfully:")
        print(f"  M15: {len(df_m15)} candles")
        print(f"  M5:  {len(df_m5)} candles") 
        print(f"  M1:  {len(df_m1)} candles")
        
        # ==================================================================
        # STEP 2: MARKET CONTEXT ANALYSIS
        # ==================================================================
        print("\n🔍 STEP 2: Market context determination...")
        
        # M15 Context Analysis - This determines our strategy mode
        context = identify_market_structure(df_m15, lookback_period=50)
        print(f"✓ M15 Market Context: {context}")
        
        # M5 Zone Analysis - Critical for both modes
        zones_m5 = find_supply_demand_zones(df_m5, lookback=20, strength_factor=1.5)
        print(f"✓ M5 Supply/Demand Zones: {len(zones_m5)} zones identified")
        
        # Current price and context
        current_price = df_m1['close'].iloc[-1]
        print(f"✓ Current Price: {current_price:.5f}")
        
        # Check for existing open trades to avoid over-exposure
        open_trades = get_open_trades('trading_journal.db')
        instrument_open_trades = [t for t in open_trades if t['instrument'] == instrument]
        
        if instrument_open_trades:
            print(f"⚠ Skipping {instrument} - {len(instrument_open_trades)} open trade(s) already exist")
            return False
        
        # ==================================================================
        # STEP 3: DUAL-MODE STRATEGY LOGIC
        # ==================================================================
        print(f"\n⚡ STEP 3: Dual-mode strategy execution...")
        
        # TREND-FOLLOWING MODE (Original high-confluence strategy)
        if context == 'Uptrend':
            print("📈 Strategy Mode: Trend-Following (Long)")
            return execute_trend_following_long(client, instrument, zones_m5, df_m1, current_price)
            
        elif context == 'Downtrend':
            print("📉 Strategy Mode: Trend-Following (Short)")
            return execute_trend_following_short(client, instrument, zones_m5, df_m1, current_price)
            
        # RANGE-BOUND MODE (New mean-reversion strategy)
        elif context == 'Range':
            print("🔄 Strategy Mode: Range-Bound Reversal")
            return execute_range_bound_strategy(client, instrument, zones_m5, df_m1, current_price)
            
        else:
            print(f"❓ Unknown market context: {context}")
            return False
            
    except Exception as e:
        print(f"✗ Error in strategy check for {instrument}: {e}")
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return False

def execute_trend_following_long(client, instrument: str, zones_m5: List[Dict], df_m1: pd.DataFrame, current_price: float) -> bool:
    """
    Execute the original high-confluence trend-following long strategy.
    
    Requirements:
    1. M15 context is 'Uptrend' (already confirmed)
    2. Current price near demand zone (M5)
    3. Recent bullish FVG or Order Block (M1) 
    4. Strong bullish M1 momentum confirmation
    """
    try:
        print("\n🔍 Analyzing trend-following LONG opportunity...")
        
        # Get demand zones
        demand_zones = [z for z in zones_m5 if z['type'] == 'demand']
        if not demand_zones:
            print("❌ No demand zones available for trend-following long")
            return False
        
        # Check if price is near any strong demand zone (within 0.2%)
        near_demand = False
        target_zone = None
        
        for zone in demand_zones:
            distance_pct = abs(current_price - zone['price_level']) / current_price * 100
            if current_price >= zone['price_level'] * 0.999 and distance_pct <= 0.2:
                near_demand = True
                target_zone = zone
                print(f"✅ Price near demand zone at {zone['price_level']:.5f} (strength: {zone['strength']:.2f})")
                break
        
        if not near_demand:
            nearest_demand = min(demand_zones, key=lambda z: abs(z['price_level'] - current_price))
            distance = abs(current_price - nearest_demand['price_level']) / current_price * 100
            print(f"❌ Not near demand zone (nearest: {distance:.2f}% away)")
            return False
        
        # Get ICT analysis for confluence
        fvg_list, ob_list = identify_fvg_and_ob(df_m1)
        
        # Check for recent bullish ICT patterns
        recent_bullish_confluence = False
        confluence_reason = ""
        
        # Check bullish FVGs
        recent_fvgs = [fvg for fvg in fvg_list if fvg['type'] == 'bullish']
        for fvg in recent_fvgs[-3:]:
            if (current_price >= fvg['lower_level'] and current_price <= fvg['upper_level'] * 1.001):
                recent_bullish_confluence = True
                confluence_reason += "Bullish FVG + "
                break
        
        # Check bullish Order Blocks
        recent_obs = [ob for ob in ob_list if ob['type'] == 'bullish']
        for ob in recent_obs[-3:]:
            if (current_price >= ob['zone_low'] and current_price <= ob['zone_high'] * 1.001):
                recent_bullish_confluence = True
                confluence_reason += "Bullish OB + "
                break
        
        if not recent_bullish_confluence:
            print("❌ No recent bullish FVG or Order Block confluence")
            return False
        
        # Look for bullish momentum confirmation
        recent_candles = df_m1.tail(3)
        bullish_momentum = False
        
        for _, candle in recent_candles.iterrows():
            if candle['close'] > candle['open']:
                body_size = candle['close'] - candle['open']
                candle_range = candle['high'] - candle['low']
                if candle_range > 0 and body_size / candle_range > 0.6:
                    bullish_momentum = True
                    break
        
        if not bullish_momentum:
            print("❌ No strong bullish momentum confirmation")
            return False
        
        confluence_reason += "Trend-following + Demand zone + Bullish momentum"
        print(f"✅ High-confluence LONG signal confirmed!")
        print(f"📋 Entry reason: {confluence_reason}")
        
        # Execute the trade
        signal = {'entry_reason': confluence_reason, 'confidence': 85}
        return execute_trade(client, instrument, 'LONG', signal, df_m1)
        
    except Exception as e:
        print(f"❌ Error in trend-following long analysis: {e}")
        return False

def execute_trend_following_short(client, instrument: str, zones_m5: List[Dict], df_m1: pd.DataFrame, current_price: float) -> bool:
    """
    Execute the original high-confluence trend-following short strategy.
    
    Requirements:
    1. M15 context is 'Downtrend' (already confirmed)
    2. Current price near supply zone (M5)
    3. Recent bearish FVG or Order Block (M1)
    4. Strong bearish M1 momentum confirmation
    """
    try:
        print("\n🔍 Analyzing trend-following SHORT opportunity...")
        
        # Get supply zones
        supply_zones = [z for z in zones_m5 if z['type'] == 'supply']
        if not supply_zones:
            print("❌ No supply zones available for trend-following short")
            return False
        
        # Check if price is near any strong supply zone (within 0.2%)
        near_supply = False
        target_zone = None
        
        for zone in supply_zones:
            distance_pct = abs(current_price - zone['price_level']) / current_price * 100
            if current_price <= zone['price_level'] * 1.001 and distance_pct <= 0.2:
                near_supply = True
                target_zone = zone
                print(f"✅ Price near supply zone at {zone['price_level']:.5f} (strength: {zone['strength']:.2f})")
                break
        
        if not near_supply:
            nearest_supply = min(supply_zones, key=lambda z: abs(z['price_level'] - current_price))
            distance = abs(current_price - nearest_supply['price_level']) / current_price * 100
            print(f"❌ Not near supply zone (nearest: {distance:.2f}% away)")
            return False
        
        # Get ICT analysis for confluence
        fvg_list, ob_list = identify_fvg_and_ob(df_m1)
        
        # Check for recent bearish ICT patterns
        recent_bearish_confluence = False
        confluence_reason = ""
        
        # Check bearish FVGs
        recent_fvgs = [fvg for fvg in fvg_list if fvg['type'] == 'bearish']
        for fvg in recent_fvgs[-3:]:
            if (current_price <= fvg['upper_level'] and current_price >= fvg['lower_level'] * 0.999):
                recent_bearish_confluence = True
                confluence_reason += "Bearish FVG + "
                break
        
        # Check bearish Order Blocks
        recent_obs = [ob for ob in ob_list if ob['type'] == 'bearish']
        for ob in recent_obs[-3:]:
            if (current_price <= ob['zone_high'] and current_price >= ob['zone_low'] * 0.999):
                recent_bearish_confluence = True
                confluence_reason += "Bearish OB + "
                break
        
        if not recent_bearish_confluence:
            print("❌ No recent bearish FVG or Order Block confluence")
            return False
        
        # Look for bearish momentum confirmation
        recent_candles = df_m1.tail(3)
        bearish_momentum = False
        
        for _, candle in recent_candles.iterrows():
            if candle['close'] < candle['open']:
                body_size = candle['open'] - candle['close']
                candle_range = candle['high'] - candle['low']
                if candle_range > 0 and body_size / candle_range > 0.6:
                    bearish_momentum = True
                    break
        
        if not bearish_momentum:
            print("❌ No strong bearish momentum confirmation")
            return False
        
        confluence_reason += "Trend-following + Supply zone + Bearish momentum"
        print(f"✅ High-confluence SHORT signal confirmed!")
        print(f"📋 Entry reason: {confluence_reason}")
        
        # Execute the trade
        signal = {'entry_reason': confluence_reason, 'confidence': 85}
        return execute_trade(client, instrument, 'SHORT', signal, df_m1)
        
    except Exception as e:
        print(f"❌ Error in trend-following short analysis: {e}")
        return False

def execute_range_bound_strategy(client, instrument: str, zones_m5: List[Dict], df_m1: pd.DataFrame, current_price: float) -> bool:
    """
    Execute the NEW range-bound mean-reversion strategy.
    
    This strategy trades during consolidation periods by:
    1. Selling at strong M5 supply zones (expecting reversion down)
    2. Buying at strong M5 demand zones (expecting reversion up)
    3. Using M1 reversal confirmation for precise entries
    """
    try:
        print("\n🔍 Analyzing range-bound mean-reversion opportunity...")
        
        if not zones_m5:
            print("❌ No M5 zones available for range-bound strategy")
            return False
        
        # Check if price is very near a strong supply zone (SELL signal)
        supply_zones = [z for z in zones_m5 if z['type'] == 'supply']
        for zone in supply_zones:
            distance_pct = abs(current_price - zone['price_level']) / current_price * 100
            
            # Price must be very close to supply zone (within 0.15% for mean reversion)
            if current_price >= zone['price_level'] * 0.998 and distance_pct <= 0.15:
                print(f"✅ Price near supply zone at {zone['price_level']:.5f} for mean-reversion SHORT")
                
                # Check for M1 bearish reversal confirmation
                reversal_signal = confirm_m1_reversal_signal(df_m1)
                if reversal_signal == 'Bearish Reversal':
                    print("✅ M1 bearish reversal signal confirmed!")
                    
                    entry_reason = f"Range-bound mean-reversion + Supply zone rejection + M1 bearish reversal"
                    signal = {'entry_reason': entry_reason, 'confidence': 75}
                    return execute_trade(client, instrument, 'SHORT', signal, df_m1)
                else:
                    print(f"❌ No M1 bearish reversal confirmation (got: {reversal_signal})")
        
        # Check if price is very near a strong demand zone (BUY signal)
        demand_zones = [z for z in zones_m5 if z['type'] == 'demand']
        for zone in demand_zones:
            distance_pct = abs(current_price - zone['price_level']) / current_price * 100
            
            # Price must be very close to demand zone (within 0.15% for mean reversion)
            if current_price <= zone['price_level'] * 1.002 and distance_pct <= 0.15:
                print(f"✅ Price near demand zone at {zone['price_level']:.5f} for mean-reversion LONG")
                
                # Check for M1 bullish reversal confirmation
                reversal_signal = confirm_m1_reversal_signal(df_m1)
                if reversal_signal == 'Bullish Reversal':
                    print("✅ M1 bullish reversal signal confirmed!")
                    
                    entry_reason = f"Range-bound mean-reversion + Demand zone bounce + M1 bullish reversal"
                    signal = {'entry_reason': entry_reason, 'confidence': 75}
                    return execute_trade(client, instrument, 'LONG', signal, df_m1)
                else:
                    print(f"❌ No M1 bullish reversal confirmation (got: {reversal_signal})")
        
        print("❌ No valid range-bound mean-reversion opportunities found")
        print("📋 Range-bound strategy requires:")
        print("   • Price very close to strong M5 supply/demand zone (< 0.15%)")
        print("   • Strong M1 reversal candle confirmation")
        return False
        
    except Exception as e:
        print(f"❌ Error in range-bound strategy analysis: {e}")
        return False

def execute_trade(
    client, 
    instrument: str, 
    direction: str, 
    signal: Dict, 
    df_m1: pd.DataFrame
) -> bool:
    """
    Execute the trade with proper risk management.
    
    Args:
        client: OANDA API client
        instrument: Trading instrument  
        direction: 'LONG' or 'SHORT'
        signal: Signal information dict
        df_m1: M1 timeframe data for stop loss calculation
    
    Returns:
        bool: True if trade executed successfully
    """
    try:
        print(f"\n💼 EXECUTING {direction} TRADE:")
        print(f"📋 Entry Reason: {signal['entry_reason']}")
        print(f"🎯 Confidence: {signal['confidence']}%")
        
        # Get current price
        current_price = df_m1['close'].iloc[-1]
        
        # Calculate risk management parameters
        risk_params = calculate_risk_management(df_m1, direction, current_price)
        
        if not risk_params['valid']:
            print(f"✗ Risk management check failed: {risk_params['reason']}")
            return False
        
        # Determine position size (0.5% risk per trade)
        account_summary = get_account_summary(client)
        account_balance = float(account_summary.get('balance', 10000))
        
        risk_amount = account_balance * 0.005  # 0.5% risk
        stop_distance_pips = abs(current_price - risk_params['stop_loss']) * 10000
        
        # Calculate position size based on risk
        if stop_distance_pips > 0:
            # For EUR_USD, 1 pip = $1 per 10k units approximately
            position_size = int((risk_amount / stop_distance_pips) * 10000)
            
            # Apply position size limits
            max_position = 50000  # Maximum position size
            min_position = 1000   # Minimum position size
            
            position_size = max(min_position, min(position_size, max_position))
        else:
            position_size = 10000  # Default position size
        
        # Set direction for units (positive = long, negative = short)
        units = position_size if direction == 'LONG' else -position_size
        
        print(f"💰 Position Details:")
        print(f"  Units: {abs(units):,}")
        print(f"  Entry: {current_price:.5f}")
        print(f"  Stop Loss: {risk_params['stop_loss']:.5f}")
        print(f"  Take Profit: {risk_params['take_profit']:.5f}")
        print(f"  Risk Amount: ${risk_amount:.2f}")
        
        # Execute the trade
        order_response = place_market_order(
            client=client,
            instrument=instrument,
            units=units,
            stop_loss_price=risk_params['stop_loss'],
            take_profit_price=risk_params['take_profit']
        )
        
        # Log the trade if successful
        if order_response and 'orderFillTransaction' in order_response:
            fill_transaction = order_response['orderFillTransaction']
            trade_id = fill_transaction.get('tradeOpened', {}).get('tradeID', 'N/A')
            fill_price = float(fill_transaction.get('price', current_price))
            
            # Log to journal
            log_success = log_new_trade(
                db_name='trading_journal.db',
                trade_id=int(trade_id) if trade_id != 'N/A' else 0,
                instrument=instrument,
                units=units,
                direction=direction,
                entry_price=fill_price,
                sl_price=risk_params['stop_loss'],
                tp_price=risk_params['take_profit'],
                reason=signal['entry_reason']
            )
            
            if log_success:
                print(f"✅ Trade executed and logged successfully!")
                return True
            else:
                print(f"⚠ Trade executed but logging failed")
                return True
        else:
            print(f"✗ Trade execution failed")
            return False
            
    except Exception as e:
        print(f"✗ Error executing trade: {e}")
        return False

def calculate_risk_management(df_m1: pd.DataFrame, direction: str, entry_price: float) -> Dict:
    """
    Calculate stop loss and take profit levels based on market structure.
    
    Args:
        df_m1: M1 timeframe data
        direction: Trade direction ('LONG' or 'SHORT')
        entry_price: Entry price
    
    Returns:
        Dict with stop_loss, take_profit, and validity
    """
    try:
        # Calculate ATR for dynamic stop placement
        df_copy = df_m1.copy()
        df_copy['tr'] = np.maximum(
            df_copy['high'] - df_copy['low'],
            np.maximum(
                abs(df_copy['high'] - df_copy['close'].shift(1)),
                abs(df_copy['low'] - df_copy['close'].shift(1))
            )
        )
        atr = df_copy['tr'].rolling(window=14).mean().iloc[-1]
        
        if direction == 'LONG':
            # For long trades: Stop below recent swing low or 2*ATR
            recent_lows = df_m1['low'].tail(10)
            swing_low = recent_lows.min()
            
            # Use the more conservative (further) stop
            atr_stop = entry_price - (2 * atr)
            structural_stop = swing_low - (0.5 * atr)  # Add buffer below swing low
            
            stop_loss = min(atr_stop, structural_stop)
            
            # Take profit at 1.5:1 reward-to-risk ratio
            risk_distance = entry_price - stop_loss
            take_profit = entry_price + (risk_distance * 1.5)
            
        else:  # SHORT
            # For short trades: Stop above recent swing high or 2*ATR
            recent_highs = df_m1['high'].tail(10)
            swing_high = recent_highs.max()
            
            # Use the more conservative (further) stop
            atr_stop = entry_price + (2 * atr)
            structural_stop = swing_high + (0.5 * atr)  # Add buffer above swing high
            
            stop_loss = max(atr_stop, structural_stop)
            
            # Take profit at 1.5:1 reward-to-risk ratio
            risk_distance = stop_loss - entry_price
            take_profit = entry_price - (risk_distance * 1.5)
        
        # Validate stop loss distance (minimum 5 pips, maximum 50 pips)
        stop_distance_pips = abs(entry_price - stop_loss) * 10000
        
        if stop_distance_pips < 5:
            return {'valid': False, 'reason': f'Stop too close ({stop_distance_pips:.1f} pips)'}
        
        if stop_distance_pips > 50:
            return {'valid': False, 'reason': f'Stop too far ({stop_distance_pips:.1f} pips)'}
        
        return {
            'valid': True,
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'risk_pips': stop_distance_pips,
            'reward_pips': abs(take_profit - entry_price) * 10000,
            'risk_reward_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        }
        
    except Exception as e:
        return {'valid': False, 'reason': f'Risk calculation error: {e}'}

def test_strategy_handler():
    """
    Test the strategy handler with a demo instrument.
    """
    try:
        print("Testing Strategy Handler...")
        
        # Get API client
        client = get_api_client()
        
        # Test with EUR_USD
        result = run_strategy_check(client, 'EUR_USD')
        
        print(f"\n✅ Strategy handler test completed. Trade executed: {result}")
        
    except Exception as e:
        print(f"✗ Strategy handler test failed: {e}")

if __name__ == "__main__":
    # Run test when script is executed directly
    test_strategy_handler()