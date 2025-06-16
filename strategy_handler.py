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
    check_for_liquidity_grab
)
from journal import log_new_trade, get_open_trades

def run_strategy_check(client, instrument: str) -> bool:
    """
    Main strategy function that performs multi-timeframe analysis and executes trades.
    
    This function implements a confluence-based trading approach:
    1. Higher timeframe (M15) provides market context/bias
    2. Medium timeframe (M5) identifies key supply/demand zones  
    3. Lower timeframe (M1) provides precise entry signals
    4. ICT concepts (FVG, Order Blocks) add confluence
    5. Risk management determines position sizing
    6. Trade execution with linked SL/TP
    
    Args:
        client: Authenticated OANDA API client
        instrument (str): Trading instrument (e.g., 'EUR_USD')
    
    Returns:
        bool: True if trade was executed, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"STRATEGY CHECK: {instrument}")
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
        # STEP 2: MULTI-TIMEFRAME ANALYSIS
        # ==================================================================
        print("\n🔍 STEP 2: Multi-timeframe analysis...")
        
        # M15 Context Analysis - Determine overall market bias
        market_context = identify_market_structure(df_m15, lookback_period=50)
        print(f"✓ M15 Market Context: {market_context}")
        
        # M5 Zone Analysis - Identify key supply/demand levels
        zones_m5 = find_supply_demand_zones(df_m5, lookback=20, strength_factor=1.5)
        print(f"✓ M5 Supply/Demand Zones: {len(zones_m5)} zones identified")
        
        # M1 ICT Analysis - Find FVGs and Order Blocks
        fvg_list, ob_list = identify_fvg_and_ob(df_m1)
        print(f"✓ M1 ICT Analysis: {len(fvg_list)} FVGs, {len(ob_list)} Order Blocks")
        
        # Current price context
        current_price = df_m1['close'].iloc[-1]
        price_context = get_current_price_context(df_m5, zones_m5, current_price)
        print(f"✓ Current Price: {current_price:.5f}")
        print(f"✓ Price Context: {price_context['price_context']}")
        
        # ==================================================================
        # STEP 3: CONFLUENCE-BASED ENTRY LOGIC
        # ==================================================================
        print("\n⚡ STEP 3: Confluence-based entry analysis...")
        
        # Check for existing open trades to avoid over-exposure
        open_trades = get_open_trades('trading_journal.db')
        instrument_open_trades = [t for t in open_trades if t['instrument'] == instrument]
        
        if instrument_open_trades:
            print(f"⚠ Skipping {instrument} - {len(instrument_open_trades)} open trade(s) already exist")
            return False
        
        # Analyze long trade opportunity
        long_signal = analyze_long_opportunity(
            market_context, price_context, zones_m5, fvg_list, ob_list, df_m1, current_price
        )
        
        # Analyze short trade opportunity  
        short_signal = analyze_short_opportunity(
            market_context, price_context, zones_m5, fvg_list, ob_list, df_m1, current_price
        )
        
        # ==================================================================
        # STEP 4: TRADE EXECUTION
        # ==================================================================
        if long_signal['valid']:
            print(f"\n🚀 LONG SIGNAL DETECTED!")
            return execute_trade(client, instrument, 'LONG', long_signal, df_m1)
            
        elif short_signal['valid']:
            print(f"\n🚀 SHORT SIGNAL DETECTED!")
            return execute_trade(client, instrument, 'SHORT', short_signal, df_m1)
            
        else:
            print(f"\n⏳ No valid signals found for {instrument}")
            if long_signal['reasons'] or short_signal['reasons']:
                print("📋 Analysis summary:")
                if long_signal['reasons']:
                    print(f"  Long blocked: {', '.join(long_signal['reasons'])}")
                if short_signal['reasons']:
                    print(f"  Short blocked: {', '.join(short_signal['reasons'])}")
            return False
            
    except Exception as e:
        print(f"✗ Error in strategy check for {instrument}: {e}")
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return False

def analyze_long_opportunity(
    market_context: str, 
    price_context: Dict, 
    zones: List[Dict], 
    fvg_list: List[Dict], 
    ob_list: List[Dict],
    df_m1: pd.DataFrame,
    current_price: float
) -> Dict[str, Any]:
    """
    Analyze conditions for a long trade opportunity using confluence.
    
    Long Trade Requirements (ALL must be met):
    1. M15 context must be 'Uptrend' 
    2. Current price near or inside a demand zone (M5)
    3. Recent bullish FVG or Order Block formed (M1)
    4. Optional: Recent liquidity grab below key level
    
    Returns:
        Dict with 'valid' boolean and supporting 'reasons' list
    """
    signal = {'valid': False, 'reasons': [], 'confidence': 0, 'entry_reason': ''}
    
    try:
        # Requirement 1: M15 Context Filter
        if market_context != 'Uptrend':
            signal['reasons'].append(f"M15 context is {market_context}, not Uptrend")
            return signal
        
        signal['confidence'] += 25
        
        # Requirement 2: Demand Zone Proximity  
        demand_zones = [z for z in zones if z['type'] == 'demand']
        
        if not demand_zones:
            signal['reasons'].append("No demand zones available")
            return signal
        
        # Check if price is near any demand zone (within 0.2%)
        near_demand = False
        target_zone = None
        
        for zone in demand_zones:
            zone_price = zone['price_level']
            distance_pct = abs(current_price - zone_price) / current_price * 100
            
            # Price should be at or slightly above demand zone
            if current_price >= zone_price * 0.999 and distance_pct <= 0.2:
                near_demand = True
                target_zone = zone
                break
        
        if not near_demand:
            nearest_demand = min(demand_zones, key=lambda z: abs(z['price_level'] - current_price))
            distance = abs(current_price - nearest_demand['price_level']) / current_price * 100
            signal['reasons'].append(f"Not near demand zone (nearest: {distance:.2f}% away)")
            return signal
        
        signal['confidence'] += 25
        
        # Requirement 3: Recent Bullish ICT Pattern
        recent_bullish_fvg = False
        recent_bullish_ob = False
        
        # Check for recent bullish FVGs (last 10 candles)
        recent_fvgs = [fvg for fvg in fvg_list if fvg['type'] == 'bullish']
        if recent_fvgs:
            # Check if current price is in or near any unfilled bullish FVG
            for fvg in recent_fvgs[-3:]:  # Last 3 FVGs
                if (current_price >= fvg['lower_level'] and 
                    current_price <= fvg['upper_level'] * 1.001):
                    recent_bullish_fvg = True
                    break
        
        # Check for recent bullish Order Blocks
        recent_obs = [ob for ob in ob_list if ob['type'] == 'bullish']
        if recent_obs:
            # Check if current price is in or near any active bullish OB
            for ob in recent_obs[-3:]:  # Last 3 OBs
                if (current_price >= ob['zone_low'] and 
                    current_price <= ob['zone_high'] * 1.001):
                    recent_bullish_ob = True
                    break
        
        if not (recent_bullish_fvg or recent_bullish_ob):
            signal['reasons'].append("No recent bullish FVG or Order Block")
            return signal
        
        signal['confidence'] += 30
        
        # Requirement 4: Confirmation via price action
        recent_candles = df_m1.tail(3)
        bullish_momentum = False
        
        # Look for bullish momentum (recent green candles or bullish engulfing)
        for _, candle in recent_candles.iterrows():
            if candle['close'] > candle['open']:  # Bullish candle
                body_size = candle['close'] - candle['open']
                candle_range = candle['high'] - candle['low']
                
                # Strong bullish candle (body > 60% of range)
                if candle_range > 0 and body_size / candle_range > 0.6:
                    bullish_momentum = True
                    break
        
        if bullish_momentum:
            signal['confidence'] += 20
        
        # Bonus: Check for liquidity grab (adds confidence but not required)
        if target_zone:
            liquidity_grab = check_for_liquidity_grab(df_m1, target_zone['low_price'])
            if liquidity_grab:
                signal['confidence'] += 10
                signal['entry_reason'] += "Liquidity grab + "
        
        # Decision threshold: Need at least 70% confidence for valid signal
        if signal['confidence'] >= 70:
            signal['valid'] = True
            signal['entry_reason'] += f"M15 uptrend + Demand zone + "
            if recent_bullish_fvg:
                signal['entry_reason'] += "Bullish FVG + "
            if recent_bullish_ob:
                signal['entry_reason'] += "Bullish OB + "
            if bullish_momentum:
                signal['entry_reason'] += "Bullish momentum"
            
            signal['entry_reason'] = signal['entry_reason'].rstrip(' + ')
        else:
            signal['reasons'].append(f"Insufficient confluence (confidence: {signal['confidence']}%)")
        
        return signal
        
    except Exception as e:
        signal['reasons'].append(f"Analysis error: {e}")
        return signal

def analyze_short_opportunity(
    market_context: str,
    price_context: Dict,
    zones: List[Dict],
    fvg_list: List[Dict],
    ob_list: List[Dict],
    df_m1: pd.DataFrame,
    current_price: float
) -> Dict[str, Any]:
    """
    Analyze conditions for a short trade opportunity using confluence.
    
    Short Trade Requirements (ALL must be met):
    1. M15 context must be 'Downtrend'
    2. Current price near or inside a supply zone (M5)  
    3. Recent bearish FVG or Order Block formed (M1)
    4. Optional: Recent liquidity grab above key level
    
    Returns:
        Dict with 'valid' boolean and supporting 'reasons' list
    """
    signal = {'valid': False, 'reasons': [], 'confidence': 0, 'entry_reason': ''}
    
    try:
        # Requirement 1: M15 Context Filter
        if market_context != 'Downtrend':
            signal['reasons'].append(f"M15 context is {market_context}, not Downtrend")
            return signal
        
        signal['confidence'] += 25
        
        # Requirement 2: Supply Zone Proximity
        supply_zones = [z for z in zones if z['type'] == 'supply']
        
        if not supply_zones:
            signal['reasons'].append("No supply zones available")
            return signal
        
        # Check if price is near any supply zone (within 0.2%)
        near_supply = False
        target_zone = None
        
        for zone in supply_zones:
            zone_price = zone['price_level']
            distance_pct = abs(current_price - zone_price) / current_price * 100
            
            # Price should be at or slightly below supply zone
            if current_price <= zone_price * 1.001 and distance_pct <= 0.2:
                near_supply = True
                target_zone = zone
                break
        
        if not near_supply:
            nearest_supply = min(supply_zones, key=lambda z: abs(z['price_level'] - current_price))
            distance = abs(current_price - nearest_supply['price_level']) / current_price * 100
            signal['reasons'].append(f"Not near supply zone (nearest: {distance:.2f}% away)")
            return signal
        
        signal['confidence'] += 25
        
        # Requirement 3: Recent Bearish ICT Pattern
        recent_bearish_fvg = False
        recent_bearish_ob = False
        
        # Check for recent bearish FVGs
        recent_fvgs = [fvg for fvg in fvg_list if fvg['type'] == 'bearish']
        if recent_fvgs:
            for fvg in recent_fvgs[-3:]:  # Last 3 FVGs
                if (current_price <= fvg['upper_level'] and 
                    current_price >= fvg['lower_level'] * 0.999):
                    recent_bearish_fvg = True
                    break
        
        # Check for recent bearish Order Blocks
        recent_obs = [ob for ob in ob_list if ob['type'] == 'bearish']
        if recent_obs:
            for ob in recent_obs[-3:]:  # Last 3 OBs
                if (current_price <= ob['zone_high'] and 
                    current_price >= ob['zone_low'] * 0.999):
                    recent_bearish_ob = True
                    break
        
        if not (recent_bearish_fvg or recent_bearish_ob):
            signal['reasons'].append("No recent bearish FVG or Order Block")
            return signal
        
        signal['confidence'] += 30
        
        # Requirement 4: Confirmation via price action
        recent_candles = df_m1.tail(3)
        bearish_momentum = False
        
        # Look for bearish momentum
        for _, candle in recent_candles.iterrows():
            if candle['close'] < candle['open']:  # Bearish candle
                body_size = candle['open'] - candle['close']
                candle_range = candle['high'] - candle['low']
                
                # Strong bearish candle
                if candle_range > 0 and body_size / candle_range > 0.6:
                    bearish_momentum = True
                    break
        
        if bearish_momentum:
            signal['confidence'] += 20
        
        # Bonus: Check for liquidity grab
        if target_zone:
            liquidity_grab = check_for_liquidity_grab(df_m1, target_zone['high_price'])
            if liquidity_grab:
                signal['confidence'] += 10
                signal['entry_reason'] += "Liquidity grab + "
        
        # Decision threshold
        if signal['confidence'] >= 70:
            signal['valid'] = True
            signal['entry_reason'] += f"M15 downtrend + Supply zone + "
            if recent_bearish_fvg:
                signal['entry_reason'] += "Bearish FVG + "
            if recent_bearish_ob:
                signal['entry_reason'] += "Bearish OB + "
            if bearish_momentum:
                signal['entry_reason'] += "Bearish momentum"
            
            signal['entry_reason'] = signal['entry_reason'].rstrip(' + ')
        else:
            signal['reasons'].append(f"Insufficient confluence (confidence: {signal['confidence']}%)")
        
        return signal
        
    except Exception as e:
        signal['reasons'].append(f"Analysis error: {e}")
        return signal

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