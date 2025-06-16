"""
Price Action Analysis Engine (Core Functions)

This module provides core analytical functions for identifying market structure,
supply/demand zones, and other key price action patterns used by the trading bot.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional, Any
import warnings

# Import smart money concepts library for ICT analysis
try:
    import smartmoneyconcepts as smc_lib
    smc = smc_lib.smc()  # Create instance of the smc class
    SMC_AVAILABLE = True
    print("✓ Smart Money Concepts library loaded successfully")
except ImportError:
    SMC_AVAILABLE = False
    smc = None
    print("⚠ Smart Money Concepts library not available - using manual ICT detection")

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

def identify_market_structure(df: pd.DataFrame, lookback_period: int = 50) -> str:
    """
    Identify the current market structure based on swing highs and lows analysis.
    
    This function uses scipy's find_peaks to identify significant swing points
    and analyzes their progression to determine if the market is in an uptrend,
    downtrend, or ranging condition.
    
    Args:
        df (pd.DataFrame): OHLC data with columns: open, high, low, close
        lookback_period (int): Number of bars to look back for analysis
    
    Returns:
        str: Market structure classification ('Uptrend', 'Downtrend', or 'Range')
    """
    try:
        # Ensure we have enough data
        if len(df) < lookback_period:
            print(f"⚠ Insufficient data for market structure analysis. Need {lookback_period}, got {len(df)}")
            return 'Range'  # Default to range when insufficient data
        
        # Get the last 'lookback_period' bars for analysis
        recent_data = df.tail(lookback_period).copy()
        
        # Find swing highs using scipy's find_peaks
        # We look for peaks in the 'high' column with minimum distance between peaks
        high_peaks, high_properties = find_peaks(
            recent_data['high'].values,
            distance=5,  # Minimum distance between peaks (prevents noise)
            prominence=recent_data['high'].std() * 0.5  # Minimum prominence for a valid peak
        )
        
        # Find swing lows by finding peaks in the inverted 'low' column
        low_peaks, low_properties = find_peaks(
            -recent_data['low'].values,  # Invert lows to find valleys as peaks
            distance=5,  # Minimum distance between valleys
            prominence=recent_data['low'].std() * 0.5  # Minimum prominence for a valid valley
        )
        
        # Get the actual price values for the identified swing points
        swing_highs = recent_data['high'].iloc[high_peaks].values
        swing_lows = recent_data['low'].iloc[low_peaks].values
        
        # Need at least 2 swing highs and 2 swing lows for trend analysis
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'Range'
        
        # Analyze the trend of swing highs (are they getting higher or lower?)
        high_trend = analyze_swing_trend(swing_highs)
        
        # Analyze the trend of swing lows (are they getting higher or lower?)
        low_trend = analyze_swing_trend(swing_lows)
        
        # Determine market structure based on swing analysis
        if high_trend == 'rising' and low_trend == 'rising':
            # Higher highs and higher lows = Uptrend
            return 'Uptrend'
        elif high_trend == 'falling' and low_trend == 'falling':
            # Lower highs and lower lows = Downtrend
            return 'Downtrend'
        else:
            # Mixed signals or sideways movement = Range
            return 'Range'
            
    except Exception as e:
        print(f"✗ Error in market structure analysis: {e}")
        return 'Range'  # Default to range on error

def analyze_swing_trend(swing_points: np.ndarray) -> str:
    """
    Analyze the trend direction of a series of swing points.
    
    Args:
        swing_points (np.ndarray): Array of swing high or low values
    
    Returns:
        str: Trend direction ('rising', 'falling', or 'sideways')
    """
    if len(swing_points) < 2:
        return 'sideways'
    
    # Take the last 3-4 swing points for recent trend analysis
    recent_swings = swing_points[-4:] if len(swing_points) >= 4 else swing_points
    
    # Calculate the linear trend using numpy's polyfit (degree 1 = linear)
    x = np.arange(len(recent_swings))
    slope, _ = np.polyfit(x, recent_swings, 1)
    
    # Determine trend direction based on slope
    if slope > 0.0001:  # Small threshold to filter out noise
        return 'rising'
    elif slope < -0.0001:
        return 'falling'
    else:
        return 'sideways'

def find_supply_demand_zones(df: pd.DataFrame, lookback: int = 20, strength_factor: float = 1.5) -> List[Dict[str, Any]]:
    """
    Identify supply and demand zones based on strong candle formations.
    
    A supply zone is identified by a strong bearish candle that precedes a significant
    downward move. A demand zone is identified by a strong bullish candle that precedes 
    a significant upward move.
    
    Args:
        df (pd.DataFrame): OHLC data with columns: open, high, low, close
        lookback (int): Window size for calculating average candle body size
        strength_factor (float): Multiplier for determining "strong" candles
    
    Returns:
        List[Dict]: List of zones with type, price level, strength, and timestamp
    """
    try:
        zones = []
        
        # Ensure we have enough data
        if len(df) < lookback + 10:
            print(f"⚠ Insufficient data for supply/demand analysis. Need {lookback + 10}, got {len(df)}")
            return zones
        
        # Calculate candle body sizes (absolute difference between open and close)
        df_copy = df.copy()
        df_copy['body_size'] = abs(df_copy['close'] - df_copy['open'])
        
        # Calculate rolling average body size over the lookback period
        df_copy['avg_body_size'] = df_copy['body_size'].rolling(window=lookback).mean()
        
        # Calculate candle direction (1 for bullish, -1 for bearish)
        df_copy['direction'] = np.where(df_copy['close'] > df_copy['open'], 1, -1)
        
        # Calculate the range of each candle (high - low)
        df_copy['candle_range'] = df_copy['high'] - df_copy['low']
        
        # Start analysis from lookback period onwards (need rolling averages)
        for i in range(lookback, len(df_copy) - 5):  # Leave 5 bars for confirmation
            current_candle = df_copy.iloc[i]
            avg_body = current_candle['avg_body_size']
            
            # Skip if average body size is too small (low volatility period)
            if avg_body <= 0:
                continue
            
            # Check if current candle is significantly larger than average
            is_strong_candle = current_candle['body_size'] >= (avg_body * strength_factor)
            
            if not is_strong_candle:
                continue
            
            # Analyze the next few candles to confirm the zone
            next_candles = df_copy.iloc[i+1:i+6]  # Next 5 candles
            
            if len(next_candles) < 3:
                continue
            
            # For DEMAND zones: Look for strong bullish candle followed by upward movement
            if current_candle['direction'] == 1:  # Bullish candle
                # Check if price moved significantly higher in the following candles
                future_high = next_candles['high'].max()
                current_close = current_candle['close']
                
                # Calculate the upward movement as percentage
                upward_move = (future_high - current_close) / current_close
                
                # If price moved up significantly (>0.1%), it's a demand zone
                if upward_move > 0.001:  # 0.1% minimum move
                    zone_price = (current_candle['low'] + current_candle['open']) / 2
                    
                    zones.append({
                        'type': 'demand',
                        'price_level': zone_price,
                        'high_price': current_candle['high'],
                        'low_price': current_candle['low'], 
                        'strength': current_candle['body_size'] / avg_body,
                        'timestamp': df_copy.index[i],
                        'candle_index': i,
                        'confirmation_move': upward_move * 100  # Store as percentage
                    })
            
            # For SUPPLY zones: Look for strong bearish candle followed by downward movement
            elif current_candle['direction'] == -1:  # Bearish candle
                # Check if price moved significantly lower in the following candles
                future_low = next_candles['low'].min()
                current_close = current_candle['close']
                
                # Calculate the downward movement as percentage
                downward_move = (current_close - future_low) / current_close
                
                # If price moved down significantly (>0.1%), it's a supply zone
                if downward_move > 0.001:  # 0.1% minimum move
                    zone_price = (current_candle['high'] + current_candle['open']) / 2
                    
                    zones.append({
                        'type': 'supply',
                        'price_level': zone_price,
                        'high_price': current_candle['high'],
                        'low_price': current_candle['low'],
                        'strength': current_candle['body_size'] / avg_body,
                        'timestamp': df_copy.index[i],
                        'candle_index': i,
                        'confirmation_move': downward_move * 100  # Store as percentage
                    })
        
        # Sort zones by strength (strongest first)
        zones.sort(key=lambda x: x['strength'], reverse=True)
        
        # Keep only the strongest zones (max 10 of each type)
        demand_zones = [z for z in zones if z['type'] == 'demand'][:10]
        supply_zones = [z for z in zones if z['type'] == 'supply'][:10]
        
        final_zones = demand_zones + supply_zones
        
        # Suppress output during backtesting to reduce noise
        pass
        
        return final_zones
        
    except Exception as e:
        print(f"✗ Error in supply/demand zone analysis: {e}")
        return []

def get_current_price_context(df: pd.DataFrame, zones: List[Dict[str, Any]], current_price: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze the current price in relation to identified supply/demand zones.
    
    Args:
        df (pd.DataFrame): OHLC data
        zones (List[Dict]): List of supply/demand zones
        current_price (float, optional): Current price (uses last close if not provided)
    
    Returns:
        Dict: Context information about current price position
    """
    try:
        if current_price is None:
            current_price = df['close'].iloc[-1]
        
        if not zones:
            return {
                'current_price': current_price,
                'nearest_supply': None,
                'nearest_demand': None,
                'price_context': 'neutral'
            }
        
        # Find nearest supply zone above current price
        supply_zones = [z for z in zones if z['type'] == 'supply' and z['price_level'] > current_price]
        nearest_supply = min(supply_zones, key=lambda x: abs(x['price_level'] - current_price)) if supply_zones else None
        
        # Find nearest demand zone below current price
        demand_zones = [z for z in zones if z['type'] == 'demand' and z['price_level'] < current_price]
        nearest_demand = max(demand_zones, key=lambda x: x['price_level']) if demand_zones else None
        
        # Determine price context
        price_context = 'neutral'
        
        # Check if price is near a supply zone (within 0.1%)
        if nearest_supply and abs(current_price - nearest_supply['price_level']) / current_price < 0.001:
            price_context = 'near_supply'
        
        # Check if price is near a demand zone (within 0.1%)
        elif nearest_demand and abs(current_price - nearest_demand['price_level']) / current_price < 0.001:
            price_context = 'near_demand'
        
        return {
            'current_price': current_price,
            'nearest_supply': nearest_supply,
            'nearest_demand': nearest_demand,
            'price_context': price_context,
            'supply_distance': (nearest_supply['price_level'] - current_price) / current_price * 100 if nearest_supply else None,
            'demand_distance': (current_price - nearest_demand['price_level']) / current_price * 100 if nearest_demand else None
        }
        
    except Exception as e:
        print(f"✗ Error in price context analysis: {e}")
        return {'current_price': current_price, 'error': str(e)}

def identify_fvg_and_ob(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Identify Fair Value Gaps (FVGs) and Order Blocks (OBs) using ICT concepts.
    
    This function uses the smart-money-concepts library to identify:
    - Fair Value Gaps: Price inefficiencies that need to be filled
    - Order Blocks: Institutional order zones where smart money placed orders
    
    Args:
        df (pd.DataFrame): OHLC data with columns: open, high, low, close
    
    Returns:
        Tuple[List[Dict], List[Dict]]: (FVG list, Order Block list)
    """
    try:
        fvg_list = []
        ob_list = []
        
        # Ensure DataFrame has the required columns
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            print("✗ DataFrame missing required OHLC columns")
            return fvg_list, ob_list
        
        # Need at least 10 candles for meaningful analysis
        if len(df) < 10:
            return fvg_list, ob_list
        
        if SMC_AVAILABLE and smc is not None:
            # Try using smart-money-concepts library first
            try:
                # Prepare data for smart-money-concepts library
                df_smc = df.copy()
                df_smc.columns = df_smc.columns.str.capitalize()  # Capitalize column names
                
                # Set data on the SMC instance
                smc.ohlc = df_smc
                
                # Identify Fair Value Gaps using smart-money-concepts
                fvg_data = smc.fvg()
                
                # Process FVG results - this library returns a Series, not DataFrame
                if isinstance(fvg_data, pd.Series) and len(fvg_data) > 0:
                    # Get indices where FVG exists (non-zero values)
                    fvg_indices = fvg_data[fvg_data != 0].index
                    
                    for idx in fvg_indices[-20:]:  # Last 20 FVGs
                        if idx in df.index:
                            fvg_value = fvg_data[idx]
                            fvg_type = 'bullish' if fvg_value > 0 else 'bearish'
                            
                            fvg_info = {
                                'type': fvg_type,
                                'timestamp': idx,
                                'high_price': df.loc[idx, 'high'],
                                'low_price': df.loc[idx, 'low'],
                                'gap_size': abs(fvg_value),
                                'status': 'unfilled'
                            }
                            
                            if fvg_type == 'bullish':
                                fvg_info['upper_level'] = df.loc[idx, 'low']
                                fvg_info['lower_level'] = fvg_info['upper_level'] - fvg_info['gap_size']
                            else:
                                fvg_info['lower_level'] = df.loc[idx, 'high']
                                fvg_info['upper_level'] = fvg_info['lower_level'] + fvg_info['gap_size']
                            
                            fvg_list.append(fvg_info)
                
                # For Order Blocks, we need swing highs/lows first
                try:
                    swing_hl = smc.swing_highs_lows()
                    ob_data = smc.ob(swing_hl)
                    
                    # Process Order Block results - also returns a Series
                    if isinstance(ob_data, pd.Series) and len(ob_data) > 0:
                        ob_indices = ob_data[ob_data != 0].index
                        
                        for idx in ob_indices[-15:]:  # Last 15 OBs
                            if idx in df.index:
                                ob_value = ob_data[idx]
                                ob_type = 'bullish' if ob_value > 0 else 'bearish'
                                
                                ob_info = {
                                    'type': ob_type,
                                    'timestamp': idx,
                                    'high_price': df.loc[idx, 'high'],
                                    'low_price': df.loc[idx, 'low'],
                                    'open_price': df.loc[idx, 'open'],
                                    'close_price': df.loc[idx, 'close'],
                                    'strength': abs(ob_value),
                                    'status': 'active'
                                }
                                
                                if ob_type == 'bullish':
                                    ob_info['zone_high'] = max(ob_info['open_price'], ob_info['close_price'])
                                    ob_info['zone_low'] = ob_info['low_price']
                                else:
                                    ob_info['zone_high'] = ob_info['high_price']
                                    ob_info['zone_low'] = min(ob_info['open_price'], ob_info['close_price'])
                                
                                ob_list.append(ob_info)
                except Exception:
                    # OB detection failed, will use manual method
                    pass
                            
            except Exception:
                # Fallback to manual detection if library fails
                pass
        
        # Use manual detection if library unavailable or failed
        if not fvg_list:
            fvg_list = detect_fvg_manually(df)
        
        if not ob_list:
            ob_list = detect_ob_manually(df)
        
        # Only print if we found significant patterns (reduce noise in backtesting)
        if len(fvg_list) > 0 or len(ob_list) > 0:
            pass  # Suppress output during backtesting to reduce noise
        
        return fvg_list, ob_list
        
    except Exception as e:
        print(f"✗ Error in FVG/OB analysis: {e}")
        return [], []

def detect_fvg_manually(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Manual Fair Value Gap detection algorithm.
    
    A bullish FVG occurs when: low[i] > high[i-2] (gap up)
    A bearish FVG occurs when: high[i] < low[i-2] (gap down)
    """
    fvg_list = []
    
    try:
        for i in range(2, len(df)):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            prev2_high = df.iloc[i-2]['high']
            prev2_low = df.iloc[i-2]['low']
            
            # Bullish FVG: Current low > Previous high (2 bars ago)
            if current_low > prev2_high:
                gap_size = current_low - prev2_high
                
                fvg_info = {
                    'type': 'bullish',
                    'timestamp': df.index[i],
                    'upper_level': current_low,
                    'lower_level': prev2_high,
                    'gap_size': gap_size,
                    'high_price': current_high,
                    'low_price': current_low,
                    'status': 'unfilled'
                }
                fvg_list.append(fvg_info)
            
            # Bearish FVG: Current high < Previous low (2 bars ago)
            elif current_high < prev2_low:
                gap_size = prev2_low - current_high
                
                fvg_info = {
                    'type': 'bearish',
                    'timestamp': df.index[i],
                    'upper_level': prev2_low,
                    'lower_level': current_high,
                    'gap_size': gap_size,
                    'high_price': current_high,
                    'low_price': current_low,
                    'status': 'unfilled'
                }
                fvg_list.append(fvg_info)
        
        # Keep only the most recent FVGs (last 20)
        fvg_list = fvg_list[-20:]
        
    except Exception as e:
        print(f"✗ Error in manual FVG detection: {e}")
    
    return fvg_list

def detect_ob_manually(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Manual Order Block detection algorithm.
    
    An Order Block is typically the last opposite-colored candle before a strong impulsive move.
    """
    ob_list = []
    
    try:
        # Calculate candle direction and strength
        df_copy = df.copy()
        df_copy['direction'] = np.where(df_copy['close'] > df_copy['open'], 1, -1)
        df_copy['body_size'] = abs(df_copy['close'] - df_copy['open'])
        df_copy['avg_body'] = df_copy['body_size'].rolling(window=10).mean()
        
        for i in range(10, len(df_copy) - 3):
            current_body = df_copy.iloc[i]['body_size']
            avg_body = df_copy.iloc[i]['avg_body']
            
            # Look for strong impulsive moves (body > 1.5x average)
            if current_body > avg_body * 1.5:
                current_direction = df_copy.iloc[i]['direction']
                
                # Look back for the last opposite candle (potential OB)
                for j in range(i-1, max(i-5, 0), -1):
                    if df_copy.iloc[j]['direction'] == -current_direction:
                        # Found potential Order Block
                        ob_candle = df_copy.iloc[j]
                        
                        ob_type = 'bullish' if current_direction == 1 else 'bearish'
                        
                        ob_info = {
                            'type': ob_type,
                            'timestamp': df_copy.index[j],
                            'high_price': ob_candle['high'],
                            'low_price': ob_candle['low'],
                            'open_price': ob_candle['open'],
                            'close_price': ob_candle['close'],
                            'strength': current_body / avg_body,
                            'status': 'active'
                        }
                        
                        # Define zone boundaries
                        if ob_type == 'bullish':
                            ob_info['zone_high'] = max(ob_candle['open'], ob_candle['close'])
                            ob_info['zone_low'] = ob_candle['low']
                        else:
                            ob_info['zone_high'] = ob_candle['high']
                            ob_info['zone_low'] = min(ob_candle['open'], ob_candle['close'])
                        
                        ob_list.append(ob_info)
                        break  # Only take the first (most recent) opposite candle
        
        # Keep only unique and recent OBs (last 15)
        seen_timestamps = set()
        unique_obs = []
        for ob in reversed(ob_list):  # Process in reverse to keep most recent
            if ob['timestamp'] not in seen_timestamps:
                unique_obs.append(ob)
                seen_timestamps.add(ob['timestamp'])
                if len(unique_obs) >= 15:
                    break
        
        ob_list = list(reversed(unique_obs))
        
    except Exception as e:
        print(f"✗ Error in manual OB detection: {e}")
    
    return ob_list

def confirm_m1_reversal_signal(df: pd.DataFrame) -> Optional[str]:
    """
    Confirm M1 reversal signal by analyzing the most recent completed candle.
    
    This function looks for strong reversal patterns in the latest M1 candle
    to confirm entry signals for the range-bound strategy.
    
    Args:
        df (pd.DataFrame): M1 OHLC data with columns: open, high, low, close
    
    Returns:
        str or None: 'Bullish Reversal', 'Bearish Reversal', or None
    """
    try:
        if len(df) < 3:
            return None
        
        # Get the last completed candle
        current_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        # Calculate candle properties
        current_open = current_candle['open']
        current_close = current_candle['close']
        current_high = current_candle['high']
        current_low = current_candle['low']
        
        prev_open = prev_candle['open']
        prev_close = prev_candle['close']
        prev_high = prev_candle['high']
        prev_low = prev_candle['low']
        
        # Calculate current candle body and range
        current_body = abs(current_close - current_open)
        current_range = current_high - current_low
        prev_body = abs(prev_close - prev_open)
        
        # Skip analysis if candle range is zero (shouldn't happen but safety check)
        if current_range == 0 or prev_body == 0:
            return None
        
        # BULLISH REVERSAL CONDITIONS
        if current_close > current_open:  # Current candle is green/bullish
            
            # Condition 1: Strong bullish body (> 60% of total range)
            body_strength = current_body / current_range
            
            # Condition 2: Small upper wick (< 30% of range)
            upper_wick = current_high - current_close
            upper_wick_ratio = upper_wick / current_range
            
            # Condition 3: Current body is larger than previous body (momentum increase)
            body_increase = current_body > prev_body
            
            # Condition 4: Lower wick shows rejection (> 20% of range)
            lower_wick = current_open - current_low
            lower_wick_ratio = lower_wick / current_range
            
            # Check all bullish conditions
            if (body_strength > 0.6 and 
                upper_wick_ratio < 0.3 and 
                body_increase and 
                lower_wick_ratio > 0.2):
                return 'Bullish Reversal'
        
        # BEARISH REVERSAL CONDITIONS  
        elif current_close < current_open:  # Current candle is red/bearish
            
            # Condition 1: Strong bearish body (> 60% of total range)
            body_strength = current_body / current_range
            
            # Condition 2: Small lower wick (< 30% of range)
            lower_wick = current_close - current_low
            lower_wick_ratio = lower_wick / current_range
            
            # Condition 3: Current body is larger than previous body (momentum increase)
            body_increase = current_body > prev_body
            
            # Condition 4: Upper wick shows rejection (> 20% of range)
            upper_wick = current_high - current_open
            upper_wick_ratio = upper_wick / current_range
            
            # Check all bearish conditions
            if (body_strength > 0.6 and 
                lower_wick_ratio < 0.3 and 
                body_increase and 
                upper_wick_ratio > 0.2):
                return 'Bearish Reversal'
        
        # No clear reversal signal detected
        return None
        
    except Exception as e:
        print(f"✗ Error in M1 reversal signal confirmation: {e}")
        return None

def check_for_liquidity_grab(df: pd.DataFrame, key_level: float, tolerance: float = 0.0005) -> bool:
    """
    Check for liquidity grab around a key level (support/resistance).
    
    A liquidity grab occurs when:
    1. Price pierces just below/above a key level (hunting stops)
    2. Price quickly reverses and closes significantly away from the breach
    3. This forms a wick/pin bar pattern
    
    Args:
        df (pd.DataFrame): OHLC data
        key_level (float): Price level to check for liquidity grab
        tolerance (float): How far price can pierce beyond the level
    
    Returns:
        bool: True if liquidity grab detected, False otherwise
    """
    try:
        if len(df) < 5:
            return False
        
        # Check the last few candles for liquidity grab patterns
        recent_candles = df.tail(5)
        
        for idx, candle in recent_candles.iterrows():
            candle_high = candle['high']
            candle_low = candle['low']
            candle_open = candle['open']
            candle_close = candle['close']
            
            # Calculate candle body and total range
            body_size = abs(candle_close - candle_open)
            total_range = candle_high - candle_low
            
            # Skip doji/small candles
            if total_range == 0 or body_size / total_range < 0.3:
                continue
            
            # Check for liquidity grab below key level (bullish reversal)
            if candle_low < (key_level - tolerance):
                # Price pierced below the key level
                
                # Check if candle closed significantly higher than the low
                close_above_low = (candle_close - candle_low) / total_range
                
                # Look for strong reversal: close in upper 60% of candle range
                if close_above_low > 0.6 and candle_close > key_level:
                    print(f"✓ Bullish liquidity grab detected at {key_level:.5f}")
                    print(f"  Candle low: {candle_low:.5f}, close: {candle_close:.5f}")
                    return True
            
            # Check for liquidity grab above key level (bearish reversal)
            elif candle_high > (key_level + tolerance):
                # Price pierced above the key level
                
                # Check if candle closed significantly lower than the high
                close_below_high = (candle_high - candle_close) / total_range
                
                # Look for strong reversal: close in lower 60% of candle range
                if close_below_high > 0.6 and candle_close < key_level:
                    print(f"✓ Bearish liquidity grab detected at {key_level:.5f}")
                    print(f"  Candle high: {candle_high:.5f}, close: {candle_close:.5f}")
                    return True
        
        return False
        
    except Exception as e:
        print(f"✗ Error checking for liquidity grab: {e}")
        return False

def test_analysis_functions():
    """
    Test the core analysis functions with sample data.
    """
    try:
        print("Testing Price Action Analysis Engine...")
        
        # Create sample OHLC data for testing
        dates = pd.date_range('2024-01-01', periods=100, freq='5T')
        
        # Generate realistic price data with trend
        base_price = 1.1000
        price_data = []
        
        for i in range(100):
            # Add some trend and randomness
            trend = i * 0.0001  # Small upward trend
            noise = np.random.normal(0, 0.0005)  # Random noise
            
            price = base_price + trend + noise
            
            # Generate OHLC from base price
            high = price + abs(np.random.normal(0, 0.0003))
            low = price - abs(np.random.normal(0, 0.0003))
            open_price = price + np.random.normal(0, 0.0002)
            close = price + np.random.normal(0, 0.0002)
            
            price_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.randint(1000, 5000)
            })
        
        # Create DataFrame
        df = pd.DataFrame(price_data, index=dates)
        
        print(f"✓ Created sample data: {len(df)} candles")
        print(f"  Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")
        
        # Test 1: Market Structure Analysis
        print("\n1. Testing market structure identification...")
        market_structure = identify_market_structure(df, lookback_period=50)
        print(f"  Market Structure: {market_structure}")
        
        # Test 2: Supply/Demand Zone Analysis
        print("\n2. Testing supply/demand zone identification...")
        zones = find_supply_demand_zones(df, lookback=20, strength_factor=1.5)
        
        if zones:
            print(f"  Found {len(zones)} zones:")
            for zone in zones[:5]:  # Show first 5 zones
                print(f"    {zone['type'].upper()} zone at {zone['price_level']:.5f} (strength: {zone['strength']:.2f})")
        else:
            print("  No significant zones found")
        
        # Test 3: Price Context Analysis
        print("\n3. Testing price context analysis...")
        context = get_current_price_context(df, zones)
        print(f"  Current Price: {context['current_price']:.5f}")
        print(f"  Price Context: {context['price_context']}")
        
        if context['nearest_supply']:
            print(f"  Nearest Supply: {context['nearest_supply']['price_level']:.5f}")
        
        if context['nearest_demand']:
            print(f"  Nearest Demand: {context['nearest_demand']['price_level']:.5f}")
        
        # Test 4: ICT Analysis (FVG and Order Blocks)
        print("\n4. Testing ICT analysis (FVG and Order Blocks)...")
        fvg_list, ob_list = identify_fvg_and_ob(df)
        
        if fvg_list:
            print(f"  Found {len(fvg_list)} Fair Value Gaps:")
            for fvg in fvg_list[:3]:  # Show first 3 FVGs
                print(f"    {fvg['type'].upper()} FVG: {fvg['lower_level']:.5f} - {fvg['upper_level']:.5f}")
        
        if ob_list:
            print(f"  Found {len(ob_list)} Order Blocks:")
            for ob in ob_list[:3]:  # Show first 3 OBs
                print(f"    {ob['type'].upper()} OB: {ob['zone_low']:.5f} - {ob['zone_high']:.5f}")
        
        # Test 5: Liquidity Grab Detection
        print("\n5. Testing liquidity grab detection...")
        if zones:
            # Test with a supply zone level
            supply_zones = [z for z in zones if z['type'] == 'supply']
            if supply_zones:
                test_level = supply_zones[0]['price_level']
                liquidity_grab = check_for_liquidity_grab(df, test_level)
                print(f"  Liquidity grab at {test_level:.5f}: {liquidity_grab}")
            else:
                print("  No supply zones available for liquidity grab test")
        else:
            print("  No zones available for liquidity grab test")
        
        # Test 6: M1 Reversal Signal Detection
        print("\n6. Testing M1 reversal signal detection...")
        reversal_signal = confirm_m1_reversal_signal(df)
        if reversal_signal:
            print(f"  M1 Reversal Signal: {reversal_signal}")
        else:
            print("  No M1 reversal signal detected")
        
        print("\n✓ All analysis engine tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Analysis engine test failed: {e}")
        raise

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_analysis_functions()