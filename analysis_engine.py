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
        
        print(f"✓ Identified {len(demand_zones)} demand zones and {len(supply_zones)} supply zones")
        
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
        
        print("\n✓ All analysis engine tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Analysis engine test failed: {e}")
        raise

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_analysis_functions()