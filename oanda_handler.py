"""
OANDA API Handler Module

This module provides functions to interact with the OANDA v20 REST API.
It handles authentication, data fetching, order placement, and account management.
"""

import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints import instruments, orders, accounts, trades, positions
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

# Load environment variables from .env file
load_dotenv()

def get_api_client():
    """
    Creates and returns an authenticated OANDA API client.
    
    Returns:
        oandapyV20.API: Authenticated API client instance
    """
    try:
        # Get credentials from environment variables
        api_key = os.getenv('OANDA_API_KEY')
        environment = os.getenv('OANDA_ENVIRONMENT', 'practice')
        
        if not api_key:
            raise ValueError("OANDA_API_KEY not found in environment variables")
        
        # Create API client with appropriate environment
        if environment == 'live':
            client = API(access_token=api_key, environment="live")
        else:
            client = API(access_token=api_key, environment="practice")
        
        print(f"✓ OANDA API client created successfully ({environment} environment)")
        return client
        
    except Exception as e:
        print(f"✗ Error creating OANDA API client: {e}")
        raise

def get_historical_data(client, instrument, count=100, granularity='M5'):
    """
    Fetches historical candlestick data from OANDA.
    
    Args:
        client (oandapyV20.API): Authenticated API client
        instrument (str): Instrument name (e.g., 'EUR_USD')
        count (int): Number of candles to fetch (max 5000)
        granularity (str): Timeframe ('M1', 'M5', 'M15', 'H1', etc.)
    
    Returns:
        pd.DataFrame: DataFrame with columns: time, open, high, low, close, volume
    """
    try:
        # Create request for historical data
        params = {
            "count": count,
            "granularity": granularity,
            "price": "MBA"  # Mid, Bid, Ask prices
        }
        
        # Make API request
        request = instruments.InstrumentsCandles(instrument=instrument, params=params)
        response = client.request(request)
        
        # Extract candle data
        candles = response['candles']
        
        # Convert to DataFrame
        data = []
        for candle in candles:
            if candle['complete']:  # Only use complete candles
                data.append({
                    'time': candle['time'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Set time as index
        df.set_index('time', inplace=True)
        
        print(f"✓ Successfully fetched {len(df)} candles for {instrument} ({granularity})")
        return df
        
    except Exception as e:
        print(f"✗ Error fetching historical data for {instrument}: {e}")
        raise

def place_market_order(client, instrument, units, stop_loss_price=None, take_profit_price=None):
    """
    Places a market order with optional stop loss and take profit.
    
    Args:
        client (oandapyV20.API): Authenticated API client
        instrument (str): Instrument name (e.g., 'EUR_USD')
        units (int): Number of units (positive for long, negative for short)
        stop_loss_price (float, optional): Stop loss price
        take_profit_price (float, optional): Take profit price
    
    Returns:
        dict: Order response from OANDA API
    """
    try:
        # Get account ID from environment
        account_id = os.getenv('OANDA_ACCOUNT_ID')
        if not account_id:
            raise ValueError("OANDA_ACCOUNT_ID not found in environment variables")
        
        # Create market order request
        order_data = {
            "instrument": instrument,
            "units": str(units)
        }
        
        # Add stop loss if provided
        if stop_loss_price is not None:
            order_data["stopLossOnFill"] = {
                "price": str(stop_loss_price),
                "timeInForce": "GTC"
            }
        
        # Add take profit if provided
        if take_profit_price is not None:
            order_data["takeProfitOnFill"] = {
                "price": str(take_profit_price),
                "timeInForce": "GTC"
            }
        
        # Create order request
        order_request = MarketOrderRequest(**order_data)
        
        # Place order
        request = orders.OrderCreate(accountID=account_id, data=order_request.data)
        response = client.request(request)
        
        # Extract order information
        if 'orderFillTransaction' in response:
            fill_transaction = response['orderFillTransaction']
            trade_id = fill_transaction.get('tradeOpened', {}).get('tradeID', 'N/A')
            fill_price = float(fill_transaction.get('price', 0))
            
            direction = "LONG" if int(units) > 0 else "SHORT"
            print(f"✓ Order executed successfully:")
            print(f"  - Trade ID: {trade_id}")
            print(f"  - Instrument: {instrument}")
            print(f"  - Direction: {direction}")
            print(f"  - Units: {abs(int(units))}")
            print(f"  - Fill Price: {fill_price}")
            if stop_loss_price:
                print(f"  - Stop Loss: {stop_loss_price}")
            if take_profit_price:
                print(f"  - Take Profit: {take_profit_price}")
        else:
            print(f"✓ Order placed but no fill information available")
            print(f"  - Response: {response}")
        
        return response
        
    except Exception as e:
        print(f"✗ Error placing market order: {e}")
        raise

def get_tradable_instruments(client):
    """
    Fetches all tradable currency instruments from OANDA account.
    
    Args:
        client (oandapyV20.API): Authenticated API client
    
    Returns:
        list: Sorted list of tradable currency pair names (e.g., ['AUD_USD', 'EUR_USD', ...])
    """
    try:
        # Get account ID from environment
        account_id = os.getenv('OANDA_ACCOUNT_ID')
        if not account_id:
            raise ValueError("OANDA_ACCOUNT_ID not found in environment variables")
        
        # Create request for account instruments
        request = accounts.AccountInstruments(accountID=account_id)
        response = client.request(request)
        
        # Extract instrument data
        instruments_data = response['instruments']
        
        # Filter for currency pairs only, excluding volatile exotic pairs
        excluded_currencies = {'TRY', 'SEK', 'DKK', 'HUF', 'CZK', 'PLN'}
        tradable_instruments = []
        
        for instrument in instruments_data:
            # Only include currency instruments
            if instrument.get('type') == 'CURRENCY':
                instrument_name = instrument.get('name', '')
                
                # Check if instrument contains any excluded currencies
                contains_excluded = any(currency in instrument_name for currency in excluded_currencies)
                
                if not contains_excluded:
                    tradable_instruments.append(instrument_name)
        
        # Sort the list for consistent ordering
        tradable_instruments.sort()
        
        print(f"✓ Successfully loaded {len(tradable_instruments)} tradable currency instruments")
        print(f"  Sample instruments: {tradable_instruments[:5]}{'...' if len(tradable_instruments) > 5 else ''}")
        
        return tradable_instruments
        
    except Exception as e:
        print(f"✗ Error fetching tradable instruments: {e}")
        # Return a fallback list of major pairs if API call fails
        fallback_instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD']
        print(f"  Using fallback instrument list: {fallback_instruments}")
        return fallback_instruments

def get_account_summary(client):
    """
    Fetches and displays account summary information.
    
    Args:
        client (oandapyV20.API): Authenticated API client
    
    Returns:
        dict: Account summary data
    """
    try:
        # Get account ID from environment
        account_id = os.getenv('OANDA_ACCOUNT_ID')
        if not account_id:
            raise ValueError("OANDA_ACCOUNT_ID not found in environment variables")
        
        # Create account summary request
        request = accounts.AccountSummary(accountID=account_id)
        response = client.request(request)
        
        # Extract account information
        account_data = response['account']
        
        # Display key account details
        print("=" * 50)
        print("ACCOUNT SUMMARY")
        print("=" * 50)
        print(f"Account ID: {account_data.get('id', 'N/A')}")
        print(f"Currency: {account_data.get('currency', 'N/A')}")
        print(f"Balance: {float(account_data.get('balance', 0)):,.2f}")
        print(f"NAV: {float(account_data.get('NAV', 0)):,.2f}")
        print(f"Unrealized P&L: {float(account_data.get('unrealizedPL', 0)):,.2f}")
        print(f"Realized P&L: {float(account_data.get('pl', 0)):,.2f}")
        print(f"Margin Available: {float(account_data.get('marginAvailable', 0)):,.2f}")
        print(f"Margin Used: {float(account_data.get('marginUsed', 0)):,.2f}")
        print(f"Open Trades: {account_data.get('openTradeCount', 0)}")
        print(f"Open Positions: {account_data.get('openPositionCount', 0)}")
        print("=" * 50)
        
        return account_data
        
    except Exception as e:
        print(f"✗ Error fetching account summary: {e}")
        raise

def get_open_positions_from_oanda(client):
    """
    Fetches current open positions directly from OANDA account.
    
    Args:
        client (oandapyV20.API): Authenticated API client
    
    Returns:
        list: List of open positions with instrument and units information
    """
    try:
        # Get account ID from environment
        account_id = os.getenv('OANDA_ACCOUNT_ID')
        if not account_id:
            raise ValueError("OANDA_ACCOUNT_ID not found in environment variables")
        
        # Create positions request
        request = positions.OpenPositions(accountID=account_id)
        response = client.request(request)
        
        # Extract position data
        positions_data = response.get('positions', [])
        
        open_positions = []
        for position in positions_data:
            instrument = position.get('instrument', '')
            long_units = float(position.get('long', {}).get('units', '0'))
            short_units = float(position.get('short', {}).get('units', '0'))
            
            # Only include positions with actual units
            if long_units != 0 or short_units != 0:
                open_positions.append({
                    'instrument': instrument,
                    'long_units': long_units,
                    'short_units': short_units,
                    'net_units': long_units + short_units
                })
        
        print(f"✓ Found {len(open_positions)} open positions in OANDA account")
        for pos in open_positions:
            print(f"  - {pos['instrument']}: Net {pos['net_units']} units")
        
        return open_positions
        
    except Exception as e:
        print(f"✗ Error fetching open positions from OANDA: {e}")
        return []

def get_open_trades_from_oanda(client):
    """
    Fetches current open trades directly from OANDA account.
    
    Args:
        client (oandapyV20.API): Authenticated API client
    
    Returns:
        list: List of open trades with detailed information
    """
    try:
        # Get account ID from environment
        account_id = os.getenv('OANDA_ACCOUNT_ID')
        if not account_id:
            raise ValueError("OANDA_ACCOUNT_ID not found in environment variables")
        
        # Create trades request
        request = trades.OpenTrades(accountID=account_id)
        response = client.request(request)
        
        # Extract trade data
        trades_data = response.get('trades', [])
        
        open_trades = []
        for trade in trades_data:
            open_trades.append({
                'trade_id': trade.get('id', ''),
                'instrument': trade.get('instrument', ''),
                'units': float(trade.get('currentUnits', '0')),
                'price': float(trade.get('price', '0')),
                'unrealized_pl': float(trade.get('unrealizedPL', '0')),
                'open_time': trade.get('openTime', ''),
                'state': trade.get('state', '')
            })
        
        print(f"✓ Found {len(open_trades)} open trades in OANDA account")
        for trade in open_trades:
            direction = "LONG" if trade['units'] > 0 else "SHORT"
            print(f"  - Trade {trade['trade_id']}: {trade['instrument']} {direction} {abs(trade['units'])} units @ {trade['price']}")
        
        return open_trades
        
    except Exception as e:
        print(f"✗ Error fetching open trades from OANDA: {e}")
        return []

def sync_database_with_oanda_positions(client):
    """
    Synchronizes local database trade status with actual OANDA positions.
    Updates database to close trades that are no longer open in OANDA.
    
    Args:
        client (oandapyV20.API): Authenticated API client
    
    Returns:
        dict: Summary of synchronization results
    """
    try:
        from journal import get_open_trades
        from datetime import datetime
        import sqlite3
        
        # Get open trades from both sources
        oanda_trades = get_open_trades_from_oanda(client)
        
        # Use the correct database path relative to where the function is called
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, 'trading_journal.db')
        
        db_trades = get_open_trades(db_path)
        
        # Create sets of trade IDs for comparison
        oanda_trade_ids = {trade['trade_id'] for trade in oanda_trades if trade['trade_id']}
        
        # Find trades that are in database but not in OANDA (should be closed)
        trades_to_close = []
        for db_trade in db_trades:
            db_trade_id = db_trade.get('trade_id', '')
            if db_trade_id and db_trade_id not in oanda_trade_ids:
                trades_to_close.append(db_trade)
        
        # Close orphaned trades in database using direct SQLite approach
        closed_count = 0
        if trades_to_close:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                for trade in trades_to_close:
                    try:
                        trade_id = trade.get('trade_id', '')
                        entry_price = float(trade.get('entry_price', 0))
                        exit_time = datetime.now().isoformat()
                        
                        print(f"  Closing orphaned trade in database: {trade['instrument']} (ID: {trade_id})")
                        
                        # Update the trade status to CLOSED
                        cursor.execute('''
                            UPDATE trades 
                            SET status = 'CLOSED', 
                                exit_price = ?, 
                                exit_time = ?,
                                profit_loss = 0.0
                            WHERE trade_id = ?
                        ''', (entry_price, exit_time, trade_id))
                        
                        if cursor.rowcount > 0:
                            closed_count += 1
                        else:
                            print(f"    Warning: No rows updated for trade {trade_id}")
                            
                    except Exception as e:
                        print(f"  Error closing trade {trade.get('trade_id', 'N/A')}: {e}")
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                print(f"  Database connection error: {e}")
        
        sync_summary = {
            'oanda_open_trades': len(oanda_trades),
            'database_open_trades': len(db_trades),
            'orphaned_trades_found': len(trades_to_close),
            'trades_closed_in_db': closed_count,
            'sync_timestamp': datetime.now().isoformat()
        }
        
        if trades_to_close:
            print(f"✓ Database sync completed: {closed_count} orphaned trades closed")
        else:
            print("✓ Database sync completed: No orphaned trades found")
        
        return sync_summary
        
    except Exception as e:
        print(f"✗ Error synchronizing database with OANDA positions: {e}")
        return {
            'error': str(e),
            'sync_timestamp': datetime.now().isoformat()
        }

# Test function to verify API connection
def test_connection():
    """
    Tests the OANDA API connection and basic functionality.
    """
    try:
        print("Testing OANDA API connection...")
        
        # Create API client
        client = get_api_client()
        
        # Test account access
        account_summary = get_account_summary(client)
        
        # Test data fetching
        print("\nTesting data fetch...")
        df = get_historical_data(client, 'EUR_USD', count=10, granularity='M5')
        print(f"Sample data (last 5 rows):")
        print(df.tail())
        
        # Test tradable instruments fetch
        print("\nTesting tradable instruments fetch...")
        instruments = get_tradable_instruments(client)
        print(f"Found {len(instruments)} tradable instruments")
        
        print("\n✓ All tests passed! OANDA API handler is working correctly.")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")

if __name__ == "__main__":
    # Run connection test when script is executed directly
    test_connection()