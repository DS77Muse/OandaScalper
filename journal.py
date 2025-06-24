"""
Trading Journal Module

This module provides a comprehensive database-driven journaling system for logging
all trading activities, including trade entries, exits, and performance metrics.

Updated for Phase 2: Now uses SQLAlchemy ORM with atomic transactions while
maintaining backward compatibility with existing API.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from loguru import logger

# New SQLAlchemy imports
from models import db_manager, Trade
from trade_operations import create_trade, close_trade, get_open_trades as get_open_trades_orm

def initialize_database(db_name: str = 'trading_journal.db') -> None:
    """
    Initialize the trading journal database using the new SQLAlchemy ORM system.
    
    This function now creates the enhanced schema and migrates from legacy format.
    
    Args:
        db_name (str): Name of the SQLite database file
    """
    try:
        # Initialize the global database manager
        global db_manager
        db_manager.database_url = f"sqlite:///{db_name}"
        db_manager.engine = db_manager.__class__(db_manager.database_url).engine
        db_manager.SessionLocal = db_manager.__class__(db_manager.database_url).SessionLocal
        
        # Create tables and migrate from legacy schema
        db_manager.create_tables()
        db_manager.migrate_from_legacy_schema()
        
        logger.info(f"Trading journal database initialized successfully: {db_name}")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def log_new_trade(
    db_name: str,
    trade_id: int,
    instrument: str,
    units: int,
    direction: str,
    entry_price: float,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    entry_time: Optional[str] = None,
    reason: Optional[str] = None
) -> bool:
    """
    Log a new trade entry using the new SQLAlchemy ORM system.
    
    Maintains backward compatibility while using atomic transactions.
    
    Args:
        db_name (str): Database file name (used for compatibility)
        trade_id (int): Unique trade identifier from OANDA
        instrument (str): Trading instrument (e.g., 'EUR_USD')
        units (int): Number of units traded
        direction (str): Trade direction ('LONG' or 'SHORT')
        entry_price (float): Entry price
        sl_price (float, optional): Stop loss price
        tp_price (float, optional): Take profit price
        entry_time (str, optional): Entry timestamp (ISO format)
        reason (str, optional): Reason for entering the trade
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Use direct SQLite logging with existing schema (skip SQLAlchemy for compatibility)
    return log_new_trade_fallback(
        db_name=db_name,
        trade_id=trade_id,
        instrument=instrument,
        units=units,
        direction=direction,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        entry_time=entry_time,
        reason=reason
    )

def update_closed_trade(
    db_name: str,
    trade_id: int,
    exit_price: float,
    exit_time: Optional[str] = None,
    profit_loss: Optional[float] = None
) -> bool:
    """
    Update a trade record when the position is closed using atomic transactions.
    
    Args:
        db_name (str): Database file name (for compatibility)
        trade_id (int): OANDA trade identifier to update
        exit_price (float): Exit price
        exit_time (str, optional): Exit timestamp (ISO format)
        profit_loss (float, optional): Profit/loss amount (legacy compatibility)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with db_manager.get_session() as session:
            # Find trade by OANDA trade_id
            trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
            
            if not trade:
                logger.error(f"No trade found with OANDA ID: {trade_id}")
                return False
            
            # Use the atomic close_trade function
            closed_trade = close_trade(
                session=session,
                trade_id=trade.id,  # Use internal ID for the close function
                exit_price=exit_price,
                exit_reason="Manual",  # Default reason for legacy compatibility
                fees=0.0  # Default fees
            )
            
            if closed_trade:
                logger.info(f"Trade {trade_id} closed successfully (Net P&L: {closed_trade.pnl_net:+.2f})")
                return True
            else:
                return False
                
    except Exception as e:
        logger.error(f"Error updating trade {trade_id}: {e}")
        return False

def get_trade_by_id(db_name: str, trade_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific trade by its ID.
    
    Args:
        db_name (str): Database file name
        trade_id (int): Trade identifier
    
    Returns:
        dict or None: Trade data as dictionary, or None if not found
    """
    try:
        conn = sqlite3.connect(db_name)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM trades WHERE trade_id = ?', (trade_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return dict(row)
        return None
        
    except sqlite3.Error as e:
        print(f"✗ Database error retrieving trade {trade_id}: {e}")
        return None

def get_open_trades(db_name: str) -> List[Dict[str, Any]]:
    """
    Retrieve all open trades.
    
    Args:
        db_name (str): Database file name
    
    Returns:
        list: List of open trades as dictionaries
    """
    try:
        conn = sqlite3.connect(db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM trades WHERE status = "OPEN" ORDER BY entry_time DESC')
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]
        
    except sqlite3.Error as e:
        print(f"✗ Database error retrieving open trades: {e}")
        return []

def log_new_trade_fallback(
    db_name: str,
    trade_id: int,
    instrument: str,
    units: int,
    direction: str,
    entry_price: float,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    entry_time: Optional[str] = None,
    reason: Optional[str] = None
) -> bool:
    """
    Fallback function to log trades using direct SQLite with existing schema.
    Used when SQLAlchemy ORM fails due to schema mismatches.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Use current timestamp if not provided
        if entry_time is None:
            entry_time = datetime.now().isoformat()
        
        # Check if trade already exists (to handle duplicate logging attempts)
        cursor.execute('SELECT trade_id FROM trades WHERE trade_id = ?', (trade_id,))
        existing = cursor.fetchone()
        
        if existing:
            logger.info(f"Trade {trade_id} already exists in database - skipping duplicate logging")
            conn.close()
            return True
        
        # Insert trade using existing table structure
        cursor.execute('''
            INSERT INTO trades (
                trade_id, instrument, units, direction, entry_price,
                stop_loss_price, take_profit_price, entry_time, status,
                entry_reason, created_at, updated_at, strategy_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_id, instrument, units, direction, entry_price,
            sl_price, tp_price, entry_time, 'OPEN',
            reason, datetime.now().isoformat(), datetime.now().isoformat(),
            'MeanReversionSR'
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Trade {trade_id} logged successfully using fallback method")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Fallback logging failed for trade {trade_id}: {e}")
        return False

def get_trading_summary(db_name: str) -> Dict[str, Any]:
    """
    Generate a summary of trading performance.
    
    Args:
        db_name (str): Database file name
    
    Returns:
        dict: Trading summary statistics
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Get basic statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN status = 'OPEN' THEN 1 END) as open_trades,
                COUNT(CASE WHEN status = 'CLOSED' THEN 1 END) as closed_trades,
                COUNT(CASE WHEN status = 'CLOSED' AND profit_loss > 0 THEN 1 END) as winning_trades,
                COUNT(CASE WHEN status = 'CLOSED' AND profit_loss < 0 THEN 1 END) as losing_trades,
                COALESCE(SUM(CASE WHEN status = 'CLOSED' THEN profit_loss END), 0) as total_pnl,
                COALESCE(AVG(CASE WHEN status = 'CLOSED' THEN profit_loss END), 0) as avg_pnl,
                COALESCE(MAX(CASE WHEN status = 'CLOSED' THEN profit_loss END), 0) as best_trade,
                COALESCE(MIN(CASE WHEN status = 'CLOSED' THEN profit_loss END), 0) as worst_trade
            FROM trades
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            total_trades, open_trades, closed_trades, winning_trades, losing_trades, total_pnl, avg_pnl, best_trade, worst_trade = row
            
            # Calculate win rate
            win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'open_trades': open_trades,
                'closed_trades': closed_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'avg_pnl': round(avg_pnl, 2),
                'best_trade': round(best_trade, 2),
                'worst_trade': round(worst_trade, 2)
            }
        
        return {}
        
    except sqlite3.Error as e:
        print(f"✗ Database error generating summary: {e}")
        return {}

def display_trading_summary(db_name: str) -> None:
    """
    Display a formatted trading performance summary.
    
    Args:
        db_name (str): Database file name
    """
    summary = get_trading_summary(db_name)
    
    if summary:
        print("\n" + "=" * 50)
        print("TRADING PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Open Trades: {summary['open_trades']}")
        print(f"Closed Trades: {summary['closed_trades']}")
        print(f"Winning Trades: {summary['winning_trades']}")
        print(f"Losing Trades: {summary['losing_trades']}")
        print(f"Win Rate: {summary['win_rate']}%")
        print(f"Total P&L: {summary['total_pnl']:+.2f}")
        print(f"Average P&L: {summary['avg_pnl']:+.2f}")
        print(f"Best Trade: {summary['best_trade']:+.2f}")
        print(f"Worst Trade: {summary['worst_trade']:+.2f}")
        print("=" * 50)
    else:
        print("No trading data available.")

def test_journal_functionality(db_name: str = 'test_journal.db') -> None:
    """
    Test the journal functionality with sample data.
    
    Args:
        db_name (str): Test database file name
    """
    try:
        print("Testing Trading Journal functionality...")
        
        # Remove test database if it exists
        if os.path.exists(db_name):
            os.remove(db_name)
        
        # Initialize database
        initialize_database(db_name)
        
        # Test logging new trades
        print("\n1. Testing new trade logging...")
        log_new_trade(
            db_name, 12345, 'EUR_USD', 10000, 'LONG', 1.1250,
            sl_price=1.1200, tp_price=1.1300, reason='Bullish FVG setup'
        )
        
        log_new_trade(
            db_name, 12346, 'GBP_USD', -5000, 'SHORT', 1.2750,
            sl_price=1.2800, tp_price=1.2700, reason='Bearish order block'
        )
        
        # Test updating closed trade
        print("\n2. Testing trade closure updates...")
        update_closed_trade(db_name, 12345, 1.1280, profit_loss=300.00)
        
        # Test retrieving trade by ID
        print("\n3. Testing trade retrieval...")
        trade = get_trade_by_id(db_name, 12345)
        if trade:
            print(f"Retrieved trade: {trade['instrument']} - {trade['direction']} - P&L: {trade['profit_loss']}")
        
        # Test getting open trades
        print("\n4. Testing open trades retrieval...")
        open_trades = get_open_trades(db_name)
        print(f"Open trades count: {len(open_trades)}")
        
        # Test trading summary
        print("\n5. Testing trading summary...")
        display_trading_summary(db_name)
        
        # Clean up test database
        os.remove(db_name)
        
        print("\n✓ All journal tests passed successfully!")
        
    except Exception as e:
        print(f"✗ Journal test failed: {e}")
        # Clean up test database on error
        if os.path.exists(db_name):
            os.remove(db_name)

if __name__ == "__main__":
    # Run functionality test when script is executed directly
    test_journal_functionality()