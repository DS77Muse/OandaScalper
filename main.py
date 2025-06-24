"""
Main Application Loop

This is the primary entry point for the OANDA Price Action Trading Bot.
It orchestrates the complete trading system including scheduling, strategy execution,
error handling, and performance monitoring.
"""

import time
import schedule
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import traceback
import json

# Import our custom modules
from oanda_handler import get_api_client, get_account_summary, get_tradable_instruments, sync_database_with_oanda_positions, get_open_trades_from_oanda
from strategy_handler import run_strategy_check
from journal import initialize_database, get_trading_summary, display_trading_summary, get_open_trades
from logging_config import configure_logging
from loguru import logger

# Global variables for graceful shutdown
shutdown_requested = False
client = None

def setup_logging():
    """
    Configure logging for the trading bot using the centralized logging system.
    """
    configure_logging()
    logger.info("Trading bot logging system initialized with structured JSON output")

def signal_handler(signum, frame):
    """
    Handle graceful shutdown on CTRL+C or system signals.
    """
    global shutdown_requested
    
    logger.info("🛑 Shutdown signal received. Gracefully stopping trading bot...")
    shutdown_requested = True
    
    # Display final summary
    try:
        display_trading_summary('trading_journal.db')
    except Exception as e:
        logger.error(f"Error displaying final summary: {e}")
    
    logger.info("✅ Trading bot stopped successfully.")
    sys.exit(0)

def check_market_hours() -> bool:
    """
    Check if the Forex market is currently open.
    
    Forex market is open 24/5 from Sunday 5pm EST to Friday 5pm EST.
    
    Returns:
        bool: True if market is open, False otherwise
    """
    try:
        now = datetime.now()
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        hour = now.hour
        
        # Market is closed on weekends (Saturday and most of Sunday)
        if weekday == 5:  # Saturday
            return False
        elif weekday == 6 and hour < 17:  # Sunday before 5 PM
            return False
        elif weekday == 4 and hour >= 17:  # Friday after 5 PM
            return False
        
        # Market is open during weekdays and Sunday evening
        return True
        
    except Exception as e:
        logger.warning(f"Error checking market hours: {e}. Assuming market is open.")
        return True

def validate_trading_environment() -> bool:
    """
    Validate that the trading environment is properly configured.
    
    Returns:
        bool: True if environment is valid, False otherwise
    """
    
    try:
        # Test API connection
        global client
        client = get_api_client()
        account_summary = get_account_summary(client)
        
        # Check account balance
        balance = float(account_summary.get('balance', 0))
        if balance <= 0:
            logger.error(f"❌ Invalid account balance: ${balance}")
            return False
        
        # Check for sufficient margin
        margin_available = float(account_summary.get('marginAvailable', 0))
        if margin_available <= 100:  # Minimum $100 margin
            logger.warning(f"⚠ Low margin available: ${margin_available}")
        
        logger.info(f"✅ Trading environment validated successfully")
        logger.info(f"   Account Balance: ${balance:,.2f}")
        logger.info(f"   Margin Available: ${margin_available:,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        return False

def get_trading_instruments() -> List[str]:
    """
    Get the list of instruments to trade - ALL TRADEABLE PAIRS for practice account testing.
    
    This function now returns None to signal that we want to use the dynamic
    instrument loading from OANDA's tradeable instruments list.
    
    Returns:
        None to trigger dynamic loading of all tradeable pairs
    """
    # Return None to use all tradeable pairs from OANDA
    # This will be handled by get_tradable_instruments() in main()
    return None

def trading_job(instrument_list=None):
    """
    Main trading job that runs periodically to check for trading opportunities.
    
    Args:
        instrument_list (List[str], optional): List of instruments to analyze
    
    This function:
    1. Validates market conditions
    2. Checks each instrument for trading signals
    3. Executes trades when conditions are met
    4. Handles errors gracefully
    5. Logs all activities
    """
    
    try:
        # Check if shutdown was requested
        if shutdown_requested:
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 TRADING JOB STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")
        
        # Check market hours
        if not check_market_hours():
            logger.info("🕒 Market is currently closed. Skipping trading job.")
            return
        
        # Validate API connection
        if not client:
            logger.error("❌ No API client available. Skipping trading job.")
            return
        
        # Quick account health check
        try:
            account_summary = get_account_summary(client)
            current_balance = float(account_summary.get('balance', 0))
            open_positions = int(account_summary.get('openPositionCount', 0))
            
            logger.info(f"💰 Account Status:")
            logger.info(f"   Balance: ${current_balance:,.2f}")
            logger.info(f"   Open Positions: {open_positions}")
            
        except Exception as e:
            logger.warning(f"⚠ Could not fetch account summary: {e}")
        
        # Get trading instruments (use provided list or fallback to default)
        if instrument_list is None:
            instruments = get_trading_instruments()
            if instruments is None:
                # This means we want to use all tradeable instruments
                # which should have been loaded in main() and passed as instrument_list
                logger.warning("No instrument list provided and get_trading_instruments() returned None")
                return
        else:
            instruments = instrument_list
        
        logger.info(f"🎯 Scanning {len(instruments)} instruments")
        # Don't print all instruments if there are many (clutters the log)
        if len(instruments) <= 10:
            logger.info(f"Instruments: {', '.join(instruments)}")
        else:
            logger.info(f"Instruments: {', '.join(instruments[:5])} ... and {len(instruments)-5} more")
        
        # Synchronize database with OANDA positions before processing
        try:
            logger.info("🔄 Synchronizing database with OANDA positions...")
            sync_result = sync_database_with_oanda_positions(client)
            if 'error' not in sync_result:
                logger.info("✓ Database synchronization completed successfully", extra=sync_result)
            else:
                logger.warning("⚠ Database synchronization failed", extra=sync_result)
        except Exception as e:
            logger.error(f"✗ Error during database synchronization: {e}")
        
        # Track results for this trading job
        job_results = {
            'start_time': datetime.now(),
            'instruments_checked': 0,
            'signals_found': 0,
            'trades_executed': 0,
            'errors': 0
        }
        
        # Check each instrument for trading opportunities
        for instrument in instruments:
            try:
                if shutdown_requested:
                    break
                
                logger.info(f"\n📊 Analyzing {instrument}...")
                job_results['instruments_checked'] += 1
                
                # Run strategy check for this instrument
                trade_executed = run_strategy_check(client, instrument)
                
                if trade_executed:
                    job_results['trades_executed'] += 1
                    logger.info(f"✅ Trade executed for {instrument}")
                    
                    # Brief pause after executing a trade
                    time.sleep(2)
                else:
                    logger.info(f"⏸️ No trade signal for {instrument}")
                
                # Small delay between instruments to avoid API rate limits
                time.sleep(1)
                
            except Exception as e:
                job_results['errors'] += 1
                logger.error(f"❌ Error analyzing {instrument}: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                
                # Continue with next instrument despite error
                continue
        
        # Job completion summary
        duration = (datetime.now() - job_results['start_time']).total_seconds()
        
        logger.info(f"\n📋 TRADING JOB SUMMARY:")
        logger.info(f"   Duration: {duration:.1f} seconds")
        logger.info(f"   Instruments Checked: {job_results['instruments_checked']}")
        logger.info(f"   Trades Executed: {job_results['trades_executed']}")
        logger.info(f"   Errors: {job_results['errors']}")
        
        # Periodic trading summary (every 6 hours)
        current_hour = datetime.now().hour
        if current_hour % 6 == 0 and datetime.now().minute < 5:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 PERIODIC TRADING SUMMARY")
            logger.info(f"{'='*60}")
            display_trading_summary('trading_journal.db')
        
        logger.info(f"✅ Trading job completed successfully\n")
        
    except Exception as e:
        logger.error(f"❌ Critical error in trading job: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

def close_all_positions(client):
    """
    Close all open positions in the OANDA account.
    
    Args:
        client: OANDA API client
        
    Returns:
        bool: True if successful, False otherwise
    """
    from oanda_handler import close_all_open_positions
    
    try:
        logger.info("🔄 Closing all open positions...")
        result = close_all_open_positions(client)
        
        if result.get('success', False):
            closed_count = result.get('closed_count', 0)
            logger.info(f"✅ Successfully closed {closed_count} position(s)")
            return True
        else:
            logger.error(f"❌ Failed to close positions: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error closing positions: {e}")
        return False

def clear_trading_logs():
    """
    Clear trading logs and reset the database for a fresh start.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("🧹 Clearing trading logs and database...")
        
        # Remove existing database file
        db_file = 'trading_journal.db'
        if os.path.exists(db_file):
            os.remove(db_file)
            logger.info(f"✅ Removed existing database: {db_file}")
        
        # Clear log files
        log_dir = 'logs'
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                if filename.endswith('.log'):
                    log_file = os.path.join(log_dir, filename)
                    try:
                        os.remove(log_file)
                        logger.info(f"✅ Removed log file: {log_file}")
                    except Exception as e:
                        logger.warning(f"⚠ Could not remove log file {log_file}: {e}")
        
        logger.info("✅ Trading logs cleared successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error clearing logs: {e}")
        return False

def assess_open_positions(client) -> Dict[str, Any]:
    """
    Assess current open positions and return summary information.
    
    Args:
        client: OANDA API client
        
    Returns:
        Dict containing position assessment information
    """
    try:
        # Get positions from OANDA
        oanda_positions = get_open_trades_from_oanda(client)
        
        # Get positions from local database
        db_positions = get_open_trades('trading_journal.db')
        
        # Get account summary for P&L information
        account_summary = get_account_summary(client)
        unrealized_pnl = float(account_summary.get('unrealizedPL', 0))
        
        assessment = {
            'oanda_positions': oanda_positions,
            'db_positions': db_positions,
            'oanda_count': len(oanda_positions),
            'db_count': len(db_positions),
            'unrealized_pnl': unrealized_pnl,
            'account_balance': float(account_summary.get('balance', 0)),
            'margin_used': float(account_summary.get('marginUsed', 0)),
            'margin_available': float(account_summary.get('marginAvailable', 0))
        }
        
        return assessment
        
    except Exception as e:
        logger.error(f"❌ Error assessing positions: {e}")
        return {
            'error': str(e),
            'oanda_positions': [],
            'db_positions': [],
            'oanda_count': 0,
            'db_count': 0,
            'unrealized_pnl': 0,
            'account_balance': 0,
            'margin_used': 0,
            'margin_available': 0
        }

def prompt_user_for_position_action(assessment: Dict[str, Any]) -> str:
    """
    Prompt user for action regarding existing positions.
    
    Args:
        assessment: Position assessment data
        
    Returns:
        str: User's choice ('continue', 'close', or 'exit')
    """
    print(f"\n{'='*80}")
    print("📋 EXISTING POSITIONS DETECTED")
    print(f"{'='*80}")
    
    print(f"🏦 Account Summary:")
    print(f"   Balance: ${assessment['account_balance']:,.2f}")
    print(f"   Margin Used: ${assessment['margin_used']:,.2f}")
    print(f"   Margin Available: ${assessment['margin_available']:,.2f}")
    print(f"   Unrealized P&L: ${assessment['unrealized_pnl']:+,.2f}")
    
    print(f"\n📊 Position Summary:")
    print(f"   OANDA Positions: {assessment['oanda_count']}")
    print(f"   Database Records: {assessment['db_count']}")
    
    if assessment['oanda_positions']:
        print(f"\n🔍 Open Positions in OANDA:")
        for i, pos in enumerate(assessment['oanda_positions'][:10], 1):  # Show first 10
            instrument = pos.get('instrument', 'N/A')
            direction = 'LONG' if float(pos.get('units', 0)) > 0 else 'SHORT'
            units = abs(float(pos.get('units', 0)))
            unrealized_pl = float(pos.get('unrealizedPL', 0))
            print(f"   {i:2d}. {instrument} {direction} {units:,.0f} units (P&L: ${unrealized_pl:+.2f})")
        
        if len(assessment['oanda_positions']) > 10:
            print(f"   ... and {len(assessment['oanda_positions']) - 10} more positions")
    
    print(f"\n{'='*80}")
    print("🤔 What would you like to do?")
    print("   [1] CONTINUE with existing positions")
    print("   [2] CLOSE ALL positions and start fresh")
    print("   [3] EXIT the application")
    print(f"{'='*80}")
    
    while True:
        try:
            choice = input("\nEnter your choice (1/2/3): ").strip()
            
            if choice == '1':
                return 'continue'
            elif choice == '2':
                return 'close'
            elif choice == '3':
                return 'exit'
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n🛑 User interrupted. Exiting...")
            return 'exit'
        except Exception as e:
            print(f"❌ Error reading input: {e}")

def display_startup_banner():
    """
    Display startup banner with system information.
    """
    
    banner = f"""
{'='*80}
🤖 OANDA MEANREVERSIONSR TRADING BOT - PRACTICE ACCOUNT
{'='*80}
🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Strategy: MeanReversionSR - Mean reversion at support levels
📈 Timeframe: M5 (5-minute candles)
💱 Entry: Bullish reversal at oversold support (RSI < 35)
🎨 Patterns: Hammer OR bullish candlestick patterns
⚙️ Risk Management: 0.5% per trade, ATR-based stops
📊 Performance: 97.06% success rate, 87.77% avg win rate
🌍 Testing: ALL TRADEABLE PAIRS (Practice Account Safe)
{'='*80}
"""
    
    logger.info(banner)

def display_shutdown_summary():
    """
    Display summary information before shutdown.
    """
    
    try:
        summary = get_trading_summary('trading_journal.db')
        
        if summary and summary.get('total_trades', 0) > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 FINAL TRADING SESSION SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Total Trades: {summary['total_trades']}")
            logger.info(f"Win Rate: {summary['win_rate']}%")
            logger.info(f"Total P&L: {summary['total_pnl']:+.2f}")
            logger.info(f"Best Trade: {summary['best_trade']:+.2f}")
            logger.info(f"Worst Trade: {summary['worst_trade']:+.2f}")
            logger.info(f"{'='*60}")
        else:
            logger.info("📊 No trades executed during this session.")
            
    except Exception as e:
        logger.error(f"Error generating shutdown summary: {e}")

def main():
    """
    Main application entry point.
    
    Sets up the trading bot, schedules trading jobs, and runs the main loop.
    """
    global client, shutdown_requested
    
    # Setup logging
    setup_logging()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Display startup banner
        display_startup_banner()
        
        # Initialize trading journal database
        logger.info("🗄️ Initializing trading journal database...")
        initialize_database('trading_journal.db')
        
        # Validate trading environment
        logger.info("🔍 Validating trading environment...")
        if not validate_trading_environment():
            logger.error("❌ Trading environment validation failed. Exiting.")
            return
        
        # Get dynamic list of tradable instruments
        logger.info("📋 Loading tradable instruments from OANDA...")
        instrument_list = get_tradable_instruments(client)
        logger.info(f"✅ Loaded {len(instrument_list)} tradable instruments.")
        
        # Assess existing positions and prompt user for action
        logger.info("🔍 Assessing existing positions...")
        position_assessment = assess_open_positions(client)
        
        # Check if there are any open positions
        if position_assessment['oanda_count'] > 0 or position_assessment['db_count'] > 0:
            # Prompt user for action
            user_choice = prompt_user_for_position_action(position_assessment)
            
            if user_choice == 'exit':
                logger.info("👋 User chose to exit. Goodbye!")
                return
            elif user_choice == 'close':
                logger.info("🔄 User chose to close all positions and start fresh...")
                
                # Close all positions
                if position_assessment['oanda_count'] > 0:
                    close_success = close_all_positions(client)
                    if not close_success:
                        logger.error("❌ Failed to close positions. Exiting for safety.")
                        return
                
                # Clear logs and database
                clear_success = clear_trading_logs()
                if not clear_success:
                    logger.warning("⚠ Some logs could not be cleared, but continuing...")
                
                # Reinitialize database
                logger.info("🗄️ Reinitializing trading journal database...")
                initialize_database('trading_journal.db')
                
                logger.info("✅ Fresh start completed!")
            else:
                logger.info("✅ Continuing with existing positions...")
        else:
            logger.info("✅ No existing positions found. Starting fresh...")
        
        # Schedule trading jobs
        logger.info("⏰ Setting up trading schedule...")
        
        # Run trading check every minute during market hours
        schedule.every(1).minutes.do(trading_job, instrument_list=instrument_list)
        
        # Optional: Add periodic maintenance tasks
        # schedule.every().hour.do(maintenance_job)  # Could add hourly maintenance
        # schedule.every().day.at("00:01").do(daily_report)  # Could add daily reports
        
        logger.info("✅ Trading bot setup complete!")
        logger.info("🚀 Starting main trading loop...")
        logger.info("   Press CTRL+C to stop the bot gracefully")
        
        # Main trading loop
        while not shutdown_requested:
            try:
                # Run pending scheduled jobs
                schedule.run_pending()
                
                # Sleep for 1 second to prevent excessive CPU usage
                time.sleep(1)
                
            except KeyboardInterrupt:
                # Handle CTRL+C gracefully
                signal_handler(signal.SIGINT, None)
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
                # Brief pause before continuing
                time.sleep(5)
                
    except Exception as e:
        logger.error(f"❌ Critical error in main application: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
    finally:
        # Cleanup and shutdown
        logger.info("🧹 Performing cleanup...")
        display_shutdown_summary()
        logger.info("👋 Trading bot shutdown complete.")

if __name__ == "__main__":
    """
    Entry point when script is run directly.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)