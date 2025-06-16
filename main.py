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
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import traceback
import json

# Import our custom modules
from oanda_handler import get_api_client, get_account_summary
from strategy_handler import run_strategy_check
from journal import initialize_database, get_trading_summary, display_trading_summary, get_open_trades

# Global variables for graceful shutdown
shutdown_requested = False
client = None

def setup_logging():
    """
    Configure logging for the trading bot.
    """
    # Create logs directory if it doesn't exist
    import os
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def signal_handler(signum, frame):
    """
    Handle graceful shutdown on CTRL+C or system signals.
    """
    global shutdown_requested
    logger = logging.getLogger(__name__)
    
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
        logger = logging.getLogger(__name__)
        logger.warning(f"Error checking market hours: {e}. Assuming market is open.")
        return True

def validate_trading_environment() -> bool:
    """
    Validate that the trading environment is properly configured.
    
    Returns:
        bool: True if environment is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
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
    Get the list of instruments to trade.
    
    Returns:
        List of currency pairs to monitor
    """
    # Primary major pairs with good liquidity and spreads
    instruments = [
        'EUR_USD',  # Most liquid pair
        'GBP_USD',  # High volatility, good for price action
        'USD_JPY',  # Different market dynamics
        'AUD_USD',  # Risk-on/risk-off sentiment
        'USD_CAD',  # Commodity currency
    ]
    
    return instruments

def trading_job():
    """
    Main trading job that runs periodically to check for trading opportunities.
    
    This function:
    1. Validates market conditions
    2. Checks each instrument for trading signals
    3. Executes trades when conditions are met
    4. Handles errors gracefully
    5. Logs all activities
    """
    logger = logging.getLogger(__name__)
    
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
        
        # Get trading instruments
        instruments = get_trading_instruments()
        logger.info(f"🎯 Scanning {len(instruments)} instruments: {', '.join(instruments)}")
        
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

def display_startup_banner():
    """
    Display startup banner with system information.
    """
    logger = logging.getLogger(__name__)
    
    banner = f"""
{'='*80}
🤖 OANDA PRICE ACTION TRADING BOT
{'='*80}
🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Strategy: Multi-Timeframe Price Action + ICT Concepts
📈 Timeframes: M15 (Context) + M5 (Zones) + M1 (Signals)
💱 Instruments: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD
⚙️ Risk Management: 0.5% per trade, Dynamic stops
📊 Performance Tracking: Automated journal & analytics
{'='*80}
"""
    
    logger.info(banner)

def display_shutdown_summary():
    """
    Display summary information before shutdown.
    """
    logger = logging.getLogger(__name__)
    
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
    logger = setup_logging()
    
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
        
        # Check current open trades
        open_trades = get_open_trades('trading_journal.db')
        if open_trades:
            logger.info(f"📋 Found {len(open_trades)} open trades from previous session:")
            for trade in open_trades:
                logger.info(f"   {trade['instrument']} {trade['direction']} @ {trade['entry_price']}")
        
        # Schedule trading jobs
        logger.info("⏰ Setting up trading schedule...")
        
        # Run trading check every minute during market hours
        schedule.every(1).minutes.do(trading_job)
        
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