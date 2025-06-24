"""
Automated Daily Performance Reporting System

This module implements Phase 4 of the High-Observability Trading System plan,
providing automated synthesis of trading performance and system health metrics
into structured JSON reports for AI-driven analysis.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from loguru import logger
from models import db_manager, Trade
from trade_operations import get_trades_by_strategy


def calculate_key_performance_indicators(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive KPIs from closed trades.
    
    Args:
        trades_df: DataFrame of closed trades
        
    Returns:
        Dictionary of calculated KPIs
    """
    if trades_df.empty:
        return {
            "net_profit_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_win_loss_ratio": 0.0,
            "maximum_drawdown": 0.0,
            "volatility_of_returns": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "win_rate": 0.0,
            "total_closed_trades": 0,
            "avg_holding_time_hours": 0.0
        }
    
    # Basic profitability metrics
    net_profit_loss = trades_df['pnl_net'].sum()
    winning_trades = trades_df[trades_df['pnl_net'] > 0]
    losing_trades = trades_df[trades_df['pnl_net'] < 0]
    
    # Profit Factor
    total_wins = winning_trades['pnl_net'].sum() if not winning_trades.empty else 0
    total_losses = abs(losing_trades['pnl_net'].sum()) if not losing_trades.empty else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Expectancy
    win_rate = len(winning_trades) / len(trades_df) * 100
    loss_rate = len(losing_trades) / len(trades_df) * 100
    avg_win = winning_trades['pnl_net'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['pnl_net'].mean() if not losing_trades.empty else 0
    expectancy = (win_rate / 100 * avg_win) + (loss_rate / 100 * avg_loss)
    
    # Average Win/Loss Ratio
    avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Maximum Drawdown calculation
    trades_df_sorted = trades_df.sort_values('exit_timestamp')
    trades_df_sorted['cumulative_pnl'] = trades_df_sorted['pnl_net'].cumsum()
    trades_df_sorted['running_max'] = trades_df_sorted['cumulative_pnl'].expanding().max()
    trades_df_sorted['drawdown'] = trades_df_sorted['cumulative_pnl'] - trades_df_sorted['running_max']
    maximum_drawdown = abs(trades_df_sorted['drawdown'].min()) if len(trades_df_sorted) > 0 else 0
    
    # Volatility and risk-adjusted returns
    returns = trades_df['pnl_net'].values
    volatility_of_returns = np.std(returns) if len(returns) > 1 else 0
    
    # Sharpe Ratio (assuming 0% risk-free rate)
    mean_return = np.mean(returns)
    sharpe_ratio = mean_return / volatility_of_returns if volatility_of_returns > 0 else 0
    
    # Sortino Ratio (downside deviation only)
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns) if len(negative_returns) > 1 else volatility_of_returns
    sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
    
    # Average holding time
    trades_df['holding_time'] = (
        pd.to_datetime(trades_df['exit_timestamp']) - 
        pd.to_datetime(trades_df['entry_timestamp'])
    ).dt.total_seconds() / 3600  # Convert to hours
    avg_holding_time_hours = trades_df['holding_time'].mean()
    
    return {
        "net_profit_loss": round(net_profit_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
        "avg_win_loss_ratio": round(avg_win_loss_ratio, 2),
        "maximum_drawdown": round(maximum_drawdown, 2),
        "volatility_of_returns": round(volatility_of_returns, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "win_rate": round(win_rate, 2),
        "total_closed_trades": len(trades_df),
        "avg_holding_time_hours": round(avg_holding_time_hours, 2) if pd.notna(avg_holding_time_hours) else 0.0
    }


def generate_equity_curve(trades_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate equity curve data points.
    
    Args:
        trades_df: DataFrame of closed trades
        
    Returns:
        List of equity curve data points
    """
    if trades_df.empty:
        return []
    
    trades_sorted = trades_df.sort_values('exit_timestamp').reset_index(drop=True)
    trades_sorted['cumulative_pnl'] = trades_sorted['pnl_net'].cumsum()
    
    equity_curve = []
    for idx, row in trades_sorted.iterrows():
        equity_curve.append({
            "trade_number": idx + 1,
            "cumulative_pnl": round(row['cumulative_pnl'], 2),
            "timestamp": row['exit_timestamp'].isoformat() if pd.notna(row['exit_timestamp']) else None
        })
    
    return equity_curve


def analyze_system_health(log_files: List[str]) -> Dict[str, Any]:
    """
    Analyze system health from log files.
    
    Args:
        log_files: List of log file paths to analyze
        
    Returns:
        System health metrics
    """
    health_metrics = {
        "error_count": 0,
        "warning_count": 0,
        "total_log_entries": 0,
        "exceptions": [],
        "log_files_analyzed": []
    }
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            continue
            
        health_metrics["log_files_analyzed"].append(log_file)
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                        
                    health_metrics["total_log_entries"] += 1
                    
                    try:
                        log_entry = json.loads(line)
                        level = log_entry.get('record', {}).get('level', {}).get('name', '')
                        
                        if level == 'ERROR':
                            health_metrics["error_count"] += 1
                            
                            # Extract exception information
                            message = log_entry.get('record', {}).get('message', '')
                            if 'exception' in message.lower() or 'error' in message.lower():
                                exception_info = {
                                    "timestamp": log_entry.get('record', {}).get('time'),
                                    "message": message,
                                    "module": log_entry.get('record', {}).get('name'),
                                    "function": log_entry.get('record', {}).get('function')
                                }
                                health_metrics["exceptions"].append(exception_info)
                        
                        elif level == 'WARNING':
                            health_metrics["warning_count"] += 1
                            
                    except json.JSONDecodeError:
                        # Skip non-JSON log lines
                        continue
                        
        except Exception as e:
            logger.error(f"Error analyzing log file {log_file}: {e}")
    
    return health_metrics


def generate_daily_report(target_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Generate comprehensive daily performance report.
    
    Args:
        target_date: Date to generate report for (defaults to yesterday)
        
    Returns:
        Complete performance report dictionary
    """
    if target_date is None:
        target_date = datetime.now() - timedelta(days=1)
    
    report_date = target_date.strftime('%Y-%m-%d')
    
    logger.info(f"Generating daily performance report for {report_date}")
    
    # Report metadata
    report_metadata = {
        "report_date": report_date,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "strategy_scope": "All_Strategies",
        "report_version": "1.0"
    }
    
    # Fetch closed trades for the target date
    start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    
    try:
        with db_manager.get_session() as session:
            # Query closed trades for the day
            trades = session.query(Trade).filter(
                Trade.status == 'CLOSED',
                Trade.exit_timestamp >= start_of_day,
                Trade.exit_timestamp < end_of_day
            ).all()
            
            # Convert to DataFrame for analysis
            if trades:
                trades_data = [trade.to_dict() for trade in trades]
                trades_df = pd.DataFrame(trades_data)
                
                # Convert timestamp columns
                trades_df['entry_timestamp'] = pd.to_datetime(trades_df['entry_timestamp'])
                trades_df['exit_timestamp'] = pd.to_datetime(trades_df['exit_timestamp'])
            else:
                trades_df = pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching trades for report: {e}")
        trades_df = pd.DataFrame()
    
    # Calculate KPIs
    kpis = calculate_key_performance_indicators(trades_df)
    
    # Generate equity curve
    equity_curve = generate_equity_curve(trades_df)
    
    # Analyze system health
    log_files = []
    log_dir = Path("logs")
    if log_dir.exists():
        # Today's system log
        today_log = log_dir / f"system_{target_date.strftime('%Y-%m-%d')}.log"
        if today_log.exists():
            log_files.append(str(today_log))
        
        # Error log
        error_log = log_dir / "errors.log"
        if error_log.exists():
            log_files.append(str(error_log))
    
    system_health = analyze_system_health(log_files)
    
    # Compile complete report
    report = {
        "report_metadata": report_metadata,
        "performance_summary": {
            "key_performance_indicators": kpis,
            "equity_curve_daily": equity_curve
        },
        "trade_details": {
            "total_closed_trades": len(trades_df),
            "closed_trades_list": trades_df.to_dict('records') if not trades_df.empty else []
        },
        "system_health": system_health
    }
    
    logger.info(f"Daily report generated successfully with {len(trades_df)} trades and {system_health['total_log_entries']} log entries")
    
    return report


def save_daily_report(report: Dict[str, Any], output_dir: str = "reports") -> str:
    """
    Save the daily report to a JSON file.
    
    Args:
        report: Report dictionary to save
        output_dir: Directory to save the report
        
    Returns:
        Path to the saved report file
    """
    # Create reports directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    report_date = report["report_metadata"]["report_date"]
    filename = f"performance_report_{report_date}.json"
    filepath = Path(output_dir) / filename
    
    try:
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Daily report saved to {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error saving daily report: {e}")
        raise


def main():
    """
    Main function to generate and save daily report.
    
    This function is designed to be called by a scheduler (cron job)
    to automatically generate daily reports.
    """
    try:
        # Generate report for yesterday
        report = generate_daily_report()
        
        # Save to file
        filepath = save_daily_report(report)
        
        # Log summary
        kpis = report["performance_summary"]["key_performance_indicators"]
        logger.info(f"Daily report completed successfully", extra={
            "report_file": filepath,
            "trades_analyzed": kpis["total_closed_trades"],
            "net_pnl": kpis["net_profit_loss"],
            "win_rate": kpis["win_rate"],
            "profit_factor": kpis["profit_factor"]
        })
        
        print(f"✅ Daily report generated successfully: {filepath}")
        print(f"📊 Trades analyzed: {kpis['total_closed_trades']}")
        print(f"💰 Net P&L: {kpis['net_profit_loss']:+.2f}")
        print(f"📈 Win Rate: {kpis['win_rate']:.1f}%")
        print(f"⚖️ Profit Factor: {kpis['profit_factor']:.2f}")
        
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        print(f"❌ Daily report generation failed: {e}")
        raise


if __name__ == "__main__":
    main()