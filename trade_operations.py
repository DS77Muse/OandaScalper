"""
Atomic Trade Operations Module

This module implements Phase 2 of the High-Observability Trading System plan,
providing atomic, transactional database operations that guarantee data integrity
for all trade lifecycle events.
"""

from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import event
from loguru import logger

from models import Trade, db_manager


def calculate_net_pnl(trade: Trade, fees: float = 0.0) -> Tuple[float, float, float]:
    """
    Calculate gross, net, and percentage P&L for a closed trade.
    
    Args:
        trade: The Trade instance with entry and exit prices
        fees: Trading fees and commissions
        
    Returns:
        Tuple of (gross_pnl, net_pnl, pnl_pct)
    """
    if not trade.exit_price or not trade.entry_price:
        raise ValueError("Both entry_price and exit_price must be set to calculate P&L")
    
    # Calculate gross P&L based on direction
    if trade.direction == 'LONG':
        gross_pnl = (trade.exit_price - trade.entry_price) * abs(trade.quantity)
    else:  # SHORT
        gross_pnl = (trade.entry_price - trade.exit_price) * abs(trade.quantity)
    
    # Calculate net P&L after fees
    net_pnl = gross_pnl - fees
    
    # Calculate percentage return
    capital_at_risk = trade.entry_price * abs(trade.quantity)
    pnl_pct = (net_pnl / capital_at_risk) * 100 if capital_at_risk > 0 else 0.0
    
    return gross_pnl, net_pnl, pnl_pct


def create_trade(
    session: Session,
    symbol: str,
    quantity: float,
    direction: str,
    entry_price: float,
    strategy_name: str,
    strategy_version: Optional[str] = None,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    entry_reason: Optional[str] = None,
    trade_id: Optional[int] = None
) -> Trade:
    """
    Create a new trade record within an atomic transaction.
    
    Args:
        session: Database session
        symbol: Trading instrument (e.g., 'EUR/USD')
        quantity: Position size (positive for long, negative for short)
        direction: 'LONG' or 'SHORT'
        entry_price: Entry price
        strategy_name: Name of the strategy
        strategy_version: Version of the strategy
        stop_loss_price: Stop loss price
        take_profit_price: Take profit price
        entry_reason: Detailed reason for the trade entry
        trade_id: Optional OANDA trade ID for legacy compatibility
        
    Returns:
        Trade: The created trade instance
    """
    trade_logger = logger.bind(
        symbol=symbol,
        direction=direction,
        strategy=strategy_name
    )
    
    try:
        with session.begin():
            trade = Trade(
                trade_id=trade_id,
                symbol=symbol,
                quantity=quantity,
                direction=direction,
                entry_price=entry_price,
                strategy_name=strategy_name,
                strategy_version=strategy_version,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                entry_reason=entry_reason,
                status='OPEN',
                entry_timestamp=datetime.utcnow()
            )
            
            session.add(trade)
            session.flush()  # This assigns the ID without committing
            
            trade_logger = trade_logger.bind(trade_id=trade.id)
            trade_logger.info(
                "New trade created successfully in database",
                extra={
                    "trade_data": trade.to_dict(),
                    "entry_price": entry_price,
                    "quantity": quantity
                }
            )
            
            return trade
            
    except Exception as e:
        trade_logger.exception("Failed to create trade due to database error")
        raise


def close_trade(
    session: Session,
    trade_id: int,
    exit_price: float,
    exit_reason: str,
    fees: float = 0.0
) -> Optional[Trade]:
    """
    Close a trade within a single atomic transaction.
    
    This function updates the trade status, records exit information,
    and calculates P&L metrics atomically.
    
    Args:
        session: Database session
        trade_id: ID of the trade to close
        exit_price: Price at which the trade was exited
        exit_reason: Reason for exit ('StopLoss', 'TakeProfit', 'Signal', 'Manual')
        fees: Trading fees and commissions
        
    Returns:
        Trade: The updated trade instance, or None if trade not found
    """
    trade_logger = logger.bind(trade_id=trade_id, exit_reason=exit_reason)
    
    try:
        with session.begin():
            # Lock the trade record to prevent concurrent modifications
            trade = session.query(Trade).filter(
                Trade.id == trade_id
            ).with_for_update().first()
            
            if not trade:
                trade_logger.error("Attempted to close a trade that does not exist")
                return None
            
            if trade.status == 'CLOSED':
                trade_logger.warning("Attempted to close an already closed trade")
                return trade
            
            # Update trade with exit information
            trade.status = 'CLOSED'
            trade.exit_price = exit_price
            trade.exit_timestamp = datetime.utcnow()
            trade.exit_reason = exit_reason
            trade.fees = fees
            
            # Calculate and store P&L figures
            gross_pnl, net_pnl, pnl_pct = calculate_net_pnl(trade, fees)
            trade.pnl_gross = gross_pnl
            trade.pnl_net = net_pnl
            trade.pnl_pct = pnl_pct
            
            trade_logger.info(
                "Trade closed successfully in database",
                extra={
                    "trade_data": trade.to_dict(),
                    "exit_price": exit_price,
                    "net_pnl": net_pnl,
                    "gross_pnl": gross_pnl,
                    "pnl_pct": pnl_pct
                }
            )
            
            return trade
            
    except Exception as e:
        trade_logger.exception("Failed to close trade due to database error")
        raise


def update_trade_excursions(
    session: Session,
    trade_id: int,
    current_price: float
) -> Optional[Trade]:
    """
    Update Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE).
    
    This function should be called periodically while a trade is open
    to track the maximum profit and loss levels reached.
    
    Args:
        session: Database session
        trade_id: ID of the trade to update
        current_price: Current market price
        
    Returns:
        Trade: The updated trade instance, or None if trade not found
    """
    try:
        with session.begin():
            trade = session.query(Trade).filter(
                Trade.id == trade_id,
                Trade.status == 'OPEN'
            ).with_for_update().first()
            
            if not trade:
                return None
            
            # Calculate current unrealized P&L
            if trade.direction == 'LONG':
                unrealized_pnl = (current_price - trade.entry_price) * abs(trade.quantity)
            else:  # SHORT
                unrealized_pnl = (trade.entry_price - current_price) * abs(trade.quantity)
            
            # Update MFE (Maximum Favorable Excursion)
            if trade.max_favorable_excursion is None or unrealized_pnl > trade.max_favorable_excursion:
                trade.max_favorable_excursion = unrealized_pnl
            
            # Update MAE (Maximum Adverse Excursion)
            if trade.max_adverse_excursion is None or unrealized_pnl < trade.max_adverse_excursion:
                trade.max_adverse_excursion = unrealized_pnl
            
            return trade
            
    except Exception as e:
        logger.error(f"Failed to update trade excursions for trade {trade_id}: {e}")
        raise


def get_open_trades(session: Session) -> List[Trade]:
    """
    Get all currently open trades.
    
    Args:
        session: Database session
        
    Returns:
        List of open Trade instances
    """
    return session.query(Trade).filter(Trade.status == 'OPEN').all()


def get_trades_by_strategy(
    session: Session,
    strategy_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Trade]:
    """
    Get trades filtered by strategy and optional date range.
    
    Args:
        session: Database session
        strategy_name: Name of the strategy to filter by
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        List of Trade instances matching the criteria
    """
    query = session.query(Trade).filter(Trade.strategy_name == strategy_name)
    
    if start_date:
        query = query.filter(Trade.entry_timestamp >= start_date)
    if end_date:
        query = query.filter(Trade.entry_timestamp <= end_date)
    
    return query.order_by(Trade.entry_timestamp.desc()).all()


# SQLAlchemy Event Listeners for Audit Trail
@event.listens_for(Trade, 'after_insert')
def receive_after_insert(mapper, connection, target):
    """
    Log trade creation events for audit trail.
    """
    logger.info(
        "Trade database record was created",
        extra={"trade_data": target.to_dict()}
    )


@event.listens_for(Trade, 'after_update')
def receive_after_update(mapper, connection, target):
    """
    Log trade update events for audit trail.
    """
    logger.info(
        "Trade database record was updated",
        extra={"trade_data": target.to_dict()}
    )


@event.listens_for(Trade, 'after_delete')
def receive_after_delete(mapper, connection, target):
    """
    Log trade deletion events for audit trail.
    """
    logger.warning(
        "Trade database record was deleted",
        extra={"trade_data": target.to_dict()}
    )


class ProfitabilityTracker:
    """
    Track cumulative P&L and trade statistics for an operational cycle.
    
    This class implements the real-time profitability calculation
    functionality specified in Phase 3.
    """
    
    def __init__(self, cycle_name: str = "daily"):
        """
        Initialize the profitability tracker.
        
        Args:
            cycle_name: Name of the tracking cycle (e.g., 'daily', 'weekly')
        """
        self.cycle_name = cycle_name
        self.net_pnl = 0.0
        self.gross_pnl = 0.0
        self.total_fees = 0.0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trades = []  # Store individual trade results
        
        logger.info(f"Profitability tracker for '{self.cycle_name}' cycle initialized")
    
    def record_trade(self, trade: Trade):
        """
        Update the tracker with the results of a closed trade.
        
        Args:
            trade: The closed Trade instance
        """
        if trade.status != 'CLOSED' or trade.pnl_net is None:
            logger.warning(f"Attempted to record incomplete trade {trade.id}")
            return
        
        self.net_pnl += trade.pnl_net
        self.gross_pnl += trade.pnl_gross or 0.0
        self.total_fees += trade.fees or 0.0
        self.trade_count += 1
        
        if trade.pnl_net > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.trades.append(trade.to_dict())
        
        logger.info(
            f"Trade recorded. PnL: {trade.pnl_net:+.2f}. Cumulative Cycle PnL: {self.net_pnl:+.2f}",
            extra={
                "cycle_name": self.cycle_name,
                "cumulative_net_pnl": self.net_pnl,
                "trade_count": self.trade_count,
                "trade_id": trade.id
            }
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the current cycle's performance.
        
        Returns:
            Dictionary containing performance metrics
        """
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
        
        return {
            "cycle_name": self.cycle_name,
            "net_pnl": self.net_pnl,
            "gross_pnl": self.gross_pnl,
            "total_fees": self.total_fees,
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": win_rate,
            "avg_trade_pnl": self.net_pnl / self.trade_count if self.trade_count > 0 else 0,
            "profit_factor": (
                sum(t['pnl_net'] for t in self.trades if t['pnl_net'] > 0) /
                abs(sum(t['pnl_net'] for t in self.trades if t['pnl_net'] < 0))
                if any(t['pnl_net'] < 0 for t in self.trades) else float('inf')
            )
        }
    
    def reset(self):
        """Reset the tracker for a new cycle."""
        logger.info(f"Resetting profitability tracker for '{self.cycle_name}' cycle")
        
        self.net_pnl = 0.0
        self.gross_pnl = 0.0
        self.total_fees = 0.0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trades.clear()