"""
SQLAlchemy ORM Models for the OandaScalper Trading System

This module implements Phase 2 of the High-Observability Trading System plan,
providing robust data management with atomic transactions and comprehensive
trade lifecycle tracking.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional

Base = declarative_base()


class Trade(Base):
    """
    Enhanced Trade model that captures the complete trade lifecycle.
    
    This model implements the revised schema from Phase 2, providing
    comprehensive tracking from signal generation to final P&L calculation.
    """
    __tablename__ = 'trades'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Legacy trade_id for OANDA compatibility
    trade_id = Column(Integer, nullable=True, unique=True)  # Original OANDA trade ID
    
    # Entry information (mapped to existing database columns)
    entry_timestamp = Column('entry_time', DateTime, nullable=False, default=datetime.utcnow)
    entry_price = Column(Float, nullable=False)
    symbol = Column('instrument', String(50), nullable=False)  # e.g., 'EUR/USD'
    quantity = Column('units', Float, nullable=False)  # Position size (positive for long, negative for short)
    direction = Column(String(10), nullable=False)  # 'LONG' or 'SHORT'
    status = Column(String(10), nullable=False, default='OPEN')  # 'OPEN' or 'CLOSED'
    
    # Strategy attribution
    strategy_name = Column(String(100), nullable=False)
    strategy_version = Column(String(20), nullable=True)
    
    # Risk management
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    
    # Exit information (nullable until trade is closed) (mapped to existing database columns)
    exit_timestamp = Column('exit_time', DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_reason = Column(String(50), nullable=True)  # 'StopLoss', 'TakeProfit', 'Signal', 'Manual'
    
    # P&L calculations (calculated on trade closure) - compatible with existing columns
    pnl_gross = Column(Float, nullable=True)  # Gross profit/loss before costs
    pnl_net = Column(Float, nullable=True)    # Net profit/loss after fees
    pnl_pct = Column(Float, nullable=True)    # Percentage return
    profit_loss = Column(Float, nullable=True)  # Legacy P&L column for compatibility
    
    # Additional context
    entry_reason = Column(Text, nullable=True)  # Detailed signal description
    fees = Column(Float, nullable=True, default=0.0)  # Trading fees/commissions
    
    # Audit trail
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Performance metrics (calculated during trade lifetime)
    max_favorable_excursion = Column(Float, nullable=True)  # Highest profit reached
    max_adverse_excursion = Column(Float, nullable=True)    # Largest loss experienced
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol='{self.symbol}', direction='{self.direction}', status='{self.status}')>"
    
    def to_dict(self):
        """Convert trade instance to dictionary for logging and serialization."""
        return {
            'id': self.id,
            'trade_id': self.trade_id,
            'entry_timestamp': self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            'entry_price': self.entry_price,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'direction': self.direction,
            'status': self.status,
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl_gross': self.pnl_gross,
            'pnl_net': self.pnl_net,
            'pnl_pct': self.pnl_pct,
            'entry_reason': self.entry_reason,
            'fees': self.fees,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion
        }


# Create indexes for optimal query performance
Index('idx_symbol', Trade.symbol)
Index('idx_status', Trade.status)
Index('idx_entry_timestamp', Trade.entry_timestamp)
Index('idx_strategy_name', Trade.strategy_name)
Index('idx_exit_timestamp', Trade.exit_timestamp)


class DatabaseManager:
    """
    Database connection and session management.
    
    Provides centralized database configuration and session handling
    with proper connection pooling and error handling.
    """
    
    def __init__(self, database_url: str = "sqlite:///trading_journal.db"):
        """
        Initialize the database manager.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,  # Set to True for SQL query logging during development
            pool_pre_ping=True,  # Validate connections before use
            pool_recycle=3600,   # Recycle connections every hour
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
    def create_tables(self):
        """Create all database tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """
        Get a new database session.
        
        Returns:
            sqlalchemy.orm.Session: Database session
        """
        return self.SessionLocal()
    
    def migrate_from_legacy_schema(self):
        """
        Migrate data from the legacy schema to the new enhanced schema.
        
        This function handles the transition from the old 'trades' table
        to the new enhanced schema with additional fields.
        """
        from loguru import logger
        
        try:
            # Check if we need to migrate
            with self.get_session() as session:
                # Try to query the old schema structure
                result = session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
                ).fetchone()
                
                if result:
                    # Check if new columns exist
                    pragma_result = session.execute(text("PRAGMA table_info(trades)")).fetchall()
                    existing_columns = [row[1] for row in pragma_result]
                    
                    required_new_columns = [
                        'strategy_name', 'strategy_version', 'exit_reason',
                        'pnl_gross', 'pnl_net', 'pnl_pct', 'fees',
                        'max_favorable_excursion', 'max_adverse_excursion'
                    ]
                    
                    missing_columns = [col for col in required_new_columns if col not in existing_columns]
                    
                    if missing_columns:
                        logger.info(f"Migrating database schema. Adding columns: {missing_columns}")
                        
                        # Add missing columns
                        for column in missing_columns:
                            if column == 'strategy_name':
                                session.execute(text("ALTER TABLE trades ADD COLUMN strategy_name TEXT DEFAULT 'Legacy'"))
                            elif column == 'strategy_version':
                                session.execute(text("ALTER TABLE trades ADD COLUMN strategy_version TEXT"))
                            elif column == 'exit_reason':
                                session.execute(text("ALTER TABLE trades ADD COLUMN exit_reason TEXT"))
                            elif column in ['pnl_gross', 'pnl_net', 'pnl_pct', 'fees', 'max_favorable_excursion', 'max_adverse_excursion']:
                                session.execute(text(f"ALTER TABLE trades ADD COLUMN {column} REAL"))
                        
                        session.commit()
                        logger.info("Database schema migration completed successfully")
                    else:
                        logger.info("Database schema is up to date")
                else:
                    # No existing table, create new schema
                    self.create_tables()
                    logger.info("Created new database schema")
                    
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()