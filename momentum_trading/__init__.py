"""
Momentum Trading System for Hyperliquid

A funding-aware, volatility-scaled time-series momentum (TSMOM) strategy
across Hyperliquid perpetuals.

Components:
- Scheduler: Triggers rebalances at fixed intervals
- Data Loader: Fetches candles, funding, L2 book, universe metrics
- Signal Engine: Computes TSMOM z-scores with funding penalties
- Risk Engine: Portfolio vol targeting, beta neutralization, caps
- Trade Gate: Edge vs friction filtering
- Execution Router: ALO/TWAP/IOC order routing with reduce-only logic
- Position/PnL Manager: Reconciles fills, fees, hourly funding
- State Store: Persists bars, signals, orders, fills, equity curve
- Monitoring: Metrics, logs, kill-switch orchestration
"""

__version__ = "0.1.0"
__author__ = "Gajesh Naik"

from momentum_trading.core.config import Config
from momentum_trading.core.scheduler import Scheduler

__all__ = [
    "Config",
    "Scheduler",
]

