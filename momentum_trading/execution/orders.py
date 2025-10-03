"""
Order data structures (OrderIntent, ExecutionReport).
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class OrderIntent:
    """
    Order intent from risk engine (before execution).
    
    Spec.md interface: coin, side, abs_notional, urgency, style, reduce_only, tp_sl
    """
    
    coin: str
    side: Literal["Buy", "Sell"]
    abs_notional: float  # USD notional (absolute value)
    urgency: Literal["Normal", "High"] = "Normal"
    style: Literal["ALO", "TWAP", "IOC"] = "ALO"
    reduce_only: bool = False
    
    # TP/SL (optional)
    tp_px: Optional[float] = None
    sl_px: Optional[float] = None
    sl_type: Literal["market", "limit"] = "limit"


@dataclass
class ExecutionReport:
    """
    Execution result after routing.
    
    Spec.md interface: oid/cloid, status, filled_sz, avg_px, fees, maker_ratio, twap_progress, error_code
    """
    
    oid: Optional[str] = None  # Order ID from exchange
    cloid: Optional[str] = None  # Client order ID
    status: Literal["pending", "filled", "partial", "cancelled", "rejected"] = "pending"
    filled_sz: float = 0.0  # Filled size (signed)
    avg_px: float = 0.0  # Average fill price
    fees: float = 0.0  # Fees paid (USD)
    maker_ratio: float = 0.0  # Fraction filled as maker (0-1)
    twap_progress: float = 0.0  # TWAP completion (0-1)
    error_code: Optional[str] = None
    error_msg: Optional[str] = None

