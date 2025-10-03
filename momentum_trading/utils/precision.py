"""
Precision helpers for Hyperliquid tick/lot formatting.

Enforces ≤5 significant figures for price and szDecimals for size.
"""

import math


def format_price(px: float, sz_decimals: int, max_decimals: int = 6) -> str:
    """
    Format price per Hyperliquid rules.
    
    Rules:
    - ≤5 significant figures
    - ≤(MAX_DECIMALS - szDecimals) decimal places
    
    Args:
        px: Price
        sz_decimals: Asset szDecimals
        max_decimals: MAX_DECIMALS (default 6)
    
    Returns:
        Formatted price string
    """
    if px <= 0:
        raise ValueError(f"price must be > 0, got {px}")
    
    # Limit decimal places
    max_dp = max_decimals - sz_decimals
    
    # Enforce 5 significant figures
    sig = f"{px:.5g}"
    
    # Then clamp decimals
    if "." in sig:
        whole, frac = sig.split(".")
        frac = frac[:max(0, max_dp)]
        sig = whole if not frac else f"{whole}.{frac}"
    
    return sig.rstrip("0").rstrip(".")


def format_size(sz: float, sz_decimals: int) -> str:
    """
    Format size per lot precision.
    
    Args:
        sz: Size (quantity)
        sz_decimals: Asset szDecimals
    
    Returns:
        Formatted size string
    """
    q = 10 ** sz_decimals
    rounded = math.floor(sz * q) / q
    
    if sz_decimals > 0:
        return f"{rounded:.{sz_decimals}f}".rstrip("0").rstrip(".")
    else:
        return str(int(rounded))
