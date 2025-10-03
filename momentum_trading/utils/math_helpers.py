"""
Mathematical helper functions.

Z-scores, volatility calculations, EWMA, etc.
"""

import numpy as np


def zscore(x: np.ndarray) -> float:
    """
    Compute z-score of last element vs array.
    
    Args:
        x: Array of values
    
    Returns:
        Z-score: (x[-1] - mean(x)) / std(x)
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    
    if sigma < 1e-12:
        return 0.0
    
    return float((x[-1] - mu) / sigma)


def annualized_vol(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Compute annualized volatility from returns.
    
    Args:
        returns: Array of returns
        periods_per_year: Scaling factor (252 for daily, 365*24 for hourly)
    
    Returns:
        Annualized volatility
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0
    
    std_dev = np.std(returns, ddof=1)
    return float(std_dev * np.sqrt(periods_per_year))


def ewma(arr: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Exponentially weighted moving average.
    
    Args:
        arr: Input array
        alpha: Decay factor (0 < alpha ≤ 1)
    
    Returns:
        EWMA array
    """
    arr = np.asarray(arr, dtype=float)
    out = np.empty_like(arr)
    acc = arr[0]
    
    for i, v in enumerate(arr):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    
    return out
