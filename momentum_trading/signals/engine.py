"""
Signal Engine

Builds lookback returns, z-scores, composite signal (s_i).
Computes funding projection and outputs funding-adjusted signal (s_i*).
"""

from typing import Dict, List
import time
import numpy as np
import pandas as pd

from momentum_trading.core.config import Config
from momentum_trading.data.loader import MarketDataLoader


class SignalRecord:
    """Signal data for one asset."""
    
    def __init__(
        self,
        coin: str,
        s_raw: float,
        funding_proj_per_h: float,
        s_adj: float,
        sigma: float,
        beta_to_btc: float = 0.0
    ):
        self.coin = coin
        self.s_raw = s_raw  # Raw signal before funding adjustment
        self.funding_proj_per_h = funding_proj_per_h  # Projected hourly funding
        self.s_adj = s_adj  # Funding-adjusted signal (s_i*)
        self.sigma = sigma  # Realized volatility (annualized)
        self.beta_to_btc = beta_to_btc  # Beta to BTC (for neutralization)


class SignalEngine:
    """
    Time-series momentum signal generation.
    
    Implements TSMOM with z-score normalization per Spec.md:
    1. Compute historical return series for each lookback
    2. Z-score latest return against historical distribution
    3. Average z-scores across lookbacks and clip to [-2, 2]
    4. Apply directional funding penalty
    """
    
    def __init__(self, config: Config, data_loader: MarketDataLoader):
        """
        Initialize signal engine.
        
        Args:
            config: System configuration
            data_loader: Market data loader
        """
        self.config = config
        self.data_loader = data_loader
    
    def compute_signals(self, universe: List[str]) -> Dict[str, SignalRecord]:
        """
        Compute signals for all assets in universe.
        
        Args:
            universe: List of coin names
        
        Returns:
            Dict mapping coin -> SignalRecord
        """
        signals: Dict[str, SignalRecord] = {}
        
        # Determine data requirements
        interval = self.config.bars.interval
        lookbacks = self.config.signal.lookbacks
        window = self.config.signal.zscore_window_bars
        max_lookback = max(lookbacks)
        
        # Need W + max_lookback bars for z-score computation
        bars_needed = window + max_lookback
        
        # Anchor to last closed bar (avoid partial forming bars)
        now_ms = int(time.time() * 1000)
        interval_ms = {"1d": 86_400_000, "4h": 14_400_000, "1h": 3_600_000}.get(interval, 3_600_000)
        bar_close = (now_ms // interval_ms) * interval_ms
        end_ms = bar_close - 1  # Last fully closed bar
        start_ms = end_ms - bars_needed * interval_ms
        
        # Batch fetch predicted funding once (avoid N round-trips)
        print("[SignalEngine] Fetching predicted funding for all coins...")
        funding_map = self.data_loader.get_all_predicted_funding()
        
        for coin in universe:
            try:
                # Fetch candles
                candles = self.data_loader.get_candles(coin, interval, start_ms, end_ms)
                
                # Sort and deduplicate by timestamp
                candles.sort(key=lambda x: x["t"])
                deduped: List[Dict] = []
                for c in candles:
                    if not deduped or c["t"] != deduped[-1]["t"]:
                        deduped.append(c)
                candles = deduped
                
                if len(candles) < max_lookback + window + 1:
                    print(f"[SignalEngine] Skipping {coin}: insufficient data ({len(candles)} bars, need {max_lookback + window + 1})")
                    continue
                
                # Extract close prices
                prices = np.array([c["c"] for c in candles], dtype=float)
                
                # Skip if NaN or zero prices
                if np.any(np.isnan(prices)) or np.any(prices <= 0):
                    print(f"[SignalEngine] Skipping {coin}: invalid prices (NaN or ≤0)")
                    continue
                
                # Compute z-scores for each lookback
                z_scores: List[float] = []
                for lookback in lookbacks:
                    z = self._compute_lookback_zscore(prices, lookback, window)
                    if np.isnan(z) or np.isinf(z):
                        print(f"[SignalEngine] Skipping {coin}: invalid z-score for lookback {lookback}")
                        z_scores = []
                        break
                    z_scores.append(z)
                
                if not z_scores:
                    continue
                
                # Composite signal: mean of z-scores, clipped to [-2, 2]
                s_raw = float(np.mean(z_scores))
                s_raw = np.clip(s_raw, self.config.signal.clip_min, self.config.signal.clip_max)
                
                # Get predicted funding from batch
                funding_proj = funding_map.get(coin, 0.0)
                
                # Apply directional funding penalty
                s_adj = self._apply_funding_penalty(coin, s_raw, funding_proj)
                
                # Compute realized volatility from 1h returns over ~20 days
                sigma = self._compute_realized_vol(coin, end_ms)
                
                # Store signal record
                signals[coin] = SignalRecord(
                    coin=coin,
                    s_raw=s_raw,
                    funding_proj_per_h=funding_proj,
                    s_adj=s_adj,
                    sigma=sigma,
                    beta_to_btc=0.0,  # TODO: Compute if beta neutralization enabled
                )
            
            except Exception as e:
                print(f"[SignalEngine] Error computing signal for {coin}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return signals
    
    def _compute_lookback_zscore(
        self,
        prices: np.ndarray,
        lookback: int,
        window: int
    ) -> float:
        """
        Compute z-score for a single lookback per Spec.md.
        
        For lookback L_k:
        1. Compute return series: r_j = P_j / P_{j-L_k} - 1 for j in [t-W, t]
        2. Compute μ = mean(r), σ = std(r)
        3. Z-score latest: z = (r_t - μ) / max(σ, ε)
        
        Args:
            prices: Array of historical prices (oldest to newest)
            lookback: Lookback period (bars)
            window: Rolling window for z-score (W bars)
        
        Returns:
            Z-score of latest lookback return
        """
        if len(prices) < lookback + window + 1:
            return 0.0
        
        # Vectorized return series: r_j = P_j / P_{j-L_k} - 1
        r = prices[lookback:] / prices[:-lookback] - 1.0
        
        # Take last W returns for distribution
        window_slice = r[-window:]
        
        if len(window_slice) < 2:
            return 0.0
        
        mu = np.mean(window_slice)
        sigma = np.std(window_slice, ddof=1)
        
        # Z-score with epsilon guard
        epsilon = self.config.signal.epsilon
        z = (r[-1] - mu) / max(sigma, epsilon)
        
        return float(z)
    
    def _apply_funding_penalty(
        self,
        coin: str,
        s_raw: float,
        funding_rate: float
    ) -> float:
        """
        Apply directional funding penalty per Spec.md.
        
        Only penalize the side that pays:
        - If sign(s_raw) == sign(funding_rate): s_adj = s_raw - λ × |F̂| × H
        - Otherwise: s_adj = s_raw (you receive carry)
        
        Args:
            coin: Asset name
            s_raw: Raw signal
            funding_rate: Projected hourly funding (positive = longs pay)
        
        Returns:
            Adjusted signal s_i*
        """
        if not self.config.funding.directional:
            # Apply penalty regardless of direction
            penalty = self.config.funding.lambda_penalty * abs(funding_rate) * self.config.funding.holding_hours
            return s_raw - penalty
        
        # Directional: only penalize if you're on the paying side
        # funding_rate > 0 → longs pay → penalize long signals (s_raw > 0)
        # funding_rate < 0 → shorts pay → penalize short signals (s_raw < 0)
        if (s_raw > 0 and funding_rate > 0) or (s_raw < 0 and funding_rate < 0):
            # Same sign → you pay carry → apply penalty
            penalty = self.config.funding.lambda_penalty * abs(funding_rate) * self.config.funding.holding_hours
            return s_raw - np.sign(s_raw) * penalty
        else:
            # Opposite sign → you receive carry → no penalty
            return s_raw
    
    def _compute_realized_vol(self, coin: str, now_ms: int) -> float:
        """
        Compute realized volatility from 1h returns over ~20 trading days.
        
        Args:
            coin: Asset name
            now_ms: Current timestamp (epoch ms)
        
        Returns:
            Annualized realized volatility
        """
        # Fetch 1h candles for ~20 days (20 * 24 = 480 hours)
        lookback_hours = 480
        start_ms = now_ms - lookback_hours * 60 * 60 * 1000
        
        try:
            candles = self.data_loader.get_candles(coin, "1h", start_ms, now_ms)
            if len(candles) < 2:
                return self.config.risk.sigma_min  # Floor
            
            # Compute log returns
            closes = np.array([c["c"] for c in candles], dtype=float)
            log_returns = np.diff(np.log(closes))
            
            if len(log_returns) < 2:
                return self.config.risk.sigma_min
            
            # Annualized vol: std(hourly returns) × sqrt(24 × 365)
            hourly_std = np.std(log_returns, ddof=1)
            annual_vol = hourly_std * np.sqrt(24 * 365)
            
            # Floor at sigma_min
            return max(float(annual_vol), self.config.risk.sigma_min)
        
        except Exception as e:
            print(f"[SignalEngine] Error computing vol for {coin}: {e}")
            return self.config.risk.sigma_min

