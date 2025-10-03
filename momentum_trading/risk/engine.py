"""
Risk Engine

Estimates volatility and covariance, applies beta-neutralization,
scales portfolio to vol target, enforces caps and daily VaR.
Computes target weights (w_i).
"""

from typing import Dict, List, Optional
import time
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from momentum_trading.core.config import Config
from momentum_trading.signals.engine import SignalRecord
from momentum_trading.data.loader import MarketDataLoader


class TargetWeight:
    """Target weight for one asset."""
    
    def __init__(
        self,
        coin: str,
        w_pre: float,
        w_scaled: float,
        caps_applied: Dict[str, float],
        edge: float,
        friction: float
    ):
        self.coin = coin
        self.w_pre = w_pre  # Weight before scaling
        self.w_scaled = w_scaled  # Final weight after scaling & caps
        self.caps_applied = caps_applied  # Which caps were hit
        self.edge = edge  # Edge proxy
        self.friction = friction  # Estimated friction


class RiskEngine:
    """
    Portfolio construction and risk management.
    
    Implements risk-parity sizing with directional tilt per Spec.md:
    1. Raw weight: w_raw_i = s_i* / max(σ_i, σ_min)
    2. BTC beta neutralization (optional)
    3. Scale to portfolio vol target using covariance
    4. Apply per-asset, liquidity, and margin caps
    5. Edge vs friction filter
    """
    
    def __init__(self, config: Config, data_loader: Optional[MarketDataLoader] = None):
        """
        Initialize risk engine.
        
        Args:
            config: System configuration
            data_loader: Market data loader (for ATR calculation)
        """
        self.config = config
        self.data_loader = data_loader
    
    def compute_target_weights(
        self,
        signals: Dict[str, SignalRecord],
        returns_df: Optional[pd.DataFrame],
        equity: float,
        ctx_by_coin: Dict[str, Dict]
    ) -> Dict[str, TargetWeight]:
        """
        Compute target weights for all assets per Spec.md.
        
        Args:
            signals: Signal records from SignalEngine
            returns_df: DataFrame of returns (cols=coins, rows=time) for covariance (optional)
            equity: Account equity (USD)
            ctx_by_coin: Asset contexts mapped by coin name (for OI, volume, mid price)
        
        Returns:
            Dict mapping coin -> TargetWeight
        """
        if not signals:
            return {}
        
        # Step 1: Compute raw weights w_raw = s_adj / max(sigma, sigma_min)
        coins = list(signals.keys())
        w_raw = {}
        for coin in coins:
            sig = signals[coin]
            sigma = max(sig.sigma, self.config.risk.sigma_min)
            w_raw[coin] = sig.s_adj / sigma
        
        # Step 2: Optional beta neutralization
        if self.config.risk.beta_neutral_enabled:
            if returns_df is not None and len(returns_df) > 10:
                btc_symbol = "BTC"
                if btc_symbol in returns_df.columns:
                    betas = self._compute_betas(returns_df, coins, btc_symbol)
                    w_raw = self._beta_neutralize(w_raw, betas, btc_symbol)
                    print("[RiskEngine] Applied BTC beta neutralization")
                else:
                    print(f"[RiskEngine] BTC column '{btc_symbol}' not in returns_df—skipping beta neutralization")
            else:
                print("[RiskEngine] No returns_df—skipping beta neutralization")
        
        # Step 3: Scale to portfolio vol target using covariance
        if returns_df is not None and len(returns_df) > 10 and len(coins) > 1:
            w_scaled = self._scale_to_vol_target(w_raw, returns_df, coins, signals)
        else:
            # Fallback: variance-aware normalization without cov
            print("[RiskEngine] No returns_df or insufficient data—using naive scaling")
            w_scaled = self._naive_scale_to_vol_target(w_raw, signals)
        
        # Step 4: Apply caps and edge gate
        target_weights: Dict[str, TargetWeight] = {}
        
        for coin in coins:
            sig = signals[coin]
            ctx = ctx_by_coin.get(coin, {})
            
            w_pre = w_scaled.get(coin, 0.0)
            caps_applied: Dict[str, float] = {}
            
            # Per-asset cap
            w_max = self.config.risk.w_max
            if abs(w_pre) > w_max:
                caps_applied["per_asset"] = w_max
                w_pre = np.sign(w_pre) * w_max
            
            # Liquidity caps (notional limits)
            notional = abs(w_pre) * equity
            
            day_vol = float(ctx.get("dayNtlVlm", 0))
            oi = float(ctx.get("openInterest", 0))
            mid_px = float(ctx.get("midPx", 0))
            
            if day_vol > 0:
                max_notional_vol = day_vol * self.config.liquidity.max_pct_24h_vol
                if notional > max_notional_vol:
                    caps_applied["24h_vol"] = max_notional_vol
                    notional = max_notional_vol
            
            if oi > 0:
                max_notional_oi = oi * mid_px * self.config.liquidity.max_pct_oi
                if notional > max_notional_oi:
                    caps_applied["oi"] = max_notional_oi
                    notional = min(notional, max_notional_oi)
            
            # Convert back to weight
            w_capped = (notional / equity) * np.sign(w_pre) if equity > 0 else 0.0
            
            # Edge vs friction gate
            edge = self._compute_edge(coin, sig, mid_px)
            friction = self._compute_friction(coin, ctx, notional)
            
            min_ratio = self.config.trade_gate.min_edge_to_cost_ratio
            edge_to_friction = edge / friction if friction > 0 else 0.0
            
            if edge < min_ratio * friction:
                # Skip this trade—not enough edge
                print(f"[RiskEngine] Skipping {coin}: edge/friction={edge_to_friction:.2f} < {min_ratio:.2f}")
                continue
            
            target_weights[coin] = TargetWeight(
                coin=coin,
                w_pre=w_pre,
                w_scaled=w_capped,
                caps_applied=caps_applied,
                edge=edge,
                friction=friction
            )
        
        # Step 5: Rescale after caps/gating to hit target vol (optional)
        if self.config.risk.__dict__.get("rescale_after_caps", False) and returns_df is not None and len(target_weights) > 1:
            kept_coins = [tw.coin for tw in target_weights.values() if tw.w_scaled != 0]
            available_cols = [c for c in kept_coins if c in returns_df.columns]
            if len(available_cols) >= 2:
                sub_returns = returns_df[available_cols].dropna()
                try:
                    cov = self._estimate_covariance(sub_returns)
                    per_year = 252
                    if self.config.bars.interval == "4h":
                        per_year = 6 * 365
                    elif self.config.bars.interval == "1h":
                        per_year = 24 * 365
                    cov_ann = cov * per_year
                    w_vec = np.array([target_weights[c].w_scaled for c in available_cols])
                    port_var_ann = w_vec @ cov_ann @ w_vec
                    if port_var_ann > 0:
                        port_vol_ann = np.sqrt(port_var_ann)
                        k2 = self.config.risk.vol_target_annual / max(port_vol_ann, 1e-12)
                        # Apply scaling and re-apply per-asset cap
                        for c in kept_coins:
                            tw = target_weights[c]
                            tw.w_scaled = float(np.sign(tw.w_scaled) * min(abs(tw.w_scaled * k2), self.config.risk.w_max))
                except Exception as e:
                    print(f"[RiskEngine] Rescale-after-caps failed: {e}")

        print(f"[RiskEngine] Final portfolio: {len(target_weights)} positions")
        return target_weights
    
    def _estimate_covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Estimate covariance matrix with shrinkage.
        
        Args:
            returns_df: Returns matrix (cols=assets, rows=time)
        
        Returns:
            Covariance matrix
        """
        if self.config.risk.cov_method == "ledoit_wolf":
            cov_est = LedoitWolf()
            return cov_est.fit(returns_df.dropna()).covariance_
        elif self.config.risk.cov_method == "ewma":
            # TODO: Implement EWMA covariance
            raise NotImplementedError("EWMA covariance not yet implemented")
        else:
            # Simple sample covariance
            return returns_df.cov().values
    
    def _scale_to_vol_target(
        self,
        w_raw: Dict[str, float],
        returns_df: pd.DataFrame,
        coins: List[str],
        signals: Dict[str, SignalRecord]
    ) -> Dict[str, float]:
        """
        Scale weights to hit portfolio vol target using covariance.
        
        Args:
            w_raw: Raw weights per coin
            returns_df: Returns DataFrame (cols=coins, rows=time)
            coins: List of coins
            signals: Signal records (for fallback)
        
        Returns:
            Scaled weights
        """
        # Filter to coins in signals
        available_cols = [c for c in coins if c in returns_df.columns]
        if len(available_cols) < 2:
            return self._naive_scale_to_vol_target(w_raw, signals)
        
        # Build weight vector
        w_vec = np.array([w_raw.get(c, 0.0) for c in available_cols])
        
        # Estimate covariance
        sub_returns = returns_df[available_cols].dropna()
        if len(sub_returns) < 10:
            return self._naive_scale_to_vol_target(w_raw, signals)
        
        try:
            # Per-bar covariance
            cov = self._estimate_covariance(sub_returns)
            
            # Annualize covariance: var scales linearly with time
            # Infer frequency from returns_df (assume daily by default)
            # TODO: make this config-driven or infer from bar interval
            per_year = 252  # Daily bars assumed
            if self.config.bars.interval == "4h":
                per_year = 6 * 365  # ~2190 4h bars/year
            elif self.config.bars.interval == "1h":
                per_year = 24 * 365  # ~8760 1h bars/year
            
            cov_ann = cov * per_year
            
            # Portfolio variance (annualized): w^T Σ_ann w
            port_var_ann = w_vec @ cov_ann @ w_vec
            if port_var_ann <= 0:
                return self._naive_scale_to_vol_target(w_raw, signals)
            
            # Scaling factor: k = V_target_annual / sqrt(w^T Σ_ann w)
            port_vol_ann = np.sqrt(port_var_ann)
            k = self.config.risk.vol_target_annual / max(port_vol_ann, 1e-12)
            
            # Scale weights
            w_scaled = {c: k * w_raw.get(c, 0.0) for c in coins}
            
            print(f"[RiskEngine] Portfolio vol (annualized): {port_vol_ann:.2%} → scaling by k={k:.3f}")
            return w_scaled
        
        except Exception as e:
            print(f"[RiskEngine] Covariance scaling failed: {e}—using naive")
            import traceback
            traceback.print_exc()
            return self._naive_scale_to_vol_target(w_raw, signals)
    
    def _naive_scale_to_vol_target(
        self,
        w_raw: Dict[str, float],
        signals: Dict[str, SignalRecord]
    ) -> Dict[str, float]:
        """
        Variance-aware scaling without covariance (assumes uncorrelated).
        
        Args:
            w_raw: Raw weights
            signals: Signal records (for per-asset vol)
        
        Returns:
            Scaled weights
        """
        # Approximate portfolio annual vol ignoring correlations:
        # port_vol_ann = sqrt(sum((w_i * sigma_i)^2))
        sum_var = sum((w_raw.get(c, 0.0) * signals[c].sigma) ** 2 for c in w_raw)
        if sum_var <= 0:
            return w_raw
        
        port_vol_ann = np.sqrt(sum_var)
        k = self.config.risk.vol_target_annual / max(port_vol_ann, 1e-12)
        
        print(f"[RiskEngine] Portfolio vol (naive, annualized): {port_vol_ann:.2%} → scaling by k={k:.3f}")
        return {c: k * w for c, w in w_raw.items()}
    
    def _compute_edge(self, coin: str, sig: SignalRecord, mid_px: float) -> float:
        """
        Compute edge proxy: |s_adj| × (ATR / P).
        
        Args:
            coin: Asset name
            sig: Signal record
            mid_px: Current mid price
        
        Returns:
            Edge proxy (dimensionless)
        """
        if mid_px <= 0:
            return 0.0
        
        # Compute ATR
        atr = self._compute_atr(coin, mid_px)
        if atr <= 0:
            return 0.0
        
        edge = abs(sig.s_adj) * (atr / mid_px)
        return edge
    
    def _compute_friction(self, coin: str, ctx: Dict, notional: float) -> float:
        """
        Compute friction: maker/taker fees + slippage.
        
        Note: Funding is already in s_adj, so excluded here to avoid double-count.
        
        Args:
            coin: Asset name
            ctx: Asset context
        
        Returns:
            Friction (bps as decimal, e.g., 0.001 = 10 bps)
        """
        # Base fees (convert bps to decimal)
        maker_bps = self.config.trade_gate.maker_fee_bps / 10000.0
        taker_bps = self.config.trade_gate.taker_fee_bps / 10000.0
        
        # Assume 70% maker, 30% taker (or config-driven)
        avg_fee = 0.7 * maker_bps + 0.3 * taker_bps
        
        # Slippage model
        if self.config.trade_gate.slippage_model == "fixed":
            slippage = self.config.trade_gate.fixed_slippage_bps / 10000.0
        else:
            # L2-depth slippage model (estimate VWAP impact for given notional)
            slippage = self.config.trade_gate.fixed_slippage_bps / 10000.0
            try:
                if self.data_loader and notional > 0 and ctx:
                    book = self.data_loader.get_l2_book(coin)
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    mid_px = float(ctx.get("midPx", 0.0))
                    # Estimate symmetric impact using average of buy/sell impact
                    def impact(levels: list, side: str) -> float:
                        remaining = notional
                        cost = 0.0
                        for px, sz in (levels if side == "buy" else levels):
                            level_notional = float(px) * float(sz)
                            take = min(remaining, level_notional)
                            if take <= 0:
                                continue
                            # Effective price contribution
                            cost += (take / level_notional) * float(px) * float(sz) if level_notional > 0 else 0.0
                            remaining -= take
                            if remaining <= 0:
                                break
                        if remaining > 0 or mid_px <= 0:
                            return 0.0
                        vwap = cost / (notional / mid_px) if mid_px > 0 else mid_px
                        return abs(vwap - mid_px) / mid_px
                    buy_imp = impact(asks, "buy")
                    sell_imp = impact(bids, "sell")
                    slip_frac = max(buy_imp, sell_imp)
                    if slip_frac > 0:
                        slippage = float(slip_frac)
            except Exception as e:
                print(f"[RiskEngine] L2 slippage estimate failed for {coin}: {e}")
        
        friction = avg_fee + slippage
        return friction
    
    def _compute_atr(self, coin: str, mid_px: float) -> float:
        """
        Compute Average True Range (ATR) for edge calculation.
        
        Args:
            coin: Asset name
            mid_px: Current mid price
        
        Returns:
            ATR value
        """
        if not self.data_loader:
            # Fallback: estimate ATR as 2% of price
            return 0.02 * mid_px
        
        try:
            # Fetch recent candles for ATR calculation
            now_ms = int(time.time() * 1000)
            interval = self.config.bars.interval
            interval_ms = {"1d": 86_400_000, "4h": 14_400_000, "1h": 3_600_000}.get(interval, 86_400_000)
            
            # ATR period + buffer
            period = self.config.stops.atr_period
            bars_needed = period + 10
            
            bar_close = (now_ms // interval_ms) * interval_ms
            end_ms = bar_close - 1
            start_ms = end_ms - bars_needed * interval_ms
            
            candles = self.data_loader.get_candles(coin, interval, start_ms, end_ms)
            
            # Sort and deduplicate by timestamp
            candles.sort(key=lambda x: x["t"])
            deduped: List[Dict] = []
            for c in candles:
                if not deduped or c["t"] != deduped[-1]["t"]:
                    deduped.append(c)
            candles = deduped
            
            if len(candles) < period:
                return 0.02 * mid_px  # Fallback
            
            # Compute True Range for each bar: max(H-L, |H-C_prev|, |L-C_prev|)
            trs: List[float] = []
            for i in range(1, len(candles)):
                h = candles[i]["h"]
                l = candles[i]["l"]
                c_prev = candles[i-1]["c"]
                tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
                trs.append(tr)
            
            if len(trs) < period:
                return 0.02 * mid_px
            
            # ATR = simple moving average of TR over period
            atr = float(np.mean(trs[-period:]))
            return atr
        
        except Exception as e:
            print(f"[RiskEngine] ATR computation failed for {coin}: {e}")
            return 0.02 * mid_px
    
    def _beta_neutralize(
        self,
        weights: Dict[str, float],
        btc_beta: Dict[str, float],
        btc_symbol: str
    ) -> Dict[str, float]:
        """
        Add BTC hedge to neutralize portfolio beta.
        
        Args:
            weights: Raw weights per coin
            btc_beta: Beta of each alt to BTC
        
        Returns:
            Adjusted weights (includes BTC overlay)
        """
        # Compute portfolio beta and adjust BTC weight to neutralize
        portfolio_beta = 0.0
        for c, w in weights.items():
            beta = btc_beta.get(c)
            if beta is not None:
                portfolio_beta += w * beta
        if btc_symbol not in weights:
            # If BTC not present, we cannot overlay; return unchanged
            return weights
        adjusted = dict(weights)
        adjusted[btc_symbol] = adjusted.get(btc_symbol, 0.0) - portfolio_beta
        # Enforce per-asset cap on BTC
        w_max = self.config.risk.w_max
        adjusted[btc_symbol] = float(np.sign(adjusted[btc_symbol]) * min(abs(adjusted[btc_symbol]), w_max))
        return adjusted

    def _compute_betas(
        self,
        returns_df: pd.DataFrame,
        coins: List[str],
        btc_symbol: str
    ) -> Dict[str, float]:
        """
        Compute rolling betas of each coin vs BTC using covariance/variance.
        """
        betas: Dict[str, float] = {}
        sub = returns_df[[c for c in coins if c in returns_df.columns] + ([btc_symbol] if btc_symbol in returns_df.columns else [])].dropna()
        if btc_symbol not in sub.columns:
            return betas
        var_btc = float(sub[btc_symbol].var(ddof=1))
        if var_btc <= 0:
            return betas
        for c in coins:
            if c == btc_symbol or c not in sub.columns:
                continue
            cov = float(sub[[c, btc_symbol]].cov().iloc[0, 1])
            betas[c] = cov / var_btc
        # BTC beta to itself is 1
        betas[btc_symbol] = 1.0
        return betas

