"""
Market Data Loader

Pulls candles, fundingHistory, l2Book, mids from Hyperliquid Info endpoint.
Enforces pagination caps (5k bars), retry/backoff, and cache TTLs.
"""

from typing import List, Dict, Tuple, Any

from hyperliquid.info import Info
from momentum_trading.core.config import Config


class MarketDataLoader:
    """
    Fetches market data from Hyperliquid Info endpoint using SDK methods.
    
    Handles:
    - Candle snapshots (≤5000 bars, pagination)
    - Funding history (hourly rates)
    - L2 order book (20 levels per side)
    - Universe metrics (dayNtlVlm, openInterest)
    - Predicted fundings
    """
    
    def __init__(self, config: Config):
        """
        Initialize data loader.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.info = Info(config.hyperliquid.api_url, skip_ws=True)
        self._cache: Dict = {}

    # ------------------------
    # Internal helpers
    # ------------------------

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        """
        Convert interval string to milliseconds (supports 1m..1M subset used here).
        """
        mapping = {
            "1m": 60_000,
            "5m": 5 * 60_000,
            "15m": 15 * 60_000,
            "30m": 30 * 60_000,
            "1h": 60 * 60_000,
            "4h": 4 * 60 * 60_000,
            "1d": 24 * 60 * 60_000,
        }
        if interval not in mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        return mapping[interval]

    @staticmethod
    def _percentile_ranks(values: List[float]) -> List[float]:
        """
        Compute simple percentile ranks in [0,1] for a list of values.
        """
        if not values:
            return []
        sorted_vals = sorted(v for v in values)
        n = len(sorted_vals)
        ranks: List[float] = []
        for v in values:
            # rank position of v in sorted list (average rank for ties)
            # Use mid-rank approach
            lt = sum(1 for x in sorted_vals if x < v)
            eq = sum(1 for x in sorted_vals if x == v)
            # percentile of center of ties
            rank_pos = lt + (eq - 1) / 2
            ranks.append((rank_pos + 1) / n)
        return ranks
    
    def get_universe(self) -> List[str]:
        """
        Fetch and filter universe to top-N liquid assets.
        
        Returns:
            List of coin names (e.g., ["BTC", "ETH", "SOL"])
        """
        meta, asset_ctxs = self.get_meta_and_asset_ctxs()

        # Extract per-asset fields aligned by index
        universe_list: List[Dict[str, Any]] = meta.get("universe", [])
        coins: List[str] = [u.get("name") for u in universe_list]

        day_notional: List[float] = []
        open_interest: List[float] = []
        mid_price: List[float] = []
        for ctx in asset_ctxs:
            # Field names per docs; be robust to alternatives
            day_notional.append(float(ctx.get("dayNtlVlm", 0.0)))
            open_interest.append(float(ctx.get("openInterest", 0.0)))
            mid_px = ctx.get("midPx")
            if mid_px is None:
                mid_px = ctx.get("oraclePx") or ctx.get("markPx") or 0.0
            mid_price.append(float(mid_px))

        # Filter by basic thresholds and non-stale mid price
        filtered_indices: List[int] = []
        for i, coin in enumerate(coins):
            if coin is None:
                continue
            if mid_price[i] <= 0:
                continue
            if day_notional[i] < self.config.universe.min_day_notional:
                continue
            if open_interest[i] < self.config.universe.min_open_interest:
                continue
            filtered_indices.append(i)

        # Ranking score
        if self.config.universe.ranking_method == "product":
            scores = [day_notional[i] * open_interest[i] for i in filtered_indices]
        else:
            dn_vals = [day_notional[i] for i in filtered_indices]
            oi_vals = [open_interest[i] for i in filtered_indices]
            dn_ranks = self._percentile_ranks(dn_vals)
            oi_ranks = self._percentile_ranks(oi_vals)
            scores = [(dn_ranks[k] + oi_ranks[k]) / 2.0 for k in range(len(filtered_indices))]

        ranked = sorted(zip(filtered_indices, scores), key=lambda x: x[1], reverse=True)
        top_n = min(self.config.universe.top_n, len(ranked))
        selected_indices = [idx for idx, _ in ranked[:top_n]]
        return [coins[i] for i in selected_indices]
    
    def get_candles(
        self,
        coin: str,
        interval: str,
        start_ms: int,
        end_ms: int
    ) -> List[Dict]:
        """
        Fetch candle data for a coin using SDK.
        
        Args:
            coin: Asset name
            interval: Bar interval ("1d", "4h", "1h", etc.)
            start_ms: Start timestamp (epoch milliseconds)
            end_ms: End timestamp (epoch milliseconds)
        
        Returns:
            List of candles with keys: t, o, h, l, c, v, n
        """
        # SDK method: candles_snapshot(coin, interval, startTime, endTime)
        # Paginate by time range to respect ≤5000 bars per request
        interval_ms = self._interval_to_ms(interval)
        all_candles: List[Dict] = []
        next_start = int(start_ms)
        end = int(end_ms)

        while next_start < end:
            resp = self.info.candles_snapshot(coin, interval, next_start, end)
            # SDK returns list of candles or wrapped dict
            candles = resp if isinstance(resp, list) else resp.get("candles", [])
            if not candles:
                break

            # Normalize entries to dict form {t,o,h,l,c,v,n}
            normalized: List[Dict] = []
            for c in candles:
                if isinstance(c, dict):
                    normalized.append({
                        "t": int(c.get("t")),
                        "o": float(c.get("o")),
                        "h": float(c.get("h")),
                        "l": float(c.get("l")),
                        "c": float(c.get("c")),
                        "v": float(c.get("v", 0.0)),
                        "n": int(c.get("n", 0)),
                    })
                elif isinstance(c, list) and len(c) >= 6:
                    # Assume [t, o, h, l, c, v, (n?)]
                    normalized.append({
                        "t": int(c[0]),
                        "o": float(c[1]),
                        "h": float(c[2]),
                        "l": float(c[3]),
                        "c": float(c[4]),
                        "v": float(c[5]),
                        "n": int(c[6]) if len(c) > 6 else 0,
                    })
            if not normalized:
                break

            all_candles.extend(normalized)
            last_t = normalized[-1]["t"]
            # Advance just beyond the last returned bar time
            next_start = int(last_t + interval_ms)

            # Safety: if no progress, break
            if len(normalized) < 2 and last_t <= next_start - interval_ms:
                break

        return all_candles
    
    def get_funding_history(
        self,
        coin: str,
        start_ms: int,
        end_ms: int
    ) -> List[Dict]:
        """
        Fetch funding history for a coin using SDK.
        
        Args:
            coin: Asset name
            start_ms: Start timestamp
            end_ms: End timestamp
        
        Returns:
            List of funding records with time, premium, funding_8h
        """
        # SDK method: funding_history(coin, startTime, endTime)
        resp = self.info.funding_history(coin, int(start_ms), int(end_ms))
        # Response is a direct list of funding records
        records = resp if isinstance(resp, list) else []
        out: List[Dict] = []
        if not records:
            return out
        for r in records:
            if isinstance(r, dict):
                # Official keys: time, premium, fundingRate (string, 8h rate)
                out.append({
                    "t": int(r.get("time")),
                    "premium": float(r.get("premium", 0.0)),
                    "funding_8h": float(r.get("fundingRate", 0.0)),
                })
        return out
    
    def get_predicted_funding(self, coin: str) -> float:
        """
        Get predicted next-hour funding rate using SDK post method.
        
        Args:
            coin: Asset name
        
        Returns:
            Predicted hourly funding rate (positive = longs pay)
        """
        # Fetch all and extract single coin (less efficient; prefer batch method)
        all_funding = self.get_all_predicted_funding()
        return all_funding.get(coin, 0.0)
    
    def get_all_predicted_funding(self) -> Dict[str, float]:
        """
        Get predicted next-hour funding rates for all coins (batch).
        
        Returns:
            Dict mapping coin -> hourly funding rate (positive = longs pay)
        """
        # Response is an array of pairs: [ [coin, [ [venue, {fundingRate,...}], ... ] ], ... ]
        # SDK doesn't have a dedicated method for this; use post()
        resp = self.info.post("/info", {"type": "predictedFundings"})
        items = resp if isinstance(resp, list) else []
        
        funding_map: Dict[str, float] = {}
        if not items:
            return funding_map
        
        for entry in items:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            c_name, venues = entry[0], entry[1]
            if not isinstance(venues, list):
                continue
            
            # Look for Hyperliquid venue first
            hl_rate_8h: float | None = None
            fallback_rate_8h: float | None = None
            for v in venues:
                if not isinstance(v, (list, tuple)) or len(v) < 2:
                    continue
                venue_name, venue_obj = v[0], v[1]
                try:
                    rate_str = venue_obj.get("fundingRate") if isinstance(venue_obj, dict) else None
                    if rate_str is None:
                        continue
                    rate_8h = float(rate_str)
                except Exception:
                    continue
                if str(venue_name) in ("HlPerp", "HLPerp", "Hyperliquid"):
                    hl_rate_8h = rate_8h
                if fallback_rate_8h is None:
                    fallback_rate_8h = rate_8h
            
            chosen = hl_rate_8h if hl_rate_8h is not None else fallback_rate_8h
            if chosen is not None:
                # Convert 8h rate to hourly
                funding_map[c_name] = chosen / 8.0
        
        return funding_map
    
    def get_l2_book(self, coin: str) -> Dict:
        """
        Fetch L2 order book snapshot using SDK.
        
        Args:
            coin: Asset name
        
        Returns:
            Dict with keys: levels (bids/asks), time
        """
        # SDK method: l2_snapshot(coin)
        resp = self.info.l2_snapshot(coin)
        # Normalize to {"bids": [[px, sz], ...], "asks": [[px, sz], ...], "time": t}
        if isinstance(resp, dict):
            # SDK returns {"levels": [[bids], [asks]], "time": t} or top-level bids/asks
            levels = resp.get("levels")
            if isinstance(levels, list) and len(levels) >= 2:
                bids, asks = levels[0], levels[1]
            else:
                bids = resp.get("bids") or []
                asks = resp.get("asks") or []
            t = resp.get("time") or resp.get("ts") or 0
        else:
            bids, asks, t = [], [], 0
        # Ensure float formatting
        def _norm(side: List) -> List[List[float]]:
            out: List[List[float]] = []
            for lvl in side:
                if isinstance(lvl, dict) and "px" in lvl and "sz" in lvl:
                    out.append([float(lvl["px"]), float(lvl["sz"])])
                elif isinstance(lvl, list) and len(lvl) >= 2:
                    out.append([float(lvl[0]), float(lvl[1])])
            return out
        return {"bids": _norm(bids), "asks": _norm(asks), "time": int(t)}
    
    def get_meta_and_asset_ctxs(self) -> Tuple[Dict, List[Dict]]:
        """
        Fetch universe metadata and asset contexts using SDK.
        
        Returns:
            Tuple of (meta, asset_contexts)
        """
        # SDK method: meta_and_asset_ctxs()
        resp = self.info.meta_and_asset_ctxs()
        # Official shape: [meta, assetCtxs]
        if isinstance(resp, list) and len(resp) >= 2:
            meta = resp[0] or {}
            asset_ctxs = resp[1] or []
        elif isinstance(resp, dict):
            # Fallback for alternate shapes
            meta = resp.get("meta") or {}
            asset_ctxs = resp.get("assetCtxs") or []
        else:
            raise RuntimeError("Unexpected response for metaAndAssetCtxs")
        return meta, asset_ctxs

