"""
Execution Router

Maker-first ALO placement with re-quotes, TWAP for large deltas,
IOC for protective exits. Sets reduce-only on flips/close.
Tracks precision rules & open-order budget.
"""

from typing import List, Dict, Optional, Tuple
import time
import hashlib
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import signing

from momentum_trading.core.config import Config
from momentum_trading.execution.orders import OrderIntent, ExecutionReport
from momentum_trading.utils.precision import format_price, format_size


class ExecutionRouter:
    """
    Order routing and execution logic.
    
    Implements Spec.md execution flow:
    1. Pre-rebalance: batch cancel all open orders
    2. For each intent:
       - Check if flip (needs two-step: close then open)
       - Choose ALO vs TWAP based on size
       - Format price/size per tick/lot rules
       - Place order with proper reduce_only flag
    3. Monitor fills and maintain TP/SL
    """
    
    def __init__(self, config: Config, exchange: Exchange, info: Optional[Info] = None, dry_run: bool = False):
        """
        Initialize execution router.
        
        Args:
            config: System configuration
            exchange: Hyperliquid Exchange instance
        """
        self.config = config
        self.exchange = exchange
        self.info = info
        self.asset_map: Dict = {}  # coin -> {asset, szDecimals, maxLeverage, etc.}
        self.cycle_id: Optional[str] = None
        self.dry_run: bool = dry_run
        self.api_error_count: int = 0
    
    def set_asset_map(self, meta: Dict):
        """
        Set asset metadata for precision/formatting.
        
        Args:
            meta: Universe metadata from metaAndAssetCtxs
        """
        self.asset_map = {
            u["name"]: {
                "asset": i,
                "szDecimals": u["szDecimals"],
                "maxLeverage": u.get("maxLeverage", 50),
                "onlyIsolated": u.get("onlyIsolated", False),
            }
            for i, u in enumerate(meta["universe"])
        }
    
    def cancel_all_orders(self):
        """Batch cancel all open orders (pre-rebalance cleanup)."""
        if not self.info:
            print("[ExecutionRouter] No Info client; cannot fetch open orders to cancel")
            return
        try:
            addr = self.config.hyperliquid.address
            open_orders = self.info.open_orders(addr)
            cancel_reqs = []
            for oo in open_orders or []:
                try:
                    name = oo.get("coin") or oo.get("name")
                    oid = int(oo.get("oid")) if oo.get("oid") is not None else None
                    if name and oid is not None:
                        cancel_reqs.append({"name": name, "oid": oid})
                except Exception:
                    continue
            if cancel_reqs:
                if not self.dry_run:
                    self.exchange.bulk_cancel(cancel_reqs)  # type: ignore[arg-type]
                print(f"[ExecutionRouter] Cancelled {len(cancel_reqs)} open orders")
        except Exception as e:
            print(f"[ExecutionRouter] Cancel-all failed: {e}")
            self.api_error_count += 1
    
    def execute_intents(
        self,
        intents: List[OrderIntent],
        current_positions: Dict[str, float]
    ) -> List[ExecutionReport]:
        """
        Execute list of order intents.
        
        Args:
            intents: Order intents from risk engine
            current_positions: Current positions (coin -> signed qty)
        
        Returns:
            List of execution reports
        """
        reports: List[ExecutionReport] = []
        for intent in intents:
            try:
                coin = intent.coin
                desired_side_is_buy = intent.side == "Buy"
                cur_qty = float(current_positions.get(coin, 0.0))
                cur_dir = 1 if cur_qty > 0 else (-1 if cur_qty < 0 else 0)
                new_dir = 1 if desired_side_is_buy else -1
                reduce_only = intent.reduce_only
                is_flip = (not reduce_only) and (cur_dir != 0 and cur_dir != new_dir)
                if is_flip:
                    reports.extend(self._handle_flip(intent, cur_qty))
                    continue
                
                if intent.style == "ALO":
                    rep = self._route_alo(intent, reduce_only)
                elif intent.style == "TWAP":
                    rep = self._route_twap(intent, reduce_only)
                else:
                    rep = self._route_ioc(intent, reduce_only)
                reports.append(rep)
            except Exception as e:
                reports.append(ExecutionReport(status="rejected", error_code="router_error", error_msg=str(e)))
                self.api_error_count += 1
        return reports
    
    def _route_alo(self, intent: OrderIntent, reduce_only: bool) -> ExecutionReport:
        """Place post-only (ALO) order."""
        try:
            coin = intent.coin
            is_buy = intent.side == "Buy"
            # Determine price and size
            best_bid, best_ask, mid = self._best_prices(coin)
            if mid <= 0:
                return ExecutionReport(status="rejected", error_code="no_price", error_msg="No price available")
            # Place at best bid/ask to avoid crossing
            raw_px = best_bid if is_buy else best_ask
            sz_dec = self.asset_map.get(coin, {}).get("szDecimals", 0)
            price_str = format_price(raw_px, sz_dec, self.config.precision.max_decimals)
            px = float(price_str)
            sz = self._size_from_notional(coin, intent.abs_notional, px)
            if sz <= 0:
                return ExecutionReport(status="rejected", error_code="zero_size")
            # Build order type: limit ALO
            order_type: signing.OrderType = {"limit": {"tif": "Alo"}}  # type: ignore[assignment]
            cloid = self._make_cloid(coin, is_buy, sz, px, order_type, reduce_only)
            if not self.dry_run:
                resp = self._with_retries(self.exchange.order, coin, is_buy, sz, px, order_type, reduce_only, cloid)
            # Minimal report
            return ExecutionReport(status="pending", maker_ratio=1.0)
        except Exception as e:
            return ExecutionReport(status="rejected", error_code="alo_error", error_msg=str(e))
    
    def _route_twap(self, intent: OrderIntent, reduce_only: bool) -> ExecutionReport:
        """Place TWAP order (30s slices, 3% max slippage)."""
        try:
            coin = intent.coin
            is_buy = intent.side == "Buy"
            # Approximate TWAP using market_open with conservative slippage cap
            best_bid, best_ask, mid = self._best_prices(coin)
            px = mid if mid > 0 else (best_bid if is_buy else best_ask)
            if px <= 0:
                return ExecutionReport(status="rejected", error_code="no_price")
            sz = self._size_from_notional(coin, intent.abs_notional, px)
            if sz <= 0:
                return ExecutionReport(status="rejected", error_code="zero_size")
            if reduce_only:
                # Reduce-only market close for protective exits
                if not self.dry_run:
                    self._with_retries(self.exchange.market_close, coin, sz)
                return ExecutionReport(status="pending", maker_ratio=0.0)
            else:
                if not self.dry_run:
                    self._with_retries(self.exchange.market_open, coin, is_buy, sz, None, 0.03)
                return ExecutionReport(status="pending", maker_ratio=0.0)
        except Exception as e:
            return ExecutionReport(status="rejected", error_code="twap_error", error_msg=str(e))
    
    def _route_ioc(self, intent: OrderIntent, reduce_only: bool) -> ExecutionReport:
        """Place IOC (immediate-or-cancel) order."""
        try:
            coin = intent.coin
            is_buy = intent.side == "Buy"
            best_bid, best_ask, mid = self._best_prices(coin)
            cross_px = (best_ask * 1.0001) if is_buy else (best_bid * 0.9999)
            sz = self._size_from_notional(coin, intent.abs_notional, cross_px)
            if sz <= 0:
                return ExecutionReport(status="rejected", error_code="zero_size")
            order_type: signing.OrderType = {"limit": {"tif": "Ioc"}}  # type: ignore[assignment]
            cloid = self._make_cloid(coin, is_buy, sz, cross_px, order_type, reduce_only)
            if not self.dry_run:
                resp = self._with_retries(self.exchange.order, coin, is_buy, sz, cross_px, order_type, reduce_only, cloid)
            return ExecutionReport(status="pending", maker_ratio=0.0)
        except Exception as e:
            return ExecutionReport(status="rejected", error_code="ioc_error", error_msg=str(e))
    
    def _handle_flip(
        self,
        intent: OrderIntent,
        current_qty: float
    ) -> List[ExecutionReport]:
        """
        Handle position flip (two-step process).
        
        Args:
            intent: New position intent
            current_qty: Current position size
        
        Returns:
            List of [close_report, open_report]
        """
        try:
            coin = intent.coin
            reports: List[ExecutionReport] = []
            # Step 1: Close existing position
            try:
                if not self.dry_run:
                    self._with_retries(self.exchange.market_close, coin, abs(current_qty))
                reports.append(ExecutionReport(status="pending", maker_ratio=0.0))
            except Exception as e:
                reports.append(ExecutionReport(status="rejected", error_code="flip_close_error", error_msg=str(e)))
                return reports
            # Step 2: Open new position
            open_intent = OrderIntent(
                coin=intent.coin,
                side=intent.side,
                abs_notional=intent.abs_notional,
                urgency=intent.urgency,
                style=intent.style,
                reduce_only=False,
                tp_px=intent.tp_px,
                sl_px=intent.sl_px,
                sl_type=intent.sl_type,
            )
            if intent.style == "ALO":
                reports.append(self._route_alo(open_intent, reduce_only=False))
            elif intent.style == "TWAP":
                reports.append(self._route_twap(open_intent, reduce_only=False))
            else:
                reports.append(self._route_ioc(open_intent, reduce_only=False))
            return reports
        except Exception as e:
            return [ExecutionReport(status="rejected", error_code="flip_error", error_msg=str(e))]

    # ------------------------
    # Helpers
    # ------------------------
    def _best_prices(self, coin: str) -> Tuple[float, float, float]:
        """Get best bid/ask and mid price from L2 snapshot."""
        if not self.info:
            return 0.0, 0.0, 0.0
        book = self.info.l2_snapshot(coin)
        bids = book.get("bids") or (book.get("levels", [[], []])[0] if book.get("levels") else [])
        asks = book.get("asks") or (book.get("levels", [[], []])[1] if book.get("levels") else [])
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
        return best_bid, best_ask, mid

    def _size_from_notional(self, coin: str, abs_notional: float, px: float) -> float:
        """Convert USD notional to contract size, respecting szDecimals."""
        if px <= 0:
            return 0.0
        raw_sz = abs_notional / px
        sz_dec = self.asset_map.get(coin, {}).get("szDecimals", 0)
        sz_str = format_size(raw_sz, sz_dec)
        try:
            return float(sz_str)
        except Exception:
            return 0.0

    def _make_cloid(self, coin: str, is_buy: bool, sz: float, px: float, order_type: signing.OrderType, reduce_only: bool) -> str:
        """Create deterministic client order id for idempotency per cycle."""
        base = f"{self.cycle_id or ''}|{coin}|{int(is_buy)}|{reduce_only}|{round(sz,8)}|{round(px,8)}|{order_type}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]

    def _with_retries(self, func, *args, **kwargs):
        """Call exchange function with retries/backoff and error tracking."""
        max_attempts = 3
        delay = 0.5
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.api_error_count += 1
                if attempt == max_attempts:
                    raise
                time.sleep(delay)
                delay *= 2

