"""
Metrics Collector

Tracks and logs system metrics: portfolio vol, maker ratio, funding PnL,
API errors, rate limit utilization, etc.
"""

from momentum_trading.core.config import Config


class MetricsCollector:
    """
    Collects and logs system metrics.
    
    SLOs from Spec.md:
    - Maker share ≥70%
    - Funding PnL reconciliation error < 1 bp/day
    - Rebalance jitter < ±10s from target
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = {}
    
    def record_edge_friction(self, edge_to_friction: float):
        arr = self.metrics.get("edge_to_friction", [])
        arr.append(edge_to_friction)
        self.metrics["edge_to_friction"] = arr
    
    def snapshot(self) -> dict:
        return dict(self.metrics)
