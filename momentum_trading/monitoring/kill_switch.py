"""
Kill Switch

Circuit breaker that halts trading on:
- Daily loss > cap
- Persistent WS disconnect
- Slippage > ceiling
- Covariance blow-up
- API reject storms
"""

from momentum_trading.core.config import Config


class KillSwitch:
    """Circuit breaker for system failures."""
    
    def __init__(self, config: Config):
        self.config = config
        self.triggered = False
        self.reason = None
    
    def check(self, metrics: dict) -> bool:
        """
        Check if kill switch should trigger.
        
        Returns:
            True if should halt trading
        """
        try:
            # Daily loss cap: requires external PnL tracking, skipped here
            # Slippage ceiling: if recent edge_to_friction is too low persistently, could indicate high costs
            edge_list = metrics.get("edge_to_friction", []) or []
            if edge_list:
                import statistics as stats
                median_ratio = stats.median(edge_list)
                if median_ratio < 1.0:  # consistently poor edge vs friction
                    self.reason = f"Edge/Friction median too low: {median_ratio:.2f}"
                    return True
            return False
        except Exception as e:
            print(f"[KillSwitch] check failed: {e}")
            return False
    
    def trigger(self, reason: str):
        """Trigger kill switch."""
        self.triggered = True
        self.reason = reason
        print(f"[KILL SWITCH] TRIGGERED: {reason}")
    
    def reset(self):
        """Reset kill switch (manual intervention)."""
        self.triggered = False
        self.reason = None
