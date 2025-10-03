"""
Configuration management for momentum trading system.

All parameters from Spec.md "Configuration (keys & ranges)" section.
Supports loading from YAML/JSON and environment variable overrides.
"""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class UniverseConfig:
    """Universe selection parameters."""
    
    min_day_notional: float = 10_000_000.0  # $10M minimum daily volume
    min_open_interest: float = 5_000_000.0  # $5M minimum OI
    top_n: int = 40  # Take top N by score
    ranking_method: Literal["product", "percentile"] = "product"  # volume × OI or percentile avg


@dataclass
class BarsConfig:
    """Bar and rebalance timing."""
    
    interval: Literal["1d", "4h"] = "1d"  # Daily or 4h bars
    rebalance_at: str = "00:05:00"  # UTC time for daily; "+5min" relative for 4h


@dataclass
class SignalConfig:
    """Signal generation parameters."""
    
    lookbacks: list[int] = field(default_factory=lambda: [10, 30, 90])  # Lookback periods (bars)
    zscore_window_bars: int = 126  # Rolling window for z-score (126d ≈ 6mo, 360 for 4h)
    clip_min: float = -2.0  # Clip signals to [-2, 2]
    clip_max: float = 2.0
    epsilon: float = 1e-9  # Guard against zero std dev


@dataclass
class FundingConfig:
    """Funding-aware adjustment parameters."""
    
    lambda_penalty: float = 1.0  # Funding penalty multiplier [0.5, 1.5]
    holding_hours: int = 24  # Expected holding horizon (24 for daily, 4 for 4h)
    directional: bool = True  # Only penalize the side that pays


@dataclass
class RiskConfig:
    """Risk management and sizing parameters."""
    
    vol_target_annual: float = 0.15  # 15% annualized portfolio vol
    w_max: float = 0.05  # Max 5% notional per asset
    sigma_min: float = 0.01  # Floor on realized vol (1% annual)
    
    # Beta neutralization (optional)
    beta_neutral_enabled: bool = False
    beta_lookback_days: int = 60
    
    # Covariance estimation
    cov_method: Literal["ledoit_wolf", "ewma", "sample"] = "ledoit_wolf"
    ewma_halflife: int = 60  # For EWMA cov


@dataclass
class LiquidityConfig:
    """Liquidity constraints."""
    
    max_pct_24h_vol: float = 0.005  # Max 0.5% of daily volume per trade
    max_pct_oi: float = 0.02  # Max 2% of open interest per trade


@dataclass
class StopsConfig:
    """Stop-loss parameters."""
    
    atr_multiple: float = 2.5  # 2.5× ATR stop (0 = disabled)
    atr_period: int = 14  # 14-period ATR


@dataclass
class TradeGateConfig:
    """Edge vs friction filtering."""
    
    min_edge_to_cost_ratio: float = 2.5  # Trade only if edge ≥ 2.5× friction
    maker_fee_bps: float = 0.15  # 0.015% maker fee (base tier)
    taker_fee_bps: float = 4.5  # 0.045% taker fee
    slippage_model: Literal["fixed", "l2_depth"] = "l2_depth"
    fixed_slippage_bps: float = 2.0  # If using fixed model


@dataclass
class ExecutionConfig:
    """Execution routing parameters."""
    
    alo_refresh_sec: int = 10  # Requote ALO after N seconds if unfilled
    twap_enabled: bool = True
    twap_total_minutes: int = 8  # TWAP duration for large orders
    twap_threshold_usd: float = 50_000.0  # Use TWAP if notional > $50k
    
    # Order precision
    enforce_sigfigs: bool = True
    max_decimals: int = 6  # Hyperliquid MAX_DECIMALS


@dataclass
class PrecisionConfig:
    """Tick/lot precision enforcement."""
    
    enforce_sigfigs: bool = True
    max_sig_figs: int = 5  # ≤5 significant figures for price
    max_decimals: int = 6  # MAX_DECIMALS constant


@dataclass
class KillSwitchConfig:
    """Circuit breaker parameters."""
    
    daily_loss_pct: float = 0.02  # Halt if lose >2% equity in day
    ws_heartbeat_misses: int = 5  # Halt after N missed heartbeats
    slip_ceiling_bps: float = 100.0  # Halt if slippage >1%
    max_api_errors: int = 10  # Halt after N consecutive API errors


@dataclass
class MonitoringConfig:
    """Monitoring and alerting."""
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str = "logs"
    metrics_enabled: bool = True
    alert_on_kill_switch: bool = True


@dataclass
class HyperliquidConfig:
    """Hyperliquid-specific settings."""
    
    network: Literal["testnet", "mainnet"] = "testnet"
    address: str = ""  # Main wallet address (from env)
    secret_key: str = ""  # API wallet private key (from env)
    
    # API endpoints (auto-set by network)
    api_url: str = ""
    ws_url: str = ""
    
    def __post_init__(self):
        """Set API URLs based on network."""
        from hyperliquid.utils import constants
        
        if self.network == "testnet":
            self.api_url = constants.TESTNET_API_URL
            self.ws_url = constants.TESTNET_API_URL.replace("http", "ws")
        else:
            self.api_url = constants.MAINNET_API_URL
            self.ws_url = constants.MAINNET_API_URL.replace("http", "ws")


@dataclass
class Config:
    """
    Complete system configuration.
    
    Load from YAML/environment variables. All parameters match Spec.md.
    
    Environment variables (override config file):
    - HL_NETWORK: "testnet" or "mainnet"
    - HL_ADDRESS: Main wallet address
    - HL_SECRET_KEY: API wallet private key
    """
    
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    bars: BarsConfig = field(default_factory=BarsConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    funding: FundingConfig = field(default_factory=FundingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    stops: StopsConfig = field(default_factory=StopsConfig)
    trade_gate: TradeGateConfig = field(default_factory=TradeGateConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    kill_switch: KillSwitchConfig = field(default_factory=KillSwitchConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    hyperliquid: HyperliquidConfig = field(default_factory=HyperliquidConfig)
    
    def __post_init__(self):
        """Load environment variable overrides."""
        # Hyperliquid credentials from env
        if os.getenv("HL_NETWORK"):
            self.hyperliquid.network = os.getenv("HL_NETWORK", "testnet")
        
        if os.getenv("HL_ADDRESS"):
            self.hyperliquid.address = os.getenv("HL_ADDRESS", "")
        
        if os.getenv("HL_SECRET_KEY"):
            self.hyperliquid.secret_key = os.getenv("HL_SECRET_KEY", "")
        
        # Re-initialize to set API URLs
        self.hyperliquid.__post_init__()
        
        # Adjust signal window for 4h bars
        if self.bars.interval == "4h":
            if self.signal.zscore_window_bars == 126:  # Default for daily
                self.signal.zscore_window_bars = 360  # ~60 days of 4h bars
            self.funding.holding_hours = 4
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        import yaml
        from dataclasses import is_dataclass, fields
        
        def build(dc_type, data):
            if not is_dataclass(dc_type):
                return data
            kwargs = {}
            for f in fields(dc_type):
                if f.name in data:
                    val = data[f.name]
                    if hasattr(f.type, "__dataclass_fields__"):
                        kwargs[f.name] = build(f.type, val)
                    else:
                        kwargs[f.name] = val
            return dc_type(**kwargs)
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return build(cls, data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Load config from dictionary."""
        from dataclasses import is_dataclass, fields
        def build(dc_type, data):
            if not is_dataclass(dc_type):
                return data
            kwargs = {}
            for f in fields(dc_type):
                if f.name in data:
                    val = data[f.name]
                    if hasattr(f.type, "__dataclass_fields__"):
                        kwargs[f.name] = build(f.type, val)
                    else:
                        kwargs[f.name] = val
            return dc_type(**kwargs)
        return build(cls, data)
    
    def validate(self) -> list[str]:
        """
        Validate configuration parameters.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required credentials
        if not self.hyperliquid.address:
            errors.append("HL_ADDRESS environment variable required")
        
        if not self.hyperliquid.secret_key:
            errors.append("HL_SECRET_KEY environment variable required")
        
        # Validate ranges
        if not (0.5 <= self.funding.lambda_penalty <= 1.5):
            errors.append("funding.lambda_penalty must be in [0.5, 1.5]")
        
        if not (0.10 <= self.risk.vol_target_annual <= 0.30):
            errors.append("risk.vol_target_annual should be in [0.10, 0.30]")
        
        if not (0.03 <= self.risk.w_max <= 0.10):
            errors.append("risk.w_max should be in [0.03, 0.10]")
        
        if self.trade_gate.min_edge_to_cost_ratio < 1.0:
            errors.append("trade_gate.min_edge_to_cost_ratio must be >= 1.0")
        
        return errors

