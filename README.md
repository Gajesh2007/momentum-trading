# Momentum Trading System - Hyperliquid

A funding-aware, volatility-scaled time-series momentum (TSMOM) strategy for Hyperliquid perpetuals.

## 📐 Specification

Full technical specification: [`Spec.md`](./Spec.md)

## 🏗️ Architecture

```
momentum_trading/
├── core/               # Core system (config, scheduler, orchestrator)
│   ├── config.py       # Configuration management
│   └── scheduler.py    # Rebalance scheduling (daily/4h)
│
├── data/               # Market data loading
│   └── loader.py       # Candles, funding, L2 book, universe
│
├── signals/            # Signal generation
│   └── engine.py       # TSMOM z-scores + funding penalties
│
├── risk/               # Portfolio construction
│   └── engine.py       # Vol targeting, beta neutralization, caps
│
├── execution/          # Order routing
│   ├── orders.py       # OrderIntent, ExecutionReport
│   └── router.py       # ALO/TWAP/IOC routing + reduce-only logic
│
├── monitoring/         # Observability
│   ├── metrics.py      # SLO tracking (maker %, funding PnL, etc.)
│   └── kill_switch.py  # Circuit breakers
│
├── utils/              # Helpers
│   ├── precision.py    # Tick/lot formatting (≤5 sig figs)
│   ├── math_helpers.py # Z-scores, vol calculations, EWMA
│   └── state_store.py  # Persistent state (bars, orders, PnL)
│
└── main.py             # Main orchestrator + CLI entry point
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Set Environment Variables

```bash
export HL_NETWORK=testnet  # or mainnet
export HL_ADDRESS=0xYourMainWallet
export HL_SECRET_KEY=0xYourAPIWalletPrivateKey
```

### 3. Run (Testnet)

```bash
poetry run python -m momentum_trading.main
```

## 📋 Configuration

All parameters in `momentum_trading/core/config.py` match the spec:

### Universe Selection
- `universe.top_n`: Number of assets (default 40)
- `universe.min_day_notional`: Min daily volume
- `universe.min_open_interest`: Min OI

### Signal Generation
- `signal.lookbacks`: Lookback periods `[10, 30, 90]` (daily) or `[24, 96, 240]` (4h bars)
- `signal.zscore_window_bars`: Rolling window for z-scores (126 for daily, 360 for 4h)
- `funding.lambda_penalty`: Funding penalty multiplier `[0.5, 1.5]`

### Risk Management
- `risk.vol_target_annual`: Portfolio vol target (default 15%)
- `risk.w_max`: Max per-asset notional (default 5%)
- `stops.atr_multiple`: ATR stop (default 2.5×)

### Execution
- `execution.twap_threshold_usd`: Use TWAP if notional > threshold (default $50k)
- `execution.alo_refresh_sec`: Requote ALO after N seconds (default 10s)

### Kill Switch
- `kill_switch.daily_loss_pct`: Halt if lose >X% equity/day (default 2%)
- `kill_switch.slip_ceiling_bps`: Halt if slippage >X bps (default 100)

## 🔐 Security

- API wallet (not main wallet) signs orders
- Testnet by default
- Kill switches for loss/slippage/disconnect
- Never commits credentials to git

## 📚 References

- [Spec.md](./Spec.md) - Complete technical specification
- [Strategy.md](./Strategy.md) - Original strategy design
- [Hyperliquid API Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api)
- [TSMOM Research (Moskowitz et al)](https://docs.lhpedersen.com/TimeSeriesMomentum.pdf)

## 📝 License

MIT

---

**Status:** 🚧 Under Development - Testnet Only

