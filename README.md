# Team 27 — MegaGem (HKUST)

HK University Web3 Quant Trading Hackathon • Roostoo Mock Exchange • Crypto Momentum + Risk Sizing Bot

---

## Repository Overview

This repo contains a working prototype of a momentum-driven, entry‑only trading bot designed for the Roostoo mock exchange. It scans a fixed USDT universe on Binance, ranks coins by momentum, confirms trend, sizes positions by risk, and places simulated or live market orders. A lightweight scheduler runs periodic tasks (scan, rebalance, manage TP/SL, sync balances, heartbeat).

Key components:
- `trading_bot.py` — core framework (Config, OrderManager, PortfolioManager, Strategy, Runner/scheduler)
- `market_scanner.py` — momentum and price utilities from Binance public REST (24h, rolling window, custom window, spot prices)
- `BinanceAPI.py` — historical OHLCV fetch via python-binance; lot step/min qty map used for order rounding
- `RoostooAPI.py` — signed client for Roostoo mock exchange (balance, orders)
- `backtesting_strategy.py` — simple MACD-based backtest example using `backtesting` + TA‑Lib (standalone)
- `volatility_scanner.py` — realized volatility (annualized) across the same symbol universe

Universe: 66 USDT pairs defined in `market_scanner.py` (e.g., BTCUSDT, ETHUSDT, SOLUSDT, …).

---

## Strategy at a Glance

- Signal: Momentum score = 0.7 × long-window change + 0.3 × 24h change
- Filters: Entry-only (skip if already held), per-symbol cooldown, optional volume min, multi‑timeframe uptrend confirmation
- Cost-aware: Requires score to exceed estimated round‑trip costs + buffer (unless test mode)
- Sizing: Risk-based. Risk per trade = `risk_per_trade × equity`; position sized so a stop at `stop_loss_pct` ≈ risk
- Execution: Market orders; quantities rounded to lot step/min qty
- Exits: Lightweight TP/SL watcher + deeper check; sells full position on TP/SL cross
- Modes: `dry_run` for simulation; `test_force_trades`/`test_ignore_trend` for quick iteration
- Scheduling: Periodic price refresh, scan/score, rebalance (entries), manage positions, sync balances, persist snapshot, heartbeat

---

## High-Level Data Flow

```
Binance (public REST) ──> market_scanner ──> Strategy.score_symbols
                                  │
                                  └─> get_price (quotes)

BinanceAPI (python-binance OHLCV) ──> checks/fallbacks (klines)

RoostooAPI (mock exchange) <── OrderManager (orders) / PortfolioManager (balance sync)

Runner (scheduler) orchestrates: refresh prices → scan → entries → manage TP/SL → sync → persist → heartbeat
```

---

## Quick Start (Local)

- Python 3.11+ recommended; use a virtual environment.
- Install essentials: `python-binance`, `pandas`, `requests` (optional for backtests: `backtesting`, `TA-Lib`).
- Optional: set Binance API key/secret (or leave placeholders if only using public endpoints and `dry_run`).
- Run a scan (example): import `market_scanner.get_24h_change()` and call `.head(10)`.
- Run the scheduler: execute `trading_bot.py` (defaults to `dry_run=True`).

Note: Move Roostoo/Binance API keys to environment variables for safety.

---

## Volatility (Optional Input)

`volatility_scanner.py` computes realized (annualized) volatility from OHLCV log returns for the same symbol list. You can merge this with momentum to build a risk‑adjusted ranking.

---

## Notes & Disclaimers
- API keys in code are placeholders—move secrets to environment variables
- Respect API rate limits (Binance public REST and Roostoo mock)
- This prototype is optimized for clarity and iteration, not HFT/latency

---

## Extro

- A progress check for the two weeks of deployment. Our model successfully identified a period of significant negative overall crypto market momentum. In strict adherence to our strategy's core logic, which includes a market-regime filter, this resulted in no trade executions during the review period. The absence of trades is therefore not a system failure but a deliberate outcome of our risk-management protocol.


## Team
Team 27 — MegaGem (HKUST)
