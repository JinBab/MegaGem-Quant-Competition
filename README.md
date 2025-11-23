# Team 27 — MegaGem (HKUST)

HK University Web3 Quant Trading Hackathon • Roostoo Mock Exchange

---

This repository contains a working prototype of a momentum-driven, entry‑first trading system built for the Roostoo mock exchange. It is designed to be clear, extensible, and easy to run locally or adapt to another exchange. The implementation focuses on ranking a fixed USDT trading universe, selecting top momentum candidates, applying conservative risk sizing, and executing market entries (entry-only mode).

Contents
- `trading_bot.py` — high-level scaffold (Config, OrderManager, PortfolioManager, Strategy, Runner). Primary orchestrator for conservative entry-only flow and scheduling.
- `structured_bot.py` — alternative object-oriented TradingBot class with modular loops (scan, strategy, order management). Useful as a simpler single-file bot skeleton for experimentation.
- `market_scanner.py` — market utilities and universe definition; provides multiple window change scanners (24h, custom windows) and quick price lookup helpers.
- `BinanceAPI.py` — helper to fetch historical OHLCV klines (public REST) and a pre-built LOT_STEP_INFO map used to round order sizes.
- `RoostooAPI.py` — signed client for the Roostoo mock exchange (balance, order placement, cancel, helper utilities). NOTE: example API keys are included in the repository for the mock; move any real credentials to environment variables.
- `HorusAPI.py` — alternate historical price fetcher (third‑party API) used for quick data pulls in development.
- `backtesting_strategy.py` — (referenced in the original README) example backtest using MACD; useful for offline validation (not required to run live).
- `volatility_scanner.py` — realized volatility utilities to compute annualized volatility for risk adjustment.
- `requirements.txt` — minimal dependencies used in the project.

Universe
- The bot scans a fixed universe of 66 USDT pairs (defined in `market_scanner.py`, e.g., BTCUSDT, ETHUSDT, ADAUSDT, …). This focused list simplifies rate-limit management and risk controls for the hackathon.

Design goals
- Clarity over complexity: easy-to-follow control flow and modular components to iterate quickly.
- Conservative, entry‑first behavior: the production-ish runner enforces entry-only rebalances and avoids forced selling; exits are handled via TP/SL checks or manual intervention.
- Risk-awareness: position sizing is calculated from per‑trade risk and stop-loss assumptions rather than blind equal weighting.
- Extensibility: swap in your preferred execution client, scoring function, or risk model with minimal changes.

Strategy overview (what the code implements)
- Signal (Momentum score)
  - Score is computed from multi-window percent changes (examples: 5d, 1d, 12h in `trading_bot.py`; 1d and 6h in `structured_bot.py`).
  - Scores are combined by configurable weights to form a single "Score" (expressed in percent).
- Filters
  - Entry-only (no automatic sells by rebalance).
  - Per-symbol cooldown to avoid repeatedly entering the same symbol.
  - Optional minimum 24h quote-volume filter (configurable).
  - Required edge filter: the score must be greater than estimated round-trip costs + an additional buffer before buying.
  - Optional multi‑timeframe uptrend confirmation (implemented as a helper `_is_uptrend` in Runner; commented by default).
- Sizing (risk-based)
  - Risk per trade = `risk_per_trade * equity` (Config.risk_per_trade).
  - Position size derived so that an assumed stop_loss_pct corresponds roughly to the dollar risk per trade.
  - Size is limited by available cash and rounded down to symbol-specific LOT_STEP_INFO (step size and min qty).
- Execution
  - Market orders by default (`OrderManager.place_market` / `RoostooAPI.place_order`).
  - Dry-run mode available (Config.dry_run) — simulates fills locally and updates the in-memory PortfolioManager.
  - OrderManager and Runner extract fills and update the PortfolioManager state.
- Exits
  - Basic TP/SL logic exists in the single-file bot (structured_bot.py) and the Runner/PortfolioManager includes hooks to implement TP/SL actions.
  - The runner's conservative flow leaves active exits to TP/SL checker or manual operations (entry-only rebalance mode).
- Scheduling / Orchestration
  - Runner in `trading_bot.py` implements a tick-based scheduler that:
    - refreshes prices,
    - scans and scores periodically,
    - executes entry-only rebalance flows,
    - manages positions (TP/SL checks),
    - syncs balances and persists a lightweight snapshot,
    - sends heartbeat logs.
  - `structured_bot.py` offers a similar single-threaded main loop with three cadences: periodic_scan, main_strategy, order_management.

Key implementation details and notes
- Data sources:
  - Market scanner and quick price lookups use Binance public REST endpoints (`market_scanner.py` / `get_all_market_prices`).
  - Historical candles are fetched via `BinanceAPI.fetch_data` which calls Binance's public klines endpoint and returns a compact OHLCV DataFrame.
  - A small Horus API helper is included for alternate historical fetches (`HorusAPI.py`).
- Order sizing and rounding:
  - `LOT_STEP_INFO` in `BinanceAPI.py` is used to round quantities to allowed exchange steps and ensure they meet the exchange min quantity.
  - Rounding uses Decimal arithmetic to avoid floating point rounding-up errors; quantities are truncated/rounded in a conservative direction.
- Portfolio model:
  - `PortfolioManager` keeps a local in-memory representation of cash and positions; `refresh_from_exchange` reconciles with the mock exchange balances.
  - `equity()` computes the current account equity estimating holdings' market values by fetching latest prices when necessary.
- Safety and production caveats:
  - The provided Roostoo API keys and Horus API key in code are illustrative. Move secrets to environment variables before any real deployment.
  - Respect API rate limits — the code uses threaded requests in a few places (market_scanner) and may trigger limits on public endpoints if run at high concurrency.
  - This prototype is optimized for clarity and iteration, not HFT or low-latency trading.

Quick start (local development)
1. Python 3.11+ recommended. Create and activate a virtual environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Configuration:
   - Edit `trading_bot.py` Config object or construct a Config and pass to Runner.
   - Set `dry_run=True` to simulate trading locally (highly recommended).
   - Move any real API keys to environment variables and adapt `RoostooAPI.py`/callers to read them securely.
4. Run a simple scan:
   - python -c "from market_scanner import get_24h_change; print(get_24h_change().head(10))"
5. Run the runner (dry-run):
   - python trading_bot.py
   - This will perform one rebalance run (`runner.run_once()` in the example) and print portfolio status. To run the scheduler loop, instantiate Runner and call `run_loop()`.

Configuration knobs (high-impact parameters)
- Config.risk_per_trade — fraction of equity risked per trade (default 2%).
- Config.stop_loss_pct — assumed stop distance used for sizing (default 5%).
- Config.max_positions / Config.top_k — limits how many concurrent names to hold.
- Config.min_volume — minimum 24h volume (USDT) required for consideration.
- Config.estimated_slippage_pct and Config.commission — influence required edge checks.
- Config.dry_run — simulate trades locally when True.

Development tips
- Use `dry_run=True` while iterating on scoring and sizing logic.
- The `Runner.rebalance` flow includes many defensive checks and is a good place to add extra risk filters (correlation, open interest, volatility caps).
- For faster iteration, stub `get_price()` to return deterministic prices in tests.
- Backtest strategy ideas locally first (e.g., `backtesting_strategy.py`) before toggling live/dry-run execution.

Dependencies
- See requirements.txt — primary runtime packages: requests, pandas, numpy, python-binance (for optional fallback).

Notes & Disclaimers
- API keys in code are placeholders—move secrets to environment variables.
- Respect API rate limits (Binance public REST and Roostoo mock).
- This prototype is optimized for clarity and iteration, not HFT/latency.

## Extro

- A progress check for the two weeks of deployment. Our model successfully identified a period of significant negative overall crypto market momentum. In strict adherence to our strategy's core logic, which includes a market-regime filter, this resulted in no trade executions during the review period. The absence of trades is therefore not a system failure but a deliberate outcome of our risk-management protocol.

