"""Trading bot scaffold for Roostoo using project endpoints.

This file provides a clear structure to plug in momentum-based entry/exit
algorithms. It does not implement a production-ready strategy; it gives
well-documented hooks and simple risk management utilities.

How to use:
 - Implement `score_symbols` and `entry_signals`/`exit_signals` with your
   algorithmic logic.
 - Instantiate Runner(config) and call runner.run_once() or run_loop().
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time
import logging
import random
import pandas as pd

# local project imports (scanners + exchange wrapper)
from RoostooAPI import place_order, get_balance
from BinanceAPI import fetch_data, LOT_STEP_INFO
from decimal import Decimal, ROUND_HALF_UP, getcontext

# Normal import for the scanner (renamed to a valid module name)
from market_scanner import get_x_change, get_24h_change, get_price

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    seed_cash: float = 44946
    commission: float = 0.001  # 0.1% -> 0.001
    trading_window_days: int = 5
    scan_intervals: List[str] = field(default_factory=lambda: ["1d", "12h", "1h"])
    max_positions: int = 10
    risk_per_trade: float = 0.02  # fraction of equity
    min_cash_reserve: float = 0
    stop_loss_pct: float = 0.05  # assumed stop-loss per trade (5%) used for sizing
    top_k: int = 6  # allocate only to top K scanned coins
    trend_hours: int = 6  # lookback hours to verify uptrend
    trend_intervals: List[str] = field(default_factory=lambda: ['5m', '1h'])  # multi-timeframe intervals to confirm trend
    # When True, simulate orders locally and don't call RoostooAPI.place_order
    dry_run: bool = True
    # Minimum 24h quote volume (USDT) required to consider a symbol for trading
    min_volume: float = 10000.0
    # Default take-profit percent for reporting (10% -> 0.10). Used for display and TP suggestions.
    take_profit_pct: float = 0.10
    # Conservative entry-only rebalancer knobs
    rebalance_mode: str = 'entry_only'  # 'entry_only' or 'target'
    min_allocation_delta: float = 0.05  # fraction (5%) - only top up if allocation gap larger
    symbol_cooldown_minutes: int = 60
    max_trades_per_cycle: int = 2
    max_trades_per_hour: int = 8
    estimated_slippage_pct: float = 0.002  # 0.2%
    expected_edge_buffer_pct: float = 0.002  # extra buffer beyond estimated costs
    min_cash_per_trade: float = 50.0
    # test helper: when True, ignore required edge checks to force simulated buys in dry-run
    test_force_trades: bool = False
    # when True in a test run, skip the uptrend/multi-timeframe confirmation
    test_ignore_trend: bool = False
    # scheduler intervals (seconds)
    tick_interval_seconds: int = 15
    price_refresh_seconds: int = 30
    scan_seconds: int = 300
    rebalance_seconds: int = 300
    manage_positions_seconds: int = 60
    sync_balances_seconds: int = 60
    persist_seconds: int = 60
    heartbeat_seconds: int = 30


class OrderManager:
    def __init__(self, config: Config):
        self.config = config

    def place_market(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        logger.info(f"Placing market order: {side} {qty} {symbol} (dry_run={self.config.dry_run})")
        # Roostoo expects coin name without USDT, e.g., 'BTC' and pair will be coin/USD
        coin = symbol.replace('USDT', '')

        # Dry-run: simulate an immediate fill at the current price
        if self.config.dry_run:
            try:
                price = float(get_price(symbol))
            except Exception:
                price = None
            return {"ok": True, "api_response": {"simulated": True}, "filled_qty": qty, "fill_price": price}

        # Live mode: call the exchange
        res = place_order(coin, side, qty, price=None)

        if res.get('status_code') == 'error':
            logger.error(f"Failed to place order: {res.get('text')}")
            return {"ok": False, "error": res.get('text')}
        if res.get('Success') is False:
            logger.error(f"Order rejected: {res}")
            return {"ok": False, "error": res}

        # Try to extract executed quantity and fill price from the API response.
        fill_qty = None
        fill_price = None
        try:
            if 'OrderDetail' in res:
                fill_qty = res['OrderDetail'].get('FilledQuantity')
                fill_price = res['OrderDetail'].get('FilledAverPrice')
        except Exception:
            pass

        return {"ok": True, "api_response": res, "filled_qty": fill_qty, "fill_price": fill_price}


class PortfolioManager:
    """Tracks portfolio allocations and positions (simple in-memory).

    In a real system you'd persist this to a database and reconcile with the
    exchange. Here it's an in-memory model to attach trading logic.
    """

    def __init__(self, config: Config):
        self.config = config
        self.cash = config.seed_cash
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {qty, avg_price, side}

    def update_after_fill(self, symbol: str, side: str, qty: float, fill_price: float):
        cost = qty * fill_price
        fee = cost * self.config.commission
        if side.upper() == 'BUY':
            # Reduce cash and add position
            self.cash -= (cost + fee)
            pos = self.positions.get(symbol)
            if pos:
                # update weighted average price
                total_qty = pos['qty'] + qty
                pos['avg_price'] = (pos['avg_price'] * pos['qty'] + fill_price * qty) / total_qty
                pos['qty'] = total_qty
            else:
                self.positions[symbol] = {'qty': qty, 'avg_price': fill_price}
        else:
            # SELL: reduce position and add cash
            pos = self.positions.get(symbol)
            proceeds = qty * fill_price
            self.cash += (proceeds - fee)
            if pos:
                pos['qty'] -= qty
                if pos['qty'] <= 0:
                    del self.positions[symbol]

    def equity(self, price_map: Optional[Dict[str, float]] = None) -> float:
        """Return current account equity using optional price_map for positions."""
        eq = self.cash
        if price_map is None:
            # fallback: query recent OHLCV from Binance for each held symbol
            price_map = {}
            for s, p in self.positions.items():
                try:
                    # fetch a few minutes of 1m candles and take the last close
                        # first try quick REST price
                        try:
                            price_map[s] = float(get_price(s))
                        except Exception:
                            end = datetime.now()
                            start = end - timedelta(minutes=5)
                            df = fetch_data(s, '1m', start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
                            price_map[s] = float(df['Close'].iloc[-1])
                except Exception:
                    # fallback to position average price
                    price_map[s] = p['avg_price']

        for s, p in self.positions.items():
            price = price_map.get(s, p['avg_price'])
            eq += p['qty'] * price
        return eq

    def refresh_from_exchange(self):
        """Refresh local cash and positions from Roostoo `get_balance()`.

        - Sets self.cash from 'USD' free balance when present.
        - Updates/creates positions for any non-USD spot balances using current market price
          as an estimated average price if we don't already have one.
        """
        try:
            bal = get_balance()
        except Exception:
            logger.exception("Failed to fetch balance from Roostoo")
            return

        # structure: {'Success': True, 'SpotWallet': { 'BTC': {'Free':..., 'Lock':...}, ...}}
        spot = bal.get('SpotWallet', {}) if isinstance(bal, dict) else {}

        # Update cash if USD present
        usd = spot.get('USD')
        if usd:
            try:
                self.cash = float(usd.get('Free', self.cash))
            except Exception:
                pass

        # Update asset positions
        for coin, asset in spot.items():
            if coin == 'USD':
                continue
            try:
                free = float(asset.get('Free', 0))
                lock = float(asset.get('Lock', 0))
                total_qty = free + lock
            except Exception:
                total_qty = 0.0

            symbol = coin + 'USDT'
            if total_qty <= 0:
                # remove if exists
                if symbol in self.positions:
                    del self.positions[symbol]
                continue

            # if we already have an avg price, keep it; otherwise estimate with current market price
            pos = self.positions.get(symbol)
            if pos and pos.get('avg_price', 0) > 0:
                avg_price = pos['avg_price']
            else:
                try:
                    avg_price = float(get_price(symbol))
                except Exception:
                    # fallback to 0 to avoid exceptions; caller should handle
                    avg_price = 0.0

            self.positions[symbol] = {'qty': total_qty, 'avg_price': avg_price}

    def get_portfolio_df(self) -> pd.DataFrame:
        """Return a DataFrame summarizing current portfolio allocations.

        Columns: Symbol, Qty, AvgPrice, CurrentPrice, Change%, StopLossPrice, StopLossChange%, TakeProfitPrice, TakeProfitChange%, Allocation%
        """
        rows = []
        # compute equity snapshot for allocation percent
        try:
            equity_now = self.equity()
        except Exception:
            equity_now = self.cash

        for sym, pos in self.positions.items():
            qty = float(pos.get('qty', 0))
            avg_price = float(pos.get('avg_price', 0) or 0)
            try:
                current = float(get_price(sym))
            except Exception:
                # try quick OHLCV fetch
                try:
                    end = datetime.now()
                    start = end - timedelta(minutes=5)
                    df = fetch_data(sym, '1m', start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
                    current = float(df['Close'].iloc[-1])
                except Exception:
                    current = avg_price

            change_pct = ((current - avg_price) / avg_price * 100) if avg_price > 0 else 0.0
            stop_price = avg_price * (1 - self.config.stop_loss_pct) if avg_price > 0 else 0.0
            stop_change_pct = ((stop_price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0
            tp_price = avg_price * (1 + self.config.take_profit_pct) if avg_price > 0 else 0.0
            tp_change_pct = ((tp_price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0
            value = qty * current
            alloc_pct = (value / equity_now * 100) if equity_now > 0 else 0.0

            rows.append({
                'Symbol': sym,
                'Qty': qty,
                'AvgPrice': avg_price,
                'CurrentPrice': current,
                'Change%': change_pct,
                'StopLossPrice': stop_price,
                'StopLossChange%': stop_change_pct,
                'TakeProfitPrice': tp_price,
                'TakeProfitChange%': tp_change_pct,
                'Allocation%': alloc_pct,
            })

        if not rows:
            return pd.DataFrame(columns=['Symbol','Qty','AvgPrice','CurrentPrice','Change%','StopLossPrice','StopLossChange%','TakeProfitPrice','TakeProfitChange%','Allocation%'])

        df = pd.DataFrame(rows)
        # sort by allocation descending
        df = df.sort_values('Allocation%', ascending=False).reset_index(drop=True)
        return df

    def print_portfolio(self):
        df = self.get_portfolio_df()
        if df.empty:
            logger.info(f"Portfolio: Cash={self.cash:.2f}, no positions")
            return
        # nice formatting
        with pd.option_context('display.float_format', '{:,.4f}'.format):
            logger.info(f"Portfolio snapshot: Cash={self.cash:.2f}, Equity={self.equity():.2f}")
            print(df.to_string(index=False))


class Strategy:
    """Base class for strategies. Override methods to implement logic."""

    def __init__(self, config: Config, pm: PortfolioManager, om: OrderManager):
        self.config = config
        self.pm = pm
        self.om = om

    def score_symbols(self) -> pd.DataFrame:
        """Return a DataFrame with columns ['Coin','Score','Price'] sorted by Score desc.

        Default: use get_24h_change as placeholder.
        """
        # Compute momentum using a long window (config.trading_window_days) and short window (1d)
        long_window = f"{self.config.trading_window_days}d"
        long_df = get_x_change(long_window)
        short_df = get_24h_change()

        # long_df columns: ['Coin', 'Change %', 'Price']
        long_df = long_df.rename(columns={ 'Change %': 'LongChange' })
        short_df = short_df.rename(columns={ 'Change %': 'ShortChange' })

        merged = pd.merge(long_df, short_df[['Coin','ShortChange']], on='Coin', how='left')
        merged['ShortChange'] = merged['ShortChange'].fillna(0.0)

        # Combined score: weight longer-term momentum higher
        merged['Score'] = merged['LongChange'].astype(float) * 0.7 + merged['ShortChange'].astype(float) * 0.3
        merged = merged.sort_values('Score', ascending=False).reset_index(drop=True)
        print(merged)
        return merged[['Coin', 'Score']]

    def generate_entry_signals(self, scored: pd.DataFrame) -> List[str]:

        ############## NEEDS WORK ###############
        """Return list of candidate coins (no sizing).

        The strategy selects the top `max_positions` coins by Score and returns
        their coin symbols (e.g., 'BTC', 'ETH'). Runner will fetch OHLCV and
        compute sizing per coin.
        """
        top = scored.head(self.config.max_positions)
        if top.empty:
            return []
        return [row['Coin'] for _, row in top.iterrows()]

    def generate_exit_signals(self) -> List[str]:
        """Return list of symbol strings to exit. Override to implement exits."""
        return []


class Runner:
    def __init__(self, config: Config):
        self.config = config
        self.om = OrderManager(config)
        self.pm = PortfolioManager(config)
        self.strategy = Strategy(config, self.pm, self.om)
        # tracking for conservative entry rules
        self.last_trade_times: Dict[str, float] = {}
        self.trades_executed_in_cycle: int = 0
        self.trades_executed_timestamps: List[float] = []
        # scheduler state
        self.latest_prices: Dict[str, float] = {}
        # last run timestamps for scheduled tasks
        now_ts = time.time()
        self.last_run: Dict[str, float] = {
            'price_refresh': now_ts - self.config.price_refresh_seconds,
            'scan': now_ts - self.config.scan_seconds,
            'rebalance': now_ts - self.config.rebalance_seconds,
            'manage_positions': now_ts - self.config.manage_positions_seconds,
            'sync_balances': now_ts - self.config.sync_balances_seconds,
            'persist': now_ts - self.config.persist_seconds,
            'heartbeat': now_ts - self.config.heartbeat_seconds,
        }
        self.intervals: Dict[str, int] = {
            'price_refresh': self.config.price_refresh_seconds,
            'scan': self.config.scan_seconds,
            'rebalance': self.config.rebalance_seconds,
            'manage_positions': self.config.manage_positions_seconds,
            'sync_balances': self.config.sync_balances_seconds,
            'persist': self.config.persist_seconds,
            'heartbeat': self.config.heartbeat_seconds,
        }
        self.last_score: Optional[pd.DataFrame] = None

    def rebalance(self, end_dt: datetime):
        """Conservative, entry-only allocation flow.

        This version enforces:
        - entry-only behavior (no forced sells)
        - per-symbol cooldowns
        - max trades per cycle and per hour caps
        - minimum volume filter (if available)
        - expected-edge threshold (cost-aware)
        - dry-run simulation when enabled
        """
        logger.info("Scanning and scoring symbols (conservative entry-only)...")
        scored = self.strategy.score_symbols()

        candidates = self.strategy.generate_entry_signals(scored)
        candidates = candidates[: self.config.top_k]
        logger.info(f"Planned candidates: {candidates}")

        if not candidates:
            return

        equity = self.pm.equity()
        per_trade_risk = equity * self.config.risk_per_trade
        stop_pct = self.config.stop_loss_pct

        total_available_cash = max(0.0, self.pm.cash - self.config.min_cash_reserve)
        if total_available_cash <= 0:
            logger.info("No available cash to allocate after reserve")
            return
        if total_available_cash < self.config.min_cash_per_trade:
            logger.info(f"Available cash {total_available_cash:.2f} below min_cash_per_trade {self.config.min_cash_per_trade}")
            return

        # compute required edge in percent (Score uses percent units)
        round_trip_cost_frac = 2 * self.config.commission + self.config.estimated_slippage_pct
        required_edge_frac = round_trip_cost_frac + self.config.expected_edge_buffer_pct
        required_edge_pct = required_edge_frac * 100.0

        # reset cycle counter
        self.trades_executed_in_cycle = 0
        now_ts = time.time()

        for coin in candidates:
            if self.trades_executed_in_cycle >= self.config.max_trades_per_cycle:
                logger.info("Reached max trades for this cycle")
                break

            symbol = coin + 'USDT'

            # entry-only: skip if already held
            if symbol in self.pm.positions:
                logger.debug(f"Skipping {symbol}: already in portfolio (entry-only)")
                continue

            # cooldown
            last_ts = self.last_trade_times.get(symbol, 0)
            if now_ts - last_ts < (self.config.symbol_cooldown_minutes * 60):
                logger.debug(f"Skipping {symbol}: in cooldown")
                continue

            # optional volume filter (try to read from scanner)
            try:
                vol_df = get_24h_change()
                vol_row = vol_df[vol_df['Coin'] == coin]
                if not vol_row.empty and 'Volume' in vol_row.columns:
                    vol = float(vol_row['Volume'].iloc[0])
                    if self.config.min_volume > 0 and vol < self.config.min_volume:
                        logger.info(f"Skipping {symbol}: volume {vol:.0f} below min_volume {self.config.min_volume}")
                        continue
            except Exception:
                logger.debug(f"Volume check skipped for {symbol}")

            # score check (score is percent)
            try:
                score_row = scored[scored['Coin'] == coin]
                score_val = float(score_row['Score'].iloc[0]) if not score_row.empty else 0.0
            except Exception:
                score_val = 0.0

            # allow forced trades in test mode
            if (not self.config.test_force_trades) and (score_val < required_edge_pct):
                logger.info(f"Skipping {symbol}: score {score_val:.3f}% below required edge {required_edge_pct:.3f}%")
                continue

            # price lookup
            try:
                price = float(get_price(symbol))
            except Exception:
                price = None
            if price is None or price <= 0:
                try:
                    end = datetime.now()
                    start = end - timedelta(minutes=60)
                    ohlcv = fetch_data(symbol, '1m', start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
                    price = float(ohlcv['Close'].iloc[-1])
                except Exception:
                    logger.info(f"Skipping {symbol}: could not fetch price")
                    continue

            # uptrend confirmation
                # validate uptrend (can be bypassed in test mode)
                try:
                    if not getattr(self.config, 'test_ignore_trend', False):
                        if not self._is_uptrend(symbol, hours=self.config.trend_hours, intervals=self.config.trend_intervals):
                            logger.info(f"Skipping {symbol}: not in uptrend across {self.config.trend_intervals}")
                            continue
                except Exception:
                    logger.debug(f"Trend check exception for {symbol}, continuing")

            # sizing
            this_trade_risk = per_trade_risk
            cost_for_trade = this_trade_risk / stop_pct if stop_pct > 0 else total_available_cash
            if cost_for_trade > total_available_cash:
                if total_available_cash <= 0:
                    logger.info("No remaining cash to allocate")
                    break
                this_trade_risk = total_available_cash * stop_pct
                cost_for_trade = total_available_cash

            qty = this_trade_risk / (price * stop_pct) if stop_pct > 0 else total_available_cash / price
            max_qty_by_cash = total_available_cash / price if price > 0 else 0
            qty = min(qty, max_qty_by_cash)

            # rounding
            try:
                step = LOT_STEP_INFO[symbol]['step_size']
                min_qty = LOT_STEP_INFO[symbol]['min_qty']
            except Exception:
                step, min_qty = (0.001, 0.001)

            try:
                getcontext().prec = 28
                qty_dec = Decimal(str(qty))
                step_dec = Decimal(str(step))
                n_steps = (qty_dec / step_dec).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
                qty_rounded = (n_steps * step_dec).normalize()
                if qty_rounded == 0:
                    if total_available_cash >= min_qty * price:
                        qty_rounded = Decimal(str(min_qty))
                    else:
                        logger.info(f"Rounded qty for {symbol} is zero and insufficient cash for min_qty, skipping")
                        continue
                qty_rounded = float(qty_rounded)
            except Exception:
                qty_rounded = round(qty, 3)
                if qty_rounded <= 0:
                    logger.info(f"Rounded qty for {symbol} is zero, skipping")
                    continue

            estimated_cost = qty_rounded * price * (1 + self.config.commission)
            if estimated_cost > total_available_cash:
                logger.info(f"Not enough cash for {symbol}: estimated cost {estimated_cost:.2f} > available {total_available_cash:.2f}")
                continue

            # place or simulate order
            if self.config.dry_run:
                logger.info(f"DRY-RUN: Simulating BUY {qty_rounded} {symbol} @ {price:.6f} (est cost {estimated_cost:.2f})")
                try:
                    self.pm.update_after_fill(symbol, 'BUY', float(qty_rounded), float(price))
                    self.last_trade_times[symbol] = now_ts
                    self.trades_executed_in_cycle += 1
                    self.trades_executed_timestamps.append(now_ts)
                    total_available_cash -= estimated_cost
                except Exception:
                    logger.exception("Failed to simulate portfolio update")
                continue

            res = self.om.place_market(symbol, 'BUY', float(qty_rounded))
            if not res.get('ok'):
                logger.info(f"Order failed for {symbol}: {res}")
                continue

            filled_qty = res.get('filled_qty') or res.get('filledQty') or res.get('filled_quantity') or qty_rounded
            fill_price = res.get('fill_price') or res.get('avg_price') or res.get('fillPrice') or price

            try:
                filled_qty = float(filled_qty)
            except Exception:
                filled_qty = qty_rounded
            try:
                fill_price = float(fill_price)
            except Exception:
                fill_price = price

            actual_cost = filled_qty * fill_price
            try:
                self.pm.update_after_fill(symbol, 'BUY', float(filled_qty), float(fill_price))
                total_available_cash -= actual_cost * (1 + self.config.commission)
                self.last_trade_times[symbol] = now_ts
                self.trades_executed_in_cycle += 1
                self.trades_executed_timestamps.append(now_ts)
                logger.info(f"Bought {filled_qty} {symbol} at {fill_price:.6f}, cost={actual_cost:.2f}")
            except Exception:
                logger.exception("Failed to update portfolio after live fill")

    def _is_uptrend(self, symbol: str, hours: int = 6, intervals: Optional[List[str]] = None) -> bool:
        """Multi-timeframe uptrend check.

        For each interval in `intervals` (or the config default), fetch `hours` worth
        of candles and require that (1) percent change over the window > 0 and
        (2) last close > mean(close) for that interval. Returns True only if all
        intervals confirm an uptrend.
        """
        if intervals is None:
            intervals = self.config.trend_intervals

        allowed_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        try:
            for iv in intervals:
                iv_use = iv if iv in allowed_intervals else '5m'
                end = datetime.now()
                start = end - timedelta(hours=hours)
                ohlcv = fetch_data(symbol, iv_use, start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
                closes = ohlcv['Close'].astype(float)
                if len(closes) < 3:
                    return False
                first = closes.iloc[0]
                last = closes.iloc[-1]
                pct = (last - first) / first
                sma = closes.mean()
                if not ((pct > 0) and (last > sma)):
                    return False
            return True
        except Exception:
            return False

    # --- lightweight task helpers used by the scheduler ---
    def fetch_market_prices(self, symbols: Optional[List[str]] = None) -> bool:
        """Refresh latest_prices for a small set of symbols (holdings + provided symbols).

        Returns True on success (best-effort) and False on error.
        """
        try:
            to_fetch = set(self.pm.positions.keys())
            if symbols:
                to_fetch.update(symbols)
            for s in list(to_fetch):
                try:
                    p = float(get_price(s))
                    self.latest_prices[s] = p
                except Exception:
                    # ignore failures for individual symbols
                    continue
            return True
        except Exception:
            logger.debug("fetch_market_prices failed")
            return False

    def manage_positions_light(self):
        """Lightweight position checks using cached latest_prices.

        This does not fetch additional klines. It uses self.latest_prices
        (populated by fetch_market_prices) to compute unrealized PnL and
        log position status. It's safe to call frequently after price refresh.
        """
        try:
            if not self.pm.positions:
                return
            for sym, pos in self.pm.positions.items():
                qty = float(pos.get('qty', 0))
                avg = float(pos.get('avg_price', 0) or 0)
                # prefer cached price
                cur = self.latest_prices.get(sym)
                if cur is None:
                    try:
                        cur = float(get_price(sym))
                    except Exception:
                        cur = avg
                unrealized = (cur - avg) * qty
                logger.info(f"[LIGHT] Position {sym}: qty={qty}, avg={avg:.6f}, cur={cur:.6f}, PnL={unrealized:.2f}")
                # quick TP/SL trigger: if price crosses TP or SL, run deep check and exit
                try:
                    tp_price = avg * (1 + self.config.take_profit_pct)
                    sl_price = avg * (1 - self.config.stop_loss_pct)
                    # if current price reached or exceeded TP, or dropped to or below SL
                    if cur >= tp_price or cur <= sl_price:
                        logger.info(f"[LIGHT] {sym} near TP/SL (cur={cur:.6f}, TP={tp_price:.6f}, SL={sl_price:.6f}), running deep check/exit")
                        # run deep check and exit for this symbol
                        try:
                            self.manage_positions_deep(symbols=[sym])
                        except Exception:
                            logger.exception(f"manage_positions_deep failed for {sym}")
                except Exception:
                    logger.debug(f"Could not evaluate TP/SL for {sym}")
        except Exception:
            logger.debug("manage_positions_light failed")

    def scan_and_score(self) -> Optional[pd.DataFrame]:
        try:
            df = self.strategy.score_symbols()
            self.last_score = df
            return df
        except Exception:
            logger.exception("scan_and_score failed")
            return None

    def sync_balances(self) -> bool:
        try:
            self.pm.refresh_from_exchange()
            return True
        except Exception:
            logger.exception("sync_balances failed")
            return False

    def persist_snapshot(self) -> bool:
        try:
            # lightweight persistence: print portfolio (could write to DB/file)
            self.pm.print_portfolio()
            return True
        except Exception:
            logger.exception("persist_snapshot failed")
            return False

    def send_heartbeat(self) -> None:
        try:
            eq = self.pm.equity()
            cash = self.pm.cash
            pos_count = len(self.pm.positions)
            logger.info(f"HEARTBEAT: Equity={eq:.2f}, Cash={cash:.2f}, Positions={pos_count}, LastScan={('yes' if self.last_score is not None else 'no')}")
        except Exception:
            logger.debug("send_heartbeat failed")

    def manage_positions(self):
        """Check ongoing positions for take-profit/stop-loss triggers. Placeholder."""
        # keep portfolio in sync and print a snapshot
        try:
            self.pm.refresh_from_exchange()
            self.pm.print_portfolio()
        except Exception:
            logger.debug("Could not refresh portfolio from exchange")

        # Query recent OHLCV for each position and compute simple PnL metrics.
        for sym, pos in list(self.pm.positions.items()):
            try:
                end = datetime.now()
                start = end - timedelta(minutes=10)
                df = fetch_data(sym, '1m', start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
                cur_price = float(df['Close'].iloc[-1])
                unrealized = (cur_price - pos['avg_price']) * pos['qty']
                logger.info(f"Position {sym}: qty={pos['qty']}, avg={pos['avg_price']}, cur={cur_price}, PnL={unrealized:.2f}")
                # TODO: implement TP/SL checks here and call OrderManager to exit when triggered
            except Exception:
                logger.debug(f"Could not fetch price for {sym}")

    def manage_positions_deep(self, symbols: Optional[List[str]] = None):
        """Deeper checks for positions. Fetches recent OHLCV and places market SELLs when TP/SL crossed.

        Executes market SELL orders via OrderManager (or simulates in dry-run). Updates portfolio on fills.
        """
        try:
            targets = symbols if symbols else list(self.pm.positions.keys())
            for sym in targets:
                pos = self.pm.positions.get(sym)
                if not pos:
                    continue
                # fetch a short window of 1m candles to check if TP/SL crossed recently
                try:
                    end = datetime.now()
                    start = end - timedelta(minutes=15)
                    df = fetch_data(sym, '1m', start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
                except Exception:
                    logger.debug(f"Could not fetch klines for deep check {sym}, falling back to latest price")
                    try:
                        cur_price = float(get_price(sym))
                    except Exception:
                        continue
                    df = None

                # determine current price and check cross
                if df is not None and not df.empty:
                    last_close = float(df['Close'].iloc[-1])
                    high = float(df['High'].max()) if 'High' in df.columns else last_close
                    low = float(df['Low'].min()) if 'Low' in df.columns else last_close
                    cur_price = last_close
                else:
                    cur_price = float(get_price(sym)) if get_price(sym) else pos['avg_price']
                    high = cur_price
                    low = cur_price

                avg = float(pos.get('avg_price', 0) or 0)
                tp_price = avg * (1 + self.config.take_profit_pct)
                sl_price = avg * (1 - self.config.stop_loss_pct)

                crossed_tp = cur_price >= tp_price or high >= tp_price
                crossed_sl = cur_price <= sl_price or low <= sl_price

                if not (crossed_tp or crossed_sl):
                    logger.debug(f"Deep check: {sym} did not cross TP/SL (cur={cur_price:.6f}, TP={tp_price:.6f}, SL={sl_price:.6f})")
                    continue

                # prepare sell quantity (all position qty)
                qty = float(pos.get('qty', 0))
                if qty <= 0:
                    continue

                # rounding to lot step
                try:
                    step = LOT_STEP_INFO[sym]['step_size']
                    min_qty = LOT_STEP_INFO[sym]['min_qty']
                except Exception:
                    step, min_qty = (0.001, 0.001)

                try:
                    getcontext().prec = 28
                    qty_dec = Decimal(str(qty))
                    step_dec = Decimal(str(step))
                    # round down when selling to avoid attempting > available
                    n_steps = (qty_dec / step_dec).quantize(Decimal('1'), rounding='ROUND_FLOOR')
                    qty_rounded = (n_steps * step_dec).normalize()
                    qty_rounded = float(qty_rounded)
                    if qty_rounded <= 0:
                        logger.info(f"Rounded sell qty for {sym} is zero, skipping")
                        continue
                except Exception:
                    qty_rounded = round(qty, 3)
                    if qty_rounded <= 0:
                        continue

                # place market SELL (Roostoo) — per your note, use market
                logger.info(f"Placing SELL market order for {sym}: qty={qty_rounded} (dry_run={self.config.dry_run})")
                res = self.om.place_market(sym, 'SELL', float(qty_rounded))

                if not res.get('ok'):
                    logger.error(f"Sell order failed for {sym}: {res}")
                    continue

                filled_qty = res.get('filled_qty') or res.get('filledQty') or res.get('filled_quantity') or qty_rounded
                fill_price = res.get('fill_price') or res.get('avg_price') or res.get('fillPrice') or cur_price

                try:
                    filled_qty = float(filled_qty)
                except Exception:
                    filled_qty = qty_rounded
                try:
                    fill_price = float(fill_price)
                except Exception:
                    fill_price = cur_price

                # update portfolio after sell
                try:
                    self.pm.update_after_fill(sym, 'SELL', filled_qty, fill_price)
                    logger.info(f"Sold {filled_qty} {sym} @ {fill_price:.6f}")
                except Exception:
                    logger.exception(f"Failed to update portfolio after selling {sym}")
        except Exception:
            logger.exception("manage_positions_deep failed")

    def run_once(self):
        end_dt = datetime.now()
        # refresh portfolio from exchange and show the user a snapshot before making decisions
        try:
            self.pm.refresh_from_exchange()
            self.pm.print_portfolio()
        except Exception:
            logger.debug("Portfolio refresh/display failed")

        self.rebalance(end_dt)
        self.manage_positions()

    # compatibility alias: execute_entries calls the conservative rebalance flow
    def execute_entries(self, end_dt: datetime):
        return self.rebalance(end_dt)

    def run_loop(self, interval_seconds: int = 60 * 60):
        """Run a lightweight scheduled loop using configured intervals.

        This scheduler wakes every `tick_interval_seconds` and dispatches
        tasks whose interval has elapsed. Each task is best-effort and
        failures are logged; a simple jitter is applied to avoid strict
        synchronization across runs.
        """
        tick = getattr(self.config, 'tick_interval_seconds', 15)
        logger.info(f"Starting scheduler loop (tick={tick}s)")
        try:
            while True:
                now = time.time()

                # price refresh
                if now - self.last_run.get('price_refresh', 0) >= self.intervals['price_refresh']:
                    try:
                        self.fetch_market_prices()
                        # immediately run lightweight position checks using cached prices
                        try:
                            self.manage_positions_light()
                        except Exception:
                            logger.debug("manage_positions_light failed after price refresh")
                        self.last_run['price_refresh'] = now + random.uniform(-0.1, 0.1) * self.intervals['price_refresh']
                    except Exception:
                        logger.debug("price_refresh task failed")

                # scan and score
                if now - self.last_run.get('scan', 0) >= self.intervals['scan']:
                    try:
                        self.scan_and_score()
                        self.last_run['scan'] = now + random.uniform(-0.05, 0.05) * self.intervals['scan']
                    except Exception:
                        logger.debug("scan task failed")

                # rebalance/execute entries
                if now - self.last_run.get('rebalance', 0) >= self.intervals['rebalance']:
                    try:
                        # call conservative entry-only flow
                        self.execute_entries(datetime.now())
                        self.last_run['rebalance'] = now + random.uniform(-0.05, 0.05) * self.intervals['rebalance']
                    except Exception:
                        logger.exception("rebalance task failed")

                # manage positions (TP/SL checks)
                if now - self.last_run.get('manage_positions', 0) >= self.intervals['manage_positions']:
                    try:
                        self.manage_positions()
                        self.last_run['manage_positions'] = now + random.uniform(-0.05, 0.05) * self.intervals['manage_positions']
                    except Exception:
                        logger.debug("manage_positions task failed")

                # sync balances
                if now - self.last_run.get('sync_balances', 0) >= self.intervals['sync_balances']:
                    try:
                        self.sync_balances()
                        self.last_run['sync_balances'] = now + random.uniform(-0.1, 0.1) * self.intervals['sync_balances']
                    except Exception:
                        logger.debug("sync_balances task failed")

                # persist snapshot
                if now - self.last_run.get('persist', 0) >= self.intervals['persist']:
                    try:
                        self.persist_snapshot()
                        self.last_run['persist'] = now + random.uniform(-0.1, 0.1) * self.intervals['persist']
                    except Exception:
                        logger.debug("persist task failed")

                # heartbeat
                if now - self.last_run.get('heartbeat', 0) >= self.intervals['heartbeat']:
                    try:
                        self.send_heartbeat()
                        self.last_run['heartbeat'] = now
                    except Exception:
                        logger.debug("heartbeat failed")

                time.sleep(tick)
        except KeyboardInterrupt:
            logger.info("Stopping scheduler loop")

    def run_dry_cycles(self, cycles: int = 3, sleep_seconds: float = 1.0, force_trades: bool = False):
        """Run `run_once()` for a small number of cycles in dry-run.

        If force_trades is True, set `config.test_force_trades` temporarily so entry checks
        will be allowed even if market score is below required edge. Restores original flag.
        """
        orig = getattr(self.config, 'test_force_trades', False)
        try:
            if force_trades:
                self.config.test_force_trades = True
            for i in range(cycles):
                logger.info(f"DRY-CYCLE {i+1}/{cycles}")
                self.run_once()
                time.sleep(sleep_seconds)
        finally:
            self.config.test_force_trades = orig


if __name__ == '__main__':
    cfg = Config()
    runner = Runner(cfg)
    # quick run
    runner.run_loop()
