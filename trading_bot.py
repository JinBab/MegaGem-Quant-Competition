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
    top_k: int = 4  # allocate only to top K scanned coins
    trend_hours: int = 6  # lookback hours to verify uptrend
    trend_intervals: List[str] = field(default_factory=lambda: ['5m', '1h'])  # multi-timeframe intervals to confirm trend
    # When True, simulate orders locally and don't call RoostooAPI.place_order
    dry_run: bool = True
    # Minimum 24h quote volume (USDT) required to consider a symbol for trading
    min_volume: float = 10000.0
    # Default take-profit percent for reporting (10% -> 0.10). Used for display and TP suggestions.
    take_profit_pct: float = 0.10


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

    def rebalance(self, end_dt: datetime):
        """Main orchestration: scan, score, allocate and place orders (placeholder)."""
        logger.info("Scanning and scoring symbols...")
        scored = self.strategy.score_symbols()

        # generate entries (signals: list of coin strings, e.g., 'BTC')
        candidates = self.strategy.generate_entry_signals(scored)
        # only keep top-k candidates to allocate across
        candidates = candidates[: self.config.top_k]
        logger.info(f"Planned candidates: {candidates}")

        # Determine per-trade risk and ensure we don't allocate more cash than available
        M = len(candidates)
        if M == 0:
            return

        equity = self.pm.equity()
        per_trade_risk = equity * self.config.risk_per_trade
        stop_pct = self.config.stop_loss_pct

        # Cash available for new allocations (respecting reserve)
        total_available_cash = max(0.0, self.pm.cash - self.config.min_cash_reserve)
        if total_available_cash <= 0:
            logger.info("No available cash to allocate after reserve")
            return

        # Required cash per trade (approx) = per_trade_risk / stop_pct
        required_per_trade_cash = per_trade_risk / stop_pct if stop_pct > 0 else total_available_cash
        total_required = required_per_trade_cash * M

        if total_required > total_available_cash:
            # scale down per-trade risk proportionally so total fits available cash
            scale = total_available_cash / total_required
            per_trade_risk *= scale
            required_per_trade_cash = per_trade_risk / stop_pct if stop_pct > 0 else total_available_cash
            logger.info(f"Scaling per-trade risk by {scale:.3f} to fit available cash: per_trade_risk={per_trade_risk:.2f}")

        remaining_cash = total_available_cash

        for coin in candidates:
            symbol = coin + 'USDT'
            # Fetch recent OHLCV for the candidate to determine price / volatility
            # try quick price lookup first (market_scanner.get_price), then fallback to OHLCV if needed
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
                    logger.info(f"Skipping {symbol}: could not fetch OHLCV")
                    continue

            # verify short-term uptrend before entering
            try:
                # require multi-timeframe confirmation: all intervals must show uptrend
                if not self._is_uptrend(symbol, hours=self.config.trend_hours, intervals=self.config.trend_intervals):
                    logger.info(f"Skipping {symbol}: not in uptrend across {self.config.trend_intervals} over last {self.config.trend_hours}h")
                    continue
            except Exception:
                logger.debug(f"Trend check failed for {symbol}, continuing with caution")

            if price is None or price <= 0:
                logger.info(f"Skipping {symbol}: invalid price")
                continue

            # Determine this trade's dollar risk (may be the scaled per_trade_risk)
            this_trade_risk = per_trade_risk

            # Cost required to size this trade = this_trade_risk / stop_pct
            cost_for_trade = this_trade_risk / stop_pct if stop_pct > 0 else remaining_cash
            if cost_for_trade > remaining_cash:
                # if not enough remaining cash, adjust trade risk down to use remaining cash proportionally
                if remaining_cash <= 0:
                    logger.info("No remaining cash to allocate")
                    break
                this_trade_risk = remaining_cash * stop_pct
                cost_for_trade = remaining_cash

            qty = this_trade_risk / (price * stop_pct) if stop_pct > 0 else remaining_cash / price

            # cap by available cash
            max_qty_by_cash = remaining_cash / price if price > 0 else 0
            qty = min(qty, max_qty_by_cash)

            # per-symbol lot step rounding: fetch allowed step and minQty
            step, min_qty = (0.001, 0.001)
            step, min_qty = LOT_STEP_INFO[symbol]['step_size'], LOT_STEP_INFO[symbol]['min_qty']

            # Round qty to nearest allowed step using Decimal for stability
            try:
                getcontext().prec = 28
                qty_dec = Decimal(str(qty))
                step_dec = Decimal(str(step))
                # number of steps (rounded to nearest integer)
                n_steps = (qty_dec / step_dec).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
                qty_rounded = (n_steps * step_dec).normalize()
                # if rounding produced zero, try to bump to min_qty if affordable
                if qty_rounded == 0:
                    if remaining_cash >= min_qty * price:
                        qty_rounded = Decimal(str(min_qty))
                    else:
                        logger.info(f"Rounded qty for {symbol} is zero and insufficient cash for min_qty, skipping")
                        continue
                qty_rounded = float(qty_rounded)
            except Exception:
                # fallback: cap to 3 decimals
                qty_rounded = round(qty, 3)
                if qty_rounded <= 0:
                    logger.info(f"Rounded qty for {symbol} is zero, skipping")
                    continue

            # place market buy via OrderManager (which calls RoostooAPI.place_order)
            res = self.om.place_market(symbol, 'BUY', float(qty_rounded))
            if not res.get('ok'):
                logger.info(f"Order failed for {symbol}: {res}")
                continue
            # If API returned executed qty and fill price, use them. Otherwise assume filled at rounded qty and price.
            filled_qty = res.get('filled_qty') or res.get('filledQty') or res.get('filled_quantity')
            fill_price = res.get('fill_price') or res.get('avg_price') or res.get('fillPrice')

            if filled_qty is None:
                # fallback to the rounded qty we sent
                filled_qty = qty_rounded
            else:
                try:
                    filled_qty = float(filled_qty)
                except Exception:
                    filled_qty = qty_rounded

            if fill_price is None:
                fill_price = price
            else:
                try:
                    fill_price = float(fill_price)
                except Exception:
                    fill_price = price

            # actual dollar risk after rounding
            actual_cost = float(filled_qty) * float(fill_price)
            actual_dollar_risk = actual_cost * stop_pct

            # update portfolio using real/simulated fill
            try:
                self.pm.update_after_fill(symbol, 'BUY', float(filled_qty), float(fill_price))
                # decrement remaining cash by the actual cost used
                remaining_cash -= actual_cost
                logger.info(f"Bought {filled_qty} {symbol} at {fill_price:.6f}, cost={actual_cost:.2f}, risk={actual_dollar_risk:.2f}")
            except Exception:
                logger.exception("Failed to update portfolio after fill")

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

    def run_loop(self, interval_seconds: int = 60 * 60):
        """Run the bot in a loop (blocking)."""
        try:
            while True:
                logger.info("Starting run cycle")
                self.run_once()
                logger.info(f"Sleeping {interval_seconds} seconds")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stopping run loop")


if __name__ == '__main__':
    cfg = Config()
    runner = Runner(cfg)
    # quick run
    runner.run_once()
