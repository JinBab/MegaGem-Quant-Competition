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
from BinanceAPI import fetch_data

# Normal import for the scanner (renamed to a valid module name)
from market_scanner import get_x_change, get_24h_change

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    seed_cash: float = 50_000.0
    commission: float = 0.001  # 0.1% -> 0.001
    trading_window_days: int = 5
    scan_intervals: List[str] = field(default_factory=lambda: ["1d", "12h", "1h"])
    max_positions: int = 10
    risk_per_trade: float = 0.02  # fraction of equity
    min_cash_reserve: float = 1000.0
    stop_loss_pct: float = 0.05  # assumed stop-loss per trade (5%) used for sizing


class OrderManager:
    """Wraps Roostoo order endpoints. Provides synchronous helpers.

    Note: RoostooAPI.place_order currently returns None (prints). We call it
    and assume success for now; replace with proper response parsing when
    the API returns structured data.
    """

    def place_market(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        logger.info(f"Placing market order: {side} {qty} {symbol}")
        # Roostoo expects coin name without USDT, e.g., 'BTC' and pair will be coin/USD
        coin = symbol.replace('USDT', '')
        try:
            res = place_order(coin, side, qty, price=None)
        except Exception as e:
            logger.exception("place_order failed")
            return {"ok": False, "error": str(e)}
        # Try to extract executed quantity and fill price from the API response.
        if isinstance(res, dict):
            # common possible fields in mock APIs
            filled_qty = None
            fill_price = None
            # check multiple possible keys
            for k in ('filledQty', 'filled_quantity', 'executedQty', 'filled_quantity', 'quantity_filled'):
                if k in res:
                    try:
                        filled_qty = float(res[k])
                        break
                    except Exception:
                        pass
            for k in ('avgPrice', 'fillPrice', 'price', 'avg_price'):
                if k in res:
                    try:
                        fill_price = float(res[k])
                        break
                    except Exception:
                        pass

            # If API didn't return executed qty/price (mock), treat as accepted and let caller simulate
            return {"ok": True, "api_response": res, "filled_qty": filled_qty, "fill_price": fill_price}

        # generic success fallback
        return {"ok": True, "symbol": symbol, "side": side, "qty": qty}


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
                self.positions[symbol] = {'qty': qty, 'avg_price': fill_price, 'side': 'LONG'}
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
        self.om = OrderManager()
        self.pm = PortfolioManager(config)
        self.strategy = Strategy(config, self.pm, self.om)

    def rebalance(self, end_dt: datetime):
        """Main orchestration: scan, score, allocate and place orders (placeholder)."""
        logger.info("Scanning and scoring symbols...")
        scored = self.strategy.score_symbols()

        # generate entries (signals: list of coin strings, e.g., 'BTC')
        candidates = self.strategy.generate_entry_signals(scored)
        logger.info(f"Planned candidates: {candidates}")

        for coin in candidates:
            symbol = coin + 'USDT'
            # Fetch recent OHLCV for the candidate to determine price / volatility
            try:
                end = datetime.now()
                start = end - timedelta(minutes=60)
                # use BinanceAPI.fetch_data which returns a DataFrame indexed by Close time
                ohlcv = fetch_data(symbol, '1m', start.strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
                price = float(ohlcv['Close'].iloc[-1])
            except Exception:
                # fallback: try the scanner's last known price if present in scored
                match = scored.loc[scored['Coin'] == coin]
                price = float(match['Price'].iloc[0]) if (not match.empty and 'Price' in match.columns) else None
            if price is None or price <= 0:
                # retry OHLCV with a larger window as a fallback (avoid using Roostoo ticker)
                try:
                    end2 = datetime.now()
                    start2 = end2 - timedelta(hours=2)
                    df2 = fetch_data(symbol, '1m', start2.strftime("%d %b %Y %H:%M:%S"), end2.strftime("%d %b %Y %H:%M:%S"))
                    price = float(df2['Close'].iloc[-1])
                except Exception:
                    price = None
            if price is None or price <= 0:
                logger.info(f"Skipping {symbol}: no price available")
                continue

            # compute sizing using risk-per-trade and stop-loss
            equity = self.pm.equity()
            dollar_risk = equity * self.config.risk_per_trade
            stop_pct = self.config.stop_loss_pct
            qty = (dollar_risk) / (price * stop_pct)

            # Limit by available cash after reserving min_cash_reserve
            max_dollar_alloc = max(0.0, self.pm.cash - self.config.min_cash_reserve)
            if max_dollar_alloc <= 0:
                logger.info("Insufficient free cash to allocate any new positions")
                continue
            max_qty_by_cash = max_dollar_alloc / price
            qty = min(qty, max_qty_by_cash)

            # place market buy via OrderManager (which calls RoostooAPI.place_order)
            res = self.om.place_market(symbol, 'BUY', float(qty))
            if not res.get('ok'):
                logger.info(f"Order failed for {symbol}: {res}")
                continue

            # If API returned executed qty and fill price, use them. Otherwise assume filled at 'price'.
            filled_qty = res.get('filled_qty')
            fill_price = res.get('fill_price')
            if filled_qty is None or fill_price is None:
                # try to read api_response for common keys
                api_resp = res.get('api_response') if isinstance(res.get('api_response'), dict) else {}
                if isinstance(api_resp, dict):
                    def _safe_get_float(d, keys):
                        for k in keys:
                            v = d.get(k)
                            if v is not None:
                                try:
                                    return float(v)
                                except Exception:
                                    continue
                        return None

                    filled_qty = filled_qty or _safe_get_float(api_resp, ('filledQty','executedQty','filled_quantity'))
                    fill_price = fill_price or _safe_get_float(api_resp, ('avgPrice','price','fillPrice'))

            if filled_qty is None:
                filled_qty = qty
            if fill_price is None:
                fill_price = price

            # update portfolio using real/simulated fill
            try:
                self.pm.update_after_fill(symbol, 'BUY', float(filled_qty), float(fill_price))
            except Exception:
                logger.exception("Failed to update portfolio after fill")

    def manage_positions(self):
        """Check ongoing positions for take-profit/stop-loss triggers. Placeholder."""
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
