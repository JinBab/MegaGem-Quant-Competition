"""Structured trading bot skeleton

This module provides a single `TradingBot` class with clear method
stubs for a periodic market scan, main strategy, order management,
market buy/sell operations, and portfolio status reporting.

Fill in the strategy and API integration points. The class is written
to be run on scheduled loops: scan less frequently, strategy moderately,
and order_management frequently.
"""
import requests
import numpy as np
import pandas as pd
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict,List, Optional, Any
from RoostooAPI import place_order, get_balance, get_ticker, sell_all  # Roostoo exchange API for placing order, getting portfolio balance, and getting all tickers
from BinanceAPI import fetch_data as fetch_OHLCV # fetches OHLCV data from Binance
from BinanceAPI import LOT_STEP_INFO  # Roostoo exchange API for lot size and step size info
from decimal import Decimal, ROUND_DOWN
from market_scanner import get_custom_change, SYMBOLS  # market scanning utility and ALL symbols list

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class Trade:
    """Represents a trade entity tracked by the bot.

    Fields are intentionally minimal — extend as needed.
    """
    symbol: str
    size: float = 0.0
    entry_price: Optional[float] = None  # price to enter (for pending orders)
    filled_price: Optional[float] = None  # actual executed price
    filled_quantity: Optional[float] = None  # actual executed quantity
    tp: Optional[float] = None
    sl: Optional[float] = None
    status: str = "pending"  # pending, open
    quantity: Optional[float] = None  # rounded/executed quantity


class TradingBot:
    """Single class encapsulating the trading bot.

    Contract (high level):
    - Inputs: exchange API client (Roostoo), configuration dictionary
    - Outputs: orders submitted via API, persistent state kept in-memory
    - Error modes: network/API errors should be caught by caller or retried

    Typical usage: instantiate, then run scheduled loops that call
    `periodic_scan`, `main_strategy`, and `order_management` at different
    cadences. Order management must run the most frequently.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # In-memory state
        self.pending_queue: List[Trade] = []  # trades waiting for entry execution
        self.trades: Dict[str, Trade] = {}  # all tracked trades by trade_id
        self.positions: Dict[str, Trade] = {}  # currently open positions keyed by symbol or trade_id

        # Tunables / cadence (seconds)
        self.scan_interval = self.config.get("scan_interval", 60*60*12)  # e.g., 12 hour
        self.strategy_interval = self.config.get("strategy_interval", 60)  # e.g., 1 minute
        self.order_mgmt_interval = self.config.get("order_mgmt_interval", 5)  # e.g., 5 seconds

        self.last_scan_df : pd.DataFrame = pd.DataFrame()
        self.periodic_scan()

        self.portfolio_value : float = 0.0
        self.available_cash : float = 0.0

        self.last_selected_coins = []
        self.update_portfolio_value()
        print( "VAL: ",self.portfolio_value)
        print( "AVA",self.available_cash)

        logger.info("TradingBot created with intervals: scan=%s strategy=%s order_mgmt=%s",
                    self.scan_interval, self.strategy_interval, self.order_mgmt_interval)

    # --------------------------- Required Loops ---------------------------
    def periodic_scan(self) -> None:
        """Scan the market and return ranked momentum (or similar) data.

        Returns a list of dicts, each containing at minimum:
        - symbol: str
        - score: float (ranking / momentum)
        - other metrics for strategy

        This function should run relatively infrequently (e.g., every 5m).
        """
        logger.debug("periodic_scan: start")
        # TODO: Use market_scanner.py's custom scanning functions
        # Get percent change data over last 5h and 24h to see momentum. Higher weight for recent changes. Rank based on momentum
        windows = ['1d', '6h']
        weights = [0.2, 0.4]

        dfs = []
        for w in windows:
            try:
                dfw = get_custom_change(w, datetime.now())
                # normalize column name
                dfw = dfw.rename(columns={'Change %': f'Change_{w}'})
                dfs.append(dfw[['Coin', f'Change_{w}']])
            except Exception:
                dfs.append(pd.DataFrame(columns=['Coin', f'Change_{w}']))

        # merge all results
        merged = dfs[0]
        for dfw in dfs[1:]:
            merged = pd.merge(merged, dfw, on='Coin', how='outer')

        # ensure numeric columns exist and fill missing
        for w in windows:
            col = f'Change_{w}'
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].astype(float).fillna(0.0)

        # normalize weights
        try:
            wsum = sum(weights)
            norm_weights = [float(w) / wsum for w in weights]
        except Exception:
            norm_weights = [0.2, 0.3, 0.5]

        merged['Score'] = 0.0
        for i, w in enumerate(windows):
            merged['Score'] += merged[f'Change_{w}'].astype(float) * norm_weights[i]

        merged = merged.sort_values('Score', ascending=False).reset_index(drop=True)
        self.last_scan_df =  merged[['Coin', 'Score']]
        
    def main_strategy(self):
        """Main strategy executed on schedule.

        - Uses `ranked_data` from periodic_scan
        - Pulls OHLCV for top coins
        - Decides whether to buy, size, TP/SL
        - If entry price != market price, enqueue a Trade in `pending_queue`
        - If entry should be immediate, call buy_market() or create immediate order
        """
        logger.debug("main_strategy: received %d ranked symbols", len(self.last_scan_df))

        #clear pending queue to avoid duplicates
        self.pending_queue = []

        # pick top 6 coins to consider
        top_n = 8
        selected_coins = self.last_scan_df[:top_n]
        # turn selected coins into a list of coins
        selected_coin_names = selected_coins['Coin'].tolist()
        self.last_selected_coins = selected_coin_names
        # Fetch OHLCV data for selected coins
        for item in selected_coin_names:
            print(f"Evaluating {item} for entry")
            ohlcv_1h = fetch_OHLCV(f"{item}USDT", '1h', start=datetime.now() - timedelta(hours=24), end=datetime.now())
            if (ohlcv_1h is None or ohlcv_1h.empty):
                logger.debug("main_strategy: no OHLCV data for %s, skipping", item)
                continue
            entry_price = ohlcv_1h['Close'].max()
            # determine entry price from market; skip symbol if price not available
            # ### TESTING PURPOSE ####
            # entry_price = None
            # price_map = self.get_all_market_prices()
            # price = self.get_price_for_symbol(price_map, f"{item}USDT")
            # if price is None or price == 0:
            #     # skip this symbol if we cannot obtain a valid market price
            #     logger.debug("main_strategy: no valid price for %s, skipping", item)
            #     continue
            # entry_price = float(price)

            ### TEST CODE END ####

            # find volatility
            ohlcv_1h['log_return'] = np.log(ohlcv_1h['Close'] / ohlcv_1h['Close'].shift(1))

            # Drop NaN from the first shift
            ohlcv_1h = ohlcv_1h.dropna()

            # Calculate volatility (standard deviation of returns)
            daily_volatility = ohlcv_1h['log_return'].std()

            # Define position size and stop loss based on volatility (example: inverse volatility scaling)
            stop_loss_pct = daily_volatility * 1.5 if daily_volatility else 0.05

            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + 4*stop_loss_pct)

            # #### TESTING PURPOSE #####
            # take_profit = entry_price * 1.001

            # given maximum portfolio loss of 1% per trade, calculate position size
            max_portfolio_loss_pct = 0.01
            position_size = (max_portfolio_loss_pct) / stop_loss_pct
    
            print("aaaaa")
            if (entry_price > 0.001 and position_size < 0.3 and position_size > 0.1): 
                # Determine decimal precision from LOT_STEP_INFO for this symbol and round prices accordingly
                # try:
                #     symbol_key = f"{item}USDT".replace('/', '').replace('-', '').upper()
                #     info = LOT_STEP_INFO.get(symbol_key, {})
                #     step = float(info.get('step_size', 0.01))
                #     exp = int(Decimal(str(step)).as_tuple().exponent)
                #     decimals = max(0, -exp)
                # except Exception:
                #     decimals = 8
                # Round prices to the determined precision
                # try:
                #     entry_price = round(float(entry_price), decimals)
                #     take_profit = round(float(take_profit), decimals)
                #     stop_loss = round(float(stop_loss), decimals)
                # except Exception:
                #     pass
                print("bbbb")
                if entry_price < 0.001:
                    continue

                # Create a trade object and add to pending queue. Quantity will be calculated and rounded at execution
                trade = Trade(symbol=item, status='pending', entry_price=entry_price, size=position_size, tp=take_profit, sl=stop_loss)

                if item in self.positions.keys():
                    continue 
                print("AHFLJ")
                self.add_pending_trade(trade)

                print(f"Added pending trade: {trade}, SL%: {stop_loss_pct}, Size: {position_size}, Entry: {entry_price}, TP: {take_profit}, SL: {stop_loss}")

    # --------------------------- Order Management ---------------------------
    def order_management(self):
        """Frequent loop: check pending orders and open positions and act.

        - Retrieves current market prices (call to the API)
        - For pending trades: check if entry price has been reached -> call buy_market
        - For open positions: check TP/SL -> call sell_market

        This should run more frequently than the strategy.
        """
        logger.debug("order_management: running")
        # 1) get all tickers/prices from Roostoo
        market_prices = self.get_all_market_prices()
        print("Pending Queue: ")
        print(self.pending_queue)
        print("Open Positions: ")
        print(self.positions)
        print("Selected Coins: ")
        print(self.last_selected_coins)


        # 2) handle pending queue (copy and clear for processing)
        
        # clear the pending queue (we're single-threaded so this is safe)


        # fix entry condition to be breakout (coming up towards the entry price, not just when it is below the entry price)
        for trade in self.pending_queue:
            logger.debug("Checking pending trade: %s", trade)
            price = self.get_price_for_symbol(market_prices, f"{trade.symbol}USDT")
            
            if price is None:
                continue
            # If entry_price is None or close to current market, execute market buy
            if trade.entry_price is None or self._entry_reached(price, trade.entry_price):
                logger.info("order_management: executing buy for %s at market %s", trade.symbol, price)
                self.buy_market(trade)
                self.pending_queue.remove(trade)

        # 3) check open positions for TP/SL
        current_positions = list(self.positions.values())
        for pos in current_positions:
            logger.debug("order_management: checking position %s", pos.symbol)
            price = self.get_price_for_symbol(market_prices, f"{pos.symbol}USDT")
            if price is None:
                continue
            if pos.tp and price >= pos.tp:
                # calculate profit
                profit = (price - pos.entry_price) * pos.filled_quantity if pos.entry_price and pos.filled_quantity else 0.0
                logger.info("order_management: TP hit for %s. Profit:%s", pos.symbol, profit)
                try: self.sell_market(pos, reason="tp")
                except Exception as e: logger.error("order_management: error selling %s: %s", pos.symbol, e)

                del self.positions[pos.symbol]
                
            elif pos.sl and price <= pos.sl:
                # calculate loss
                loss = (pos.entry_price - price) * pos.filled_quantity if pos.entry_price and pos.filled_quantity else 0.0
                logger.info("order_management: SL loss for %s. Loss: %s", pos.symbol, loss)
          
                try: self.sell_market(pos, reason="sl")
                except Exception as e: logger.error("order_management: error selling %s: %s", pos.symbol, e)
            
                del self.positions[pos.symbol]

    # --------------------------- Market Execution ---------------------------
    def buy_market(self, trade: Trade):
        """Execute a market buy via the Roostoo API and update state.

        This function should only submit market orders. After a successful fill,
        create/update a Trade object representing the open position and store it
        in `self.positions` for ongoing checking.
        """

        # check if enough cash / buying power is available
        if trade.size * self.portfolio_value > self.available_cash:
            logger.warning("buy_market: insufficient funds for %s size %s", trade.symbol, trade.size)
            return
        value = trade.size * self.portfolio_value
        if not trade.entry_price:
            return

        # raw quantity before exchange rounding
        raw_quantity = value / trade.entry_price

        # determine lot step and min qty for this symbol
        symbol_key = f"{trade.symbol}USDT".replace('/', '').replace('-', '').upper()
        info = LOT_STEP_INFO.get(symbol_key, {})
        step = float(info.get('step_size', 0.001))
        min_qty = float(info.get('min_qty', 0.001))

        # determine decimals from step (e.g., step=0.001 -> decimals=3)
        # step = 1.0? 
        try:
            exp = int(Decimal(str(step)).as_tuple().exponent)
            decimals = max(0, -exp)
        except Exception:
            decimals = 2

        # truncate quantity down to allowed decimals (round down)
        try:
            if decimals > 0:
                quant = Decimal(1).scaleb(-decimals)
            else:
                quant = Decimal(1)
            rounded_qty_dec = Decimal(str(raw_quantity)).quantize(quant, rounding=ROUND_DOWN)
            rounded_qty = float(rounded_qty_dec)
        except Exception:
            rounded_qty = float(raw_quantity)

        if rounded_qty < min_qty or rounded_qty <= 0:
            logger.warning("buy_market: rounded quantity %s for %s below min %s; skipping order", rounded_qty, trade.symbol, min_qty)
            return

        # store rounded quantity on the trade for bookkeeping
        trade.quantity = rounded_qty

        logger.info("buy_market: %s %s @ %s (qty=%s)", trade.symbol, trade.size, trade.entry_price, rounded_qty)
        
        resp = place_order(trade.symbol, "BUY", rounded_qty)
        if resp and resp.get('Success'):
            try:
                trade.filled_price = resp["OrderDetail"].get("FilledAverPrice")
                trade.filled_quantity = resp["OrderDetail"].get("FilledQuantity")
            except Exception:
                trade.filled_quantity = rounded_qty
        else:
            trade.filled_quantity = rounded_qty
        trade.status = "open"

        self.update_portfolio_value()

        # store trade and open position (single-threaded)
        self.positions[trade.symbol] = trade

        logger.debug("buy_market: trade stored %s", trade.symbol)

    def sell_market(self, trade: Trade, reason: Optional[str] = None):
        """Execute market sell via Roostoo API and update portfolio/trade state.

        Called when TP/SL or manual close is required.
        """
        logger.info("sell_market: closing %s reason=%s", trade.symbol, reason)

        # TODO: call the real api: self.api.create_market_order(symbol=..., side='sell', size=...)
        # On success, update trade.status='closed', filled_price, filled_at, and remove from positions
        bal = get_balance()["SpotWallet"]
        free_qty = float(bal[trade.symbol]['Free'])
        resp = place_order(trade.symbol, "SELL", free_qty)
        # if resp['Success']:
        #     sold_price = resp["OrderDetail"].get("FilledAverPrice", 0.0) 
        #     print(f"Sold price: {sold_price}")
        #     trade.status = "closed"
        # else:
        #     place_order(trade.symbol, "SELL", free_qty)
        
        self.update_portfolio_value()
        logger.debug("sell_market: trade closed %s", trade.symbol)

    # --------------------------- Portfolio / Utilities ---------------------------
    def update_portfolio_value(self):
        balance = get_balance()
        if balance['Success']:
            pricemap = self.get_all_market_prices()
            try:
                self.available_cash = float(balance['SpotWallet']['USD'].get('Free', 0))
            except Exception:
                # fallback if structure differs
                try:
                    self.available_cash = float(balance['SpotWallet']['USD'])
                except Exception:
                    self.available_cash = 0.0
            self.portfolio_value = float(self.available_cash)
            # loop through all other coins and sum up their USD value
            for coin, asset in balance['SpotWallet'].items():
                if coin != 'USD':
                    # try to parse free amount
                    try:
                        free_amt = float(asset.get('Free', 0))
                    except Exception:
                        try:
                            free_amt = float(asset)
                        except Exception:
                            free_amt = 0.0

                    if free_amt == 0:
                        continue

                    price = self.get_price_for_symbol(pricemap, f"{coin}USDT")
                    if price is None:
                        continue
                    # price is returned as float by get_price_for_symbol
                    self.portfolio_value += float(price) * free_amt

    def portfolio_status(self) -> Dict[str, Any]:
        """Return and log a human-readable snapshot of current portfolio and pending orders.

        Should include: open positions, pending orders, net value (if API provides balances),
        and other metrics.
        """


        # single-threaded snapshot of positions and pending orders
        positions = list(self.positions.values())
        pending = list(self.pending_queue)

        # Fetch balances if supported
        balances = None
        try:
            # prefer the module-level get_balance if available
            if 'get_balance' in globals():
                balances = get_balance()
        except Exception as e:
            logger.debug("portfolio_status: could not fetch balances: %s", e)

        snapshot = {
            "positions": positions,
            "pending": pending,
            "balances": balances,
            "timestamp": datetime.now(),
        }

        # Print/log in an easy to read format
        logger.info("Portfolio snapshot: %d positions, %d pending", len(positions), len(pending))
        for p in positions:
            logger.info("POS %s: size=%s entry=%s tp=%s sl=%s status=%s", p.symbol, p.size, p.filled_price, p.tp, p.sl, p.status)

        return snapshot

    # --------------------------- Helper Methods ---------------------------
    def add_pending_trade(self, trade: Trade):
        """Enqueue a pending trade for order management to monitor and execute later."""
        logger.debug("add_pending_trade: %s", trade)
        self.pending_queue.append(trade)

    def get_all_market_prices(self) -> pd.DataFrame:
        response = requests.get("https://api.binance.com/api/v3/ticker/price")
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            # get only symbols in our trading list
            df = df[df['symbol'].isin(SYMBOLS)]
            df.reset_index(drop=True, inplace=True)
            return df
        return pd.DataFrame()

    def get_price_for_symbol(self, price_map: pd.DataFrame, symbol: str) -> Optional[float]:
        try:
            series = price_map.loc[price_map['symbol'] == symbol, 'price']
            if series.empty:
                return None
            price = series.iloc[0]
            return float(price)
        except Exception:
            return None

    def _entry_reached(self, market_price: float, entry_price: float) -> bool:
        """Return True when market price meets or passes the entry price.

        `tol` is a small tolerance to account for tiny differences.
        """

        return market_price >= (entry_price)

    def run_loop(self):
        """Run the main scheduling loop (single-threaded).

        This will repeatedly call:
        - periodic_scan at `self.scan_interval`
        - main_strategy at `self.strategy_interval`
        - order_management at `self.order_mgmt_interval`

        The implementation is intentionally simple and single-threaded.
        Use Ctrl-C to stop.
        """
        logger.info("run_loop: starting main loop (scan=%s strategy=%s order_mgmt=%s)",
                    self.scan_interval, self.strategy_interval, self.order_mgmt_interval)
        last_scan = 0.0
        last_strategy = 0.0
        last_order = 0.0

        try:
            while True:
                now = time.time()

                # periodic scan
                if now - last_scan >= self.scan_interval:
                    try:
                        logger.debug("run_loop: running periodic_scan")
                        self.periodic_scan()
                    except Exception as e:
                        logger.exception("run_loop: periodic_scan error: %s", e)
                    last_scan = now

                # strategy
                if now - last_strategy >= self.strategy_interval:
                    try:
                        logger.debug("run_loop: running main_strategy")
                        self.main_strategy()
                    except Exception as e:
                        logger.exception("run_loop: main_strategy error: %s", e)
                    last_strategy = now

                # order management (runs most frequently)
                if now - last_order >= self.order_mgmt_interval:
                    try:
                        logger.debug("run_loop: running order_management")
                        self.order_management()
                    except Exception as e:
                        logger.exception("run_loop: order_management error: %s", e)
                    last_order = now

                print("Last Intervals")
                print(f"Last Scan: {int(now - last_scan)} seconds ago")
                print(f"Last Strategy: {int(now - last_strategy)} seconds ago")

                # short sleep to avoid busy loop; resolution smaller than order interval
                
                time.sleep(min(1.0, self.order_mgmt_interval))


        except KeyboardInterrupt:
            sell_all()
            logger.info("run_loop: interrupted by user, stopping loop")
        except Exception:
            sell_all()
            logger.exception("run_loop: unexpected error, stopping")


if __name__ == "__main__":
    # minimal example runner — for development only

    bot = TradingBot(config={
        "scan_interval": 60*60*2,
        "strategy_interval": 60*30,
        "order_mgmt_interval": 2,
        "default_size": 0.001,
    })
    # simple demo: call portfolio snapshot once
    bot.portfolio_status()
    # start the main run loop (will run until interrupted)
    bot.run_loop()
    # print(bot.last_scan_df)
    # market_prices = bot.get_all_market_prices()
    # print(market_prices)
