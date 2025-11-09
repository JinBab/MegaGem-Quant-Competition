import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
# Telegram notifications removed

# ============================= CONFIG =============================
API_KEY = 'YOUR_BINANCE_API_KEY'
API_SECRET = 'YOUR_BINANCE_API_SECRET'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'
N_DONCHIAN = 20
VOLUME_MULT = 2.0
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
CHAND_MULT = 3.0
CHAND_PERIOD = 22
RISK_PERCENT = 1.0  # % of balance per trade
LEVERAGE = 5

# Notifications disabled

# =================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger()

# Initialize
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})
exchange.set_sandbox_mode(False)  # Set True for testnet

def _notify(msg: str):
    # Notification stub (disabled); keep console output for visibility
    print(msg)

def fetch_ohlcv(symbol, timeframe, limit=500):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def get_balance():
    balance = exchange.fetch_balance()
    usdt = balance['total'].get('USDT', 0)
    return float(usdt)

def get_funding_rate(symbol):
    try:
        funding = exchange.fetch_funding_rate(symbol)
        return funding['fundingRate']
    except:
        return 0.0

def get_position():
    positions = exchange.fetch_positions([SYMBOL])
    for pos in positions:
        if pos['symbol'] == SYMBOL.replace('/', ''):
            return {
                'side': 'long' if float(pos['contracts']) > 0 else 'short' if float(pos['contracts']) < 0 else None,
                'size': abs(float(pos['contracts'])),
                'entry': float(pos['entryPrice']) if pos['entryPrice'] else 0
            }
    return {'side': None, 'size': 0, 'entry': 0}

def place_order(side, amount, stop_loss=None):
    try:
        params = {'leverage': LEVERAGE}
        order = exchange.create_order(SYMBOL, 'market', side, amount, None, params)
        logger.info(f"ORDER: {side.upper()} {amount} {SYMBOL}")
        if stop_loss:
            sl_side = 'sell' if side == 'buy' else 'buy'
            exchange.create_order(SYMBOL, 'stop_market', sl_side, amount, stop_loss)
        return order
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return None

# ============================= STRATEGY =============================
def donchian_strategy():
    # Fetch data
    bars = fetch_ohlcv(SYMBOL, TIMEFRAME, N_DONCHIAN * 3)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    if len(df) < N_DONCHIAN:
        return

    # Indicators
    df['upper'] = df['high'].rolling(N_DONCHIAN).max()
    df['lower'] = df['low'].rolling(N_DONCHIAN).min()
    df['middle'] = (df['upper'] + df['lower']) / 2
    df['vol_avg'] = df['volume'].rolling(N_DONCHIAN).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD).mean()

    # Chandelier Exit
    df['chandelier_long'] = df['high'].rolling(CHAND_PERIOD).max() - CHAND_MULT * df['atr']
    df['chandelier_short'] = df['low'].rolling(CHAND_PERIOD).min() + CHAND_MULT * df['atr']

    row = df.iloc[-2]  # Previous closed candle
    close = df['close'].iloc[-1]
    funding = get_funding_rate(SYMBOL)
    position = get_position()
    balance = get_balance()

    # Risk management
    risk_amount = balance * (RISK_PERCENT / 100)
    price = close
    atr = row['atr']
    position_size = risk_amount / (ATR_STOP_MULT * atr) if atr > 0 else 0
    position_size = round(position_size, 6)

    # Avoid tiny positions
    if position_size < 0.001:
        return

    # Current state
    in_long = position['side'] == 'long'
    in_short = position['side'] == 'short'

    # === ENTRY LOGIC ===
    long_signal = (
        close > row['upper'] and
        df['volume'].iloc[-1] > row['vol_avg'] * VOLUME_MULT and
        close > row['ema200'] and
        funding < 0.0008  # < 0.08%
    )

    short_signal = (
        close < row['lower'] and
        df['volume'].iloc[-1] > row['vol_avg'] * VOLUME_MULT and
        close < row['ema200'] and
        funding > -0.0005
    )

    # === EXIT LOGIC ===
    exit_long = in_long and close < row['middle']
    exit_short = in_short and close > row['middle']

    # Chandelier trail
    trail_exit_long = in_long and close < row['chandelier_long']
    trail_exit_short = in_short and close > row['chandelier_short']

    # === EXECUTE ===
    if long_signal and not in_long and not in_short:
        # Close short if exists
        if in_short:
            place_order('sell', position['size'])
        # Enter long
        stop_loss = price - ATR_STOP_MULT * atr
        place_order('buy', position_size, stop_loss)
        _notify(f"LONG ENTRY @ {price:.1f} | SL: {stop_loss:.1f}")

    elif short_signal and not in_short and not in_long:
        if in_long:
            place_order('sell', position['size'])
        stop_loss = price + ATR_STOP_MULT * atr
        place_order('sell', position_size, stop_loss)
        _notify(f"SHORT ENTRY @ {price:.1f} | SL: {stop_loss:.1f}")

    # Exit
    elif exit_long or trail_exit_long:
        place_order('sell', position['size'])
        _notify(f"LONG EXIT @ {price:.1f}")

    elif exit_short or trail_exit_short:
        place_order('buy', position['size'])
        _notify(f"SHORT EXIT @ {price:.1f}")

# ============================= MAIN LOOP =============================
if __name__ == "__main__":
    _notify("Donchian Bot STARTED")
    logger.info("Bot started. Waiting for signals...")

    while True:
        try:
            donchian_strategy()
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error: {e}")
            _notify(f"ERROR: {e}")
            time.sleep(60)