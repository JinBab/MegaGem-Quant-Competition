import argparse
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from datetime import timedelta
from BinanceAPI import fetch_data as binance_fetch

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

def _notify(msg: str):
    # Notification stub (disabled); keep console output for visibility
    print(msg)

# ------------------- Data helpers using BinanceAPI -------------------
def _to_binance_symbol(sym: str) -> str:
    return sym.replace('/', '')

def _interval_to_seconds(interval: str) -> int:
    s = interval.strip()
    unit = s[-1]
    val = int(s[:-1]) if len(s) > 1 else 1
    if unit == 'm':
        return val * 60
    if unit == 'h':
        return val * 60 * 60
    if unit == 'd':
        return val * 24 * 60 * 60
    if unit == 'w':
        return val * 7 * 24 * 60 * 60
    if unit == 'M':  # Binance monthly
        return val * 30 * 24 * 60 * 60
    return val * 60

def _fmt(dt: datetime) -> str:
    return dt.strftime("%d %b %Y %H:%M:%S")

def fetch_ohlcv(symbol: str, timeframe: str, bars: int = 200) -> pd.DataFrame:
    """Fetch OHLCV via BinanceAPI.fetch_data and return DataFrame with lower-case columns."""
    end = datetime.utcnow()
    seconds = _interval_to_seconds(timeframe)
    start = end - timedelta(seconds=seconds * bars)
    bsymbol = _to_binance_symbol(symbol)
    df = binance_fetch(bsymbol, timeframe, _fmt(start), _fmt(end))
    # Normalize column names to match prior usage
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    return df

# ============================= EVALUATION =============================
def evaluate_symbol(symbol: str,
                     timeframe: str = TIMEFRAME,
                     n_donchian: int = N_DONCHIAN,
                     risk_percent: float = RISK_PERCENT,
                     atr_stop_mult: float = ATR_STOP_MULT,
                     tp_multiplier: float = 2.0) -> dict:
    """Evaluate trade setup for a single symbol.

    Returns dict with:
        symbol, side ('long'/'short'/None), entry_price, stop_loss, take_profit,
        allocation_pct (percent of portfolio), position_size, reason, atr
    """
    df = fetch_ohlcv(symbol, timeframe, n_donchian * 3)
    if df is None or df.empty or len(df) < n_donchian:
        return {"symbol": symbol, "side": None, "reason": "insufficient data"}

    # Indicators
    df['upper'] = df['high'].rolling(n_donchian).max()
    df['lower'] = df['low'].rolling(n_donchian).min()
    df['middle'] = (df['upper'] + df['lower']) / 2
    df['vol_avg'] = df['volume'].rolling(n_donchian).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD).mean()

    # Chandelier (for exits info)
    df['chandelier_long'] = df['high'].rolling(CHAND_PERIOD).max() - CHAND_MULT * df['atr']
    df['chandelier_short'] = df['low'].rolling(CHAND_PERIOD).min() + CHAND_MULT * df['atr']

    row = df.iloc[-2]
    close = df['close'].iloc[-1]
    atr = float(row['atr']) if not pd.isna(row['atr']) else 0.0

    # Approximate allocation percent (of portfolio) based on stop distance
    # Allocation% ≈ risk_percent * price / (atr_stop_mult * atr)
    allocation_pct = 0.0
    if atr > 0:
        allocation_pct = risk_percent * (close / (atr_stop_mult * atr))
        allocation_pct = float(max(0.0, min(100.0, allocation_pct)))

    # Long-only signal (no shorts)
    long_signal = (
        close > row['upper'] and
        df['volume'].iloc[-1] > row['vol_avg'] * VOLUME_MULT and
        close > row['ema200']
    )

    side = None  # 'buy' or None
    stop_loss = None
    take_profit = None
    reason = "no signal"

    if long_signal:
        side = 'buy'
        stop_loss = close - atr_stop_mult * atr if atr > 0 else None
        take_profit = close + tp_multiplier * atr if atr > 0 else None
        reason = "donchian breakout buy"

    return {
        "symbol": symbol,
    "action": side,  # 'buy' or None
        "entry_price": close if side else None,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "allocation_pct": allocation_pct if side else 0,
        "reason": reason,
        "atr": atr,
        "upper": float(row['upper']),
        "lower": float(row['lower']),
    }


# ============================= MAIN LOOP =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Donchian evaluation (Binance fetch)")
    parser.add_argument("symbol", nargs="?", default=SYMBOL, help="Symbol like BTC/USDT")
    parser.add_argument("--live", action="store_true", help="Run continuous live loop (not recommended in this stripped mode)")
    args = parser.parse_args()

    if not args.live:
        res = evaluate_symbol(args.symbol)
        print("--- Evaluation ---")
        for k,v in res.items():
            print(f"{k}: {v}")
    else:
        _notify("Live mode is disabled in this simplified version.")