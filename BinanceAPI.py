import requests
import pandas as pd
from datetime import datetime

# Use the public Binance REST API directly via requests. This removes the
# dependency on python-binance's Client for simple historical klines fetches.
BASE_URL = "https://api.binance.us"

# All valid interval strings supported by Binance. We use the same string
# values when calling the /api/v3/klines endpoint.
INTERVAL_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}


def _to_ms(ts):
    """Convert a datetime or string to milliseconds since epoch."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        # assume already seconds or milliseconds; prefer milliseconds if large
        if ts > 1e12:
            return int(ts)
        return int(ts * 1000)
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1000)
    # try parse with pandas (handles many common formats)
    try:
        dt = pd.to_datetime(ts)
        return int(dt.timestamp() * 1000)
    except Exception:
        raise ValueError(f"Could not parse time: {ts}")


def fetch_data(symbol, interval, start=None, end=None, limit: int = 1000):
    """Fetch historical klines from Binance using the public REST API.

    Parameters
    - symbol: e.g. 'BTCUSDT'
    - interval: one of the keys in INTERVAL_MAP (e.g. '1m','1h','1d')
    - start, end: datetime or parseable time string (optional). If omitted
      Binance will return klines up to the latest time (limit-controlled).
    - limit: maximum number of klines to return (default 1000)

    Returns a pandas.DataFrame with columns: ['Open','High','Low','Close','Volume']
    and the index set to the close time (pd.Timestamp).
    """
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Invalid interval. Choose from: {list(INTERVAL_MAP.keys())}")

    params = {
        "symbol": symbol,
        "interval": INTERVAL_MAP[interval],
        "limit": int(limit),
    }

    start_ms = _to_ms(start) if start is not None else None
    end_ms = _to_ms(end) if end is not None else None
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    url = BASE_URL + "/api/v3/klines"
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Binance klines request failed: {r.status_code} {r.text}")

    data = r.json()
    # Binance returns list of arrays: [ OpenTime, Open, High, Low, Close, Volume, CloseTime, ... ]
    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base", "Taker buy quote", "Ignore"
    ])

    # Convert timestamps and numeric types
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
    df.set_index("Close time", inplace=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ensure integer volume where reasonable
    try:
        df["Volume"] = df["Volume"].astype(float)
    except Exception:
        pass

    # keep only a compact set of columns, rounded similarly to prior behaviour
    df = df[["Open", "High", "Low", "Close", "Volume"]].round(6)
    return df


# def get_lot_step(symbol: str):
#     """Return the lot step size and minQty for a given symbol using exchangeInfo.

#     Returns (stepSize: float, minQty: float). If unavailable, returns (0.001, 0.001)
#     as a sensible default.
#     """
#     try:
#         info = client.get_symbol_info(symbol)
#         if not info:
#             return 0.001, 0.001
#         for f in info.get('filters', []):
#             if f.get('filterType') == 'LOT_SIZE':
#                 step = float(f.get('stepSize', '0.001'))
#                 min_qty = float(f.get('minQty', '0.001'))
#                 return step, min_qty
#     except Exception:
#         # Fall back to a conservative default
#         return 0.001, 0.001

#     return 0.001, 0.001


LOT_STEP_INFO = {'1000CHEEMSUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'AAVEUSDT': {'step_size': 0.001, 'min_qty': 0.001}, 'ADAUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'APTUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'ARBUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'ASTERUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'AVAXUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'AVNTUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'BIOUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'BMTUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'BNBUSDT': {'step_size': 0.001, 'min_qty': 0.001}, 'BONKUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'BTCUSDT': {'step_size': 1e-05, 'min_qty': 1e-05}, 'CAKEUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'CFXUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'CRVUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'DOGEUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'DOTUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'EDENUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'EIGENUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'ENAUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'ETHUSDT': {'step_size': 0.0001, 'min_qty': 0.0001}, 'FETUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'FILUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'FLOKIUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'FORMUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'HBARUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'HEMIUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'ICPUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'LINEAUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'LINKUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'LISTAUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'LTCUSDT': {'step_size': 0.001, 'min_qty': 0.001}, 'MIRAUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'NEARUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'ONDOUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'OPENUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'PAXGUSDT': {'step_size': 0.0001, 'min_qty': 0.0001}, 'PENDLEUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'PENGUUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'PEPEUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'PLUMEUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'POLUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'PUMPUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'SUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'SEIUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'SHIBUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'SOLUSDT': {'step_size': 0.001, 'min_qty': 0.001}, 'SOMIUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'STOUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'SUIUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'TAOUSDT': {'step_size': 0.0001, 'min_qty': 0.0001}, 'TONUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'TRUMPUSDT': {'step_size': 0.001, 'min_qty': 0.001}, 'TRXUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'TUTUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'UNIUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'VIRTUALUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'WIFUSDT': {'step_size': 0.01, 'min_qty': 0.01}, 'WLDUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'WLFIUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'XLMUSDT': {'step_size': 1.0, 'min_qty': 1.0}, 'XPLUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'XRPUSDT': {'step_size': 0.1, 'min_qty': 0.1}, 'ZECUSDT': {'step_size': 0.001, 'min_qty': 0.001}, 'ZENUSDT': {'step_size': 0.01, 'min_qty': 0.01}}


# Example time formats that can be used for start and end
# "1 Nov 2025"           # just date
# "1 Nov 2025 15:30"     # date + hour:minute
# "2025-11-01 15:30:45"  # full date + time + seconds
# "1 Jan 2023 00:00:00"  # for very precise start

# If no start time is given, it will just give maximum number of klines up to end time

# data = fetch_data("BTCUSDT", "1m", None, "8 Nov 2025")

