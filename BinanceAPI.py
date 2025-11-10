from binance.client import Client
# pip install python-binance
import pandas as pd

api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"

client = Client(api_key, api_secret)

# all possible time intervals in binance
INTERVAL_MAP = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "3m": Client.KLINE_INTERVAL_3MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "2h": Client.KLINE_INTERVAL_2HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "8h": Client.KLINE_INTERVAL_8HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "3d": Client.KLINE_INTERVAL_3DAY,
    "1w": Client.KLINE_INTERVAL_1WEEK,
    "1M": Client.KLINE_INTERVAL_1MONTH
}

def fetch_data(symbol, interval, start, end):
    # convert start and end datetimes to str
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")

    if interval not in INTERVAL_MAP:
        raise ValueError(f"Invalid interval. Choose from: {list(INTERVAL_MAP.keys())}")
    
    data = client.get_historical_klines(
        symbol,
        INTERVAL_MAP[interval],
        start_str,
        end_str
    )
    # Convert to DataFrame
    data = pd.DataFrame(data, columns=[
    "Open time", "Open", "High", "Low", "Close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base", "Taker buy quote", "Ignore"
    ])
    
    # Convert timestamps
    data["Open time"] = pd.to_datetime(data["Open time"], unit="ms")
    data["Close time"] = pd.to_datetime(data["Close time"], unit="ms")
    data.set_index("Close time", inplace=True)
    data["Close"] = pd.to_numeric(data["Close"])
    data["High"] = pd.to_numeric(data["High"])
    data["Low"] = pd.to_numeric(data["Low"])
    data["Open"] = pd.to_numeric(data["Open"])
    data["Volume"] = pd.to_numeric(data["Volume"])
    data["Volume"] = data["Volume"].astype(int)
    data = data.round(2)
    return data[["Open", "High", "Low", "Close", "Volume"]]


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

