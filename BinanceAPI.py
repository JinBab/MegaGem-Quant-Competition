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
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Invalid interval. Choose from: {list(INTERVAL_MAP.keys())}")
    
    data = client.get_historical_klines(
        symbol,
        INTERVAL_MAP[interval],
        start,
        end
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

# Example time formats that can be used for start and end
# "1 Nov 2025"           # just date
# "1 Nov 2025 15:30"     # date + hour:minute
# "2025-11-01 15:30:45"  # full date + time + seconds
# "1 Jan 2023 00:00:00"  # for very precise start

# If no start time is given, it will just give maximum number of klines up to end time

# data = fetch_data("BTCUSDT", "1m", None, "8 Nov 2025")

