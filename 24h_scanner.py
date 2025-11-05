# roostoo_24h_scanner.py
# Run this in Roostoo notebook or your laptop
# Shows TOP 10 gainers + full CSV export

import requests
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures

# === 66 ROOSTOO CRYPTOS (USDT pairs) ===
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT",
    "TRXUSDT","AVAXUSDT","SHIBUSDT","LINKUSDT","DOTUSDT","BCHUSDT","NEARUSDT",
    "LTCUSDT","MATICUSDT","ICPUSDT","UNIUSDT","APTUSDT","HBARUSDT","VETUSDT",
    "FILUSDT","ETCUSDT","ATOMUSDT","ARBUSDT","IMXUSDT","OPUSDT","INJUSDT",
    "FTTUSDT","ALGOUSDT","THETAUSDT","FLOWUSDT","SANDUSDT","AXSUSDT","MANAUSDT",
    "GALAUSDT","CHZUSDT","EOSUSDT","XTZUSDT","ZECUSDT","AAVEUSDT","MKRUSDT",
    "KSMUSDT","CRVUSDT","GRTUSDT","RUNEUSDT","SNXUSDT","COMPUSDT","ZILUSDT",
    "ENJUSDT","BATUSDT","LRCUSDT","YFIUSDT","SUSHIUSDT","1INCHUSDT","REEFUSDT",
    "OCEANUSDT","RNDRUSDT","CELOUSDT","KAVAUSDT","ANKRUSDT","ONTUSDT","IOSTUSDT",
    "SKLUSDT","DENTUSDT","HOTUSDT","CELRUSDT","STMXUSDT","RLCUSDT"
]

def get_24h_change():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    resp = requests.get(url).json()
    
    # Filter only our 66
    data = [d for d in resp if d['symbol'] in SYMBOLS]
    


    df = pd.DataFrame(data)[['symbol', 'priceChangePercent', 'lastPrice', 'quoteVolume']]
    df['priceChangePercent'] = df['priceChangePercent'].astype(float)
    df = df.sort_values('priceChangePercent', ascending=False).reset_index(drop=True)
    
    # Clean names
    df['coin'] = df['symbol'].str.replace('USDT', '')
    df = df[['coin', 'priceChangePercent', 'lastPrice', 'quoteVolume']]
    df.columns = ['Coin', '24h %', 'Price', 'Volume']
    
    return df

def get_custom_change(window, end_datetime):
    """Return change over a custom window ending at end_datetime.

    Parameters
    - window: seconds (int) or interval string like '5m','1h','1d'
    - end_datetime: a datetime object representing the end time

    Approach: compute start = end - window, then for each symbol fetch only the
    start and end prices (avoid downloading the full range). We try to get a
    kline/trade at or immediately before each timestamp.
    """
    url_klines = "https://api.binance.com/api/v3/klines"
    url_agg = "https://api.binance.com/api/v3/aggTrades"
    url_ticker = "https://api.binance.com/api/v3/ticker/price"

    results = []

    # helper: convert binance-style interval string to seconds
    def _interval_to_seconds(iv):
        try:
            if isinstance(iv, (int, float)):
                return int(iv)
            s = str(iv)
            num = int(s[:-1]) if len(s) > 1 else 1
            unit = s[-1]
            if unit == 'm':
                return num * 60
            if unit == 'h':
                return num * 60 * 60
            if unit == 'd':
                return num * 24 * 60 * 60
            if unit == 'w':
                return num * 7 * 24 * 60 * 60
            if unit == 'M':
                return num * 30 * 24 * 60 * 60
        except Exception:
            pass
        return int(iv)

    def _price_at(symbol, ts_ms):
        # 1) try a small-interval kline that ends at or before ts_ms
        params = {'symbol': symbol, 'interval': '1m', 'endTime': ts_ms, 'limit': 1}
        r = requests.get(url_klines, params=params).json()
        return float(r[0][4])

    # compute start and end in ms
    window_seconds = _interval_to_seconds(window)
    start_dt = end_datetime - timedelta(seconds=window_seconds)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_datetime.timestamp() * 1000)

    # Run start and end price fetches in parallel across symbols and across the two timestamps.
    with concurrent.futures.ThreadPoolExecutor(max_workers=140) as executor:
        # submit all start-time fetches
        start_futures = {symbol: executor.submit(_price_at, symbol, start_ms) for symbol in SYMBOLS}
        # submit all end-time fetches
        end_futures = {symbol: executor.submit(_price_at, symbol, end_ms) for symbol in SYMBOLS}

        for symbol in SYMBOLS:
            try:
                start_price = start_futures[symbol].result(timeout=10)
            except Exception:
                start_price = None
            try:
                end_price = end_futures[symbol].result(timeout=10)
            except Exception:
                end_price = None

            if start_price is None or end_price is None or start_price == 0:
                continue

            price_change_percent = ((end_price - start_price) / start_price) * 100
            results.append({
                'symbol': symbol,
                'priceChangePercent': price_change_percent,
                'lastPrice': end_price
            })

    df = pd.DataFrame(results)
    df['priceChangePercent'] = df['priceChangePercent'].astype(float)
    df = df.sort_values('priceChangePercent', ascending=False).reset_index(drop=True)

    # Clean names
    df['coin'] = df['symbol'].str.replace('USDT', '')
    df = df[['coin', 'priceChangePercent', 'lastPrice']]
    df.columns = ['Coin', f'Change %', 'Price']

    return df


# df = get_24h_change()
end_time = "2025-11-05 00:00"
#convert date input into datetime object
end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M")

# example: 1 day window ending at midnight UTC today
custom_df = get_custom_change("1d", end_time)

# print("TOP 10 GAINERS 🔥")
# print(df.head(10).to_string(index=False, float_format="{:,.2f}".format))

print(f"{end_time.strftime('%Y-%m-%d %H:%M')}")
print("CUSTOM WINDOW TOP 10 🔥")
print(custom_df.head(10).to_string(index=False, float_format="{:,.2f}".format))