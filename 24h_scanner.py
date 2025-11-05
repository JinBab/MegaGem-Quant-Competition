# roostoo_24h_scanner.py
# Run this in Roostoo notebook or your laptop
# Shows TOP 10 gainers + full CSV export

import requests
import pandas as pd
from datetime import datetime

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

# === RUN IT ===
print(f"Scanning 66 cryptos @ {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC\n")
df = get_24h_change()

print("TOP 10 GAINERS 🔥")
print(df.head(10).to_string(index=False, float_format="{:,.2f}".format))

print("\nBOTTOM 5 LOSERS 💀")
print(df.tail(5).to_string(index=False, float_format="{:,.2f}".format))

# SAVE FOR YOUR BOT
df.to_csv('roostoo_24h_scan.csv', index=False)
print("\n✅ Full list saved → roostoo_24h_scan.csv")
