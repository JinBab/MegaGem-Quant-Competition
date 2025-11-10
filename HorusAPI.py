########### USE HORUS TO GET HISTORICAL CRYPTO DATA ###################

import requests
import pandas as pd

url = "https://api-horus.com/market/price" 
api_key = "42290f5b23af3ce503804ca20b5e46305cbd6ebe6a3070fe8bfed0c829829807"

headers = {
    "X-API-Key": api_key
}

def get_data(symbol, interval, start=None, end=None):
    #convert datetime to seconds time
    if start:
        start = int(pd.Timestamp(start).timestamp())
    if end: 
        end = int(pd.Timestamp(end).timestamp())
    params = {
        "asset": symbol,
        "interval": interval,
        "start": start,
        "end": end
    }
    return requests.get(url, headers=headers, params=params)

    # Send GET request

# enter start and end date as "YYYY-MM-DD HH:MM:SS" can exclude time portion 
response = get_data("BTC", "1d", start="2014-11-01", end="2016-11-05")

# Check for success
if response.status_code == 200:
    data = response.json()
    data = pd.DataFrame(data)

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data = data[::-1]
    data.set_index('timestamp', inplace=True)
    print(data)
else:
    print(f"Error {response.status_code}: {response.text}")
