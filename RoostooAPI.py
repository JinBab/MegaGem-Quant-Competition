
import requests
import hashlib
import hmac
import time


API_KEY = "UVVCgOdMjpH3yjoHFTIJLun9ODTwmPjYMCDYGuOYmjHAGPKDcTWfGtU4jAqYJw32"
SECRET = "16QegeXbzXsgGhNBzVheqsQnz35vv1YRIR8ZHxCBJWxwfkMMoAyNjmuiwZKbLgFc"

BASE_URL = "https://mock-api.roostoo.com"


def generate_signature(params):
    query_string = '&'.join(["{}={}".format(k, params[k])
                             for k in sorted(params.keys())])
    us = SECRET.encode('utf-8')
    m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
    return m.hexdigest()


def get_server_time():
    r = requests.get(
        BASE_URL + "/v3/serverTime",
    )
    print(r.status_code, r.text)
    return r.json()


def get_ex_info():
    r = requests.get(
        BASE_URL + "/v3/exchangeInfo",
    )
    print(r.status_code, r.text)
    return r.json()


def get_ticker(pair=None):
    payload = {
        "timestamp": int(time.time()),
    }
    if pair:
        payload["pair"] = pair

    r = requests.get(
        BASE_URL + "/v3/ticker",
        params=payload,
    )
    print(r.status_code, r.text)
    return r.json()


def get_balance():
    payload = {
        "timestamp": int(time.time()) * 1000,
    }

    r = requests.get(
        BASE_URL + "/v3/balance",
        params=payload,
        headers={"RST-API-KEY": API_KEY,
                 "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)
    return r.json()


def place_order(coin, side, qty, price=None):
    payload = {
        "timestamp": int(time.time()) * 1000,
        "pair": coin + "/USD",
        "side": side,
        "quantity": qty,
    }

    if not price:
        payload['type'] = "MARKET"
    else:
        payload['type'] = "LIMIT"
        payload['price'] = price

    r = requests.post(
        BASE_URL + "/v3/place_order",
        data=payload,
        headers={"RST-API-KEY": API_KEY,
                 "MSG-SIGNATURE": generate_signature(payload)}
    )
    
    # Return parsed JSON when possible so callers can inspect order details.
    print(r.status_code, r.text)
    try:
        return r.json()
    except Exception:
        # Fall back to raw text and status
        return {"status_code": r.status_code, "text": r.text}


def cancel_order():
    payload = {
        "timestamp": int(time.time()) * 1000,
        # "order_id": 77,
        "pair": "TRUMP/USD",
    }

    r = requests.post(
        BASE_URL + "/v3/cancel_order",
        data=payload,
        headers={"RST-API-KEY": API_KEY,
                 "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)


def query_order():
    payload = {
        "timestamp": int(time.time())*1000,
        # "order_id": 77,
        # "pair": "DASH/USD",
        # "pending_only": True,
    }

    r = requests.post(
        BASE_URL + "/v3/query_order",
        data=payload,
        headers={"RST-API-KEY": API_KEY,
                 "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)


def pending_count():
    payload = {
        "timestamp": int(time.time()) * 1000,
    }

    r = requests.get(
        BASE_URL + "/v3/pending_count",
        params=payload,
        headers={"RST-API-KEY": API_KEY,
                 "MSG-SIGNATURE": generate_signature(payload)}
    )
    print( r.status_code, r.text)
    return r.json()

#200 {"Success":true,"ErrMsg":"","SpotWallet":{"BTC":{"Free":0.08324,"Lock":0},"DOGE":{"Free":28742,"Lock":0},"ETH":{"Free":0.008,"Lock":0},"FET":{"Free":17182.1,"Lock":0},"FIL":{"Free":1930.5,"Lock":0},"HEMI":{"Free":0,"Lock":0},"ICP":{"Free":627.97,"Lock":0},"NEAR":{"Free":1846.4,"Lock":0},"S":{"Free":31585.6,"Lock":0},"TRUMP":{"Free":0,"Lock":0},"USD":{"Free":1431.71,"Lock":0},"ZEC":{"Free":8.63,"Lock":0}},"MarginWallet":{}}
def sell_all():
    cur_balance = get_balance()
    for coin, asset in cur_balance['SpotWallet'].items():
        free_qty = float(asset['Free'])
        if coin != 'USD' and free_qty > 0:
            print(f"Selling all {free_qty} of {coin}")
            place_order(coin, "SELL", free_qty)



if __name__ == '__main__':
    # get_server_time()
    # get_ex_info()
    # data = get_ticker()
    # extract all ticker symbol names into an array
    # symbols = list(data["Data"].keys())
    # print(symbols)
    
    ticker = get_ticker("ZEC/USD")

    # if ticker['Success']:
    #     print(ticker['Data'].values())
    #     place_order("ZEC", "BUY", 1, list(ticker['Data'].values())[0]['LastPrice'])

    # sell_all()
    place_order("LINEA", "BUY", 100.0)
    # place_order("ETH", "BUY", 0.004)
    # cancel_order()
    get_balance()
    # query_order()
    # pending_count()