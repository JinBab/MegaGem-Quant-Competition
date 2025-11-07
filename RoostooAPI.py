
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


if __name__ == '__main__':
    # get_server_time()
    # get_ex_info()
    # data = get_ticker()
    # extract all ticker symbol names into an array
    # symbols = list(data["Data"].keys())
    # print(symbols)

    # place_order("TRUMP", "BUY", 1, 7)
    # place_order("HEMI", "SELL", 224200.2)
    # cancel_order()
    get_balance()
    query_order()
    # pending_count()