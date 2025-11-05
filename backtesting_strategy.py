# pip install backtesting
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG
from BinanceAPI import fetch_data
import pandas as pd

# adjust the data fetching parameters here, the 2nd parameter is the interval at which we will trade
fetched_data = fetch_data("ETHUSDT", "1h", "1 Jan 2023", "11 NOV 2025")

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.sma1 = self.I(SMA, price, 10)
        self.sma2 = self.I(SMA, price, 20)

    def next(self):

        ## DEFINE STRATEGY LOGIC HERE ##
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

print(fetched_data.dtypes)
print(GOOG.dtypes)

bt = Backtest(fetched_data, SmaCross, cash=1000000, commission=0.001, exclusive_orders=True, finalize_trades=True)
stats = bt.run()

# bt.plot()
print(stats)
