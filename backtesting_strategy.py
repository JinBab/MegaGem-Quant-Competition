
# pip install TA-Lib
import talib
# pip install backtesting
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG
from BinanceAPI import fetch_data
import pandas as pd

# adjust the data fetching parameters here, the 2nd parameter is the interval at which we will trade
fetched_data = fetch_data("ETHUSDT", "1d", "1 Nov 2023", "11 NOV 2025")
fetched_data = fetched_data.iloc[:-1]

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        # the first method called once at the start of the strategy

        # Define indicators here
        self.macd = self.I(lambda x: talib.MACD(x)[0],price)
        self.macd_signal = self.I(lambda x: talib.MACD(x)[1],price)


        # self.sma1 = self.I(SMA, price, 10)
        # self.sma2 = self.I(SMA, price, 20)
        # self.buy()

    def next(self):
        if (crossover(self.macd, self.macd_signal)):
            self.buy()
        elif (crossover(self.macd_signal, self.macd)):
            self.sell()
        pass
        # self.buy()
        ## DEFINE STRATEGY LOGIC HERE ##
        # if crossover(self.sma1, self.sma2):
        #     self.buy()
        # elif crossover(self.sma2, self.sma1):
        #     self.sell()

print(fetched_data.dtypes)
print(GOOG.dtypes)

bt = Backtest(fetched_data, SmaCross, cash=1000000, commission=0.001, exclusive_orders=True, finalize_trades=True)
stats = bt.run()

# bt.plot()
print(stats)
