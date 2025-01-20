import src.backtest as bt

from src.backtest import RsiStrategy
from src.backtest import FeatureBuilder
from src.backtest import yf_helper
df = yf_helper.get_ohlc_data("SPY", "1d")

atr = FeatureBuilder(df).with_atr(20).build()

print(atr.tail(5))



