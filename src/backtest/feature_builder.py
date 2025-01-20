import pandas as pd
import numpy as np
import numba


class FeatureBuilder:
    """Builder class to construct feature sets for strategies"""

    def __init__(self, ohlc_data: pd.DataFrame):
        # self.base_data = ohlc_data.copy()
        self.features = ohlc_data.copy()

    def with_pct_change(self) -> "FeatureBuilder":
        calc_pct_chg(self.features)
        return self

    def with_rsi(self, window: int) -> "FeatureBuilder":
        # column_name = f'rsi_{window}'
        calc_rsi_sma(self.features, window)
        return self

    def with_macd(self, fast: int = 12, slow: int = 26) -> "FeatureBuilder":
        return self

    def with_atr(self, window: int) -> "FeatureBuilder":
        # column_name = f'atr_{window}'
        calc_atr_vol(self.features, window)
        return self

    def build(self) -> pd.DataFrame:
        return self.features


def calc_pct_chg(ticker_data: pd.DataFrame) -> None:
    ticker_data['pct_chg'] = ticker_data['Close'].pct_change()


# Not currently used
def calc_atr_regular(ticker_data: pd.DataFrame) -> None:
    ticker_data['TR'] = np.maximum.reduce([
        ticker_data['High'] - ticker_data['Low'],
        abs(ticker_data['High'] - ticker_data['Close'].shift(1)),
        abs(ticker_data['Low'] - ticker_data['Close'].shift(1))
    ])
    # Rolling one day ATR
    ticker_data['ATR1D'] = (ticker_data['TR'] + ticker_data['TR'].shift(1)) * 1 / 2


# Calculates ATR for a given window
# Can also be used to calculate the rolling 1D ATR
def calc_atr_vol(ticker_data: pd.DataFrame, window: int) -> None:
    # This is different from the usual TR formula as it focuses on volatility
    df = ticker_data.copy()
    df['TR'] = ticker_data['High'] - ticker_data['Low']
    df['ATR1D_vol'] = (
            ((df['TR'] + df['TR'].shift(1)) / 2)
            / df['Open'].shift(1)
    )
    df['ATR'] = df['ATR1D_vol'].rolling(window=window).sum() / window
    ticker_data['ATR'] = df['ATR'].copy()


@numba.jit
def rma(x, n):
    """Running moving average"""
    a = np.full_like(x, np.nan)
    a[n] = x[1:n + 1].mean()
    for i in range(n + 1, len(x)):
        a[i] = (a[i - 1] * (n - 1) + x[i]) / n
    return a

def calc_rsi_sma(ticker_data: pd.DataFrame, window: int=14):
    rsi_column = f'rsi_{window}'
    ticker_data[rsi_column] = 100 - (100 / (
                1 + ticker_data['Close'].diff(1).mask(ticker_data['Close'].diff(1) < 0, 0)
                .ewm(alpha=1 / window, adjust=False).mean() /
                ticker_data['Close'].diff(1).mask(ticker_data['Close']
                .diff(1) > 0, -0.0).abs().ewm(alpha=1 / window, adjust=False).mean()))

# Returns a features dataframe that contains the RSI (feature) and date as the index
def calc_rsi_x(ticker_data: pd.DataFrame, window: int = 14) -> None:
    if "Close" not in ticker_data:
        raise ValueError("No close data in the provided DataFrame in StatsCalc.calc_rsi_sma()")

    df = pd.DataFrame()
    df['Change'] = pd.DataFrame(ticker_data['Close'].diff())
    df['Gain'] = df.Change.mask(df.Change < 0, 0.0)
    df['Loss'] = -df.Change.mask(df.Change > 0, -0.0)

    df['avg_gain'] = rma(df.Gain.to_numpy(), window)
    df['avg_loss'] = rma(df.Loss.to_numpy(), window)

    df['rs'] = df.avg_gain / df.avg_loss
    df['RSI'] = 100 - (100 / (1 + df['rs']))
    feature_df = pd.DataFrame(index=ticker_data.index.copy())
    feature_df['RSI'] = df['RSI'].copy()
    feature_df['Close'] = ticker_data['Close'].copy()

    rsi_column = f'rsi_{window}'
    ticker_data[rsi_column] = feature_df['RSI'].copy()
    # return feature_df


# Returns a dataframe with the SMA
def calc_sma(ticker_data: pd.DataFrame, window: int) -> None:
    ticker_data['SMA'] = ticker_data['Close'].rolling(window=window).sum() / window


def calc_ema(ticker_data: pd.DataFrame, window: int) -> None:
    ema_column = f'EMA_{window}'
    smoothing = 2
    multiplier = smoothing / (window + 1)
    ticker_data[ema_column] = ticker_data['Close'].rolling(window=window, min_periods=1).mean()

    for i in range(window, len(ticker_data)):
        ticker_data.loc[ticker_data.index[i], ema_column] = (
                ticker_data.loc[ticker_data.index[i], 'Close'] * multiplier +
                ticker_data.loc[ticker_data.index[i - 1], ema_column] * (1 - multiplier)
        )