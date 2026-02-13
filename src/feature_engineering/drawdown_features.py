import pandas as pd
import numpy as np


class DrawdownFeatures:
    """
    Drawdown and ATR calculations.
    """

    @staticmethod
    def compute_drawdown(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute drawdown from cumulative max.
        """
        cumulative_max = prices.cummax()
        drawdown = (prices - cumulative_max) / cumulative_max

        return drawdown

    @staticmethod
    def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Compute Average True Range.
        Requires High, Low, Close columns.
        """

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=window).mean()

        return atr.dropna()
