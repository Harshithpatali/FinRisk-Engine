import pandas as pd
import numpy as np


class VolatilityFeatures:
    """
    Volatility-related feature engineering.
    """

    @staticmethod
    def rolling_volatility(log_returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """
        Annualized rolling volatility.
        """
        rolling_std = log_returns.rolling(window=window).std()
        annualized_vol = rolling_std * np.sqrt(252)

        return annualized_vol.dropna()
