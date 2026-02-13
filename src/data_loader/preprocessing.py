import pandas as pd


class DataValidator:
    """
    Validates financial time series before modeling.
    """

    @staticmethod
    def check_missing(df: pd.DataFrame) -> None:
        if df.isnull().sum().sum() > 0:
            raise ValueError("Dataset contains missing values.")

    @staticmethod
    def check_stationarity_ready(df: pd.DataFrame) -> None:
        if df.shape[0] < 100:
            raise ValueError("Insufficient data length for modeling.")

    @staticmethod
    def validate(df: pd.DataFrame) -> None:
        DataValidator.check_missing(df)
        DataValidator.check_stationarity_ready(df)
