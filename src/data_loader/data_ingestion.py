import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf
import numpy as np


class MarketDataLoader:
    """
    Production-grade multi-asset market data ingestion.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_data(self) -> pd.DataFrame:
        """
        Download OHLCV data for multiple assets.
        """
        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            raise ValueError("No data downloaded. Check ticker symbols or date range.")

        return data

    def compute_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log returns for Close prices.
        Vectorized and production-safe.
        """
        close_prices = df["Close"]

        log_returns = np.log(close_prices / close_prices.shift(1))

        log_returns = log_returns.dropna()

        return log_returns


    def save_raw(self, df: pd.DataFrame):
        filepath = os.path.join(self.raw_dir, "market_data_raw.csv")
        df.to_csv(filepath)

    def save_processed(self, df: pd.DataFrame):
        filepath = os.path.join(self.processed_dir, "log_returns.csv")
        df.to_csv(filepath)

    def run(self) -> pd.DataFrame:
        """
        Full ingestion pipeline.
        """
        raw_data = self.download_data()
        self.save_raw(raw_data)

        log_returns = self.compute_log_returns(raw_data)
        self.save_processed(log_returns)

        return log_returns
