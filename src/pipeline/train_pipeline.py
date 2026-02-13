import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow

from src.data_loader.data_ingestion import MarketDataLoader
from src.models.lstm_model import LSTMVolatility
from src.models.model_io import save_model
from src.models.garch_model import GARCHModel
from src.evaluation.metrics import RiskMetrics


class TrainingPipeline:
    """
    Production-grade volatility forecasting training pipeline.
    """

    def __init__(
        self,
        tickers,
        start_date,
        experiment_name="FinRisk-Engine",
        window=20,
        epochs=10
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.window = window
        self.epochs = epochs

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)

    # ===============================
    # Data Preparation
    # ===============================
    def prepare_volatility_series(self):
        loader = MarketDataLoader(self.tickers, self.start_date)
        returns = loader.run().squeeze()

        # Rolling volatility target
        vol = returns.rolling(self.window).std().dropna()

        return returns, vol

    # ===============================
    # LSTM Volatility Training
    # ===============================
    def train_lstm_volatility(self, vol_series):

        series = vol_series.values.reshape(-1, 1)

        X, y = [], []

        for i in range(len(series) - self.window):
            X.append(series[i:i+self.window])
            y.append(series[i+self.window])

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        model = LSTMVolatility(input_size=1)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        return model, float(loss.item())

    # ===============================
    # GARCH Baseline
    # ===============================
    def train_garch(self, returns):
        model = GARCHModel()
        model.fit(returns)

        forecast = model.forecast(horizon=1)

        return model, float(forecast.values[0])

    # ===============================
    # Run Pipeline
    # ===============================
    def run(self):

        returns, vol = self.prepare_volatility_series()

        with mlflow.start_run():

            garch_model, garch_vol_forecast = self.train_garch(returns)
            lstm_model, lstm_loss = self.train_lstm_volatility(vol)

            sharpe = RiskMetrics.sharpe_ratio(returns.values)

            mlflow.log_param("tickers", self.tickers)
            mlflow.log_param("window", self.window)
            mlflow.log_metric("garch_vol_forecast", garch_vol_forecast)
            mlflow.log_metric("lstm_loss", lstm_loss)
            mlflow.log_metric("sharpe_ratio", float(sharpe))

            os.makedirs("models_artifacts", exist_ok=True)
            save_model(lstm_model, "models_artifacts/lstm_vol_model.pth")

        return {
            "garch_vol_forecast": garch_vol_forecast,
            "lstm_loss": lstm_loss,
            "sharpe": float(sharpe)
        }
