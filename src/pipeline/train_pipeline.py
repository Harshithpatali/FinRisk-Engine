import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

from src.data_loader.data_ingestion import MarketDataLoader
from src.feature_engineering.volatility_features import VolatilityFeatures
from src.models.garch_model import GARCHModel
from src.models.lstm_model import LSTMVolatility
from src.evaluation.metrics import RiskMetrics
from src.models.model_io import save_model


class TrainingPipeline:

    def __init__(
        self,
        tickers,
        start_date,
        experiment_name="FinRisk-Engine"
    ):
        self.tickers = tickers
        self.start_date = start_date

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)

    def prepare_data(self):
        loader = MarketDataLoader(self.tickers, self.start_date)
        returns = loader.run()

        vol = VolatilityFeatures.rolling_volatility(returns)

        return returns, vol

    def train_garch(self, returns: pd.Series):
        model = GARCHModel()
        model.fit(returns.squeeze())
        forecast = model.forecast()

        return model, forecast.mean()

    def train_lstm(self, returns: pd.Series, epochs=5):

        series = returns.values.reshape(-1, 1)

        X, y = [], []

        window = 20
        for i in range(len(series) - window):
            X.append(series[i:i+window])
            y.append(series[i+window])

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        model = LSTMVolatility()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        return model, float(loss.item())

    def run(self):

        returns, vol = self.prepare_data()

        with mlflow.start_run():

            garch_model, garch_forecast = self.train_garch(returns)
            lstm_model, lstm_loss = self.train_lstm(returns)

            sharpe = RiskMetrics.sharpe_ratio(returns.squeeze().values)

            mlflow.log_param("tickers", self.tickers)
            mlflow.log_metric("garch_forecast_mean", float(garch_forecast))
            mlflow.log_metric("lstm_final_loss", lstm_loss)
            mlflow.log_metric("sharpe_ratio", float(sharpe))

            # Save models
            os.makedirs("models_artifacts", exist_ok=True)

            save_model(lstm_model, "models_artifacts/lstm_model.pth")
            mlflow.pytorch.log_model(lstm_model, "lstm_model")

        return {
            "garch_forecast": float(garch_forecast),
            "lstm_loss": lstm_loss,
            "sharpe": float(sharpe),
        }
