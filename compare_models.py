import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.data_loader.data_ingestion import MarketDataLoader
from src.models.garch_model import GARCHModel
from src.models.lstm_model import LSTMVolatility
from src.models.temporal_cnn import TemporalCNN


class ModelComparator:

    def __init__(self, ticker="AAPL", start_date="2020-01-01"):
        self.ticker = ticker
        self.start_date = start_date

    def load_data(self):
        loader = MarketDataLoader([self.ticker], self.start_date)
        returns = loader.run()
        return returns.squeeze()

    def train_test_split(self, series, split_ratio=0.8):
        split = int(len(series) * split_ratio)
        return series[:split], series[split:]

    # ===============================
    # GARCH
    # ===============================
    def garch_predict(self, train, test):
        model = GARCHModel()
        model.fit(train)

        preds = []
        for _ in range(len(test)):
            forecast = model.forecast(horizon=1)
            preds.append(forecast.values[0])

        return np.array(preds)

    # ===============================
    # LSTM
    # ===============================
    def lstm_predict(self, train, test):

        series = train.values.reshape(-1, 1)
        window = 20

        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i+window])
            y.append(series[i+window])

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        model = LSTMVolatility()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(5):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Forecast next values
        preds = []
        input_seq = series[-window:]

        for _ in range(len(test)):
            x = torch.tensor(input_seq.reshape(1, window, 1), dtype=torch.float32)
            pred = model(x).item()
            preds.append(pred)

            input_seq = np.append(input_seq[1:], [[pred]], axis=0)

        return np.array(preds)

    # ===============================
    # Temporal CNN
    # ===============================
    def tcnn_predict(self, train, test):

        series = train.values.reshape(-1, 1)
        window = 20

        X = []
        for i in range(len(series) - window):
            X.append(series[i:i+window])

        X = torch.tensor(np.array(X), dtype=torch.float32).permute(0, 2, 1)

        model = TemporalCNN()
        output = model(X)

        preds = output.detach().numpy().flatten()

        return preds[:len(test)]

    # ===============================
    # RUN
    # ===============================
    def run(self):

        returns = self.load_data()
        train, test = self.train_test_split(returns)

        garch_preds = self.garch_predict(train, test)
        lstm_preds = self.lstm_predict(train, test)
        tcnn_preds = self.tcnn_predict(train, test)

        actual = test.values

        min_len = min(len(actual), len(garch_preds), len(lstm_preds), len(tcnn_preds))

        actual = actual[:min_len]
        garch_preds = garch_preds[:min_len]
        lstm_preds = lstm_preds[:min_len]
        tcnn_preds = tcnn_preds[:min_len]

        results = pd.DataFrame({
            "Actual": actual,
            "GARCH": garch_preds,
            "LSTM": lstm_preds,
            "TemporalCNN": tcnn_preds
        })

        return results


if __name__ == "__main__":

    comparator = ModelComparator("AAPL", "2020-01-01")
    df = comparator.run()

    print("\n===== ACTUAL VS PREDICTED =====\n")
    print(df.head())

    df.to_csv("actual_vs_predicted.csv", index=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["Actual"], label="Actual")
    plt.plot(df["GARCH"], label="GARCH")
    plt.plot(df["LSTM"], label="LSTM")
    plt.plot(df["TemporalCNN"], label="Temporal CNN")

    plt.legend()
    plt.title("Actual vs Predicted Returns")
    plt.savefig("actual_vs_predicted_plot.png")
    plt.show()

    print("\nSaved:")
    print(" - actual_vs_predicted.csv")
    print(" - actual_vs_predicted_plot.png")
