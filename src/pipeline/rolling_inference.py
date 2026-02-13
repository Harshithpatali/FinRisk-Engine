import numpy as np
import torch
import torch.nn as nn

from src.models.lstm_model import LSTMVolatility


class RollingVolatilityForecaster:

    def __init__(self, window=20, epochs=3):
        self.window = window
        self.epochs = epochs

    def walk_forward(self, series):

        preds = []
        actual = []

        for i in range(self.window, len(series)-1):

            train = series[:i]
            test_value = series[i]

            X = []
            y = []

            for j in range(len(train) - self.window):
                X.append(train[j:j+self.window])
                y.append(train[j+self.window])

            X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
            y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

            model = LSTMVolatility()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            for _ in range(self.epochs):
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            model.eval()

            latest_window = train[-self.window:]
            x_input = torch.tensor(
                latest_window.reshape(1, self.window, 1),
                dtype=torch.float32
            )

            with torch.no_grad():
                pred = model(x_input).item()

            preds.append(pred)
            actual.append(test_value)

        return np.array(preds), np.array(actual)
