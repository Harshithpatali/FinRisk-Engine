import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    """
    1D CNN for time series volatility prediction.
    """

    def __init__(self, input_channels=1):
        super(TemporalCNN, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x
