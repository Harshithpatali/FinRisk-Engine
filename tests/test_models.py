import torch
import numpy as np
import pandas as pd

from src.models.lstm_model import LSTMVolatility
from src.models.temporal_cnn import TemporalCNN
from src.models.hmm_model import RegimeHMM
from src.models.garch_model import GARCHModel


def test_lstm_forward():
    model = LSTMVolatility()
    x = torch.randn(10, 20, 1)
    out = model(x)
    assert out.shape == (10, 1)


def test_tcnn_forward():
    model = TemporalCNN()
    x = torch.randn(10, 1, 20)
    out = model(x)
    assert out.shape == (10, 1)


def test_hmm():
    returns = np.random.randn(500)
    hmm = RegimeHMM(n_states=2)
    hmm.fit(returns)
    states, probs = hmm.predict(returns)
    assert len(states) == 500


def test_garch():
    returns = pd.Series(np.random.randn(500))
    garch = GARCHModel()
    garch.fit(returns)
    forecast = garch.forecast()
    assert forecast.shape[0] > 0
