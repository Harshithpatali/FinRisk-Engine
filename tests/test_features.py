import pandas as pd
import numpy as np

from src.feature_engineering.volatility_features import VolatilityFeatures
from src.feature_engineering.drawdown_features import DrawdownFeatures
from src.feature_engineering.wavelet_features import WaveletFeatures


def test_rolling_volatility():
    data = pd.DataFrame(
        {"AAPL": np.random.normal(0, 0.01, 200)}
    )

    vol = VolatilityFeatures.rolling_volatility(data, window=21)

    assert vol.shape[0] > 0


def test_drawdown():
    prices = pd.DataFrame(
        {"AAPL": np.linspace(100, 150, 200)}
    )

    dd = DrawdownFeatures.compute_drawdown(prices)

    assert dd.min().min() <= 0


def test_wavelet():
    series = pd.Series(np.random.randn(256))

    coeffs = WaveletFeatures.haar_decomposition(series, level=2)
    energy = WaveletFeatures.wavelet_energy(coeffs)

    assert isinstance(coeffs, dict)
    assert isinstance(energy, dict)
    assert len(coeffs) > 0
