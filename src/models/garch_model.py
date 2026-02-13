import pandas as pd
import numpy as np
from arch import arch_model


class GARCHModel:
    """
    Production-safe GARCH(1,1) volatility model.
    """

    def __init__(self, p: int = 1, q: int = 1, scale_factor: float = 100.0):
        self.p = p
        self.q = q
        self.scale_factor = scale_factor
        self.model = None
        self.fitted = None

    def fit(self, returns: pd.Series):
        scaled_returns = returns * self.scale_factor

        self.model = arch_model(
            scaled_returns,
            vol="Garch",
            p=self.p,
            q=self.q,
            dist="normal",
            rescale=False,
        )

        self.fitted = self.model.fit(disp="off")
        return self.fitted

    def forecast(self, horizon: int = 5):
        forecast = self.fitted.forecast(horizon=horizon)
        variance = forecast.variance.iloc[-1]

        # Reverse scaling effect
        variance = variance / (self.scale_factor ** 2)

        return variance
