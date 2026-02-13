import numpy as np


class RiskScorer:
    """
    Composite risk score between 0 and 100.
    """

    @staticmethod
    def compute(volatility: float, drawdown: float):
        vol_component = min(volatility * 100, 50)
        dd_component = min(abs(drawdown) * 100, 50)

        score = vol_component + dd_component

        return min(score, 100)
