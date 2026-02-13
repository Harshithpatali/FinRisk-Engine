import numpy as np


class VolatilityStrategy:
    """
    Risk-adjusted exposure strategy.
    """

    def __init__(self, vol_threshold: float = 0.02):
        self.vol_threshold = vol_threshold

    def generate_positions(self, predicted_vol: np.ndarray):
        """
        Position sizing rule.
        """
        positions = np.where(predicted_vol > self.vol_threshold, 0.5, 1.0)
        return positions

    def apply_strategy(self, returns: np.ndarray, positions: np.ndarray):
        """
        Compute strategy returns.
        """
        strategy_returns = returns * positions
        return strategy_returns
