import numpy as np


class RiskMetrics:

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0):
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    @staticmethod
    def max_drawdown(returns: np.ndarray):
        cumulative = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
