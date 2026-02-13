import numpy as np

from src.backtesting.strategy_engine import VolatilityStrategy
from src.evaluation.metrics import RiskMetrics
from src.risk_engine.risk_score import RiskScorer


def test_strategy():
    returns = np.random.normal(0, 0.01, 100)
    predicted_vol = np.random.uniform(0.01, 0.03, 100)

    strategy = VolatilityStrategy(vol_threshold=0.02)
    positions = strategy.generate_positions(predicted_vol)
    strat_returns = strategy.apply_strategy(returns, positions)

    assert len(strat_returns) == 100


def test_metrics():
    returns = np.random.normal(0, 0.01, 252)

    sharpe = RiskMetrics.sharpe_ratio(returns)
    mdd = RiskMetrics.max_drawdown(returns)

    assert isinstance(sharpe, float)
    assert isinstance(mdd, float)


def test_risk_score():
    score = RiskScorer.compute(volatility=0.25, drawdown=-0.2)
    assert 0 <= score <= 100
