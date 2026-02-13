import numpy as np
from hmmlearn.hmm import GaussianHMM


class RegimeHMM:
    """
    Hidden Markov Model for market regime detection.
    """

    def __init__(self, n_states=2):
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)

    def fit(self, returns: np.ndarray):
        returns = returns.reshape(-1, 1)
        self.model.fit(returns)

    def predict(self, returns: np.ndarray):
        returns = returns.reshape(-1, 1)
        states = self.model.predict(returns)
        probs = self.model.predict_proba(returns)
        return states, probs
