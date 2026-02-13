import numpy as np
import pandas as pd
from typing import Callable


class WalkForwardValidator:
    """
    Expanding window walk-forward validation.
    """

    def __init__(self, train_size: int, test_size: int, step_size: int):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def run(
        self,
        data: pd.Series,
        model_fit_function: Callable,
        model_predict_function: Callable,
    ):
        """
        Walk-forward evaluation loop.
        """

        predictions = []
        actuals = []

        start = 0

        while start + self.train_size + self.test_size <= len(data):

            train = data[start : start + self.train_size]
            test = data[
                start + self.train_size :
                start + self.train_size + self.test_size
            ]

            model = model_fit_function(train)
            preds = model_predict_function(model, test)

            predictions.extend(preds)
            actuals.extend(test.values)

            start += self.step_size

        return np.array(predictions), np.array(actuals)
