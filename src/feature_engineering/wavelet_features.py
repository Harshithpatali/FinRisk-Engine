import numpy as np
import pandas as pd
import pywt


class WaveletFeatures:
    """
    Haar wavelet decomposition for multi-scale volatility analysis.
    """

    @staticmethod
    def haar_decomposition(
        series: pd.Series,
        level: int = 2
    ) -> dict:
        """
        Perform multi-level Haar decomposition.
        Returns approximation and detail coefficients.
        """

        coeffs = pywt.wavedec(series.values, "haar", level=level)

        result = {}

        for i, coeff in enumerate(coeffs):
            result[f"level_{i}"] = coeff

        return result

    @staticmethod
    def wavelet_energy(coeffs: dict) -> dict:
        """
        Compute energy at each decomposition level.
        """

        energy = {}
        for level, values in coeffs.items():
            energy[level] = np.sum(np.square(values))

        return energy
