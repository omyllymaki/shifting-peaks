import numpy as np


def linear_correction(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return (coefficients[0] + 1) * x + coefficients[1]


def quadratic_correction(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] * x ** 2 + (coefficients[1] + 1) * x + coefficients[2]
