import numpy as np


def linear_correction(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p[0] + 1) * x + p[1]


def quadratic_correction(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[0] * x ** 2 + (p[1] + 1) * x + p[2]
