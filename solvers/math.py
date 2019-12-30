from typing import Callable

import numpy as np
from numpy.linalg import pinv
from scipy.optimize import nnls


def interpolate_signal(signal: np.ndarray,
                       x_original: np.ndarray,
                       x_target: np.ndarray,
                       left_value: float = np.nan,
                       right_value: float = np.nan) -> np.ndarray:
    return np.interp(x_target, x_original, signal, left=left_value, right=right_value)


def interpolate_array(signal_array: np.ndarray,
                      x_original: np.ndarray,
                      x_target: np.ndarray,
                      left_value: float = np.nan,
                      right_value: float = np.nan) -> np.ndarray:
    return np.array([interpolate_signal(s, x_original, x_target, left_value, right_value)
                     for s in signal_array])


def calculate_signal(contributions: np.ndarray, pure_component_signals: np.ndarray) -> np.ndarray:
    return contributions.T @ pure_component_signals


def rsme(residuals: np.ndarray) -> float:
    return np.sqrt(sum(residuals ** 2) / len(residuals))


def calculate_pseudoinverse(x: np.ndarray) -> np.ndarray:
    return pinv(x.T @ x) @ x.T


def nnls_fit(signal: np.ndarray, pure_component_signals: np.ndarray) -> np.ndarray:
    return nnls(pure_component_signals.T, signal)[0]


def ls_fit(signal: np.ndarray, pure_component_signals: np.ndarray) -> np.ndarray:
    return calculate_pseudoinverse(pure_component_signals.T) @ signal


def calculate_partial_derivatives(x0: np.ndarray, func: Callable, step: float = 10 ** (-6)) -> np.ndarray:
    y0 = func(x0)

    partial_derivatives = []
    for i, parameter in enumerate(x0):
        x = x0.copy()
        x[i] += step
        y = func(x)
        derivative = (y - y0) / step
        partial_derivatives.append(derivative)
    partial_derivatives = np.array(partial_derivatives).T

    return partial_derivatives
