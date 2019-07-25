from typing import Tuple, Callable

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


def nnls_fit_with_interpolated_library(x_original: np.ndarray,
                                       x_target: np.ndarray,
                                       pure_components: np.ndarray,
                                       signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pure_components_interpolated = interpolate_array(pure_components, x_original, x_target)
    signal_copy = signal.copy()

    # Signal is NaN outside interpolation range
    # Zero these channels so that they don't affect to fit result
    is_nan = np.isnan(sum(pure_components_interpolated))
    signal_copy[is_nan] = 0
    pure_components_interpolated[:, is_nan] = 0

    prediction = nnls_fit(signal_copy, pure_components_interpolated)
    estimate = calculate_signal(prediction, pure_components_interpolated)
    residual = estimate - signal_copy

    return prediction, residual


def calculate_gradient(x: np.ndarray, func: Callable, step: float = 10 ** (-6)) -> np.ndarray:
    _, residual = func(x)

    gradient = []
    for i, parameter in enumerate(x):
        xt = x.copy()
        xt[i] += step
        _, residual_after_step = func(xt)
        derivative = (residual_after_step - residual) / step
        gradient.append(derivative)
    gradient = np.array(gradient).T

    return gradient
