import logging
from typing import Tuple, Optional, Callable

import numpy as np

from solvers.math import interpolate_array, calculate_signal, ls_fit


class BaseSolver:
    logger = logging.getLogger(__name__)
    correction_model = None
    x = None
    pure_components = None
    signal = None
    parameters = None
    max_x_deviation = float(np.inf)

    def __init__(self,
                 correction_model: Callable,
                 fit_function: Callable = ls_fit):
        """
        :param correction_model:
        Model for x axis correction.
        (x: np.ndarray, coefficients: np.ndarray) -> x_corrected: np.ndarray
        E.g.
        def linear_correction(x, coefficients):
            return (coefficients[0] + 1) * x + coefficients[1]

        :param fit_function:
        Curve fitting function.
        (signal: np.ndarray, pure_component_signals: np.ndarray) -> contributions: np.ndarray
        E.g.
        def nnls_fit(signal, pure_component_signals):
            return scipy.optimize.nnls(pure_component_signals.T, signal)[0]

        """
        self.correction_model = correction_model
        self.fit_function = fit_function

    def solve(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param signal: Mixture signal of pure components.
        :return: Tuple[Contributions of pure components, fitted x axis correction parameters].
        """
        raise NotImplementedError

    def fit_with_shifted_axis(self, parameters: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x_target = self.correction_model(self.x, parameters)

        if not self._is_x_within_target_range(x_target):
            return None, None

        prediction, residual = self._fit_with_interpolated_library(x_target)

        return prediction, residual

    def _is_x_within_target_range(self, x):
        if abs(min(x) - min(self.x)) > self.max_x_deviation:
            return False
        if abs(max(x) - max(self.x)) > self.max_x_deviation:
            return False
        return True

    def _fit_with_interpolated_library(self, x_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pure_components_interpolated = interpolate_array(self.pure_components, self.x, x_target)
        signal_copy = self.signal.copy()

        # Signal is NaN outside interpolation range
        # Zero these channels so that they don't affect to fit result
        is_nan = np.isnan(sum(pure_components_interpolated))
        signal_copy[is_nan] = 0
        pure_components_interpolated[:, is_nan] = 0

        prediction = self.fit_function(signal_copy, pure_components_interpolated)
        estimate = calculate_signal(prediction, pure_components_interpolated)
        residual = estimate - signal_copy

        return prediction, residual
