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
        self.correction_model = correction_model
        self.fit_function = fit_function

    def solve(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
