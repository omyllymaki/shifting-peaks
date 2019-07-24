import logging
from typing import Tuple, Optional

import numpy as np

from solvers.math import nnls_fit_with_interpolated_library


class BaseSolver:
    logger = logging.getLogger(__name__)
    correction_model = None
    x = None
    pure_components = None
    signal = None
    parameters = None
    max_x_deviation = float(np.inf)

    def solve(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fit_with_shifted_axis(self, parameters: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x_target = self.correction_model(self.x, parameters)

        if not self._is_x_within_target_range(x_target):
            return None, None

        prediction, residual = nnls_fit_with_interpolated_library(self.x,
                                                                  x_target,
                                                                  self.pure_components,
                                                                  self.signal)

        return prediction, residual

    def _is_x_within_target_range(self, x):
        if abs(min(x) - min(self.x)) > self.max_x_deviation:
            return False
        if abs(max(x) - max(self.x)) > self.max_x_deviation:
            return False
        return True
