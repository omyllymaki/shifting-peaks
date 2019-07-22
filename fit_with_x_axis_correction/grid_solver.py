from typing import Callable, Tuple

import numpy as np

from fit_with_x_axis_correction.common import nnls_fit_with_interpolated_library, rsme
from fit_with_x_axis_correction.solver_interface import SolverInterface


class GridSolver(SolverInterface):
    def __init__(self,
                 x: np.ndarray,
                 pure_components: np.ndarray,
                 candidates: np.ndarray,
                 correction_model: Callable):
        self.x = x
        self.pure_components = pure_components
        self.candidates = candidates
        self.correction_model = correction_model

    def solve(self, signal: np.ndarray, *args) -> Tuple[np.ndarray, np.ndarray]:

        min_rsme = float('inf')
        solution = None
        best_parameters = None

        for parameters in self.candidates:

            x_target = self.correction_model(self.x, parameters)
            prediction, residual = nnls_fit_with_interpolated_library(self.x, x_target, self.pure_components, signal)
            rsme_current = rsme(residual)

            if rsme_current < min_rsme:
                min_rsme = rsme_current
                solution = prediction
                best_parameters = parameters

            self.logger.debug(f'''
               RSS: {rsme_current}
               Parameters: {parameters}
               Prediction: {prediction}
               ''')

        return solution, best_parameters
