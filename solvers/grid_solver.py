from typing import Callable, Tuple

import numpy as np

from solvers.base_solver import BaseSolver
from solvers.math import rsme, ls_fit


class GridSolver(BaseSolver):
    def __init__(self,
                 x: np.ndarray,
                 pure_components: np.ndarray,
                 candidates: np.ndarray,
                 correction_model: Callable,
                 fit_function: Callable = ls_fit,
                 ):
        self.x = x
        self.pure_components = pure_components
        self.candidates = candidates

        super().__init__(correction_model, fit_function)

    def solve(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        self.signal = signal
        min_rsme = float('inf')
        solution = None
        best_parameters = None

        for parameters in self.candidates:

            prediction, residual = self.fit_with_shifted_axis(parameters)
            rsme_current = rsme(residual)

            if rsme_current < min_rsme:
                min_rsme = rsme_current
                solution = prediction
                best_parameters = parameters

            self.logger.debug(f'''
               RSS: {rsme_current}
               Parameters: {self.parameters}
               Prediction: {prediction}
               ''')

        return solution, best_parameters
