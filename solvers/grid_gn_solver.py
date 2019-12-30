import itertools
from typing import Tuple

import numpy as np

from solvers.base_solver import BaseSolver
from solvers.correction_models import linear_correction, quadratic_correction
from solvers.gn_solver import GNSolver
from solvers.grid_solver import GridSolver
from solvers.math import nnls_fit


class GridGNSolver(BaseSolver):
    def __init__(self,
                 x: np.ndarray,
                 pure_components: np.ndarray):
        offset_candidates = np.arange(-10, 10, 1)
        slope_candidates = np.arange(-0.1, 0.1, 0.01)
        candidates_array = self.get_combinations(slope_candidates, offset_candidates)

        self.grid_solver = GridSolver(x=x,
                                      pure_components=pure_components,
                                      candidates=candidates_array,
                                      correction_model=linear_correction,
                                      fit_function=nnls_fit)
        self.gn_solver = GNSolver(x=x,
                                  pure_components=pure_components,
                                  correction_model=quadratic_correction,
                                  fit_function=nnls_fit)

    def solve(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Find rough estimates for slope and offset using grid search
        _, parameters = self.grid_solver.solve(signal)

        # Use grid search output as initial guess for Gauss-Newton method
        # Assume that quadratic term is close to zero
        init_guess = (0, parameters[0], parameters[1])  # (x^2 , slope, offset)

        # Use Gauss-Newton to calculate final solution
        solution, parameters = self.gn_solver.solve(signal=signal, initial_parameters=init_guess)

        return solution, parameters

    @staticmethod
    def get_combinations(*args: np.ndarray) -> np.ndarray:
        return np.array(list(itertools.product(*args)))
