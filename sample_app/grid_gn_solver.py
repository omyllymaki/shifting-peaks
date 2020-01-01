import itertools
from typing import Tuple

import numpy as np
from scipy.optimize import nnls

import os
import sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from solvers.gn_solver import GNSolver
from solvers.grid_solver import GridSolver


class GridGNSolver():
    def __init__(self,
                 x: np.ndarray,
                 pure_components: np.ndarray):
        offset_candidates = np.arange(-10, 10, 2)
        slope_candidates = np.arange(-0.1, 0.1, 0.02)
        candidates_array = self.get_combinations(slope_candidates, offset_candidates)

        self.grid_solver = GridSolver(x=x,
                                      pure_components=pure_components,
                                      candidates=candidates_array,
                                      correction_model=self.linear_correction,
                                      fit_function=self.nnls_fit)
        self.gn_solver = GNSolver(x=x,
                                  pure_components=pure_components,
                                  correction_model=self.quadratic_correction,
                                  fit_function=self.nnls_fit)

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

    @staticmethod
    def linear_correction(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        return (coefficients[0] + 1) * x + coefficients[1]

    @staticmethod
    def quadratic_correction(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        return coefficients[0] * x ** 2 + (coefficients[1] + 1) * x + coefficients[2]

    @staticmethod
    def nnls_fit(signal: np.ndarray, pure_component_signals: np.ndarray) -> np.ndarray:
        return nnls(pure_component_signals.T, signal)[0]
