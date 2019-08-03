from typing import Callable, Tuple

import numpy as np

from solvers.base_solver import BaseSolver
from solvers.math import rsme, calculate_pseudoinverse, calculate_partial_derivatives


class GNSolver(BaseSolver):
    def __init__(self,
                 x: np.ndarray,
                 pure_components: np.ndarray,
                 correction_model: Callable,
                 min_iter: int = 10,
                 max_iter: int = 100,
                 relative_tolerance: float = 10 ** (-5)
                 ):
        self.x = x
        self.pure_components = pure_components
        self.correction_model = correction_model
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.relative_tolerance = relative_tolerance

        self.iteration_round = None
        self.rsme_previous = None
        self.rsme_current = None

    def solve(self,
              signal,
              initial_parameters: Tuple = (0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:

        prediction = None
        self.rsme_previous = float(np.inf)
        parameters = np.array(initial_parameters)
        self.signal = signal

        for self.iteration_round in range(1, self.max_iter + 1):

            prediction, residual = self.fit_with_shifted_axis(parameters)
            self.rsme_current = rsme(residual)

            self.logger.debug(f'''
                Iteration: {self.iteration_round}
                RSME: {self.rsme_current}
                Parameters: {self.parameters}
                Prediction: {prediction}
                ''')

            if self._is_termination_condition_filled():
                break

            parameters = self._update_parameters(parameters, residual)
            self.rsme_previous = self.rsme_current

        return prediction, parameters

    def _update_parameters(self, parameters, residual):
        func = lambda x: self.fit_with_shifted_axis(x)[1]       # returns only residual
        jacobian = calculate_partial_derivatives(parameters, func)
        inverse_jacobian = calculate_pseudoinverse(jacobian)
        parameter_update = inverse_jacobian @ residual
        parameters = parameters - parameter_update
        return parameters

    def _is_termination_condition_filled(self):
        rsme_relative_change = (self.rsme_previous - self.rsme_current) / self.rsme_current
        if self.iteration_round >= self.min_iter and rsme_relative_change < self.relative_tolerance:
            self.logger.info(f'Fit converged: iteration {self.iteration_round}, RSME {self.rsme_current}')
            return True
        if self.iteration_round == self.max_iter - 1:
            self.logger.warning("Maximum number of iterations reached. Fit didn't converge.")
            return True
        return False
