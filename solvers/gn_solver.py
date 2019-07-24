from typing import Callable, Tuple

import numpy as np

from solvers.base_solver import BaseSolver
from solvers.math import rsme, calculate_pseudoinverse


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

    def solve(self,
              signal,
              initial_parameters: Tuple = (0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        step = 10 ** (-6)
        prediction = None
        rsme_previous = float(np.inf)
        parameters = np.array(initial_parameters)
        self.signal = signal

        for k in range(1, self.max_iter + 1):

            prediction, residual = self.fit_with_shifted_axis(parameters)

            jacobian = []
            for i, parameter in enumerate(parameters):
                test_parameters = parameters.copy()
                test_parameters[i] += step
                _, residual_after_step = self.fit_with_shifted_axis(test_parameters)
                derivative = (residual_after_step - residual) / step
                jacobian.append(derivative)
            jacobian = np.array(jacobian).T

            rsme_current = rsme(residual)
            if k >= self.min_iter and (rsme_previous - rsme_current) / rsme_current < self.relative_tolerance:
                self.logger.info(f'Fit converged: iteration {k}, RSME {rsme_current}')
                break

            inverse_jacobian = calculate_pseudoinverse(jacobian)
            parameter_update = inverse_jacobian @ residual
            parameters = parameters - parameter_update

            rsme_previous = rsme_current

            self.logger.debug(f'''
                Iteration: {k}
                RSME: {rsme_current}
                Parameters: {self.parameters}
                Prediction: {prediction}
                ''')

            if k == self.max_iter - 1:
                self.logger.warning("Maximum number of iterations reached. Fit didn't converge.")

        return prediction, parameters
