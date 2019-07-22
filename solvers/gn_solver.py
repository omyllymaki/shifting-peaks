from typing import Callable, Tuple

import numpy as np

from solvers.common import nnls_fit_with_interpolated_library, rsme, calculate_pseudoinverse
from solvers.solver_interface import SolverInterface


class GNSolver(SolverInterface):
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
        parameters = np.array(initial_parameters)
        prediction = None
        rsme_previous = float(np.inf)

        for k in range(1, self.max_iter + 1):

            x_target = self.correction_model(self.x, parameters)
            prediction, residual = nnls_fit_with_interpolated_library(self.x, x_target, self.pure_components, signal)

            jacobian = []
            for i, parameter in enumerate(parameters):
                test_parameters = parameters.copy()
                test_parameters[i] += step
                x_target = self.correction_model(self.x, test_parameters)
                _, residual_after_step = nnls_fit_with_interpolated_library(self.x,
                                                                            x_target,
                                                                            self.pure_components,
                                                                            signal)
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
                Parameters: {parameters}
                Prediction: {prediction}
                ''')

            if k == self.max_iter - 1:
                self.logger.warning("Maximum number of iterations reached. Fit didn't converge.")

        return prediction, parameters
