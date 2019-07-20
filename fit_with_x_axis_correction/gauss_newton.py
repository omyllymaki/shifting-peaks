from typing import Callable, Tuple

import numpy as np

from fit_with_x_axis_correction.common import nnls_fit_with_interpolated_library, rsme, calculate_pseudoinverse
import logging

logger = logging.getLogger(__name__)


def solve_with_gauss_newton(x_original: np.ndarray,
                            signal: np.ndarray,
                            pure_components: np.ndarray,
                            correction_model: Callable,
                            min_iter: int = 10,
                            max_iter: int = 100,
                            initial_parameters: tuple = (0, 0),
                            relative_tolerance: float = 10 ** (-5)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyzes given signal with NLLS fit combined with x axis correction. x axis correction is done using given model.
    X axis correction parameters are found using Gauss-Newton optimization method.

    :param x_original: Nominal x axis without any errors.
    :param signal: Signal that needs to be analyzed.
    :param pure_components: Pure component signals. It is assumed that signal is mixture of these.
    :param correction_model: Model used for x axis correction.
    :param min_iter: Minimum number of iterations.
    :param max_iter: Maximum number of iterations.
    :param initial_parameters: Initial guess for correction parameters in correction model.
    :param relative_tolerance: Tolerance argument for termination of optimization. Optimization is terminated if
    relative difference of RSS between current and previous iteration is smaller than this value.
    :return: Tuple containing estimated pure component contributions and parameters used to correct x axis.
    """
    step = 10 ** (-6)
    parameters = np.array(initial_parameters)
    prediction = None
    rsme_previous = float(np.inf)

    for k in range(1, max_iter + 1):

        x_target = correction_model(x_original, parameters)
        prediction, residual = nnls_fit_with_interpolated_library(x_original, x_target, pure_components, signal)

        jacobian = []
        for i, parameter in enumerate(parameters):
            test_parameters = parameters.copy()
            test_parameters[i] += step
            x_target = correction_model(x_original, test_parameters)
            _, residual_after_step = nnls_fit_with_interpolated_library(x_original, x_target, pure_components, signal)
            derivative = (residual_after_step - residual) / step
            jacobian.append(derivative)
        jacobian = np.array(jacobian).T

        rsme_current = rsme(residual)
        if k >= min_iter and (rsme_previous - rsme_current) / rsme_current < relative_tolerance:
            logger.info(f'Fit converged: iteration {k}, RSME {rsme_current}')
            break

        inverse_jacobian = calculate_pseudoinverse(jacobian)
        parameter_update = inverse_jacobian @ residual
        parameters = parameters - parameter_update

        rsme_previous = rsme_current

        logger.debug(f'''
        Iteration: {k}
        RSME: {rsme_current}
        Parameters: {parameters}
        Prediction: {prediction}
        ''')

        if k == max_iter - 1:
            logger.warning("Maximum number of iterations reached. Fit didn't converge.")

    return prediction, parameters
