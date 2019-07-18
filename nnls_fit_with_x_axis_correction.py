import logging
from typing import Callable, Tuple

import numpy as np

from correction_models import linear_correction, quadratic_correction
from utils import calculate_signal, rsme, interpolate_array, calculate_pseudoinverse, nnls_fit, get_combinations

logger = logging.getLogger(__name__)


def nnls_fit_with_interpolated_library(x_original: np.ndarray,
                                       x_target: np.ndarray,
                                       pure_components: np.ndarray,
                                       signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pure_components_interpolated = interpolate_array(pure_components, x_original, x_target)
    signal_copy = signal.copy()

    # Signal is NaN outside interpolation range
    # Zero these channels so that they don't affect to fit result
    is_nan = np.isnan(sum(pure_components_interpolated))
    signal_copy[is_nan] = 0
    pure_components_interpolated[:, is_nan] = 0

    prediction = nnls_fit(signal_copy, pure_components_interpolated)
    estimate = calculate_signal(prediction, pure_components_interpolated)
    residual = estimate - signal_copy

    return prediction, residual


def solve_with_grid_search(x_original: np.ndarray,
                           signal: np.ndarray,
                           pure_components: np.ndarray,
                           candidates: np.ndarray,
                           correction_model: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyzes given signal with NLLS fit combined with x axis correction. x axis correction is done using given model.
    X axis correction parameters are found using grid search, meaning that all options are tested and best option is
    returned as final solution.

    :param x_original: Nominal x axis without any errors.
    :param signal: Signal that needs to be analyzed.
    :param pure_components: Pure component signals. It is assumed that signal is mixture of these.
    :param candidates: Array of x axis correction parameter candidates to be tested. For each row, there is one
    parameter combination that will be tested.
    :param correction_model: Model used for x axis correction.
    :return: Tuple containing estimated pure component contributions and parameters used to correct x axis.
    """
    min_rsme = float('inf')
    solution = None
    best_parameters = None

    for parameters in candidates:

        x_target = correction_model(x_original, parameters)
        prediction, residual = nnls_fit_with_interpolated_library(x_original, x_target, pure_components, signal)
        rsme_current = rsme(residual)

        if rsme_current < min_rsme:
            min_rsme = rsme_current
            solution = prediction
            best_parameters = parameters

        logger.debug(f'''
        RSS: {rsme_current}
        Parameters: {parameters}
        Prediction: {prediction}
        ''')

    return solution, best_parameters


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


def analyze(x_original: np.ndarray,
            signal: np.ndarray,
            pure_components: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyzes given signal with NLLS fit combined with x axis correction. x axis correction is done using quadratic
    model. Analysis uses grid search to make rough estimates at first and then continues with Gauss-Newton optimization
    method to get final solution.

    :param x_original: Nominal x axis without any errors.
    :param signal: Signal that needs to be analyzed.
    :param pure_components: Pure component signals. It is assumed that signal is mixture of these.
    :return: Tuple containing estimated pure component contributions and parameters used to correct x axis.
    """
    offset_candidates = np.arange(-10, 10, 1)
    slope_candidates = np.arange(-0.1, 0.1, 0.01)
    candidates_array = get_combinations(slope_candidates, offset_candidates)

    # Find rough estimates for slope and offset using grid search
    _, parameters = solve_with_grid_search(x_original,
                                           signal,
                                           pure_components,
                                           candidates=candidates_array,
                                           correction_model=linear_correction)

    # Use grid search output as initial guess for Gauss-Newton method
    init_guess = (0, parameters[0], parameters[1])  # (x^2 , slope, offset)

    # Use Gauss-Newton to calculate final solution
    solution, parameters = solve_with_gauss_newton(x_original,
                                                   signal,
                                                   pure_components,
                                                   initial_parameters=init_guess,
                                                   correction_model=quadratic_correction)
    return solution, parameters
