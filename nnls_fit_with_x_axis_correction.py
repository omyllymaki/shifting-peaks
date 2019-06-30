import logging
from typing import Callable, Tuple

import numpy as np

from correction_models import linear_correction, quadratic_correction
from utils import calculate_signal, rss, interpolate_array, calculate_pseudoinverse, nnls_fit, get_combinations

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
    min_rss = float('inf')
    solution = None
    best_parameters = None

    for parameters in candidates:

        x_target = correction_model(x_original, parameters)
        prediction, residual = nnls_fit_with_interpolated_library(x_original, x_target, pure_components, signal)
        rss_current = rss(residual)

        if rss_current < min_rss:
            min_rss = rss_current
            solution = prediction
            best_parameters = parameters

        logger.debug(f'''
        RSS: {rss_current}
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
    step = 10 ** (-6)
    parameters = np.array(initial_parameters)
    prediction = None
    rss_previous = float(np.inf)

    for k in range(1, max_iter + 1):

        x_target = correction_model(x_original, parameters)
        prediction, residual = nnls_fit_with_interpolated_library(x_original, x_target, pure_components, signal)

        jacobian = []
        for i, parameter in enumerate(parameters):
            test_parameters = parameters.copy()
            test_parameters[i] += step
            x_target = correction_model(x_original, test_parameters)
            _, residual_after_step = nnls_fit_with_interpolated_library(x_original, x_target, pure_components, signal)
            gradient = (residual_after_step - residual) / step
            jacobian.append(gradient)
        jacobian = np.array(jacobian).T

        rss_current = rss(residual)
        if k >= min_iter and (rss_previous - rss_current) / rss_current < relative_tolerance:
            logger.info(f'Fit converged: iteration {k}, RSS {rss_current}')
            break

        inverse_jacobian = calculate_pseudoinverse(jacobian)
        parameter_update = inverse_jacobian @ residual
        parameters = parameters - parameter_update

        rss_previous = rss_current

        logger.debug(f'''
        Iteration: {k}
        RSS: {rss_current}
        Parameters: {parameters}
        Prediction: {prediction}
        ''')

        if k == max_iter - 1:
            logger.warning("Maximum number of iterations reached. Fit didn't converge.")

    return prediction, parameters


def analysis(x_original: np.ndarray,
             signal: np.ndarray,
             pure_components: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
