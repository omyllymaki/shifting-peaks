import itertools
from typing import Callable

import numpy as np

from analysis import nnls_fit
from correction_models import linear_correction
from utils import calculate_signal, rss, interpolate_array, calculate_pseudoinverse


def nnls_fit_with_interpolated_library(x_original, x_target, library, signal):
    library_interpolated = interpolate_array(library, x_original, x_target)
    signal_copy = signal.copy()

    is_nan = np.isnan(sum(library_interpolated))
    signal_copy[is_nan] = 0
    library_interpolated[:, is_nan] = 0

    prediction = nnls_fit(signal_copy, library_interpolated)
    estimate = calculate_signal(prediction, library_interpolated)
    residual = estimate - signal_copy

    return prediction, residual


def solve_with_grid_search(x_original: np.ndarray,
                           signal: np.ndarray,
                           library: np.ndarray,
                           candidates: np.ndarray,
                           correction_model=linear_correction):
    min_residual_sum = float('inf')
    solution = None
    best_parameters = None

    for parameters in candidates:

        x_target = correction_model(x_original, parameters)
        prediction, residual = nnls_fit_with_interpolated_library(x_original, x_target, library, signal)
        residual_sum = rss(residual)

        if residual_sum < min_residual_sum:
            min_residual_sum = residual_sum
            solution = prediction
            best_parameters = parameters

    return solution, best_parameters


def solve_with_gauss_newton(x_original: np.ndarray,
                            signal: np.ndarray,
                            library: np.ndarray,
                            correction_model: Callable = linear_correction,
                            max_iter: int = 30,
                            initial_parameters: tuple = (0, 0, 0, 0, 0),
                            relative_tolerance=10 ** (-5)):
    step = 10 ** (-6)
    parameters = np.array(initial_parameters)
    prediction = None
    rss_list = []

    for k in range(max_iter):

        x_target = correction_model(x_original, parameters)
        prediction, residual = nnls_fit_with_interpolated_library(x_original, x_target, library, signal)

        jacobian = []
        for i, parameter in enumerate(parameters):
            test_parameters = parameters.copy()
            test_parameters[i] += step
            x_target = correction_model(x_original, test_parameters)
            _, residual_after_step = nnls_fit_with_interpolated_library(x_original, x_target, library, signal)
            gradient = (residual_after_step - residual) / step
            jacobian.append(gradient)
        jacobian = np.array(jacobian).T

        jacobian_pseudoinverse = calculate_pseudoinverse(jacobian)
        parameter_update = jacobian_pseudoinverse @ residual
        parameters = parameters - parameter_update

        rss_list.append(rss(residual))

        if len(rss_list) > 2:
            if abs(rss_list[-2] - rss_list[-1]) / rss_list[-2] < relative_tolerance:
                break

    return prediction, parameters


def analysis(x_original: np.ndarray,
             signal: np.ndarray,
             library: np.ndarray,
             ):
    offset_candidates = np.arange(-10, 10, 1)
    slope_candidates = np.arange(-0.1, 0.1, 0.01)
    candidates = list(itertools.product(slope_candidates, offset_candidates))

    _, parameters = solve_with_grid_search(x_original,
                                           signal,
                                           library,
                                           np.array(candidates))
    solution, parameters = solve_with_gauss_newton(x_original,
                                                   signal,
                                                   library,
                                                   initial_parameters=parameters)
    return solution, parameters
