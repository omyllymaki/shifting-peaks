import itertools

import numpy as np
from numpy.linalg import pinv

from analysis import nnls_fit
from utils import estimate_signal, rss, interpolate_array


def nnls_fit_with_interpolation(x_original, x_target, library, signal):
    library_interpolated = interpolate_array(library, x_original, x_target)
    signal_copy = signal.copy()

    is_nan = np.isnan(sum(library_interpolated))
    signal_copy[is_nan] = 0
    library_interpolated[:, is_nan] = 0

    prediction = nnls_fit(signal_copy, library_interpolated)
    estimate = estimate_signal(prediction, library_interpolated)
    residual = estimate - signal_copy

    return prediction, residual


def get_x_axis_for_interpolation(x_original, parameters, x0):
    return (parameters[0] + 1) * (x_original - x0) + parameters[1] + x0


def nnls_fit_with_x_axis_correction_trial_and_error(x_original: np.ndarray,
                                                    signal: np.ndarray,
                                                    library: np.ndarray,
                                                    offset_candidates: np.ndarray,
                                                    slope_candidates: np.ndarray,
                                                    x0: float = 0):
    parameters_grid = list(itertools.product(slope_candidates, offset_candidates))
    min_residual_sum = float('inf')
    solution = None
    best_parameters = None

    for parameters in parameters_grid:

        x_target = get_x_axis_for_interpolation(x_original, parameters, x0)
        prediction, residual = nnls_fit_with_interpolation(x_original, x_target, library, signal)
        residual_sum = rss(residual)

        if residual_sum < min_residual_sum:
            min_residual_sum = residual_sum
            solution = prediction
            best_parameters = parameters

    return solution, best_parameters


def nnls_fit_with_x_axis_correction_gauss_newton(x_original: np.ndarray,
                                                 signal: np.ndarray,
                                                 library: np.ndarray,
                                                 x0: float = 0,
                                                 max_iter: int = 30,
                                                 initial_parameters: tuple = (0,0)):
    d_offset = 10 ** (-6)
    d_slope = 10 ** (-6)
    parameters = initial_parameters
    prediction = None

    for k in range(max_iter):
        test_parameters = parameters
        x_target = get_x_axis_for_interpolation(x_original, test_parameters, x0)
        prediction, residual = nnls_fit_with_interpolation(x_original, x_target, library, signal)

        test_parameters = parameters + np.array([d_slope, 0])
        x_target = get_x_axis_for_interpolation(x_original, test_parameters, x0)
        _, residual_slope_changed = nnls_fit_with_interpolation(x_original, x_target, library, signal)

        test_parameters = parameters + np.array([0, d_offset])
        x_target = get_x_axis_for_interpolation(x_original, test_parameters, x0)
        _, residual_offset_changed = nnls_fit_with_interpolation(x_original, x_target, library, signal)

        slope_gradient = (residual_slope_changed - residual) / d_slope
        offset_gradient = (residual_offset_changed - residual) / d_offset
        J = np.array([slope_gradient, offset_gradient]).T

        J_pseudoinverse = pinv(J.T @ J) @ J.T
        parameter_update = J_pseudoinverse @ residual
        parameters = parameters - parameter_update

    return prediction, parameters
