import logging
from typing import Tuple

import numpy as np

from solvers.correction_models import linear_correction, quadratic_correction
from solvers.gauss_newton import solve_with_gauss_newton
from solvers.grid_search import solve_with_grid_search
from solvers.common import get_combinations

logger = logging.getLogger(__name__)


def solve_with_grid_search_gauss_newton(x_original: np.ndarray,
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
