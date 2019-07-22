import logging
from typing import Callable, Tuple

import numpy as np

from solvers.common import nnls_fit_with_interpolated_library, rsme

logger = logging.getLogger(__name__)


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
