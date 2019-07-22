import logging
from typing import Callable, Tuple

import numpy as np
from numpy.random import normal

from solvers.common import nnls_fit_with_interpolated_library, rsme
from solvers.correction_models import quadratic_correction

logger = logging.getLogger(__name__)


def solve_with_EA(x_original: np.ndarray,
                  signal: np.ndarray,
                  pure_components: np.ndarray,
                  correction_model: Callable = quadratic_correction,
                  n_population: int = 200,
                  scaled_proportion: float = 0.7,
                  init_guess: Tuple[float] = (0, 0, 0),
                  deviations: Tuple[float] = (0.01, 0.1, 1),
                  deviations_scaling: np.ndarray = None,
                  n_max_generations: int = 500,
                  rsme_threshold: float = 1,
                  n_no_change_threshold: int = 50,
                  max_x_deviation: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyzes given signal with NLLS fit combined with x axis correction. x axis correction is done using given model.
    x axis correction parameters are found using evolutionary algorithm.


    """

    # Initialize optional arguments if not given
    if not max_x_deviation:
        max_x_deviation = (max(x_original) - min(x_original)) // 2
    if not deviations_scaling:
        deviations_scaling = np.array([1 / k for k in range(1, n_max_generations + 1)])

    # Initialize parameter candidates
    parameters = []
    for _ in range(n_population):
        candidate = [normal(loc=mean, scale=stdev) for mean, stdev in zip(init_guess, deviations)]
        parameters.append(candidate)
    parameters = np.array(parameters)

    # Initialize optimization
    min_rsme = float(np.inf)
    return_solution, return_parameters = None, None
    counter = 0

    # Search for best parameter combination
    for round in range(n_max_generations):
        counter += 1
        rsme_values = []

        for candidate in parameters:
            x_target = correction_model(x_original, candidate)

            # Check that interpolated x axis is within accepted range
            if abs(min(x_target) - min(x_original)) > max_x_deviation:
                rsme_values.append(float(np.inf))
                continue
            if abs(max(x_target) - max(x_original)) > max_x_deviation:
                rsme_values.append(float(np.inf))
                continue

            solution, residual = nnls_fit_with_interpolated_library(x_original, x_target, pure_components, signal)
            rsme_value = rsme(residual)
            rsme_values.append(rsme_value)

            # Update solution if RSME is smaller than current minimum
            if rsme_value < min_rsme:
                min_rsme = rsme_value
                return_solution = solution
                return_parameters = candidate
                counter = 0
                logger.info(f'Solution updated: Round {round}, RSME {min_rsme}')

        # Check termination conditions
        if min_rsme < rsme_threshold:
            logger.info(f'Target RSME reached. Iteration terminated at round {round}.')
            break
        if counter > n_no_change_threshold:
            logger.info(
                f"RSME didn't change in last {n_no_change_threshold} rounds. Iteration terminated at round {round}.")
            break
        if round == n_max_generations - 1:
            logger.warning(f"Maximum number of generations reached. Iteration terminated at round {round}.")
            break

        # Find best parameter combination
        index_lowest_rsme = np.argmin(rsme_values)
        best_parameters = parameters[index_lowest_rsme]

        # Generate new parameter combinations
        parameters = []
        parameters.append(best_parameters)
        scale = deviations_scaling[round]
        n_scaled = int(scaled_proportion * n_population)
        for _ in range(n_scaled):
            candidate = [normal(loc=mean, scale=scale * stdev) for mean, stdev in zip(best_parameters, deviations)]
            parameters.append(candidate)
        for _ in range(n_population - len(parameters)):
            candidate = [normal(loc=mean, scale=stdev) for mean, stdev in zip(best_parameters, deviations)]
            parameters.append(candidate)
        parameters = np.array(parameters)

    return return_solution, return_parameters
