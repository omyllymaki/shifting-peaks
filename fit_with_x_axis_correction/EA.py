import logging
import random
from typing import Callable, Tuple

import numpy as np
from numpy.random import normal

from fit_with_x_axis_correction.common import nnls_fit_with_interpolated_library, rsme
from fit_with_x_axis_correction.correction_models import quadratic_correction

logger = logging.getLogger(__name__)


def solve_with_EA(x_original: np.ndarray,
                  signal: np.ndarray,
                  pure_components: np.ndarray,
                  correction_model: Callable = quadratic_correction,
                  n_population: int = 200,
                  n_survivors: int = None,
                  n_crossovers: int = None,
                  n_mutations: int = None,
                  init_guess: Tuple[float] = (0, 0, 0),
                  deviations: Tuple[float] = (0.01, 0.1, 1),
                  n_max_generations: int = 500,
                  rsme_threshold: float = 2,
                  n_no_change_threshold: int = 50,
                  max_x_deviation: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyzes given signal with NLLS fit combined with x axis correction. x axis correction is done using given model.
    x axis correction parameters are found using evolutionary algorithm.


    :param x_original:
    :param signal:
    :param pure_components:
    :param correction_model:
    :param n_population:
    :param n_survivors:
    :param n_crossovers:
    :param n_mutations:
    :param init_guess:
    :param deviations:
    :param n_max_generations:
    :param rsme_threshold:
    :param n_no_change_threshold:
    :param max_x_deviation:
    :return:
    """

    # Initialize arguments
    if not n_survivors:
        n_survivors = n_population // 10
    if not n_crossovers:
        n_crossovers = n_survivors
    if not n_mutations:
        n_mutations = n_population - n_crossovers
    if not max_x_deviation:
        max_x_deviation = (max(x_original) - min(x_original)) // 2

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
    for round in range(1, n_max_generations + 1):
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
        if round == n_max_generations:
            logger.warning(f"Maximum number of generations reached. Iteration terminated at round {round}.")
            break

        # Take some best parameters (survivors)
        sorted_indices = np.argsort(rsme_values)
        survivors = parameters[sorted_indices][:n_survivors]
        best_parameters = survivors[0]

        # Generate new parameter combinations
        parameters = []
        parameters.append(best_parameters)
        for _ in range(n_crossovers):
            candidate = [random.choice(column) for column in survivors.T]
            parameters.append(candidate)
        for _ in range(n_mutations // 3):
            candidate = [normal(loc=mean, scale=stdev) for mean, stdev in zip(best_parameters, deviations)]
            parameters.append(candidate)
        for _ in range(n_mutations // 3):
            candidate = [normal(loc=mean, scale=round * stdev) for mean, stdev in zip(best_parameters, deviations)]
            parameters.append(candidate)
        for _ in range(n_mutations // 3):
            candidate = [normal(loc=mean, scale=stdev / round) for mean, stdev in zip(best_parameters, deviations)]
            parameters.append(candidate)
        for _ in range(n_population - len(parameters)):
            candidate = [normal(loc=mean, scale=stdev) for mean, stdev in zip(init_guess, deviations)]
            parameters.append(candidate)
        parameters = np.array(parameters)

    return return_solution, return_parameters
