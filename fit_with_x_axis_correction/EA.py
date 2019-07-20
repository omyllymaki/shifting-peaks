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
                  n_population=200,
                  n_survivors=None,
                  n_crossovers=None,
                  n_mutations=None,
                  init_guess=(0, 0, 0),
                  deviations=(0.01, 0.1, 1),
                  n_max_generations=500,
                  rsme_threshold=2,
                  n_no_change_threshold=50,
                  max_x_deviation=None) -> Tuple[np.ndarray, np.ndarray]:
    if not n_survivors:
        n_survivors = n_population // 10
    if not n_crossovers:
        n_crossovers = n_survivors
    if not n_mutations:
        n_mutations = n_population - n_crossovers
    if not max_x_deviation:
        max_x_deviation = (max(x_original) - min(x_original)) // 2

    parameters = []
    for _ in range(n_population):
        candidate = [normal(loc=mean, scale=stdev) for mean, stdev in zip(init_guess, deviations)]
        parameters.append(candidate)
    parameters = np.array(parameters)

    min_rsme = float(np.inf)
    return_solution, return_parameters = None, None
    all_rsme_values = []
    counter = 0

    for round in range(1, n_max_generations + 1):
        counter += 1
        rsme_values = []

        for candidate in parameters:
            x_target = correction_model(x_original, candidate)

            if abs(min(x_target) - min(x_original)) > max_x_deviation:
                rsme_values.append(float(np.inf))
                continue
            if abs(max(x_target) - max(x_original)) > max_x_deviation:
                rsme_values.append(float(np.inf))
                continue

            solution, residual = nnls_fit_with_interpolated_library(x_original, x_target, pure_components, signal)
            rsme_value = rsme(residual)
            rsme_values.append(rsme_value)
            if rsme_value < min_rsme:
                min_rsme = rsme_value
                return_solution = solution
                return_parameters = candidate
                counter = 0
                logger.info(f'Solution updated: Round {round}, RSME {min_rsme}')

        all_rsme_values.append(rsme_values)

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

        sorted_indices = np.argsort(rsme_values)
        survivors = parameters[sorted_indices][:n_survivors]
        best_parameters = survivors[0]

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
