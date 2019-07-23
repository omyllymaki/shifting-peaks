from typing import Callable, Tuple

import numpy as np
from numpy.random import normal

from solvers.base_solver import BaseSolver
from solvers.common import nnls_fit_with_interpolated_library, rsme


class EASolver(BaseSolver):
    def __init__(self,
                 x: np.ndarray,
                 pure_components: np.ndarray,
                 correction_model: Callable,
                 n_population: int = 200,
                 scaled_proportion: float = 0.7,
                 init_guess: Tuple[float] = (0, 0, 0),
                 deviations: Tuple[float] = (0.01, 0.1, 1),
                 deviations_scaling: np.ndarray = None,
                 n_max_generations: int = 500,
                 rsme_threshold: float = 1,
                 n_no_change_threshold: int = 50,
                 max_x_deviation: float = None
                 ):
        self.x = x
        self.pure_components = pure_components
        self.correction_model = correction_model
        self.n_population = n_population
        self.scaled_proportion = scaled_proportion
        self.init_guess = init_guess
        self.deviations = deviations
        self.n_max_generations = n_max_generations
        self.rsme_threshold = rsme_threshold
        self.n_no_change_threshold = n_no_change_threshold
        self.max_x_deviation = max_x_deviation
        self.deviations_scaling = deviations_scaling

        if not max_x_deviation:
            self.max_x_deviation = (max(x) - min(x)) // 2
        if not deviations_scaling:
            self.deviations_scaling = np.array([1 / k for k in range(1, n_max_generations + 1)])

    def solve(self, signal) -> Tuple[np.ndarray, np.ndarray]:

        self.signal = signal

        # Initialize parameter candidates
        parameters = []
        for _ in range(self.n_population):
            candidate = [normal(loc=mean, scale=stdev) for mean, stdev
                         in zip(self.init_guess, self.deviations)]
            parameters.append(candidate)
        parameters = np.array(parameters)

        # Initialize optimization
        min_rsme = float(np.inf)
        return_solution, return_parameters = None, None
        counter = 0

        # Search for best parameter combination
        for round in range(self.n_max_generations):
            counter += 1
            rsme_values = []

            for candidate in parameters:
                solution, residual = self.fit_with_shifted_axis(candidate)
                if residual is None:
                    rsme_value = float(np.inf)
                else:
                    rsme_value = rsme(residual)
                rsme_values.append(rsme_value)

                # Update solution if RSME is smaller than current minimum
                if rsme_value < min_rsme:
                    min_rsme = rsme_value
                    return_solution = solution
                    return_parameters = candidate
                    counter = 0
                    self.logger.info(f'Solution updated: Round {round}, RSME {min_rsme}')

            # Check termination conditions
            if min_rsme < self.rsme_threshold:
                self.logger.info(f'Target RSME reached. Iteration terminated at round {round}.')
                break
            if counter > self.n_no_change_threshold:
                self.logger.info(
                    f"RSME didn't change in last {self.n_no_change_threshold} rounds. Iteration terminated at round {round}.")
                break
            if round == self.n_max_generations - 1:
                self.logger.warning(f"Maximum number of generations reached. Iteration terminated at round {round}.")
                break

            # Find best parameter combination
            index_lowest_rsme = np.argmin(rsme_values)
            best_parameters = parameters[index_lowest_rsme]

            # Generate new parameter combinations
            parameters = []
            parameters.append(best_parameters)
            scale = self.deviations_scaling[round]
            n_scaled = int(self.scaled_proportion * self.n_population)
            for _ in range(n_scaled):
                candidate = [normal(loc=mean, scale=scale * stdev) for mean, stdev
                             in zip(best_parameters, self.deviations)]
                parameters.append(candidate)
            for _ in range(self.n_population - len(parameters)):
                candidate = [normal(loc=mean, scale=stdev) for mean, stdev
                             in zip(best_parameters, self.deviations)]
                parameters.append(candidate)
            parameters = np.array(parameters)

        return return_solution, return_parameters
