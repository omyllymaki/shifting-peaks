import logging

import matplotlib.pyplot as plt
import numpy as np
from constants import PATH_PURE_COMPONENTS, PATH_MIXTURES, X
from file_io import load_pickle_file
from grid_gn_solver import GridGNSolver
from scipy.optimize import nnls

from solvers.gn_solver import GNSolver

logging.basicConfig(level=logging.WARNING)


def quadratic_correction(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] * x ** 2 + (coefficients[1] + 1) * x + coefficients[2]


def nnls_fit(signal: np.ndarray, pure_component_signals: np.ndarray) -> np.ndarray:
    return nnls(pure_component_signals.T, signal)[0]


def main():
    pure_components = load_pickle_file(PATH_PURE_COMPONENTS)
    mixtures_data = load_pickle_file(PATH_MIXTURES)

    solver = GNSolver(X, pure_components, quadratic_correction, fit_function=nnls_fit)

    results = []
    for i, sample in enumerate(mixtures_data, 1):
        print(f'{i}/{len(mixtures_data)}')
        true_contributions = sample['contributions']
        signal = sample['signal']
        result = [true_contributions[-1],
                  nnls_fit(signal, pure_components)[-1],
                  solver.solve(signal)[0][-1],
                  ]
        results.append(result)
    results = np.array(results)

    mean_abs_error_no_correction = np.mean(abs(results[:, 1] - results[:, 0]))
    mean_abs_error_gauss_newton = np.mean(abs(results[:, 2] - results[:, 0]))

    print("Mean absolute errors:")
    print(f'No correction: {mean_abs_error_no_correction}')
    print(f'With correction: {mean_abs_error_gauss_newton}')

    _ = plt.figure()
    _ = plt.subplot(2, 1, 1)
    _ = plt.plot(results[:, 0], 'b-', results[:, 1], 'r-')
    plt.grid()
    _ = plt.title('Without X axis correction')
    _ = plt.legend(['True', 'Predicted'])
    plt.xlabel("Sample")

    plt.subplot(2, 1, 2)
    _ = plt.plot(results[:, 0], 'b-', results[:, 2], 'r-')
    plt.grid()
    _ = plt.title('With X axis correction')
    _ = plt.legend(['True', 'Predicted'])
    plt.xlabel("Sample")
    plt.show()


if __name__ == "__main__":
    main()
