import logging

import matplotlib.pyplot as plt
import numpy as np

from constants import PATH_PURE_COMPONENTS, PATH_MIXTURES, X
from file_io import load_pickle_file
from solvers.correction_models import quadratic_correction
from solvers.gn_solver import GNSolver
from solvers.math import nnls_fit

logging.basicConfig(level=logging.WARNING)


def main():
    pure_components = load_pickle_file(PATH_PURE_COMPONENTS)
    mixtures_data = load_pickle_file(PATH_MIXTURES)

    gn_solver = GNSolver(X, pure_components, correction_model=quadratic_correction, fit_function=nnls_fit)

    results = []
    for i, sample in enumerate(mixtures_data, 1):
        print(f'{i}/{len(mixtures_data)}')
        true_contributions = sample['contributions']
        signal = sample['signal']
        result = [true_contributions[-1],
                  nnls_fit(signal, pure_components)[-1],
                  gn_solver.solve(signal)[0][-1],
                  ]
        results.append(result)
    results = np.array(results)

    mean_abs_error_no_correction = np.mean(abs(results[:, 1] - results[:, 0]))
    mean_abs_error_gauss_newton = np.mean(abs(results[:, 2] - results[:, 0]))

    print("Mean absolute errors:")
    print(f'No correction: {mean_abs_error_no_correction}')
    print(f'Gauss-Newton: {mean_abs_error_gauss_newton}')

    _ = plt.figure()
    _ = plt.subplot(2, 1, 1)
    _ = plt.plot(results[:, 0], 'b-', results[:, 1], 'r-', results[:, 1] - results[:, 0], 'g-')
    plt.grid()
    _ = plt.title('NNLS without X axis correction')
    _ = plt.legend(['True', 'Predicted', 'Error'])

    plt.subplot(2, 1, 2)
    _ = plt.plot(results[:, 0], 'b-', results[:, 2], 'r-', results[:, 2] - results[:, 0], 'g-')
    plt.grid()
    _ = plt.title('NNLS with Gauss-Newton correction')
    _ = plt.legend(['True', 'Predicted', 'Error'])
    plt.show()


if __name__ == "__main__":
    main()
