import logging

import matplotlib.pyplot as plt
import numpy as np

from constants import PATH_PURE_COMPONENTS, PATH_MIXTURES, X
from file_io import load_pickle_file
from solvers.correction_models import quadratic_correction
from solvers.ea_solver import EASolver
from solvers.grid_gn_solver import GridGNSolver
from solvers.math import nnls_fit, calculate_signal

logging.basicConfig(level=logging.WARNING)


def main():
    pure_components = load_pickle_file(PATH_PURE_COMPONENTS)
    mixtures_data = load_pickle_file(PATH_MIXTURES)

    grid_gn_solver = GridGNSolver(X, pure_components)
    ea_solver = EASolver(X, pure_components, quadratic_correction)

    results = []
    for i, sample in enumerate(mixtures_data, 1):
        print(f'{i}/{len(mixtures_data)}')
        true_contributions = sample['contributions']
        signal = sample['signal']
        result = [true_contributions[-1],
                  nnls_fit(signal, pure_components)[-1],
                  grid_gn_solver.solve(signal)[0][-1],
                  ea_solver.solve(signal)[0][-1]]
        results.append(result)
    results = np.array(results)

    mean_abs_error_no_correction = np.mean(abs(results[:, 1] - results[:, 0]))
    mean_abs_error_grid_search_and_gauss_newton = np.mean(abs(results[:, 2] - results[:, 0]))
    mean_abs_error_EA = np.mean(abs(results[:, 3] - results[:, 0]))

    print("Mean absolute errors:")
    print(f'No correction: {mean_abs_error_no_correction}')
    print(f'Grid_search + Gauss-Newton: {mean_abs_error_grid_search_and_gauss_newton}')
    print(f'EA: {mean_abs_error_EA}')

    _ = plt.figure()
    _ = plt.plot(results[:, 0], 'b-', results[:, 1], 'r-', results[:, 1] - results[:, 0], 'g-')
    plt.grid()
    _ = plt.title('NNLS without correction')
    _ = plt.legend(['True', 'Predicted', 'Error'])
    plt.show()

    _ = plt.figure()
    _ = plt.plot(results[:, 0], 'b-', results[:, 2], 'r-', results[:, 2] - results[:, 0], 'g-')
    plt.grid()
    _ = plt.title('NNLS with Grid search + Gauss-Newton correction')
    _ = plt.legend(['True', 'Predicted', 'Error'])
    plt.show()

    _ = plt.figure()
    _ = plt.plot(results[:, 0], 'b-', results[:, 3], 'r-', results[:, 3] - results[:, 0], 'g-')
    plt.grid()
    _ = plt.title('NNLS with EA correction')
    _ = plt.legend(['True', 'Predicted', 'Error'])
    plt.show()

    index = 1
    true_concentrations = mixtures_data[index]['contributions']
    signal = mixtures_data[index]['signal']

    prediction0 = nnls_fit(signal, pure_components)
    prediction1 = grid_gn_solver.solve(signal)[0]
    prediction2 = ea_solver.solve(signal)[0]

    signal_estimate = calculate_signal(prediction0, pure_components)
    residual = signal_estimate - signal

    _ = plt.plot(signal, 'b', linewidth=2, label='True')
    _ = plt.plot(signal_estimate, 'r', linewidth=2, label='Estimate')
    _ = plt.plot(residual, 'g', linewidth=2, label='Residual')
    plt.grid()
    _ = plt.title('True signal, estimated signal and residual')
    _ = plt.legend()
    plt.show()

    print('Prediction without correction', prediction0)
    print('Prediction with Grid search + Gauss-Newton correction', prediction1)
    print('Prediction with EA correction', prediction2)
    print('True', true_concentrations)


if __name__ == "__main__":
    main()
