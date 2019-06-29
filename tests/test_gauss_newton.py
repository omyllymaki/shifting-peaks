import os
import timeit
import unittest
from functools import partial

import numpy as np
from numpy.testing import assert_almost_equal

from correction_models import linear_correction, quadratic_correction
from file_io import load_pickle_file
from nnls_fit_with_x_axis_correction import solve_with_gauss_newton
from utils import interpolate_signal, calculate_signal


class TestGaussNewton(unittest.TestCase):

    def setUp(self):
        root_path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(root_path, 'data', 'pure_components.p')
        self.library = load_pickle_file(file_path)
        self.x = np.arange(1, 100)
        self.concentrations = np.array([1, 3, 5])
        self.mixture_signal = calculate_signal(self.concentrations, self.library)

    def test_no_x_axis_errors_should_pass(self) -> None:
        self.method = partial(solve_with_gauss_newton, correction_model=linear_correction)
        x_distorted = self.x
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        self.run_test(signal)

    def test_offset_error_should_pass(self) -> None:
        self.method = partial(solve_with_gauss_newton, correction_model=linear_correction)
        x_distorted = self.x + 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        self.run_test(signal)

    def test_slope_error_should_pass(self) -> None:
        self.method = partial(solve_with_gauss_newton, correction_model=linear_correction)
        x_distorted = 1.01 * self.x
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        self.run_test(signal)

    def test_slope_and_offset_error_should_pass(self) -> None:
        self.method = partial(solve_with_gauss_newton, correction_model=linear_correction)
        x_distorted = 1.01 * self.x - 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        self.run_test(signal)

    def test_quadratic_error_should_pass(self) -> None:
        self.method = partial(solve_with_gauss_newton, correction_model=quadratic_correction)
        x_distorted = 0.005 * self.x ** 2 + 1.01 * self.x + 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        self.run_test(signal)

    def test_performance(self):
        self.method = partial(solve_with_gauss_newton, correction_model=linear_correction)
        x_distorted = 1.01 * self.x - 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        f = partial(self.method, x_original=self.x, signal=signal, library=self.library)
        time = timeit.timeit(f, number=50)
        self.assertLess(time, 1)

    def test_with_random_concentrations_and_random_linear_x_errors_should_pass(self):
        self.method = partial(solve_with_gauss_newton, correction_model=linear_correction)
        root_path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(root_path, 'data', 'mixtures.p')
        mixture_data = load_pickle_file(file_path)
        for item in mixture_data:
            self.concentrations = item['concentrations']
            signal = item['signal']
            self.run_test(signal, decimal=1)

    def run_test(self, signal, decimal: int = 2):
        actual, _ = self.method(self.x,
                                signal,
                                self.library)
        expected = self.concentrations
        assert_almost_equal(actual, expected, decimal=decimal)
