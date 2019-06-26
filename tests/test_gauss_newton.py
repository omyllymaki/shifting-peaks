import os
import timeit
import unittest
from functools import partial

import numpy as np
from numpy.random import normal as rand
from numpy.testing import assert_almost_equal

from correction_models import linear_correction, quadratic_correction
from file_io import load_pickle_file
from nnls_fit_with_x_axis_correction import solve_with_gauss_newton
from utils import dict_to_array, calculate_mixture_signal, interpolate_signal


class TestGaussNewton(unittest.TestCase):
    method = partial(solve_with_gauss_newton, correction_model=linear_correction)

    def setUp(self):
        root_path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(root_path, 'data', 'pure_components.p')
        self.pure_components = load_pickle_file(file_path)
        self.library = dict_to_array(self.pure_components)
        self.x = np.arange(1, 100)
        self.concentrations = {'A': 1, 'B': 3, 'C': 5}
        self.mixture_signal = calculate_mixture_signal(self.concentrations, self.pure_components)

    def test_no_x_axis_errors_should_pass(self) -> None:
        x_distorted = self.x
        self.run_test(x_distorted)

    def test_offset_error_should_pass(self) -> None:
        x_distorted = self.x + 2
        self.run_test(x_distorted)

    def test_slope_error_should_pass(self) -> None:
        x_distorted = 1.01 * self.x
        self.run_test(x_distorted)

    def test_slope_and_offset_error_should_pass(self) -> None:
        x_distorted = 1.01 * self.x - 2
        self.run_test(x_distorted)

    def test_performance(self):
        x_distorted = 1.01 * self.x - 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        f = partial(self.method, x_original=self.x, signal=signal, library=self.library)
        time = timeit.timeit(f, number=50)
        self.assertLess(time, 1)

    def test_small_random_offset_and_slope_errors_should_pass(self):
        for k in range(50):
            x_distorted = (1 + rand(scale=0.1)) * self.x + rand(scale=1)
            self.run_test(x_distorted, decimal=1)

    def run_test(self, x_distorted, decimal: int = 7):
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        actual, _ = self.method(self.x,
                                signal,
                                self.library)
        expected = np.array(list(self.concentrations.values()))
        assert_almost_equal(actual, expected, decimal=decimal)
