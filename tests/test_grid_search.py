import itertools
import os
import unittest
from functools import partial

import numpy as np
from numpy.testing import assert_almost_equal

from correction_models import linear_correction
from file_io import load_pickle_file
from nnls_fit_with_x_axis_correction import solve_with_grid_search
from utils import interpolate_signal, calculate_signal


class TestGridSearch(unittest.TestCase):
    offset_candidates = np.arange(-3, 3, 0.1)
    slope_candidates = np.arange(-0.03, 0.03, 0.001)
    candidates = list(itertools.product(slope_candidates, offset_candidates))
    method = partial(solve_with_grid_search, candidates=candidates, correction_model=linear_correction)

    def setUp(self):
        root_path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(root_path, 'data', 'pure_components.p')
        self.library = load_pickle_file(file_path)
        self.x = np.arange(1, 100)
        self.concentrations = np.array([1, 3, 5])
        self.mixture_signal = calculate_signal(self.concentrations, self.library)

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

    def run_test(self, x_distorted, decimal: int = 7):
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        actual, _ = self.method(self.x,
                                signal,
                                self.library)
        expected = self.concentrations
        assert_almost_equal(actual, expected, decimal=decimal)
