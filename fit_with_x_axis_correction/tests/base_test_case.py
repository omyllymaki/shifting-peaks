import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from file_io import load_pickle_file
from fit_with_x_axis_correction.common import calculate_signal


class BaseTestCase(unittest.TestCase):
    solver = None

    def setUp(self):
        root_path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(root_path, 'data', 'pure_components.p')
        self.pure_components = load_pickle_file(file_path)
        self.x = np.arange(0, 100)
        self.contributions = np.array([1, 3, 5])
        self.mixture_signal = calculate_signal(self.contributions, self.pure_components)

    def run_test(self, signal, decimal: int = 1):
        actual, _ = self.solver.solve(signal)
        expected = self.contributions
        assert_almost_equal(actual, expected, decimal=decimal)
