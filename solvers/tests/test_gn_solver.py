import os
import timeit
from functools import partial

from file_io import load_pickle_file
from solvers.gn_solver import GNSolver
from solvers.math import interpolate_signal, ls_fit
from solvers.tests.base_test_case import BaseTestCase
from solvers.tests.correction_models import quadratic_correction


class TestGNSolver(BaseTestCase):

    def setUp(self):
        super().setUp()
        self.solver = GNSolver(x=self.x,
                               pure_components=self.pure_components,
                               correction_model=quadratic_correction,
                               fit_function=ls_fit)

    def test_no_x_axis_errors_should_pass(self) -> None:
        self.run_test(self.mixture_signal)

    def test_offset_error_should_pass(self) -> None:
        x_distorted = self.x + 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        self.run_test(signal)

    def test_slope_error_should_pass(self) -> None:
        x_distorted = 1.01 * self.x
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        self.run_test(signal)

    def test_slope_and_offset_error_should_pass(self) -> None:
        x_distorted = 1.01 * self.x - 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        self.run_test(signal)

    def test_quadratic_error_should_pass(self) -> None:
        x_distorted = 0.01 * self.x ** 2 + 1.01 * self.x + 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        self.run_test(signal)

    def test_speed(self):
        x_distorted = 1.01 * self.x - 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        f = partial(self.solver.solve, signal=signal)
        time = timeit.timeit(f, number=50)
        self.assertLess(time, 1)

    def test_with_random_concentrations_and_random_linear_x_errors_should_pass(self):
        root_path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(root_path, 'data', 'mixtures.p')
        mixture_data = load_pickle_file(file_path)
        for item in mixture_data:
            self.contributions = item['contributions']
            signal = item['signal']
            self.run_test(signal, decimal=1)
