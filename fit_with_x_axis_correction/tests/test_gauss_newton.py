import os
import timeit
from functools import partial

from fit_with_x_axis_correction.correction_models import linear_correction, quadratic_correction
from file_io import load_pickle_file
from fit_with_x_axis_correction.gauss_newton import solve_with_gauss_newton
from fit_with_x_axis_correction.tests.base_test_case import BaseTestCase
from fit_with_x_axis_correction.common import interpolate_signal


class TestGaussNewton(BaseTestCase):

    def test_no_x_axis_errors_should_pass(self) -> None:
        self.method = partial(solve_with_gauss_newton, correction_model=linear_correction)
        self.run_test(self.mixture_signal)

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
        self.method = partial(solve_with_gauss_newton,
                              correction_model=quadratic_correction,
                              initial_parameters=(0, 0, 0))
        x_distorted = 0.005 * self.x ** 2 + 1.01 * self.x + 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        self.run_test(signal)

    def test_speed(self):
        self.method = partial(solve_with_gauss_newton, correction_model=linear_correction)
        x_distorted = 1.01 * self.x - 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        f = partial(self.method, x_original=self.x, signal=signal, pure_components=self.pure_components)
        time = timeit.timeit(f, number=50)
        self.assertLess(time, 1)

    def test_with_random_concentrations_and_random_linear_x_errors_should_pass(self):
        self.method = partial(solve_with_gauss_newton, correction_model=linear_correction)
        root_path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(root_path, 'data', 'mixtures.p')
        mixture_data = load_pickle_file(file_path)
        for item in mixture_data:
            self.contributions = item['contributions']
            signal = item['signal']
            self.run_test(signal, decimal=1)
