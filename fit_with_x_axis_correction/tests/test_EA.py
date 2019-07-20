from functools import partial

from fit_with_x_axis_correction.EA import solve_with_EA
from fit_with_x_axis_correction.common import interpolate_signal
from fit_with_x_axis_correction.correction_models import quadratic_correction, linear_correction
from fit_with_x_axis_correction.tests.base_test_case import BaseTestCase


class TestEA(BaseTestCase):
    method = partial(solve_with_EA,
                     correction_model=linear_correction,
                     rsme_threshold=0.01,
                     init_guess=(0, 0),
                     deviations=(0.1, 1))

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
