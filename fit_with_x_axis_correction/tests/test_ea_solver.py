import timeit
from functools import partial

from fit_with_x_axis_correction.ea_solver import EASolver
from fit_with_x_axis_correction.common import interpolate_signal
from fit_with_x_axis_correction.correction_models import linear_correction
from fit_with_x_axis_correction.tests.base_test_case import BaseTestCase


class TestEASolver(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.solver = EASolver(x=self.x,
                               pure_components=self.pure_components,
                               correction_model=linear_correction,
                               rsme_threshold=0.1)

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

    def test_speed(self):
        x_distorted = 1.01 * self.x - 2
        signal = interpolate_signal(self.mixture_signal, self.x, x_distorted, 0, 0)
        f = partial(self.solver.solve, signal=signal)
        time = timeit.timeit(f, number=20)
        self.assertLess(time, 10)
