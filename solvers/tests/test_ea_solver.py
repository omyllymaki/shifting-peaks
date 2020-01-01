from solvers.ea_solver import EASolver
from solvers.math import interpolate_signal, ls_fit
from solvers.tests.base_test_case import BaseTestCase
from solvers.tests.correction_models import linear_correction


class TestEASolver(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.solver = EASolver(x=self.x,
                               pure_components=self.pure_components,
                               correction_model=linear_correction,
                               rsme_threshold=0.1,
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
