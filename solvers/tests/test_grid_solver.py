import itertools

import numpy as np

from solvers.grid_solver import GridSolver
from solvers.math import interpolate_signal, ls_fit
from solvers.tests.base_test_case import BaseTestCase
from solvers.tests.correction_models import linear_correction


class TestGridSolver(BaseTestCase):
    offset_candidates = np.arange(-3, 3, 0.1)
    slope_candidates = np.arange(-0.03, 0.03, 0.001)
    candidates = np.array(list(itertools.product(slope_candidates, offset_candidates)))

    def setUp(self):
        super().setUp()
        self.solver = GridSolver(x=self.x,
                                 pure_components=self.pure_components,
                                 candidates=self.candidates,
                                 correction_model=linear_correction,
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
