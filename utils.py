import numpy as np

from numpy.linalg import pinv
from scipy.optimize import nnls


def interpolate_signal(signal, x_original, x_target, left=np.nan, right=np.nan):
    return np.interp(x_target, x_original, signal, left=left, right=right)


def interpolate_array(signal_array, x_original, x_target, left=np.nan, right=np.nan):
    return np.array([interpolate_signal(s, x_original, x_target, left, right) for s in signal_array])


def calculate_signal(concentration: np.ndarray, L: np.ndarray) -> np.ndarray:
    return concentration.T @ L


def rss(residuals: np.ndarray) -> float:
    return sum(residuals ** 2)


def calculate_pseudoinverse(x: np.ndarray):
    return pinv(x.T @ x) @ x.T


def nnls_fit(s: np.ndarray, L: np.ndarray) -> np.ndarray:
    return nnls(L.T, s)[0]