import numpy as np
from scipy.optimize import nnls


def nnls_fit(s: np.ndarray, L: np.ndarray) -> np.ndarray:
    return nnls(L.T, s)[0]
