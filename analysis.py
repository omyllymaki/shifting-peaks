import numpy as np
from scipy.optimize import nnls

from utils import dict_to_array, array_to_dict


def nnls_fit(s: np.ndarray, L: np.ndarray) -> np.ndarray:
    return nnls(L.T, s)[0]
