import numpy as np
from scipy.optimize import nnls

from utils import dict_to_array, array_to_dict


def nnls_fit(s: np.ndarray, L: np.ndarray) -> np.ndarray:
    return nnls(L.T, s)[0]


def nnls_analysis(signal: np.ndarray, pure_components: dict):
    L = dict_to_array(pure_components)
    prediction = nnls_fit(signal, L)
    results = array_to_dict(prediction, list(pure_components.keys()))
    return results
