import numpy as np
from typing import Dict, List

from numpy.linalg import pinv


def calculate_mixture_signal(concentrations: Dict[str, float],
                             component_signals: Dict[str, np.ndarray]) -> np.ndarray:
    mixture_signal = 0
    for component, concentration in concentrations.items():
        mixture_signal += concentration * component_signals[component]
    return mixture_signal


def interpolate_signal(signal, x_original, x_target, left=np.nan, right=np.nan):
    return np.interp(x_target, x_original, signal, left=left, right=right)


def interpolate_array(signal_array, x_original, x_target, left=np.nan, right=np.nan):
    return np.array([interpolate_signal(s, x_original, x_target, left, right) for s in signal_array])


def dict_to_array(dictionary: dict) -> np.ndarray:
    return np.array(list(dictionary.values()))


def array_to_dict(array: np.ndarray, keys: List[str]) -> dict:
    return dict(zip(keys, array))


def estimate_signal(prediction: np.ndarray, L: np.ndarray) -> np.ndarray:
    return prediction.T @ L


def rss(residuals: np.ndarray) -> float:
    return sum(residuals ** 2)


def calculate_pseudoinverse(x: np.ndarray):
    return pinv(x.T @ x) @ x.T