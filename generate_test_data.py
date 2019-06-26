import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal as rand

from constants import X, PATH_PURE_COMPONENTS, PATH_MIXTURES
from file_io import save_pickle_file
from utils import calculate_mixture_signal, interpolate_signal

N_SAMPLES = 100
OFFSET_ERROR_STDEV = 2
SLOPE_ERROR_STDEV = 0.01
QUADRATIC_ERROR_STDEV = 0
CONCENTRATIONS = {
    'A': 100,
    'B': 500,
    'C': 35,
}


def gaussian(x: np.ndarray,
             amplitude: float,
             center: float,
             sigma: float) -> np.ndarray:
    multiplier = amplitude * (1 / (sigma * (np.sqrt(2 * np.pi))))
    exponent = -((x - center) ** 2) / ((2 * sigma) ** 2)
    return multiplier * np.exp(exponent)


def generate_pure_component_signals() -> dict:
    return {
        'A': gaussian(X, 7, 20, 7),
        'B': gaussian(X, 6, 50, 19),
        'C': gaussian(X, 2, 55, 2),
    }


def generate_distorted_axis(x: np.ndarray,
                            offset_error: float,
                            slope_error: float,
                            quadratic_error: float) -> np.ndarray:
    return rand(scale=offset_error) + (1 + rand(scale=slope_error)) * x + rand(scale=quadratic_error) * x ** 2


def generate_random_concentrations() -> dict:
    concentrations = {}
    for component, max_concentration in CONCENTRATIONS.items():
        concentrations[component] = max_concentration * random.random()
    return concentrations


def generate_mixtures(component_signals):
    mixtures_data = []
    for _ in range(N_SAMPLES):
        concentrations = generate_random_concentrations()
        mixture_signal = calculate_mixture_signal(concentrations, component_signals)
        x_distorted = generate_distorted_axis(X, OFFSET_ERROR_STDEV, SLOPE_ERROR_STDEV, QUADRATIC_ERROR_STDEV)
        mixture_signal = interpolate_signal(mixture_signal, X, x_distorted, 0, 0)
        mixtures_data.append(
            {'concentrations': concentrations,
             'signal': mixture_signal})
    return mixtures_data


def main():
    component_signals = generate_pure_component_signals()
    mixtures_data = generate_mixtures(component_signals)

    plt.figure()
    for component_name, signal in component_signals.items():
        plt.plot(X, signal, label=component_name)
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    for sample in mixtures_data:
        plt.plot(X, sample['signal'])
    plt.grid()
    plt.show()

    save_pickle_file(component_signals, PATH_PURE_COMPONENTS)
    save_pickle_file(mixtures_data, PATH_MIXTURES)


if __name__ == "__main__":
    main()
