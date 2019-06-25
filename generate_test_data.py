import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal as rand

from file_io import save_pickle_file
from utils import calculate_mixture_signal, interpolate_signal

X = np.arange(1, 100)
PATH_PURE_COMPONENTS = 'pure_components.p'
PATH_MIXTURES = 'mixtures.p'
N_SAMPLES = 100
OFFSET_ERROR = 2
SLOPE_ERROR = 0
POLYNOMIAL_ERROR = 0
CONCENTRATIONS = {
    'A': 100,
    'B': 500,
    'C': 35,
}


def gaussian(x: np.ndarray,
             amplitude: float,
             center: float,
             sigma: float) -> np.ndarray:
    return amplitude * (1 / (sigma * (np.sqrt(2 * np.pi)))) * (np.exp(-((x - center) ** 2) / ((2 * sigma) ** 2)))


def generate_pure_component_signals():
    component_signals = {
        'A': gaussian(X, 7, 20, 7),
        'B': gaussian(X, 6, 50, 5),
        'C': gaussian(X, 2, 85, 2),
    }
    return component_signals


def generate_distorted_axis(x, offset_error, slope_error, polynomial_error):
    return (1 + rand(scale=slope_error)) * x + rand(scale=offset_error) + rand(scale=polynomial_error)


def generate_random_concentrations():
    concentrations = {}
    for component, max_concentration in CONCENTRATIONS.items():
        concentrations[component] = max_concentration * random.random()
    return concentrations


def generate_mixtures():
    mixtures_data = []
    for _ in range(N_SAMPLES):
        concentrations = generate_random_concentrations()
        mixture_signal = calculate_mixture_signal(concentrations, component_signals)
        x_distorted = generate_distorted_axis(X, OFFSET_ERROR, SLOPE_ERROR, POLYNOMIAL_ERROR)
        mixture_signal = interpolate_signal(mixture_signal, X, x_distorted, 0, 0)
        mixtures_data.append(
            {'concentrations': concentrations,
             'signal': mixture_signal,
             'errors': [OFFSET_ERROR, SLOPE_ERROR, POLYNOMIAL_ERROR]})
    return mixtures_data


def main():
    component_signals = generate_pure_component_signals()
    mixtures_data = generate_mixtures()

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
