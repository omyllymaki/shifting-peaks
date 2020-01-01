import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal as rand

from constants import PATH_PURE_COMPONENTS, PATH_MIXTURES
from file_io import save_pickle_file

X = np.arange(-50, 150)
KEEP_CHANNELS = np.arange(50, 150)
N_SAMPLES = 100
OFFSET_ERROR_STDEV = 2
SLOPE_ERROR_STDEV = 0.01
QUADRATIC_ERROR_STDEV = 0.0005
MAX_CONTRIBUTIONS = [100, 500, 35]
AMPLITUDE_NOISE = 1


def interpolate_signal(signal: np.ndarray,
                       x_original: np.ndarray,
                       x_target: np.ndarray,
                       left_value: float = np.nan,
                       right_value: float = np.nan) -> np.ndarray:
    return np.interp(x_target, x_original, signal, left=left_value, right=right_value)


def calculate_signal(contributions: np.ndarray, pure_component_signals: np.ndarray) -> np.ndarray:
    return contributions.T @ pure_component_signals


def gaussian(x: np.ndarray,
             amplitude: float,
             center: float,
             fwhm: float) -> np.ndarray:
    width = fwhm / (2 * np.log(2))
    exponent = - ((x - center) / width) ** 2
    return amplitude * np.exp(exponent)


def generate_pure_components() -> np.ndarray:
    return np.array([
        gaussian(X, 7, 20, 7),
        gaussian(X, 6, 50, 19),
        gaussian(X, 5, 55, 2),
    ])


def generate_distorted_axis(x: np.ndarray,
                            offset_error: float,
                            slope_error: float,
                            quadratic_error: float) -> np.ndarray:
    return rand(scale=offset_error) + (1 + rand(scale=slope_error)) * x + rand(scale=quadratic_error) * x ** 2


def generate_random_contributions(max_contributions: list) -> np.ndarray:
    return np.array([c * random.random() for c in max_contributions])


def add_amplitude_noise_to_signal(signal: np.ndarray, noise: float) -> np.ndarray:
    return signal + rand(scale=noise, size=len(signal))


def main():
    pure_components = generate_pure_components()

    mixtures_data = []
    for _ in range(N_SAMPLES):
        contributions = generate_random_contributions(MAX_CONTRIBUTIONS)
        mixture_signal = calculate_signal(contributions, pure_components)
        x_distorted = generate_distorted_axis(X, OFFSET_ERROR_STDEV, SLOPE_ERROR_STDEV, QUADRATIC_ERROR_STDEV)
        mixture_signal = interpolate_signal(mixture_signal, X, x_distorted, 0, 0)
        mixture_signal = mixture_signal[KEEP_CHANNELS]
        mixture_signal = add_amplitude_noise_to_signal(mixture_signal, AMPLITUDE_NOISE)
        mixtures_data.append(
            {'contributions': contributions,
             'signal': mixture_signal})

    pure_components = pure_components[:, KEEP_CHANNELS]

    plt.figure()
    for signal in pure_components:
        plt.plot(range(len(signal)), signal)
    plt.grid()
    plt.show()

    plt.figure()
    for sample in mixtures_data:
        signal = sample['signal']
        plt.plot(range(len(signal)), sample['signal'])
    plt.grid()
    plt.show()

    save_pickle_file(pure_components, PATH_PURE_COMPONENTS)
    save_pickle_file(mixtures_data, PATH_MIXTURES)


if __name__ == "__main__":
    main()
