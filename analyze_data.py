import matplotlib.pyplot as plt
import numpy as np

from analysis import nnls_fit
from file_io import load_pickle_file
from nnls_fit_with_x_axis_correction import nnls_fit_with_x_axis_correction_trial_and_error, \
    nnls_fit_with_x_axis_correction_gauss_newton
from utils import dict_to_array, estimate_signal

PATH_PURE_COMPONENTS = 'pure_components.p'
PATH_MIXTURES = 'mixtures.p'
X = np.arange(1, 100)
OFFSET_CANDIDATES = np.arange(-7, 7, 0.1)
SLOPE_CANDIDATES = np.arange(-0.05, 0.05, 0.001)

component_signals = load_pickle_file(PATH_PURE_COMPONENTS)
mixtures_data = load_pickle_file(PATH_MIXTURES)

index = -1
true_concentrations = mixtures_data[index]['concentrations']
signal = mixtures_data[index]['signal']
errors = mixtures_data[index]['errors']
L = dict_to_array(component_signals)

prediction0 = nnls_fit(signal, L)
prediction1 = nnls_fit_with_x_axis_correction_trial_and_error(X, signal, L, OFFSET_CANDIDATES, SLOPE_CANDIDATES)
prediction2 = nnls_fit_with_x_axis_correction_gauss_newton(X, signal, L)

signal_estimate = estimate_signal(prediction0, L)
residual = signal_estimate - signal

print('errors', errors)
print('prediction no correction', prediction0)
print('prediction brute force', prediction1)
print('prediction gauss-newton', prediction2)
print('true concentration', true_concentrations)

plt.plot(signal, 'b', linewidth=2, label='True')
plt.plot(signal_estimate, 'r', linewidth=2, label='Estimate')
plt.plot(residual, 'g', linewidth=2, label='Residual')
plt.grid()
plt.legend()
plt.show()
