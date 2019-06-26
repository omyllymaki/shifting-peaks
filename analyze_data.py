import matplotlib.pyplot as plt
import numpy as np

from constants import PATH_PURE_COMPONENTS, PATH_MIXTURES, X
from file_io import load_pickle_file
from nnls_fit_with_x_axis_correction import analysis
from utils import calculate_signal, nnls_fit

library = load_pickle_file(PATH_PURE_COMPONENTS)
mixtures_data = load_pickle_file(PATH_MIXTURES)

results = []
for sample in mixtures_data:
    true_concentrations = sample['concentrations']
    signal = sample['signal']
    result = [true_concentrations[-1],
              nnls_fit(signal, library)[-1],
              analysis(X, signal, library)[0][-1]]
    results.append(result)
results = np.array(results)

plt.figure()
plt.plot(results[:, 0], 'b-', results[:, 1], 'r-', results[:, 1] - results[:, 0], 'g-')
plt.grid()
plt.legend(['True', 'Predicted (no correction)', 'Error'])
plt.show()

plt.figure()
plt.plot(results[:, 0], 'b-', results[:, 2], 'r-', results[:, 2] - results[:, 0], 'g-')
plt.grid()
plt.legend(['True', 'Predicted (with correction)', 'Error'])
plt.show()

index = 1
true_concentrations = mixtures_data[index]['concentrations']
signal = mixtures_data[index]['signal']

prediction0 = nnls_fit(signal, library)
prediction = analysis(X, signal, library)

signal_estimate = calculate_signal(prediction0, library)
residual = signal_estimate - signal

print('Prediction without correction', prediction0)
print('Prediction with correction', prediction)
print('True concentrations', true_concentrations)

plt.plot(signal, 'b', linewidth=2, label='True')
plt.plot(signal_estimate, 'r', linewidth=2, label='Estimate')
plt.plot(residual, 'g', linewidth=2, label='Residual')
plt.grid()
plt.legend()
plt.show()
