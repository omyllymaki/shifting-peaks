# Shifting peaks

This project provides methods to correct effect of peak shifting in curve fitting problem. The methods 
are illustrated with synthetic data.

## Background

The target of curve fitting is to extract amounts of different components from measured signal.

Measured signal contains always noise and errors which affect to accuracy of our estimates. 
In general, random amplitude noise (y-axis noise) is handled well by curve fitting methods. However, in some cases,
we also have error in x-axis (time, channel, pixels, wavelength, ...) and it is not random by
nature. In other words, observed peaks and shapes shift from one sample to other according to some model. This kind of 
error source will produce major inaccuracy to our output, especially if signal shapes are narrow.

This project implements model based correction method to handle x-axis errors. This kind of approach
works well when x-axis errors follow some model that is known or can be approximated. In this project,
the synthetic data set has quadratic errors (y = a*x^2 + b*x + c) but usage of the method is not limited
to polynomial models.


## Requirements

- Python 3.6 (probably works also with older and newer versions)
- Python libraries: numpy, scipy, matplotlib

Install requirements by running

```
pip install -r requirements.txt
```


## Usage

Just run analysis sample:

```
python analysis_sample.py
```

You can also generate your own data set:

```
python generate_test_data.py
```


## Algorithms

In general level, the algorithms work by looking minimum RSME by changing correction model parameters
and doing NNLS fit afterwards. This process is repeated iteratively. Accepted solution is the one which 
produces minimum RSME. At high level, the process can be described as follows:

Loop:
- Update correction model parameters
- Generate new x-axis using current guess for correction model parameters
- Interpolate signal to new axis
- Calculate solution using regular NNLS and interpolated signal
- Calculate RSME
- Check termination condition and exit loop if filled

Update of correction model parameters can be done in different ways. In this project, I have implemented three
different algorithms to update parameters.

**Grid search**

All candidates are tested and best option is returned as final solution.

**Gauss-Newton**

Parameter candidate is updated using Gauss-Newton optimization method. This method uses gradient to
find direction and step size for update.

**Evolutionary algorithm**

Parameter candidates are updated using evolutionary algorithm. This method uses population of random 
parameter combinations, evaluates candidates in population and generates new population based on
results for previous population.

These three different update algorithms can be used separately or they can be combined to produce better
results. For example, we can use grid search to make rough first estimate for parameters and then use more
accurate Gauss-Newton to optimize solution from there. Using grid search first will reduce the risk that 
gradient based Gauss-Newton optimization ends up to local minimum instead of global minimum.


