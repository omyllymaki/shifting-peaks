# Shifting peaks

This project provides methods to correct effect of peak shifting in curve fitting problem. The methods 
are illustrated with synthetic data.

## Background

The target of curve fitting is to extract amounts of different components from measured signal.

Measured signal contains always noise and errors which affect accuracy of estimates. 
In general, random amplitude noise (y-axis noise) is handled well by curve fitting methods. However, in some cases,
we also have error in x-axis (time, channel, pixels, wavelength, ...) and it is not random by nature. 
In other words, observed peaks and shapes shift from one sample to other according to some model. This kind of 
error source will produce major inaccuracy to our output, especially if signal shapes are narrow.

This project implements model based correction method to handle x-axis errors. This kind of approach
works well when x-axis errors follow some model that is known or can be approximated. 

In this project, the synthetic data set has quadratic errors (y = a*x^2 + b*x + c) but usage of the method is not limited
to polynomial models. Also, user can define what curve fitting method (e.g. NNLS) will be used in conjuction with x-axis
correction.


## Requirements

- Python 3.6 (probably works also with older and newer Python3 versions)
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

User gives x-axis correction model (e.g. quadratic model) and regular (doesn't include x-axis correction) curve fitting 
function (e.g. NNLS) as input. 

In general level, the algorithms work by looking minimum RSME by changing correction model parameters
and doing regular fit after this. This process is repeated iteratively. Accepted solution is 
the one which produces minimum RMSE. At high level, the process can be described as follows:

Loop:
- Update correction model parameters
- Generate new x-axis using current guess for correction model parameters
- Interpolate signal to new axis
- Calculate solution using regular curve fitting method and interpolated signal
- Calculate RMSE for solution
- Check termination condition and exit loop if it is filled

Update of correction model parameters can be done in different ways. In this project, three
different algorithms are implemented.

**Grid search**

All candidates are tested and best option is returned as final solution.

**Gauss-Newton**

Parameter candidate is updated using Gauss-Newton optimization method. This method uses gradient to
find direction and step size for update.

**Evolutionary algorithm**

Parameter candidates are updated using evolutionary algorithm. This method uses population of random 
parameter combinations, evaluates candidates in population and generates new population based on
results for previous population.

**Combinations**

These three different update algorithms can be used separately or they can be combined to produce better
results. For example, we can use grid search to make rough first estimate for parameters and then use more
accurate Gauss-Newton to optimize solution from there. Using grid search first will reduce the risk that 
gradient based Gauss-Newton optimization ends up to local minimum instead of global minimum.


