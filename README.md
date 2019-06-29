# Shifting peaks

This project provides methods to correct effect of peak shifting in curve fitting problem. The methods 
are illustrated with synthetic data.

## Background

Let's assume, for simplicity, that measured signal is linear combination of overlapping
components. The target is to extract amounts of different components from measured signal.

One of the common approaches to solve the problem is to do classical least squares (CLS) fit where
we estimate amount of different components by minimising residual sum of squares. We can also constrain 
this fit, e.g. we can use non-negative least squares (NNLS) to make estimated component amounts non-negative.

Measured signal contains always noise and errors which affect to accuracy of our estimates. 
In general, random amplitude noise (y-axis noise) is handled well by NNLS. However, in some cases,
we also have error in x-axis (time, channel, pixels, wavelength, ...) and it is not random by
nature. In other words, observed peaks and shapes shift from one sample to other. This kind of error 
source will produce major inaccuracy to our output, especially if signal shapes are narrow.

This project implements model based correction method to handle x-axis errors. This kind of approach
works well when x-axis errors follow some model that is known or can be approximated. In this project,
the synthetic data set has quadratic errors (y = a*x^2 + b*x + c) but usage of the method is not limited
to polynomial models.

## Algorithm

The algorithm has two stages:

1. Calculating rough estimates for correction model parameters using grid search
2. Calculating more accurate estimates using Gauss-Newton optimization method, using model parameters
from stage 1 as initial guess

Stage 1 is done in order ensure that gradient based optimization in stage 2 finds global minimum.

In both stages we are looking minimum residual sum of squares by changing correction model parameters
and doing NNLS fit afterwards. This process is repeated iteratively. Accepted solution is the one which 
produces minimum residual sum of squares. At high level, the process can be described as follows:

Loop:
- Update correction model parameters, using grid search or optimization method
- Generate new x-axis using current guess for correction model parameters
- Interpolate signal to new axis
- Calculate solution using regular NNLS and interpolated signal
- Calculate sum of residual squares
- Check termination 

Return solution with lowest sum of residual squares 

