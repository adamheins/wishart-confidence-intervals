# Wishart Matrix Eigenvalue Probabilities

This is a Python implementation of Algorithm 1 from [1] to compute the
probability that the eigenvalues of a standard Wishart-distributed random
matrix lie within a given interval. This immediately also gives us the
cumulative density functions for the distributions of the minimum and maximum
eigenvalues. In addition, we provide optimization routines to evaluate the
associated quantile functions.

## Limitations

The Python implmentation uses standard floating point precision and so runs
into numerical challenges when computing the results for matrices larger than
TODO.

## References

1. M. Chiani, "On the Probability That All Eigenvalues of Gaussian, Wishart,
   and Double Wishart Random Matrices Lie Within an Interval," in IEEE
   Transactions on Information Theory, vol. 63, no. 7, pp. 4521-4531, July
   2017, doi:
   [10.1109/TIT.2017.2694846](https://doi.org/10.1109/TIT.2017.2694846).
1. Chiani, Marco (2014). "Distribution of the Largest Eigenvalue for Real
   Wishart and Gaussian Random Matrices and a Simple Approximation for the
   Tracy–Widom Distribution". In: Journal of Multivariate Analysis 129, pp.
   69–81. doi: [10.1016/j.jmva.2014.04.002](https://doi.org.10.1016/j.jmva.2014.04.002).
