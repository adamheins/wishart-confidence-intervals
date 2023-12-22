# Confidence Intervals for Wishart Random Matrices

This is a Python implementation of Algorithm 1 from (Chiani, 2017) to compute
the probability that the eigenvalues of a standard Wishart-distributed random
matrix lie within a given interval. This immediately also gives us the
cumulative density functions for the distributions of the minimum and maximum
eigenvalues. In addition, we provide optimization routines to evaluate the
associated quantile functions. See the accompanying [blog post](TODO).

## Limitations

The Python implmentation uses standard floating point precision and so runs
into numerical challenges when computing the results for matrices larger than,
say, 10 by 10 or so.

## References

* Chiani, M. (2017). "On the Probability That All Eigenvalues of Gaussian, Wishart,
 and Double Wishart Random Matrices Lie Within an Interval," in IEEE
 Transactions on Information Theory, vol. 63, no. 7, pp. 4521-4531, July
 2017, doi:
 [10.1109/TIT.2017.2694846](https://doi.org/10.1109/TIT.2017.2694846),
 arxiv: [1502.04189](https://arxiv.org/abs/1502.04189).
