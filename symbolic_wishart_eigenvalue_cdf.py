"""Eigenvalue probability bounds for Wishart matrices.

References
----------
1. Chiani, 2014, doi: 10.1016/j.jmva.2014.04.002
2. Chiani, 2017, doi: 10.1109/TIT.2017.2694846
"""
import numpy as np
from scipy.stats import wishart

# from scipy.special import gamma, gammainc, loggamma
# from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

import sympy as sp

import IPython


def gammam(m, a):
    """Γ_m function defined in Sec. 2.1 of [1].

    Parameters
    ----------
    m : int
    a : float

    Returns
    -------
    : float
    """
    y = sp.pi ** (m * (m - 1) / 4)
    for i in range(m):
        y *= sp.gamma(a - i / 2)
    return y


def compute_Kprime2(n_min, n_max):
    # compute K, the normalizing constant for the joint distribution of
    # eigenvalues
    K_nom = sp.pi ** (n_min**2 / 2)
    K_den = (
        sp.Float(2) ** (n_min * n_max / 2) * gammam(n_min, n_max / 2) * gammam(n_min, n_min / 2)
    )
    K = K_nom / K_den

    α = (n_max - n_min - 1) / 2
    K1 = K * sp.Float(2) ** (α * n_min + n_min * (n_min + 1) / 2)
    for k in range(n_min):
        K1 *= sp.gamma(α + k + 1)
    return K1


def max_eigval_cdf(p, df, x):
    """Cumulative density function of the largest eigenvalue of a real random
    matrix with a standard Wishart-distributed matrix.

    In other words, we are computing
        Probability{ largest eigenvalue of A <= x } = cdf(p, df, x),
    where A ~ Wishart(p, df).

    Parameters
    ----------
    p : int
        The dimension of the matrix (i.e., the matrix is p x p).
    df : int
        The degrees of freedom of the Wishart distribution.
    x : float
        The upper bound on the maximum eigenvalue.

    Returns
    -------
    : float
        The probability that the maximum eigenvalue is less than or equal to
        ``x``.
    """
    return eigval_interval_probability(p, df, 0, x)


def gammainc(a, x):
    """Regularized lower incomplete gamma function."""
    return sp.lowergamma(a, x) / sp.gamma(a)

def gengammainc(a, x, y):
    """Generalized regularized incomplete gamma function."""
    return gammainc(a, y) - gammainc(a, x)


def eigval_interval_probability(p, df, a, b):
    """Compute the probability that all non-zero eigenvalues of a real random
    matrix with a standard Wishart-distributed matrix lie in the inerval [a, b].

    This is Algorithm 1 of [2].

    Parameters
    ----------
    p : int
        The dimension of the matrix (i.e., the matrix is p x p).
    df : int
        The degrees of freedom of the Wishart distribution.
    a : float
        The lower bound on the eigenvalues.
    b : float
        The upper bound on the eigenvalues.

    Returns
    -------
    : float
        The probability that all of the eigenvalues lie within the interval.
    """
    n_min = min(p, df)
    n_max = max(p, df)

    if n_min % 2 == 0:
        A = sp.zeros(n_min, n_min)
    else:
        A = sp.zeros(n_min + 1, n_min + 1)

    def α(l):
        return (n_max - n_min - 1) / 2 + l

    def g(l, x):
        return x ** α(l) * sp.exp(-x)

    for i in range(n_min - 1):
        for j in range(i, n_min - 1):
            αi = α(i + 1)
            αj = α(j + 1)
            nom1 = sp.gamma(αi + αj) * 2 ** (1 - αi - αj) * gengammainc(αi + αj, a, b)
            den1 = sp.gamma(αj + 1) * sp.gamma(αi)
            nom2 = -(g(j + 1, a / 2) + g(j + 1, b / 2)) * gengammainc(αi, a / 2, b / 2)
            den2 = sp.gamma(αj + 1)
            A[i, j + 1] = A[i, j] + nom1 / den1 + nom2 / den2

    if n_min % 2 == 1:
        for i in range(n_min):
            αi = α(i + 1)
            A[i, -1] = gengammainc(αi, a / 2, b / 2)

    A = (A - A.T).evalf()
    Kprime = compute_Kprime2(n_min, n_max)
    return Kprime * sp.sqrt(sp.det(A))


def min_eigval_cdf(p, df, x):
    return eigval_interval_probability(p, df, x, np.inf)


def chiani_2017_table2():
    """Generate the first few entries of Table 2 of [2].

    Note that we currently cannot generate more entries due to numerical
    difficulties.
    """
    print("n  probability")
    for n in [2, 5, 10, 50, 100]:
        prob = eigval_interval_probability(p=n, df=n, a=0, b=n)
        print(f"{n}  {prob.evalf()}")


def chiani_2017_fig1():
    m = 400
    s = 10
    t = np.linspace(0, 0.1, 100)
    bounds = (np.sqrt(m) - np.sqrt(s) - t * np.sqrt(m)) ** 2
    probs = [min_eigval_cdf(s, m, bound) for bound in bounds]
    print(bounds)
    print(probs)
    plt.plot(t, probs)
    plt.show()


def chiani_2014_fig3():
    p = 5
    xs = sp.linspace(0, 70, 100)
    plt.figure()
    for df in [5, 10, 15, 20, 25, 30, 35]:
        probs = [max_eigval_cdf(p, df, sp.Float(x)) for x in xs]
        plt.plot(xs, probs)
    plt.xlabel("x")
    plt.ylabel("CDF(x)")
    plt.grid()
    plt.show()


def chiani_2014_fig4():
    p = 500
    df = 500
    xs = np.linspace(1900, 2100, 100)
    plt.figure()
    probs = [max_eigval_cdf(p, df, sp.Float(x)) for x in xs]
    plt.plot(xs, probs)
    plt.xlabel("x")
    plt.ylabel("CDF(x)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    chiani_2017_table2()
    # chiani_2017_fig1()
    # chiani_2014_fig4()
