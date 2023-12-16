"""Eigenvalue probability bounds for Wishart matrices.

References
----------
1. Chiani, 2017, doi: 10.1109/TIT.2017.2694846
2. Chiani, 2014, doi: 10.1016/j.jmva.2014.04.002
"""
import numpy as np
from scipy.special import gamma, gammainc
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


def gammam(m, a):
    """Γ_m function defined in Sec. II.A of [1].

    Parameters
    ----------
    m : int
    a : float

    Returns
    -------
    : float
    """
    y = np.pi ** (m * (m - 1) / 4)
    for i in range(m):
        y *= gamma(a - i / 2)
    return y


def gengammainc(a, x, y):
    """Generalized regularized incomplete gamma function."""
    return gammainc(a, y) - gammainc(a, x)


def compute_Kprime(n_min, n_max):
    """Compute K', the normalizing constant for the joint distribution of
    eigenvalues, as defined in Theorem 1 of [1].

    Parameters
    ----------
    n_min : int
        Wishart dimension or DOFs, whichever is smaller.
    n_max : int
        Wishart dimension or DOFs, whichever is larger.

    Returns
    -------
    : float
        The normalizing constant K'.
    """
    K_nom = np.pi ** (n_min**2 / 2)
    K_den = (
        2 ** (n_min * n_max / 2) * gammam(n_min, n_max / 2) * gammam(n_min, n_min / 2)
    )
    K = K_nom / K_den

    α = (n_max - n_min - 1) / 2
    Kprime = K * 2 ** (α * n_min + n_min * (n_min + 1) / 2)
    for k in range(n_min):
        Kprime *= gamma(α + k + 1)
    return Kprime


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
        A = np.zeros((n_min, n_min))
    else:
        A = np.zeros((n_min + 1, n_min + 1))

    def α(l):
        return (n_max - n_min - 1) / 2 + l

    def g(l, x):
        if np.isinf(x):
            return 0
        return x ** α(l) * np.exp(-x)

    for i in range(n_min - 1):
        for j in range(i, n_min - 1):
            αi = α(i + 1)
            αj = α(j + 1)
            nom1 = gamma(αi + αj) * 2 ** (1 - αi - αj) * gengammainc(αi + αj, a, b)
            den1 = gamma(αj + 1) * gamma(αi)
            nom2 = -(g(j + 1, a / 2) + g(j + 1, b / 2)) * gengammainc(αi, a / 2, b / 2)
            den2 = gamma(αj + 1)
            A[i, j + 1] = A[i, j] + nom1 / den1 + nom2 / den2

    if n_min % 2 == 1:
        for i in range(n_min):
            αi = α(i + 1)
            A[i, -1] = gengammainc(αi, a / 2, b / 2)

    A = A - A.T
    Kprime = compute_Kprime(n_min, n_max)
    return Kprime * np.sqrt(np.linalg.det(A))


def max_eigval_cdf(p, df, x):
    """Cumulative density function of the largest eigenvalue of a real random
    matrix with a standard Wishart distribution.

    In other words, we are computing the probability that the maximum
    eigenvalue of the matrix is below the value ``x``.

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


def min_eigval_cdf(p, df, x):
    """Cumulative density function of the smallest eigenvalue of a real random
    matrix with a standard Wishart distribution.

    In other words, we are computing the probability that the minimum
    eigenvalue of the matrix is below the value ``x``.

    Parameters
    ----------
    p : int
        The dimension of the matrix (i.e., the matrix is p x p).
    df : int
        The degrees of freedom of the Wishart distribution.
    x : float
        The upper bound on the minimum eigenvalue.

    Returns
    -------
    : float
        The probability that the maximum eigenvalue is less than or equal to
        ``x``.
    """
    return 1 - eigval_interval_probability(p, df, x, np.inf)


def max_eigval_quantile(p, df, prob, bound=100):
    """Quantile function for the largest eigenvalue of a real random
    matrix with a standard Wishart distribution.

    In other words, we are computing the value ``x`` such that the probability
    that the largest eigenvalue is below ``x`` is ``prob``. This is the inverse
    of ``max_eigval_cdf``.

    Parameters
    ----------
    p : int
        The dimension of the matrix (i.e., the matrix is p x p).
    df : int
        The degrees of freedom of the Wishart distribution.
    prob : float
        The desired probability.
    bound : float
        The upper bound for the computed value ``x``.

    Returns
    -------
    : float
        The value ``x`` such that the probability that the largest eigenvalue
        is below ``x`` is ``prob``.
    """
    def fun(x):
        c = max_eigval_cdf(p, df, x)
        return 0.5 * (c - prob) ** 2

    res = minimize_scalar(fun, bounds=(0, bound), method="bounded")
    if not res.success:
        raise ValueError("Quantile optimization failed.")
    return res.x


def min_eigval_quantile(p, df, prob, bound=100):
    """Quantile function for the smallest eigenvalue of a real random
    matrix with a standard Wishart distribution.

    In other words, we are computing the value ``x`` such that the probability
    that the smallest eigenvalue is below ``x`` is ``prob``. This is the
    inverse of ``min_eigval_cdf``.

    Parameters
    ----------
    p : int
        The dimension of the matrix (i.e., the matrix is p x p).
    df : int
        The degrees of freedom of the Wishart distribution.
    prob : float
        The desired probability.
    bound : float
        The upper bound for the computed value ``x``.

    Returns
    -------
    : float
        The value ``x`` such that the probability that the smallest eigenvalue
        is below ``x`` is ``prob``.
    """
    def fun(x):
        c = min_eigval_cdf(p, df, x)
        return 0.5 * (c - prob) ** 2

    res = minimize_scalar(fun, bounds=(0, bound), method="bounded")
    if not res.success:
        raise ValueError("Quantile optimization failed.")
    return res.x


def chiani_2017_table2():
    """Generate the first few entries of Table 2 of [1].

    Note that we currently cannot generate more entries due to numerical
    difficulties.
    """
    print("n  probability")
    print("-- -----------")
    for n in [2, 5, 10]:
        prob = max_eigval_cdf(p=n, df=n, x=n)
        print(f"{n}  {prob}")

        # check the quantile function
        assert(np.isclose(max_eigval_quantile(n, n, prob), n))


def chiani_2014_figure3():
    """Generate (something similar to) Figure 3 of [2]."""
    p = 5
    xs = np.linspace(0, 70, 100)
    plt.figure()
    for df in [5, 10, 15, 20, 25, 30, 35]:
        probs = [max_eigval_cdf(p, df, x) for x in xs]
        plt.plot(xs, probs)
    plt.xlabel("x")
    plt.ylabel("CDF(x)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    chiani_2017_table2()
    chiani_2014_figure3()
