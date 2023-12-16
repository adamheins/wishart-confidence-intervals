"""Eigenvalue probability bounds for Wishart matrices.

References
----------
1. Chiani, 2014, doi: 10.1016/j.jmva.2014.04.002
2. Chiani, 2017, doi: 10.1109/TIT.2017.2694846
"""
import numpy as np
from scipy.stats import wishart
from scipy.special import gamma, gammainc, loggamma
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

import sympy


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
    y = np.pi ** (m * (m - 1) / 4)
    for i in range(m):
        y *= gamma(a - i / 2)
    return y


def loggammam(m, a):
    """Natural logarithm of ``gammam`` function."""
    y = m * (m - 1) / 4 * np.log(np.pi)
    for i in range(m):
        y += loggamma(a - i / 2)
    return y


def compute_logKprime1(n_min, n_max):
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
    logK = (
        n_min**2 * np.log(np.pi) / 2
        - n_min * n_max * np.log(2) / 2
        - loggammam(n_min, n_max / 2)
        - loggammam(n_min, n_min / 2)
    )

    if n_min % 2 == 0:
        n_mat = n_min
    else:
        n_mat = n_min + 1

    α = (n_max - n_min - 1) / 2
    logKprime = logK + (α * n_mat + n_mat * (n_mat + 1) / 2) * np.log(2)
    for k in range(n_mat):
        logKprime += loggamma(α + k + 1)
    return logKprime


def compute_logKprime2(n_min, n_max):
    """Compute K', the normalizing constant for the joint distribution of
    eigenvalues, as defined in Theorem 1 of [2].

    Note that this is slightly different than the definition from [1], hence
    this function is different from ``compute_Kprime1``.

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
    logK = (
        n_min**2 * np.log(np.pi) / 2
        - n_min * n_max * np.log(2) / 2
        - loggammam(n_min, n_max / 2)
        - loggammam(n_min, n_min / 2)
    )

    α = (n_max - n_min - 1) / 2
    logKprime = logK + (α * n_min + n_min * (n_min + 1) / 2) * np.log(2)
    for k in range(n_min):
        logKprime += loggamma(α + k + 1)
    return logKprime


def compute_Kprime2(n_min, n_max, α):
    # compute K, the normalizing constant for the joint distribution of
    # eigenvalues
    K_nom = np.pi ** (n_min ** 2 / 2)
    K_den = (
        2 ** (n_min * n_max / 2) * gammam(n_min, n_max / 2) * gammam(n_min, n_min / 2)
    )
    K = K_nom / K_den

    if n_min % 2 == 0:
        n_mat = n_min
    else:
        n_mat = n_min + 1

    K1 = K * 2 ** (α * n_mat + n_mat * (n_mat + 1) / 2)
    for k in range(n_mat):
        K1 *= gamma(α + k + 1)
    return K1


def max_eigval_cdf(p, df, x):
    """Cumulative density function of the largest eigenvalue of a real random
    matrix with a standard Wishart-distributed matrix.

    In other words, we are computing
        Probability{ largest eigenvalue of A <= x } = cdf(p, df, x),
    where A ~ Wishart(p, df).

    This is Algorithm 1 of [1].

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
    n_min = min(p, df)
    n_max = max(p, df)

    if n_min % 2 == 0:
        A = np.zeros((n_min, n_min))
    else:
        A = np.zeros((n_min + 1, n_min + 1))

    α = (n_max - n_min - 1) / 2

    p = np.zeros(n_min)
    γ = np.zeros(n_min)
    for i in range(n_min):
        ell = i + 1
        p[i] = gammainc(α + ell, x / 2)
        γ[i] = gamma(α + ell)

    q = np.zeros(2 * n_min - 2)
    for i in range(2 * n_min - 2):
        ell = i + 2
        β = 2 * α + ell
        q[i] = 2**-β * gamma(β) * gammainc(β, x)

    for i in range(n_min):
        b = p[i] ** 2 / 2
        for j in range(i, n_min - 1):
            b = b - q[i + j] / (γ[i] * γ[j + 1])
            A[i, j + 1] = p[i] * p[j + 1] - 2 * b

    if n_min % 2 == 1:
        for i in range(n_min):
            A[i, -1] = 2 ** -(α + n_min + 1) / gamma(α + n_min + 1)
            A[i, -1] *= gammainc(α + i + 1, x / 2)

    A = A - A.T
    logKprime = compute_logKprime1(n_min, n_max)
    s, logabsdet = np.linalg.slogdet(A)
    assert s >= 0
    return np.exp(logKprime + 0.5 * logabsdet)


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
    logKprime = compute_logKprime2(n_min, n_max)
    s, logabsdet = np.linalg.slogdet(A)
    assert s >= 0
    return np.exp(logKprime + 0.5 * logabsdet)


def min_eigval_cdf(p, df, x):
    return eigval_interval_probability(p, df, x, np.inf)


def chiani_2017_table2():
    """Generate the first few entries of Table 2 of [2].

    Note that we currently cannot generate more entries due to numerical
    difficulties.
    """
    print("n  probability")
    for n in [2, 5, 10]:
        prob = eigval_interval_probability(p=n, df=n, a=0, b=n)
        print(f"{n}  {prob}")


def chiani_2014_fig3():
    p = 5
    xs = np.linspace(0, 70, 100)
    plt.figure()
    for df in [5, 10, 15, 20, 25, 30, 35]:
        probs = [eigval_interval_probability(p, df, 0, x) for x in xs]
        plt.plot(xs, probs)
    plt.xlabel("x")
    plt.ylabel("CDF(x)")
    plt.grid()
    plt.show()


def main():
    p = 10
    df = 10  # degrees of freedom
    # Σ0 = np.diag([1, 1, 2])  # nominal scale matrix
    N = 10000  # number of simulation trials
    a = 0  # lower bound on eigenvalues
    b = np.inf  # upper bound on eigenvalues

    W = wishart(df=df, scale=np.eye(p))
    As = W.rvs(size=N)

    # bound matrix
    # B = α / df * Σ0

    # print(f"B = {B}")
    print(f"CDF(x = {b}) = {max_eigval_cdf(p, df, b)}")
    print(f"ψ(a = {a}, b = {b}) = {eigval_interval_probability(p, df, a, b)}")

    # num_psd_samples = 0
    # num_ratio = 0
    # for i in range(N):
    #     A = As[i, :, :]
    #     e, _ = np.linalg.eig(B - A)
    #     if np.min(e) >= 0:
    #         num_psd_samples += 1
    #
    # print(f"fraction of p.s.d. samples = {num_psd_samples / N}")
    #
    # # given p, find α such that CDF(α) = p using basic gradient-free
    # # optimization
    # p = 0.5
    #
    # def fun(α):
    #     c = cdf(3, df, α)
    #     return 0.5 * (c - p) ** 2
    #
    # res = minimize_scalar(fun, bounds=(0, 10), method="bounded")
    # if res.success:
    #     print(f"α = {res.x} for p = {p}")


if __name__ == "__main__":
    chiani_2017_table2()
    chiani_2014_figure3()
