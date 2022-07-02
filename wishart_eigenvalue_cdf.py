import numpy as np
from scipy.stats import wishart
from scipy.special import gamma, gammainc
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


df = 3  # degrees of freedom
Σ0 = np.diag([1, 1, 2])  # nominal scale matrix
N = 10000  # number of simulation trials
α = 10  # upper bound on max eigenvalue


def gammam(m, a):
    y = np.pi ** (m * (m - 1) / 4)
    for i in range(m):
        y *= gamma(a - i / 2)
    return y


def K1(n_min, n_max, α):
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


def cdf(p, df, x):
    """CDF of the largest eigenvalue of a Wishart matrix of dimension (p, p)
    with df degrees of freedom.

    Probability{ largest eigenvalue of A <= x } = cdf(p, df, x),

    where A ~ Wishart(p, df).
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
        q[i] = 2 ** -β * gamma(β) * gammainc(β, x)

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
    return K1(n_min, n_max, α) * np.sqrt(np.linalg.det(A))


def main():
    Σ = Σ0 / df
    W = wishart(df=df, scale=Σ)
    As = W.rvs(size=N)

    # bound matrix
    B = α / df * Σ0

    print(f"B = {B}")
    print(f"CDF(α = {α}) = {cdf(3, df, α)}")

    num_psd_samples = 0
    num_ratio = 0
    for i in range(N):
        A = As[i, :, :]
        e, _ = np.linalg.eig(B - A)
        if np.min(e) >= 0:
            num_psd_samples += 1

    print(f"fraction of p.s.d. samples = {num_psd_samples / N}")

    # given p, find α such that CDF(α) = p using basic gradient-free
    # optimization
    p = 0.5
    def fun(α):
        c = cdf(3, df, α)
        return 0.5 * (c - p)**2

    res = minimize_scalar(fun, bounds=(0, 10), method="bounded")
    if res.success:
        print(f"α = {res.x} for p = {p}")


if __name__ == "__main__":
    main()
