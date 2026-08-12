import numpy as np

from .config import ALPHA, D_F, kappa


def c_exact(x1, t):
    """
    Exact manufactured solution on Omega = (0, 1)^2:

        c_exact(x, t) = exp(t) * (2 - x_1 - x_1^2).

    The solution is independent of x_2.
    """
    return np.exp(t) * (2.0 - x1 - x1**2)


def dt_c_exact(x1, t):
    """Time derivative of the exact solution."""
    return np.exp(t) * (2.0 - x1 - x1**2)


def dc_dx1_exact(x1, t):
    """Derivative of the exact solution with respect to x_1."""
    return -np.exp(t) * (1.0 + 2.0 * x1)


def laplacian_c_exact(x1, t):
    """
    Laplacian of the exact solution.

    Since the solution is independent of x_2,

        Delta c_exact = partial_{x_1 x_1} c_exact = -2 exp(t).
    """
    return -2.0 * np.exp(t) * np.ones_like(x1)


def c_initial(x1):
    """Initial condition c_exact(x, 0)."""
    return 2.0 - x1 - x1**2


def source(x1, t):
    """
    Manufactured source for

        partial_t c - D_F Delta c = source.
    """
    return np.exp(t) * (
        2.0
        - x1
        - x1**2
        + 2.0 * D_F
    )


def c_int(t):
    """
    Reservoir concentration compatible with the Robin condition

        D_F grad(c) . nu
            = kappa(t) * (c_int(t) - ALPHA * c)

    on Gamma_R = {x_1 = 0}, where nu = (-1, 0).
    """
    return (
        2.0 * ALPHA * np.exp(t)
        + D_F * np.exp(t) / kappa(t)
    )
