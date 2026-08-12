import math

import numpy as np
from dolfin import (
    Function,
    LagrangeInterpolator,
    Measure,
    assemble,
)


def empirical_rates(h_values, errors):
    """
    Compute observed rates between consecutive thickness values.
    """

    h_values = np.asarray(h_values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if h_values.shape != errors.shape:
        raise ValueError(
            "h_values and errors must have the same shape."
        )

    rates = [np.nan]

    for i in range(1, len(h_values)):
        h_previous = h_values[i - 1]
        h_current = h_values[i]
        error_previous = errors[i - 1]
        error_current = errors[i]

        if (
            h_previous > 0.0
            and h_current > 0.0
            and error_previous > 0.0
            and error_current > 0.0
        ):
            rate = np.log(
                error_previous / error_current
            ) / np.log(
                h_previous / h_current
            )
        else:
            rate = np.nan

        rates.append(rate)

    return np.asarray(rates, dtype=float)


def validate_matching_time_grids(
    times_first,
    times_second,
    atol=1.0e-12,
):
    """
    Verify that two simulations use the same strictly increasing time grid.
    """

    times_first = np.asarray(times_first, dtype=float)
    times_second = np.asarray(times_second, dtype=float)

    if times_first.ndim != 1 or times_second.ndim != 1:
        raise ValueError("Time grids must be one-dimensional.")

    if times_first.shape != times_second.shape:
        raise ValueError(
            "The full and reduced time grids have different lengths."
        )

    if not np.all(np.isfinite(times_first)):
        raise ValueError(
            "The first time grid contains non-finite values."
        )

    if not np.all(np.isfinite(times_second)):
        raise ValueError(
            "The second time grid contains non-finite values."
        )

    if not np.all(np.diff(times_first) > 0.0):
        raise ValueError(
            "The first time grid is not strictly increasing."
        )

    if not np.all(np.diff(times_second) > 0.0):
        raise ValueError(
            "The second time grid is not strictly increasing."
        )

    if not np.allclose(
        times_first,
        times_second,
        rtol=0.0,
        atol=atol,
    ):
        maximum_difference = np.max(
            np.abs(times_first - times_second)
        )

        raise ValueError(
            "The full and reduced time grids do not match. "
            f"Maximum difference: {maximum_difference:.6e}"
        )

    return times_first


def backward_euler_cumulative(times, flux):
    """
    Reconstruct cumulative release using right-endpoint quadrature:

        M^{n+1} = M^n + dt_n * J^{n+1}.

    This is the time accumulation consistent with the backward-Euler
    discretization used by the full and reduced solvers.
    """

    times = np.asarray(times, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if times.ndim != 1 or flux.ndim != 1:
        raise ValueError(
            "times and flux must be one-dimensional."
        )

    if times.shape != flux.shape:
        raise ValueError(
            "times and flux must have the same length."
        )

    if not np.all(np.isfinite(times)):
        raise ValueError(
            "times contains non-finite values."
        )

    if not np.all(np.isfinite(flux)):
        raise ValueError(
            "flux contains non-finite values."
        )

    time_steps = np.diff(times)

    if not np.all(time_steps > 0.0):
        raise ValueError(
            "The time grid must be strictly increasing."
        )

    cumulative = np.zeros_like(times, dtype=float)

    for n in range(1, len(times)):
        cumulative[n] = (
            cumulative[n - 1]
            + time_steps[n - 1] * flux[n]
        )

    return cumulative


def compute_L2_bulk_error(c_full, c_reduced):
    """
    Compute the full-versus-reduced field error on the common bulk domain.

    The transformed full solution q equals the physical concentration c_f
    in the exterior bulk. It is interpolated onto the reduced finite-element
    space, after which the L2 error is assembled over that bulk mesh.
    """

    reduced_space = c_reduced.function_space()
    reduced_mesh = reduced_space.mesh()
    dx_reduced = Measure(
        "dx",
        domain=reduced_mesh,
    )

    # Permit evaluation at reduced-mesh nodes located on the shared boundary.
    c_full.set_allow_extrapolation(True)

    c_full_on_reduced = Function(reduced_space)

    LagrangeInterpolator.interpolate(
        c_full_on_reduced,
        c_full,
    )

    difference = c_full_on_reduced - c_reduced

    absolute_error = math.sqrt(
        assemble(
            difference
            * difference
            * dx_reduced
        )
    )

    full_norm = math.sqrt(
        assemble(
            c_full_on_reduced
            * c_full_on_reduced
            * dx_reduced
        )
    )

    relative_error = absolute_error / max(
        full_norm,
        1.0e-14,
    )

    return {
        "absolute_error": float(absolute_error),
        "relative_error": float(relative_error),
        "full_norm": float(full_norm),
        "full_on_reduced": c_full_on_reduced,
    }


def final_abs_error(first_curve, second_curve):
    """
    Return the absolute discrepancy at the final time.
    """

    first_curve = np.asarray(first_curve, dtype=float)
    second_curve = np.asarray(second_curve, dtype=float)

    if first_curve.size == 0 or second_curve.size == 0:
        raise ValueError("Curves must not be empty.")

    if not np.isfinite(first_curve[-1]):
        raise ValueError(
            "The first final value is not finite."
        )

    if not np.isfinite(second_curve[-1]):
        raise ValueError(
            "The second final value is not finite."
        )

    return float(
        abs(first_curve[-1] - second_curve[-1])
    )


def max_abs_curve_error(first_curve, second_curve):
    """
    Return the maximum absolute discrepancy over a common time grid.
    """

    first_curve = np.asarray(first_curve, dtype=float)
    second_curve = np.asarray(second_curve, dtype=float)

    if first_curve.shape != second_curve.shape:
        raise ValueError(
            "Curves must have the same shape."
        )

    if not np.all(np.isfinite(first_curve)):
        raise ValueError(
            "The first curve contains non-finite values."
        )

    if not np.all(np.isfinite(second_curve)):
        raise ValueError(
            "The second curve contains non-finite values."
        )

    return float(
        np.max(
            np.abs(first_curve - second_curve)
        )
    )


def relative_error(absolute_error, reference_value):
    """
    Normalize an absolute error by a nonzero reference magnitude.
    """

    return float(
        absolute_error
        / max(abs(float(reference_value)), 1.0e-14)
    )
