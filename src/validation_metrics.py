"""Pure numerical diagnostics that do not require FEniCS."""

import math

import numpy as np


def number_of_time_steps(t_final, dt_value, tolerance=1.0e-12):
    """Return T/dt and reject a grid that does not end exactly at T."""
    if t_final <= 0.0:
        raise ValueError("t_final must be positive.")

    if dt_value <= 0.0:
        raise ValueError("dt_value must be positive.")

    ratio = t_final / dt_value
    num_steps = int(round(ratio))

    if num_steps < 1:
        raise ValueError("The time grid must contain at least one step.")

    if not math.isclose(
        num_steps * dt_value,
        t_final,
        rel_tol=0.0,
        abs_tol=tolerance * max(1.0, abs(t_final)),
    ):
        raise ValueError(
            "t_final must be an integer multiple of dt_value: "
            f"T={t_final}, dt={dt_value}."
        )

    return num_steps


def validate_matching_time_grids(
    times_first,
    times_second,
    atol=1.0e-12,
):
    """Validate two common, strictly increasing one-dimensional grids."""
    first = np.asarray(times_first, dtype=float)
    second = np.asarray(times_second, dtype=float)

    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Time grids must be one-dimensional.")

    if first.shape != second.shape:
        raise ValueError("The time grids have different lengths.")

    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("A time grid contains non-finite values.")

    if not np.all(np.diff(first) > 0.0):
        raise ValueError("The first time grid is not strictly increasing.")

    if not np.all(np.diff(second) > 0.0):
        raise ValueError("The second time grid is not strictly increasing.")

    if not np.allclose(first, second, rtol=0.0, atol=atol):
        raise ValueError(
            "The time grids do not match; maximum difference "
            f"{np.max(np.abs(first - second)):.6e}."
        )

    return first


def backward_euler_cumulative(times, flux):
    """Accumulate M^{n+1}=M^n+dt_n J^{n+1}."""
    times = np.asarray(times, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if times.ndim != 1 or flux.ndim != 1:
        raise ValueError("times and flux must be one-dimensional.")

    if times.shape != flux.shape:
        raise ValueError("times and flux must have the same length.")

    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(flux)):
        raise ValueError("times or flux contains non-finite values.")

    time_steps = np.diff(times)

    if not np.all(time_steps > 0.0):
        raise ValueError("The time grid must be strictly increasing.")

    cumulative = np.zeros_like(times, dtype=float)
    cumulative[1:] = np.cumsum(time_steps * flux[1:])
    return cumulative


def relative_time_l2_error(times, first, reference):
    """Compute the backward-Euler-weighted relative L2(0,T) error."""
    times = np.asarray(times, dtype=float)
    first = np.asarray(first, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if times.shape != first.shape or times.shape != reference.shape:
        raise ValueError("times and both curves must have the same shape.")

    if not (
        np.all(np.isfinite(times))
        and np.all(np.isfinite(first))
        and np.all(np.isfinite(reference))
    ):
        raise ValueError("The time grid or a curve contains non-finite values.")

    time_steps = np.diff(times)

    if not np.all(time_steps > 0.0):
        raise ValueError("The time grid must be strictly increasing.")

    difference_squared = (first[1:] - reference[1:]) ** 2
    reference_squared = reference[1:] ** 2

    numerator = math.sqrt(float(np.sum(time_steps * difference_squared)))
    denominator = math.sqrt(float(np.sum(time_steps * reference_squared)))

    return numerator / max(denominator, 1.0e-14)


def normalized_balance_residual(
    residual,
    mass_rate,
    interface_inflow,
    outer_outflow,
):
    """Normalize a stepwise exterior-domain balance residual."""
    scale = (
        abs(float(mass_rate))
        + abs(float(interface_inflow))
        + abs(float(outer_outflow))
    )
    return abs(float(residual)) / max(scale, 1.0e-14)


def empirical_rates(h_values, errors):
    """Compute rates between consecutive decreasing thicknesses."""
    h_values = np.asarray(h_values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if h_values.shape != errors.shape:
        raise ValueError("h_values and errors must have the same shape.")

    rates = np.full(h_values.shape, np.nan, dtype=float)

    for index in range(1, len(h_values)):
        if (
            h_values[index - 1] > 0.0
            and h_values[index] > 0.0
            and errors[index - 1] > 0.0
            and errors[index] > 0.0
        ):
            rates[index] = (
                math.log(errors[index - 1] / errors[index])
                / math.log(h_values[index - 1] / h_values[index])
            )

    return rates
