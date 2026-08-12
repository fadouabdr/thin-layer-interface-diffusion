"""Pure helpers for selecting and comparing circular illustration results."""

import numpy as np


def affine_profile_values(inner_value, outer_values, fractions):
    """Return affine values joining an inner value to outer trace values."""
    outer_values = np.asarray(outer_values, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if outer_values.shape != fractions.shape:
        raise ValueError("outer_values and fractions must have the same shape.")
    if not np.all(np.isfinite(outer_values)) or not np.all(
        np.isfinite(fractions)
    ):
        raise ValueError("profile inputs must be finite.")
    if np.any(fractions < 0.0) or np.any(fractions > 1.0):
        raise ValueError("profile fractions must lie in [0, 1].")
    return (1.0 - fractions) * float(inner_value) + fractions * outer_values


DEFAULT_RELEASE_FRACTIONS = (0.10, 0.50, 0.90)


def release_fraction_indices(
    times,
    cumulative_release,
    fractions=DEFAULT_RELEASE_FRACTIONS,
):
    """Return the first indices reaching fractions of terminal release.

    The cumulative curve is allowed to contain small local reversals.  The
    sign of its terminal value determines the crossing direction, so the
    helper also remains well defined for a consistently negative signed
    release history.
    """
    times = np.asarray(times, dtype=float)
    cumulative_release = np.asarray(cumulative_release, dtype=float)
    fractions = np.asarray(fractions, dtype=float)

    if times.ndim != 1 or cumulative_release.ndim != 1:
        raise ValueError("times and cumulative_release must be one-dimensional.")
    if times.shape != cumulative_release.shape:
        raise ValueError("times and cumulative_release must have the same shape.")
    if len(times) < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError("times must contain at least two increasing values.")
    if not np.all(np.isfinite(times)) or not np.all(
        np.isfinite(cumulative_release)
    ):
        raise ValueError("times or cumulative_release contains non-finite values.")
    if fractions.ndim != 1 or len(fractions) == 0:
        raise ValueError("fractions must be a nonempty one-dimensional sequence.")
    if not np.all(np.isfinite(fractions)) or not np.all(
        (fractions > 0.0) & (fractions <= 1.0)
    ):
        raise ValueError("release fractions must lie in (0, 1].")
    if not np.all(np.diff(fractions) > 0.0):
        raise ValueError("release fractions must be strictly increasing.")

    terminal = float(cumulative_release[-1])
    scale = max(float(np.max(np.abs(cumulative_release))), 1.0)
    if abs(terminal) <= 1.0e-14 * scale:
        raise ValueError("terminal cumulative release is too small for fractions.")

    direction = 1.0 if terminal > 0.0 else -1.0
    directed_curve = direction * cumulative_release
    directed_terminal = direction * terminal

    indices = []
    for fraction in fractions:
        target = float(fraction) * directed_terminal
        crossings = np.flatnonzero(directed_curve >= target)
        if len(crossings) == 0:
            # The final sample reaches the target by construction, apart from
            # possible roundoff in a non-monotone signed curve.
            index = len(times) - 1
        else:
            index = int(crossings[0])
        indices.append(index)

    return np.asarray(indices, dtype=np.int64)


def validate_snapshot_indices(indices, num_time_samples):
    """Validate snapshot indices against a common time grid."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("snapshot indices must be a nonempty 1D sequence.")
    if num_time_samples < 1:
        raise ValueError("num_time_samples must be positive.")
    if np.any(indices < 0) or np.any(indices >= num_time_samples):
        raise ValueError("a snapshot index lies outside the time grid.")
    if np.any(np.diff(indices) < 0):
        raise ValueError("snapshot indices must be nondecreasing.")
    return indices


def relative_max_mismatch(first, second):
    """Return max|first-second| normalized by the largest curve magnitude."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError("curves must have the same shape.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("a curve contains non-finite values.")
    scale = max(
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
        1.0e-14,
    )
    return float(np.max(np.abs(first - second)) / scale)
