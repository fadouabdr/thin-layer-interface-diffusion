"""Pure helpers for validating planar resolution-audit coverage."""

import numpy as np


def normalized_thicknesses(values):
    """Validate and return a deterministic list of coating thicknesses."""
    result = [float(value) for value in values]
    if not result or not np.all(np.isfinite(result)):
        raise ValueError("Thickness values must be a nonempty finite list.")
    if any(value <= 0.0 for value in result):
        raise ValueError("Every coating thickness must be positive.")
    for index, value in enumerate(result):
        if any(
            np.isclose(value, earlier, rtol=0.0, atol=1.0e-12)
            for earlier in result[:index]
        ):
            raise ValueError("The coating-thickness list contains duplicates.")
    return result


def audit_covers_thicknesses(audited, required):
    """Return whether every required thickness appears in the audit."""
    audited = normalized_thicknesses(audited)
    required = normalized_thicknesses(required)
    return all(
        any(np.isclose(value, item, rtol=0.0, atol=1.0e-12)
            for item in audited)
        for value in required
    )


def is_valid_coverage_extension(existing, requested):
    """Return whether ``requested`` preserves all existing thicknesses."""
    return audit_covers_thicknesses(requested, existing)
