import numpy as np


# Physical parameters
D_F = 1.0
K_IN = 1.0
K_OUT = 1.3
ALPHA = K_OUT / K_IN

# Legacy June-2026 defaults.  They are retained so the historical scripts can
# still be inspected, but the canonical validation scripts below use the
# explicit candidate settings defined further down.
T_FINAL = 0.1
DT_VALIDATION = 1.0e-4

# Coating thicknesses used in the validation
THICKNESSES = [0.08, 0.04, 0.02, 0.01]

# Baseline bulk mesh resolution
BULK_RESOLUTION = 64

# Canonical validation scenarios
CONSTANT_SCENARIO = "constant"
TIME_DEPENDENT_SCENARIO = "time_dependent"
SCENARIOS = (
    CONSTANT_SCENARIO,
    TIME_DEPENDENT_SCENARIO,
)
DEFAULT_SCENARIO = TIME_DEPENDENT_SCENARIO

# Initial-data choices
EMPTY_INITIALIZATION = "empty"
PREPARED_INITIALIZATION = "prepared"
INITIALIZATIONS = (
    EMPTY_INITIALIZATION,
    PREPARED_INITIALIZATION,
)
DEFAULT_INITIALIZATION = PREPARED_INITIALIZATION

# Candidate settings for the new, genuinely non-autonomous benchmark.  The
# time step is deliberately labelled as a candidate: it must be accepted or
# replaced using scripts/run_planar_resolution_audit.py before production
# results are reported.
CANONICAL_T_FINAL = 1.0
CANONICAL_DT_CANDIDATE = 1.0e-3


def validate_scenario(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}."
        )

    return scenario


def validate_initialization(initialization):
    if initialization not in INITIALIZATIONS:
        raise ValueError(
            "Unknown initialization "
            f"{initialization!r}; expected one of {INITIALIZATIONS}."
        )

    return initialization


def D_m(t, scenario=DEFAULT_SCENARIO):
    """Coating diffusivity for the selected validation scenario."""
    validate_scenario(scenario)

    if scenario == CONSTANT_SCENARIO:
        return 1.0

    return 1.0 + 0.5 * np.sin(t)


def kappa_h(t, h, scenario=DEFAULT_SCENARIO):
    """Effective interface coefficient."""
    if h <= 0.0:
        raise ValueError("The coating thickness h must be positive.")

    return K_IN * D_m(t, scenario=scenario) / h


def c_int(t, scenario=DEFAULT_SCENARIO):
    """Prescribed concentration inside the granule."""
    validate_scenario(scenario)
    return np.exp(-t)


def scenario_metadata(scenario):
    """Return human-readable metadata used in result manifests."""
    validate_scenario(scenario)

    if scenario == CONSTANT_SCENARIO:
        diffusivity_law = "D_m(t) = 1"
    else:
        diffusivity_law = "D_m(t) = 1 + 0.5 sin(t)"

    return {
        "scenario": scenario,
        "diffusivity_law": diffusivity_law,
        "core_concentration_law": "c_int(t) = exp(-t)",
    }
