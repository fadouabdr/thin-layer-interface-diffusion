"""Initial data shared by the resolved and reduced planar benchmarks.

The main asymptotic study uses a well-prepared profile.  At t=0 it satisfies
the partition values, the coating flux, the reduced Robin law, and the
exterior gradient simultaneously.  The former empty initialization remains
available only as a separately named transient experiment.
"""

from .validation_config import (
    ALPHA,
    D_F,
    K_IN,
    K_OUT,
    DEFAULT_SCENARIO,
    D_m,
    c_int,
    kappa_h,
)


def prepared_interface_concentration(h, scenario=DEFAULT_SCENARIO):
    """Return A_h such that c_f(x,0)=A_h(1-x_1) is flux compatible."""
    kappa0 = kappa_h(0.0, h, scenario=scenario)
    cint0 = c_int(0.0, scenario=scenario)

    return kappa0 * cint0 / (D_F + ALPHA * kappa0)


def prepared_release_flux(h, scenario=DEFAULT_SCENARIO):
    """Common initial flux of the full and reduced prepared profiles."""
    return D_F * prepared_interface_concentration(
        h,
        scenario=scenario,
    )


def prepared_bulk_value(x1, h, scenario=DEFAULT_SCENARIO):
    """Evaluate the prepared exterior concentration A_h(1-x_1)."""
    amplitude = prepared_interface_concentration(
        h,
        scenario=scenario,
    )
    return amplitude * (1.0 - x1)


def prepared_full_transformed_value(
    x1,
    h,
    scenario=DEFAULT_SCENARIO,
):
    """Evaluate the prepared transformed full variable q.

    In the coating q=c_m/K_OUT, and in the bulk q=c_f.
    """
    if x1 < 0.0:
        q_inner = (
            K_IN
            * c_int(0.0, scenario=scenario)
            / K_OUT
        )
        q_interface = prepared_interface_concentration(
            h,
            scenario=scenario,
        )
        y = (x1 + h) / h
        return q_inner + y * (q_interface - q_inner)

    return prepared_bulk_value(
        x1,
        h,
        scenario=scenario,
    )


def prepared_coating_flux(h, scenario=DEFAULT_SCENARIO):
    """Compute the initial full-model coating flux from its gradient."""
    q_inner = (
        K_IN
        * c_int(0.0, scenario=scenario)
        / K_OUT
    )
    q_interface = prepared_interface_concentration(
        h,
        scenario=scenario,
    )

    return (
        -K_OUT
        * D_m(0.0, scenario=scenario)
        * (q_interface - q_inner)
        / h
    )


def prepared_robin_flux(h, scenario=DEFAULT_SCENARIO):
    """Compute the initial reduced-model flux from the Robin law."""
    amplitude = prepared_interface_concentration(
        h,
        scenario=scenario,
    )

    return kappa_h(
        0.0,
        h,
        scenario=scenario,
    ) * (
        c_int(0.0, scenario=scenario)
        - ALPHA * amplitude
    )
