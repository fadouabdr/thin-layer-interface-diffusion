"""Fast FEniCS checks to run before any expensive validation case."""

import numpy as np
from dolfin import (
    Expression,
    FunctionSpace,
    Measure,
    assemble,
    interpolate,
)

from src.full_mesh_utils import (
    BULK_TAG,
    COATING_TAG,
    create_fitted_full_mesh,
)
from src.full_thin_layer_solver import (
    compute_bulk_boundary_fluxes,
    create_bulk_flux_context,
)
from src.initial_data import (
    prepared_coating_flux,
    prepared_release_flux,
    prepared_robin_flux,
)
from src.validation_config import DEFAULT_SCENARIO


def require_close(name, actual, expected, tolerance=1.0e-10):
    if not np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=tolerance,
    ):
        raise RuntimeError(
            f"{name}: expected {expected}, found {actual}."
        )


def main():
    h = 0.01
    mesh, cell_markers, _ = create_fitted_full_mesh(
        h=h,
        n_layer=8,
        n_bulk=64,
        ny=64,
    )
    dx = Measure(
        "dx",
        domain=mesh,
        subdomain_data=cell_markers,
    )

    require_close(
        "coating area",
        assemble(1.0 * dx(COATING_TAG)),
        h,
    )
    require_close(
        "bulk area",
        assemble(1.0 * dx(BULK_TAG)),
        1.0,
    )

    expected_flux = prepared_release_flux(
        h,
        scenario=DEFAULT_SCENARIO,
    )
    require_close(
        "prepared coating-gradient flux",
        prepared_coating_flux(
            h,
            scenario=DEFAULT_SCENARIO,
        ),
        expected_flux,
    )
    require_close(
        "prepared Robin flux",
        prepared_robin_flux(
            h,
            scenario=DEFAULT_SCENARIO,
        ),
        expected_flux,
    )

    V = FunctionSpace(mesh, "P", 1)
    test_profile = interpolate(
        Expression("1.0 - x[0]", degree=1),
        V,
    )
    flux_context = create_bulk_flux_context(
        mesh=mesh,
        cell_markers=cell_markers,
        degree=1,
    )
    (
        interface_inflow,
        outer_outflow,
        balance_residual,
        normalized_residual,
    ) = compute_bulk_boundary_fluxes(
        q_full_new=test_profile,
        q_full_old=test_profile,
        dt_value=1.0,
        flux_context=flux_context,
    )

    require_close(
        "positive interface inflow for c=1-x",
        interface_inflow,
        1.0,
    )
    require_close(
        "positive outer outflow for c=1-x",
        outer_outflow,
        1.0,
    )
    require_close(
        "linear-profile balance residual",
        balance_residual,
        0.0,
    )
    require_close(
        "linear-profile normalized residual",
        normalized_residual,
        0.0,
    )

    print("All planar setup and flux-sign checks passed.")


if __name__ == "__main__":
    main()
