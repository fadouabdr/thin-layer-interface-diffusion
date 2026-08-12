"""Resolved transient solver for the circular coated-granule illustration."""

import numpy as np
import dolfin as df

from .circular_config import (
    BULK_CELL,
    COATING_CELL,
    CORE_BOUNDARY,
    OUTER_BOUNDARY,
)
from .circular_mesh_utils import load_dolfin_circular_mesh
from .circular_initial_data import initialize_resolved_from_common
from .circular_results import (
    DEFAULT_RELEASE_FRACTIONS,
    release_fraction_indices,
)
from .circular_solver_utils import (
    create_resolved_bulk_flux_context,
    p1_vertex_values,
    recover_boundary_fluxes,
    recover_stationary_boundary_fluxes,
)
from .validation_config import D_F, K_IN, K_OUT, D_m, c_int
from .validation_metrics import number_of_time_steps


def run_circular_full_case(
    mesh_files,
    config,
    common_initial_state,
    snapshot_fractions=DEFAULT_RELEASE_FRACTIONS,
):
    """Run the resolved annulus-plus-exterior problem.

    The continuous transformed variable is ``q=c_m/K_OUT`` in the coating
    and ``q=c_f`` in the exterior.  Both models use one common exterior
    initial datum.  The resolved coating is initialized by an affine radial
    profile satisfying the two partition traces.
    """
    config.validate()
    mesh, cell_markers, facet_markers = load_dolfin_circular_mesh(mesh_files)
    V = df.FunctionSpace(mesh, "P", config.polynomial_degree)
    dx = df.Measure("dx", domain=mesh, subdomain_data=cell_markers)

    q_n, initialization_checks = initialize_resolved_from_common(
        V,
        cell_markers,
        config,
        common_initial_state,
    )
    q_new = df.Function(V)

    flux_context = create_resolved_bulk_flux_context(
        mesh,
        cell_markers,
        config,
        config.polynomial_degree,
    )
    (
        J0,
        Q0,
        stationary_residual,
        stationary_normalized,
    ) = recover_stationary_boundary_fluxes(
        q_n,
        D_F,
        flux_context,
    )

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    Dm_value = df.Constant(D_m(0.0, scenario=config.scenario))
    inner_value = df.Constant(
        (K_IN / K_OUT) * c_int(0.0, scenario=config.scenario)
    )
    bcs = [
        df.DirichletBC(V, inner_value, facet_markers, CORE_BOUNDARY),
        df.DirichletBC(V, df.Constant(0.0), facet_markers, OUTER_BOUNDARY),
    ]

    a = (
        (K_OUT / config.dt) * u * v * dx(COATING_CELL)
        + (1.0 / config.dt) * u * v * dx(BULK_CELL)
        + K_OUT
        * Dm_value
        * df.dot(df.grad(u), df.grad(v))
        * dx(COATING_CELL)
        + D_F
        * df.dot(df.grad(u), df.grad(v))
        * dx(BULK_CELL)
    )
    L = (
        (K_OUT / config.dt) * q_n * v * dx(COATING_CELL)
        + (1.0 / config.dt) * q_n * v * dx(BULK_CELL)
    )

    num_steps = number_of_time_steps(config.t_final, config.dt)
    times = [0.0]
    interface_flux = [J0]
    outer_outflow = [Q0]
    balance_residual = [stationary_residual]
    balance_normalized = [stationary_normalized]
    cumulative_release = [0.0]
    mass_bulk = [float(df.assemble(q_n * dx(BULK_CELL)))]
    mass_coating = [
        float(df.assemble(K_OUT * q_n * dx(COATING_CELL)))
    ]
    energy_bulk = [
        0.5 * float(df.assemble(q_n * q_n * dx(BULK_CELL)))
    ]

    # Keeping all P1 vertex vectors for one 1001-step run costs about 61 MB
    # with the accepted mesh.  It avoids a second expensive resolved solve
    # merely to recover the three release-based snapshot times.
    vertex_history = [p1_vertex_values(q_n)]
    M_release = 0.0
    solver = df.LUSolver()

    for step in range(num_steps):
        t_new = (step + 1) * config.dt
        Dm_value.assign(D_m(t_new, scenario=config.scenario))
        inner_value.assign(
            (K_IN / K_OUT) * c_int(t_new, scenario=config.scenario)
        )

        A = df.assemble(a)
        b = df.assemble(L)
        for bc in bcs:
            bc.apply(A)
            bc.apply(b)
        solver.solve(A, q_new.vector(), b)

        J_new, Q_new, residual, normalized = recover_boundary_fluxes(
            q_new,
            q_n,
            config.dt,
            D_F,
            flux_context,
        )
        M_release += config.dt * J_new

        times.append(t_new)
        interface_flux.append(J_new)
        outer_outflow.append(Q_new)
        balance_residual.append(residual)
        balance_normalized.append(normalized)
        cumulative_release.append(M_release)
        mass_bulk.append(float(df.assemble(q_new * dx(BULK_CELL))))
        mass_coating.append(
            float(df.assemble(K_OUT * q_new * dx(COATING_CELL)))
        )
        energy_bulk.append(
            0.5 * float(df.assemble(q_new * q_new * dx(BULK_CELL)))
        )
        vertex_history.append(p1_vertex_values(q_new))
        q_n.assign(q_new)

    times = np.asarray(times, dtype=float)
    cumulative_release = np.asarray(cumulative_release, dtype=float)
    snapshot_indices = release_fraction_indices(
        times,
        cumulative_release,
        fractions=snapshot_fractions,
    )
    snapshot_q = np.stack(
        [vertex_history[int(index)] for index in snapshot_indices],
        axis=0,
    )

    return {
        "model": "resolved",
        "times": times,
        "J_full": np.asarray(interface_flux, dtype=float),
        "Q_outer_full": np.asarray(outer_outflow, dtype=float),
        "bulk_balance_residual": np.asarray(balance_residual, dtype=float),
        "bulk_balance_normalized": np.asarray(
            balance_normalized,
            dtype=float,
        ),
        "M_full": cumulative_release,
        "mass_bulk_full": np.asarray(mass_bulk, dtype=float),
        "mass_coating_full": np.asarray(mass_coating, dtype=float),
        "energy_bulk_full": np.asarray(energy_bulk, dtype=float),
        "stationary_balance_residual": float(stationary_residual),
        "stationary_balance_normalized": float(stationary_normalized),
        "initialization_checks": initialization_checks,
        "snapshot_fractions": np.asarray(snapshot_fractions, dtype=float),
        "snapshot_indices": snapshot_indices,
        "snapshot_times": times[snapshot_indices],
        "snapshot_q_vertex": snapshot_q,
        "coordinates": np.asarray(mesh.coordinates(), dtype=float).copy(),
        "cells": np.asarray(mesh.cells(), dtype=np.int64).copy(),
        "cell_tags": np.asarray(cell_markers.array(), dtype=np.int64).copy(),
        "num_cells": int(mesh.num_cells()),
        "num_dofs": int(V.dim()),
        "interface_measure": flux_context["interface_measure"],
        "outer_measure": flux_context["outer_measure"],
    }
