"""Reduced transient solver for the circular effective-interface model."""

import numpy as np
import dolfin as df

from .circular_config import CORE_BOUNDARY, OUTER_BOUNDARY
from .circular_mesh_utils import load_dolfin_circular_mesh
from .circular_initial_data import initialize_reduced_from_common
from .circular_results import validate_snapshot_indices
from .circular_solver_utils import (
    create_direct_flux_context,
    p1_vertex_values,
    recover_boundary_fluxes,
    recover_stationary_boundary_fluxes,
)
from .validation_config import ALPHA, D_F, c_int, kappa_h
from .validation_metrics import number_of_time_steps


def run_circular_reduced_case(
    mesh_files,
    config,
    common_initial_state,
    snapshot_indices,
):
    """Run the reduced exterior problem on the accepted circular mesh."""
    config.validate()
    num_steps = number_of_time_steps(config.t_final, config.dt)
    snapshot_indices = validate_snapshot_indices(
        snapshot_indices,
        num_steps + 1,
    )

    mesh, cell_markers, facet_markers = load_dolfin_circular_mesh(mesh_files)
    V = df.FunctionSpace(mesh, "P", config.polynomial_degree)
    dx = df.Measure("dx", domain=mesh)
    ds = df.Measure("ds", domain=mesh, subdomain_data=facet_markers)
    c_n, initialization_checks = initialize_reduced_from_common(
        V,
        common_initial_state,
    )
    c_new = df.Function(V)

    flux_context = create_direct_flux_context(
        mesh,
        facet_markers,
        CORE_BOUNDARY,
        OUTER_BOUNDARY,
        config.polynomial_degree,
    )
    J_variational0, Q0, _, _ = recover_stationary_boundary_fluxes(
        c_n,
        D_F,
        flux_context,
    )
    kappa0 = kappa_h(
        0.0,
        config.coating_thickness,
        scenario=config.scenario,
    )
    cint0 = c_int(0.0, scenario=config.scenario)
    J0 = float(
        df.assemble(
            kappa0
            * (cint0 - ALPHA * c_n)
            * ds(CORE_BOUNDARY)
        )
    )
    stationary_residual = J0 - Q0
    stationary_scale = abs(J0) + abs(Q0)
    stationary_normalized = abs(stationary_residual) / max(
        stationary_scale,
        1.0e-14,
    )
    stationary_robin_mismatch = J_variational0 - J0

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    kappa_value = df.Constant(kappa0)
    cint_value = df.Constant(cint0)
    a = (
        (1.0 / config.dt) * u * v * dx
        + D_F * df.dot(df.grad(u), df.grad(v)) * dx
        + ALPHA
        * kappa_value
        * u
        * v
        * ds(CORE_BOUNDARY)
    )
    L = (
        (1.0 / config.dt) * c_n * v * dx
        + kappa_value * cint_value * v * ds(CORE_BOUNDARY)
    )
    bc = df.DirichletBC(
        V,
        df.Constant(0.0),
        facet_markers,
        OUTER_BOUNDARY,
    )

    times = [0.0]
    robin_flux = [J0]
    variational_flux = [J_variational0]
    outer_outflow = [Q0]
    balance_residual = [stationary_residual]
    balance_normalized = [stationary_normalized]
    robin_mismatch = [stationary_robin_mismatch]
    cumulative_release = [0.0]
    mass_bulk = [float(df.assemble(c_n * dx))]
    energy_bulk = [0.5 * float(df.assemble(c_n * c_n * dx))]

    requested_steps = set(int(index) for index in snapshot_indices)
    snapshot_by_step = {}
    if 0 in requested_steps:
        snapshot_by_step[0] = p1_vertex_values(c_n)

    solver = df.LUSolver()
    M_release = 0.0
    for step in range(num_steps):
        step_number = step + 1
        t_new = step_number * config.dt
        kappa_new = kappa_h(
            t_new,
            config.coating_thickness,
            scenario=config.scenario,
        )
        cint_new = c_int(t_new, scenario=config.scenario)
        kappa_value.assign(kappa_new)
        cint_value.assign(cint_new)

        A = df.assemble(a)
        b = df.assemble(L)
        bc.apply(A)
        bc.apply(b)
        solver.solve(A, c_new.vector(), b)

        J_new = float(
            df.assemble(
                kappa_new
                * (cint_new - ALPHA * c_new)
                * ds(CORE_BOUNDARY)
            )
        )
        J_variational, Q_new, residual_variational, _ = (
            recover_boundary_fluxes(
                c_new,
                c_n,
                config.dt,
                D_F,
                flux_context,
            )
        )
        mismatch = J_variational - J_new
        residual = residual_variational - mismatch
        mass_rate = J_variational - Q_new - residual_variational
        scale = abs(J_new) + abs(Q_new) + abs(mass_rate)
        normalized = abs(residual) / max(scale, 1.0e-14)
        M_release += config.dt * J_new

        times.append(t_new)
        robin_flux.append(J_new)
        variational_flux.append(J_variational)
        outer_outflow.append(Q_new)
        balance_residual.append(residual)
        balance_normalized.append(normalized)
        robin_mismatch.append(mismatch)
        cumulative_release.append(M_release)
        mass_bulk.append(float(df.assemble(c_new * dx)))
        energy_bulk.append(0.5 * float(df.assemble(c_new * c_new * dx)))
        if step_number in requested_steps:
            snapshot_by_step[step_number] = p1_vertex_values(c_new)
        c_n.assign(c_new)

    missing = sorted(requested_steps.difference(snapshot_by_step))
    if missing:
        raise RuntimeError(f"Reduced snapshots were not captured at {missing}.")
    snapshot_c = np.stack(
        [snapshot_by_step[int(index)] for index in snapshot_indices],
        axis=0,
    )

    times = np.asarray(times, dtype=float)
    return {
        "model": "reduced",
        "times": times,
        "J_red": np.asarray(robin_flux, dtype=float),
        "J_variational_red": np.asarray(variational_flux, dtype=float),
        "Q_outer_red": np.asarray(outer_outflow, dtype=float),
        "bulk_balance_residual": np.asarray(balance_residual, dtype=float),
        "bulk_balance_normalized": np.asarray(
            balance_normalized,
            dtype=float,
        ),
        "robin_flux_mismatch": np.asarray(robin_mismatch, dtype=float),
        "M_red": np.asarray(cumulative_release, dtype=float),
        "mass_red": np.asarray(mass_bulk, dtype=float),
        "energy_red": np.asarray(energy_bulk, dtype=float),
        "stationary_balance_residual": float(stationary_residual),
        "stationary_balance_normalized": float(stationary_normalized),
        "stationary_robin_mismatch": float(stationary_robin_mismatch),
        "initialization_checks": initialization_checks,
        "snapshot_indices": snapshot_indices,
        "snapshot_times": times[snapshot_indices],
        "snapshot_c_vertex": snapshot_c,
        "coordinates": np.asarray(mesh.coordinates(), dtype=float).copy(),
        "cells": np.asarray(mesh.cells(), dtype=np.int64).copy(),
        "cell_tags": np.asarray(cell_markers.array(), dtype=np.int64).copy(),
        "num_cells": int(mesh.num_cells()),
        "num_dofs": int(V.dim()),
    }
