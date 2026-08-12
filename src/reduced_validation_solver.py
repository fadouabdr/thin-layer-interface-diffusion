import csv
from pathlib import Path

import numpy as np
from dolfin import *

from .mesh_utils import create_square_mesh, ROBIN_TAG, DIRICHLET_TAG
from .validation_config import (
    D_F,
    ALPHA,
    T_FINAL,
    DT_VALIDATION,
    BULK_RESOLUTION,
    DEFAULT_INITIALIZATION,
    DEFAULT_SCENARIO,
    EMPTY_INITIALIZATION,
    PREPARED_INITIALIZATION,
    kappa_h,
    c_int,
    validate_initialization,
    validate_scenario,
)
from .initial_data import (
    prepared_bulk_value,
    prepared_release_flux,
)
from .validation_metrics import (
    normalized_balance_residual,
    number_of_time_steps,
)


class ReducedInitialCondition(UserExpression):
    """Prepared exterior initial profile A_h(1-x_1)."""

    def __init__(self, h, scenario, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.scenario = scenario

    def eval(self, value, x):
        value[0] = prepared_bulk_value(
            x1=x[0],
            h=self.h,
            scenario=self.scenario,
        )

    def value_shape(self):
        return ()


def cumulative_release_update(M_old, J_new, dt):
    """Advance cumulative release with the backward-Euler time rule."""
    return M_old + dt * J_new


def compute_reduced_boundary_fluxes(
    c_new,
    c_old,
    dt_value,
    dx,
    psi_interface,
    psi_outer,
    robin_flux,
):
    """Recover the reduced outer flux and independently audit Robin inflow."""
    time_derivative = (c_new - c_old) / dt_value

    residual_interface_flux = assemble(
        time_derivative * psi_interface * dx
        + D_F
        * dot(grad(c_new), grad(psi_interface))
        * dx
    )

    outer_signed_flux = assemble(
        time_derivative * psi_outer * dx
        + D_F
        * dot(grad(c_new), grad(psi_outer))
        * dx
    )
    outer_outflow = -outer_signed_flux

    bulk_mass_rate = assemble(time_derivative * dx)
    balance_residual = (
        robin_flux
        - outer_outflow
        - bulk_mass_rate
    )
    normalized_residual = normalized_balance_residual(
        residual=balance_residual,
        mass_rate=bulk_mass_rate,
        interface_inflow=robin_flux,
        outer_outflow=outer_outflow,
    )

    return (
        float(outer_outflow),
        float(balance_residual),
        float(normalized_residual),
        float(residual_interface_flux - robin_flux),
    )


def run_reduced_validation_case(
    h,
    resolution=BULK_RESOLUTION,
    dt_value=DT_VALIDATION,
    degree=1,
    t_final=T_FINAL,
    scenario=DEFAULT_SCENARIO,
    initialization=DEFAULT_INITIALIZATION,
):
    """
    Solve the reduced interface model on Omega_f=(0,1)^2.

     Model:
        partial_t c - div(D_F grad c) = 0 in Omega_f.

        On Gamma_g = {x_1 = 0}, n is directed from the granule
        toward the bulk, while the outward normal of Omega_f is -n:

            -D_F grad(c) . n
                = kappa_h(t) * (c_int(t) - ALPHA * c).

        The remaining boundary conditions are

            c = 0 on Gamma_D = {x_1 = 1},

        with homogeneous Neumann conditions on x_2 = 0 and x_2 = 1.

    This is the reduced half of the validation. It must later be compared
    to the bulk component of the full thin-layer solution.
    """

    validate_scenario(scenario)
    validate_initialization(initialization)

    mesh, boundaries = create_square_mesh(resolution)

    V = FunctionSpace(mesh, "P", degree)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = Measure("dx", domain=mesh)

    u = TrialFunction(V)
    v = TestFunction(V)

    if initialization == PREPARED_INITIALIZATION:
        c_n = interpolate(
            ReducedInitialCondition(
                h=h,
                scenario=scenario,
                degree=1,
            ),
            V,
        )
    elif initialization == EMPTY_INITIALIZATION:
        c_n = interpolate(Constant(0.0), V)
    else:
        raise AssertionError("Initialization validation failed.")

    u_sol = Function(V)

    bc = DirichletBC(V, Constant(0.0), boundaries, DIRICHLET_TAG)

    kappa_const = Constant(
        kappa_h(0.0, h, scenario=scenario)
    )
    cint_const = Constant(
        c_int(0.0, scenario=scenario)
    )

    psi_interface = interpolate(
        Expression("1.0 - x[0]", degree=1),
        V,
    )
    psi_outer = interpolate(
        Expression("x[0]", degree=1),
        V,
    )

    # Weak form:
    # (u/dt,v) + D(grad u,grad v) + alpha*kappa_h<u,v>_Gamma
    # =
    # (u_old/dt,v) + kappa_h*c_int<v>_Gamma
    #
    # Here n points from the granule toward the bulk, so the outward
    # normal of the computational bulk domain is -n.
    a = (
        (1.0 / dt_value) * u * v * dx
        + D_F * dot(grad(u), grad(v)) * dx
        + ALPHA * kappa_const * u * v * ds(ROBIN_TAG)
    )

    L = (
        (1.0 / dt_value) * c_n * v * dx
        + kappa_const * cint_const * v * ds(ROBIN_TAG)
    )

    solver = LUSolver()

    num_steps = number_of_time_steps(
        t_final=t_final,
        dt_value=dt_value,
    )
    t = 0.0

    times = [0.0]

    J0 = assemble(
        kappa_h(
            0.0,
            h,
            scenario=scenario,
        )
        * (
            c_int(0.0, scenario=scenario)
            - ALPHA * c_n
        )
        * ds(ROBIN_TAG)
    )

    if initialization == PREPARED_INITIALIZATION:
        expected_J0 = prepared_release_flux(
            h,
            scenario=scenario,
        )

        if not np.isclose(
            J0,
            expected_J0,
            rtol=1.0e-11,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Prepared reduced initial flux is inconsistent: "
                f"assembled={J0}, expected={expected_J0}."
            )

        Q_outer0 = expected_J0
    else:
        Q_outer0 = 0.0

    flux = [J0]
    outer_outflow = [Q_outer0]
    bulk_balance_residual = [0.0]
    bulk_balance_normalized = [0.0]
    robin_flux_mismatch = [0.0]
    mass_bulk = [assemble(c_n * dx)]
    energy_bulk = [0.5 * assemble(c_n * c_n * dx)]
    cumulative_release = [0.0]

    M_release = 0.0

    for step in range(num_steps):
        t_new = (step + 1) * dt_value

        kappa_value = kappa_h(
            t_new,
            h,
            scenario=scenario,
        )
        cint_value = c_int(
            t_new,
            scenario=scenario,
        )

        kappa_const.assign(kappa_value)
        cint_const.assign(cint_value)

        A = assemble(a)
        b = assemble(L)

        bc.apply(A)
        bc.apply(b)

        solver.solve(A, u_sol.vector(), b)

        J_new = assemble(kappa_value * (cint_value - ALPHA * u_sol) * ds(ROBIN_TAG))
        (
            Q_outer_new,
            balance_residual_new,
            normalized_balance_residual_new,
            robin_flux_mismatch_new,
        ) = compute_reduced_boundary_fluxes(
            c_new=u_sol,
            c_old=c_n,
            dt_value=dt_value,
            dx=dx,
            psi_interface=psi_interface,
            psi_outer=psi_outer,
            robin_flux=J_new,
        )

        M_release = cumulative_release_update(
            M_release,
            J_new,
            dt_value,
        )

        times.append(t_new)
        flux.append(J_new)
        outer_outflow.append(Q_outer_new)
        bulk_balance_residual.append(
            balance_residual_new
        )
        bulk_balance_normalized.append(
            normalized_balance_residual_new
        )
        robin_flux_mismatch.append(
            robin_flux_mismatch_new
        )
        cumulative_release.append(M_release)
        mass_bulk.append(assemble(u_sol * dx))
        energy_bulk.append(0.5 * assemble(u_sol * u_sol * dx))

        c_n.assign(u_sol)
        t = t_new

    # Store final reduced concentration field for field-level validation.
    final_solution = Function(V)
    final_solution.assign(u_sol)

    return {
        "h": h,
        "scenario": scenario,
        "initialization": initialization,
        "t_final": t_final,
        "resolution": resolution,
        "dt": dt_value,
        "num_cells": mesh.num_cells(),
        "num_dofs": V.dim(),
        "times": np.array(times),
        "J_red": np.array(flux),
        "Q_outer_red": np.array(outer_outflow),
        "bulk_balance_residual": np.array(
            bulk_balance_residual
        ),
        "bulk_balance_normalized": np.array(
            bulk_balance_normalized
        ),
        "robin_flux_mismatch": np.array(
            robin_flux_mismatch
        ),
        "M_red": np.array(cumulative_release),
        "mass_red": np.array(mass_bulk),
        "energy_red": np.array(energy_bulk),
        "mesh": mesh,
        "solution": final_solution,
    }


def save_reduced_validation_case(out, outdir="results/validation_reduced"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    htag = f"h{int(round(out['h'] * 1000)):03d}"

    csv_path = outdir / f"reduced_curves_{htag}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")

        writer.writerow([
            "scenario",
            "initialization",
            "h",
            "dt",
            "t",
            "J_red",
            "Q_outer_red",
            "bulk_balance_residual",
            "bulk_balance_normalized",
            "robin_flux_mismatch",
            "M_red",
            "mass_red",
            "energy_red",
        ])

        for row in zip(
            out["times"],
            out["J_red"],
            out["Q_outer_red"],
            out["bulk_balance_residual"],
            out["bulk_balance_normalized"],
            out["robin_flux_mismatch"],
            out["M_red"],
            out["mass_red"],
            out["energy_red"],
        ):
            writer.writerow([
                out["scenario"],
                out["initialization"],
                out["h"],
                out["dt"],
                *row,
            ])

    # Save final reduced concentration field at mesh nodes.
    if "solution" in out:
        sol = out["solution"]

        V = sol.function_space()

        coords = V.tabulate_dof_coordinates().reshape((-1, 2))
        values = sol.vector().get_local()

        field_path = outdir / f"reduced_field_{htag}.csv"

        with open(field_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")

            writer.writerow(["x", "y", "c_red"])

            for (x, y), val in zip(coords, values):
                writer.writerow([x, y, val])

    return csv_path
