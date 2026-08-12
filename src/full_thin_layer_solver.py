import csv
from pathlib import Path

import numpy as np
from dolfin import *

from .full_mesh_utils import (
    create_fitted_full_mesh,
    COATING_TAG,
    BULK_TAG,
    INNER_TAG,
    OUTER_TAG,
    INTERFACE_TAG,
)
from .validation_config import (
    D_F,
    K_IN,
    K_OUT,
    T_FINAL,
    DT_VALIDATION,
    BULK_RESOLUTION,
    DEFAULT_INITIALIZATION,
    DEFAULT_SCENARIO,
    EMPTY_INITIALIZATION,
    PREPARED_INITIALIZATION,
    D_m,
    c_int,
    validate_initialization,
    validate_scenario,
)
from .initial_data import (
    prepared_full_transformed_value,
    prepared_release_flux,
)
from .validation_metrics import (
    normalized_balance_residual,
    number_of_time_steps,
)


class BetaCoefficient(UserExpression):
    """
    Time-mass coefficient for the transformed full variable q.

    q = c_f in the bulk,
    q = c_m / K_OUT in the coating.

    Therefore,

        c_f = q              in the bulk,
        c_m = K_OUT * q      in the coating.

    The time-mass coefficient is consequently

        beta = 1             in the bulk,
        beta = K_OUT         in the coating.
    """

    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self.h = h

    def eval(self, value, x):
        value[0] = K_OUT if x[0] < 0.0 else 1.0

    def value_shape(self):
        return ()


class DiffusionCoefficient(UserExpression):
    """
    Diffusion coefficient in the transformed weak formulation.

    In the bulk,

        A = D_F.

    In the coating,

        A = K_OUT * D_m(t).

    This is consistent with c_m = K_OUT * q and preserves the
    physical diffusive-flux continuity at the coating--bulk interface.
    """

    def __init__(self, h, Dm_value, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.Dm_value = Dm_value

    def eval(self, value, x):
        value[0] = (
            K_OUT * self.Dm_value
            if x[0] < 0.0
            else D_F
        )

    def value_shape(self):
        return ()


class FullInitialCondition(UserExpression):
    """Prepared transformed initial profile for the resolved model."""

    def __init__(self, h, scenario, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.scenario = scenario

    def eval(self, value, x):
        value[0] = prepared_full_transformed_value(
            x1=x[0],
            h=self.h,
            scenario=self.scenario,
        )

    def value_shape(self):
        return ()


def cumulative_release_update(M_old, J_new, dt):
    """Advance cumulative release with the backward-Euler time rule."""
    return M_old + dt * J_new


def create_bulk_flux_context(mesh, cell_markers, degree):
    """
    Construct the exterior bulk submesh used for variational flux recovery.

    Two lifting functions are used:

        psi_interface = 1 - x_1,
        psi_outer     = x_1.

    The first equals one on the coating--bulk interface and zero on the
    exterior boundary. The second does the opposite.
    """

    bulk_mesh = SubMesh(mesh, cell_markers, BULK_TAG)
    bulk_mesh.init()

    bulk_facet_markers = MeshFunction(
        "size_t",
        bulk_mesh,
        bulk_mesh.topology().dim() - 1,
        0,
    )
    bulk_facet_markers.set_all(0)

    tol = 1.0e-12

    for facet in facets(bulk_mesh):
        x = facet.midpoint().x()

        if near(x, 0.0, tol):
            bulk_facet_markers[facet] = INTERFACE_TAG
        elif near(x, 1.0, tol):
            bulk_facet_markers[facet] = OUTER_TAG

    ds_bulk = Measure(
        "ds",
        domain=bulk_mesh,
        subdomain_data=bulk_facet_markers,
    )
    dx_bulk = Measure("dx", domain=bulk_mesh)

    interface_length = assemble(
        Constant(1.0) * ds_bulk(INTERFACE_TAG)
    )
    outer_length = assemble(
        Constant(1.0) * ds_bulk(OUTER_TAG)
    )

    if not np.isclose(interface_length, 1.0, atol=1.0e-10):
        raise RuntimeError(
            "Incorrect bulk-interface measure: "
            f"{interface_length}"
        )

    if not np.isclose(outer_length, 1.0, atol=1.0e-10):
        raise RuntimeError(
            "Incorrect outer-boundary measure: "
            f"{outer_length}"
        )

    V_bulk = FunctionSpace(bulk_mesh, "P", degree)

    q_bulk_new = Function(V_bulk)
    q_bulk_old = Function(V_bulk)

    psi_interface = interpolate(
        Expression("1.0 - x[0]", degree=1),
        V_bulk,
    )
    psi_outer = interpolate(
        Expression("x[0]", degree=1),
        V_bulk,
    )

    return {
        "mesh": bulk_mesh,
        "space": V_bulk,
        "new_solution": q_bulk_new,
        "old_solution": q_bulk_old,
        "psi_interface": psi_interface,
        "psi_outer": psi_outer,
        "dx": dx_bulk,
        "interface_length": interface_length,
        "outer_length": outer_length,
    }


def compute_bulk_boundary_fluxes(
    q_full_new,
    q_full_old,
    dt_value,
    flux_context,
):
    """
    Recover the bulk-side boundary fluxes variationally.

    For any lifting psi,

        boundary flux
        =
        integral_Omega ((q_new-q_old)/dt) psi dx
        + integral_Omega D_F grad(q_new).grad(psi) dx.

    The interface lifting equals one at x=0 and zero at x=1.
    The outer lifting equals zero at x=0 and one at x=1.

    Returns
    -------
    J_interface:
        Positive release rate from the coating into the bulk.

    Q_outer:
        Positive outward loss through x=1.

    balance_residual:
        J_interface - Q_outer - dB/dt. It should be close to zero.

    normalized_residual:
        Absolute balance residual normalized by the three balance terms.
    """

    q_bulk_new = flux_context["new_solution"]
    q_bulk_old = flux_context["old_solution"]

    LagrangeInterpolator.interpolate(
        q_bulk_new,
        q_full_new,
    )
    LagrangeInterpolator.interpolate(
        q_bulk_old,
        q_full_old,
    )

    dx_bulk = flux_context["dx"]
    psi_interface = flux_context["psi_interface"]
    psi_outer = flux_context["psi_outer"]

    time_derivative = (
        q_bulk_new - q_bulk_old
    ) / dt_value

    J_interface = assemble(
        time_derivative * psi_interface * dx_bulk
        + D_F
        * dot(
            grad(q_bulk_new),
            grad(psi_interface),
        )
        * dx_bulk
    )

    outer_signed_flux = assemble(
        time_derivative * psi_outer * dx_bulk
        + D_F
        * dot(
            grad(q_bulk_new),
            grad(psi_outer),
        )
        * dx_bulk
    )

    # The signed diffusive flux at x=1 is negative for outward loss.
    Q_outer = -outer_signed_flux

    bulk_mass_rate = assemble(
        time_derivative * dx_bulk
    )

    balance_residual = (
        J_interface
        - Q_outer
        - bulk_mass_rate
    )
    normalized_residual = normalized_balance_residual(
        residual=balance_residual,
        mass_rate=bulk_mass_rate,
        interface_inflow=J_interface,
        outer_outflow=Q_outer,
    )

    return (
        float(J_interface),
        float(Q_outer),
        float(balance_residual),
        float(normalized_residual),
    )


def run_full_validation_case(
    h,
    n_layer=8,
    n_bulk=BULK_RESOLUTION,
    ny=BULK_RESOLUTION,
    dt_value=DT_VALIDATION,
    degree=1,
    t_final=T_FINAL,
    scenario=DEFAULT_SCENARIO,
    initialization=DEFAULT_INITIALIZATION,
):
    """
    Solve the full thin-layer problem on the coating and exterior bulk.

    Geometry:

        coating = [-h,0] x [0,1],
        bulk    = [0,1] x [0,1].

    Transformed unknown:

        q = c_f             in the bulk,
        q = c_m / K_OUT     in the coating.

    The variable q is continuous at x=0, while the physical partition
    condition c_m=K_OUT*c_f is incorporated into the transformation.

    At the inner boundary x=-h,

        c_m = K_IN*c_int(t),

    which becomes

        q = (K_IN/K_OUT)*c_int(t).

    At the exterior boundary x=1,

        q = 0.
    """

    validate_scenario(scenario)
    validate_initialization(initialization)

    mesh, cell_markers, facet_markers = create_fitted_full_mesh(
        h=h,
        n_layer=n_layer,
        n_bulk=n_bulk,
        ny=ny,
    )

    V = FunctionSpace(mesh, "P", degree)

    dx = Measure(
        "dx",
        domain=mesh,
        subdomain_data=cell_markers,
    )

    u = TrialFunction(V)
    v = TestFunction(V)

    beta = BetaCoefficient(
        h=h,
        degree=0,
    )

    Acoef = DiffusionCoefficient(
        h=h,
        Dm_value=D_m(0.0, scenario=scenario),
        degree=0,
    )

    if initialization == PREPARED_INITIALIZATION:
        q_n = interpolate(
            FullInitialCondition(
                h=h,
                scenario=scenario,
                degree=1,
            ),
            V,
        )
    elif initialization == EMPTY_INITIALIZATION:
        q_n = interpolate(Constant(0.0), V)
    else:
        raise AssertionError("Initialization validation failed.")

    q_sol = Function(V)

    inner_value = Constant(
        (K_IN / K_OUT)
        * c_int(0.0, scenario=scenario)
    )
    outer_value = Constant(0.0)

    bc_inner = DirichletBC(
        V,
        inner_value,
        facet_markers,
        INNER_TAG,
    )

    bc_outer = DirichletBC(
        V,
        outer_value,
        facet_markers,
        OUTER_TAG,
    )

    bcs = [bc_inner, bc_outer]

    a = (
        (1.0 / dt_value) * beta * u * v * dx
        + Acoef * dot(grad(u), grad(v)) * dx
    )

    L = (
        (1.0 / dt_value) * beta * q_n * v * dx
    )

    solver = LUSolver()

    flux_context = create_bulk_flux_context(
        mesh=mesh,
        cell_markers=cell_markers,
        degree=degree,
    )

    num_steps = number_of_time_steps(
        t_final=t_final,
        dt_value=dt_value,
    )
    t = 0.0

    times = [0.0]

    mass_bulk = [
        assemble(q_n * dx(BULK_TAG))
    ]

    energy_bulk = [
        0.5 * assemble(q_n * q_n * dx(BULK_TAG))
    ]

    mass_coating = [
        assemble(K_OUT * q_n * dx(COATING_TAG))
    ]

    if initialization == PREPARED_INITIALIZATION:
        J0 = prepared_release_flux(
            h,
            scenario=scenario,
        )
        Q_outer0 = J0
    else:
        J0 = 0.0
        Q_outer0 = 0.0

    balance_residual0 = 0.0
    normalized_balance_residual0 = 0.0

    flux = [J0]
    outer_outflow = [Q_outer0]
    bulk_balance_residual = [balance_residual0]
    bulk_balance_normalized = [
        normalized_balance_residual0
    ]
    cumulative_release = [0.0]

    M_release = 0.0

    for step in range(num_steps):
        t_new = (step + 1) * dt_value

        Acoef.Dm_value = D_m(
            t_new,
            scenario=scenario,
        )

        inner_value.assign(
            (K_IN / K_OUT)
            * c_int(t_new, scenario=scenario)
        )

        A_mat = assemble(a)
        b_vec = assemble(L)

        for bc in bcs:
            bc.apply(A_mat)
            bc.apply(b_vec)

        solver.solve(
            A_mat,
            q_sol.vector(),
            b_vec,
        )

        (
            J_new,
            Q_outer_new,
            balance_residual_new,
            normalized_balance_residual_new,
        ) = compute_bulk_boundary_fluxes(
            q_full_new=q_sol,
            q_full_old=q_n,
            dt_value=dt_value,
            flux_context=flux_context,
        )

        M_release = cumulative_release_update(
            M_old=M_release,
            J_new=J_new,
            dt=dt_value,
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
        cumulative_release.append(M_release)

        mass_bulk.append(
            assemble(q_sol * dx(BULK_TAG))
        )

        energy_bulk.append(
            0.5 * assemble(
                q_sol * q_sol * dx(BULK_TAG)
            )
        )

        mass_coating.append(
            assemble(
                K_OUT * q_sol * dx(COATING_TAG)
            )
        )

        q_n.assign(q_sol)
        t = t_new

    final_solution = Function(V)
    final_solution.assign(q_sol)

    return {
        "h": h,
        "scenario": scenario,
        "initialization": initialization,
        "t_final": t_final,
        "n_layer": n_layer,
        "n_bulk": n_bulk,
        "ny": ny,
        "dt": dt_value,
        "num_cells": mesh.num_cells(),
        "num_dofs": V.dim(),
        "interface_length": flux_context["interface_length"],
        "times": np.array(times),
        "J_full": np.array(flux),
        "Q_outer_full": np.array(outer_outflow),
        "bulk_balance_residual": np.array(
            bulk_balance_residual
        ),
        "bulk_balance_normalized": np.array(
            bulk_balance_normalized
        ),
        "M_full": np.array(cumulative_release),
        "mass_bulk_full": np.array(mass_bulk),
        "mass_coating_full": np.array(mass_coating),
        "energy_bulk_full": np.array(energy_bulk),
        "mesh": mesh,
        "cell_markers": cell_markers,
        "solution": final_solution,
    }


def save_full_validation_case(
    out,
    outdir="results/validation_full",
):
    """Save the full-model curves and final bulk field."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    htag = f"h{int(round(out['h'] * 1000)):03d}"

    csv_path = outdir / f"full_curves_{htag}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(
            f,
            lineterminator="\n",
        )

        writer.writerow([
            "scenario",
            "initialization",
            "h",
            "dt",
            "t",
            "J_full",
            "Q_outer_full",
            "bulk_balance_residual",
            "bulk_balance_normalized",
            "M_full",
            "mass_bulk_full",
            "energy_bulk_full",
            "mass_coating_full",
        ])

        for row in zip(
            out["times"],
            out["J_full"],
            out["Q_outer_full"],
            out["bulk_balance_residual"],
            out["bulk_balance_normalized"],
            out["M_full"],
            out["mass_bulk_full"],
            out["energy_bulk_full"],
            out["mass_coating_full"],
        ):
            writer.writerow([
                out["scenario"],
                out["initialization"],
                out["h"],
                out["dt"],
                *row,
            ])

    # In the bulk region, the transformed variable q equals c_f.
    if "solution" in out:
        sol = out["solution"]
        V = sol.function_space()

        coords = V.tabulate_dof_coordinates().reshape((-1, 2))
        values = sol.vector().get_local()

        field_path = outdir / f"full_bulk_field_{htag}.csv"
        tol = 1.0e-12

        with open(field_path, "w", newline="") as f:
            writer = csv.writer(
                f,
                lineterminator="\n",
            )

            writer.writerow([
                "x",
                "y",
                "c_full",
            ])

            for (x, y), value in zip(coords, values):
                if x >= -tol:
                    writer.writerow([
                        x,
                        y,
                        value,
                    ])

    return csv_path
