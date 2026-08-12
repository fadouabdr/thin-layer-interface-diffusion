from dolfin import *

from .config import ALPHA, D_F, T_FINAL, kappa
from .manufactured import c_int
from .mesh_utils import (
    DIRICHLET_TAG,
    NEUMANN_TAG,
    ROBIN_TAG,
    create_square_mesh,
)


def run_reduced_verification(resolution, dt_value, degree=1):
    """
    Solve the manufactured reduced-interface problem on the unit square.

    The exact solution is

        c_exact(x, t) = exp(t) * (2 - x_1 - x_1^2).

    The test is used to verify the spatial finite-element discretization
    and the backward Euler time discretization.
    """

    mesh, boundaries = create_square_mesh(resolution)

    V = FunctionSpace(mesh, "P", degree)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Check the boundary markers on the unit square.
    print("measure Robin      =", assemble(Constant(1.0) * ds(ROBIN_TAG)))
    print("measure Dirichlet  =", assemble(Constant(1.0) * ds(DIRICHLET_TAG)))
    print("measure Neumann    =", assemble(Constant(1.0) * ds(NEUMANN_TAG)))

    u = TrialFunction(V)
    v = TestFunction(V)

    # Exact initial condition:
    #
    #     c_exact(x, 0) = 2 - x_1 - x_1^2.
    c_n = interpolate(
        Expression(
            "2.0 - x[0] - x[0]*x[0]",
            degree=8,
        ),
        V,
    )

    # Homogeneous Dirichlet condition on Gamma_D = {x_1 = 1}.
    bc = DirichletBC(
        V,
        Constant(0.0),
        boundaries,
        DIRICHLET_TAG,
    )

    u_sol = Function(V)

    num_steps = int(round(T_FINAL / dt_value))
    final_time = num_steps * dt_value
    t = 0.0

    print("T_FINAL            =", T_FINAL)
    print("num_steps * dt     =", final_time)

    # Time-dependent Robin data.
    #
    # On Gamma_R = {x_1 = 0}, the manufactured condition is
    #
    #     D_F grad(c) . nu
    #         = kappa(t) * (c_int(t) - ALPHA * c).
    kappa_const = Constant(kappa(0.0))
    cint_const = Constant(c_int(0.0))

    # Manufactured source:
    #
    #     s(x, t)
    #       = exp(t) * (2 - x_1 - x_1^2 + 2 D_F).
    f_expr = Expression(
        "exp(t) * (2.0 - x[0] - x[0]*x[0] + 2.0*Df)",
        degree=8,
        t=0.0,
        Df=D_F,
    )

    # Backward Euler weak form:
    #
    # (u^{n+1}/dt, v)
    # + (D_F grad(u^{n+1}), grad(v))
    # + <ALPHA kappa u^{n+1}, v>_GammaR
    #
    # =
    #
    # (u^n/dt, v)
    # + (s^{n+1}, v)
    # + <kappa c_int, v>_GammaR.
    a = (
        (1.0 / dt_value) * u * v * dx
        + D_F * dot(grad(u), grad(v)) * dx
        + ALPHA * kappa_const * u * v * ds(ROBIN_TAG)
    )

    L = (
        (1.0 / dt_value) * c_n * v * dx
        + f_expr * v * dx
        + kappa_const * cint_const * v * ds(ROBIN_TAG)
    )

    solver = LUSolver()

    for step in range(num_steps):
        t_new = t + dt_value

        # Evaluate all time-dependent data at t^{n+1}.
        kappa_const.assign(kappa(t_new))
        cint_const.assign(c_int(t_new))
        f_expr.t = t_new

        # Reassemble because the Robin coefficient depends on time.
        A = assemble(a)
        b = assemble(L)

        bc.apply(A)
        bc.apply(b)

        solver.solve(A, u_sol.vector(), b)

        if (step + 1) % 50 == 0 or (step + 1) == num_steps:
            print("step =", step + 1, "time =", t_new)

        c_n.assign(u_sol)
        t = t_new

    # Exact solution at the final numerical time.
    u_exact_T = Expression(
        "exp(T) * (2.0 - x[0] - x[0]*x[0])",
        degree=8,
        T=t,
    )

    error_L2 = errornorm(
        u_exact_T,
        u_sol,
        norm_type="L2",
        degree_rise=5,
    )

    return {
        "resolution": resolution,
        "dt": dt_value,
        "num_cells": mesh.num_cells(),
        "L2_error": error_L2,
    }