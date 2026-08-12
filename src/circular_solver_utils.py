"""Shared DOLFIN utilities for the paired circular transient solvers."""

import numpy as np
import dolfin as df

from .circular_config import (
    BULK_CELL,
    COATING_BULK_INTERFACE,
    OUTER_BOUNDARY,
)


def _harmonic_liftings(mesh, facet_markers, inner_tag, outer_tag, degree):
    """Build complementary boundary liftings on a two-boundary domain."""
    V = df.FunctionSpace(mesh, "P", degree)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    dx = df.Measure("dx", domain=mesh)

    a = df.dot(df.grad(u), df.grad(v)) * dx
    L = df.Constant(0.0) * v * dx
    bcs = [
        df.DirichletBC(V, df.Constant(1.0), facet_markers, inner_tag),
        df.DirichletBC(V, df.Constant(0.0), facet_markers, outer_tag),
    ]

    A = df.assemble(a)
    b = df.assemble(L)
    for bc in bcs:
        bc.apply(A)
        bc.apply(b)

    psi_inner = df.Function(V)
    df.LUSolver().solve(A, psi_inner.vector(), b)

    psi_outer = df.interpolate(df.Constant(1.0), V)
    psi_outer.vector().axpy(-1.0, psi_inner.vector())

    return {
        "mesh": mesh,
        "space": V,
        "dx": dx,
        "psi_inner": psi_inner,
        "psi_outer": psi_outer,
    }


def create_direct_flux_context(
    mesh,
    facet_markers,
    inner_tag,
    outer_tag,
    degree,
):
    """Create boundary liftings when the solution already lives on ``mesh``."""
    context = _harmonic_liftings(
        mesh,
        facet_markers,
        inner_tag,
        outer_tag,
        degree,
    )
    context["new_solution"] = df.Function(context["space"])
    context["old_solution"] = df.Function(context["space"])
    context["requires_interpolation"] = False
    return context


def create_resolved_bulk_flux_context(
    mesh,
    cell_markers,
    config,
    degree,
):
    """Create the exterior submesh and its two boundary liftings."""
    bulk_mesh = df.SubMesh(mesh, cell_markers, BULK_CELL)
    bulk_mesh.init()

    facet_markers = df.MeshFunction(
        "size_t",
        bulk_mesh,
        bulk_mesh.topology().dim() - 1,
        0,
    )
    facet_markers.set_all(0)

    tolerance = 1.0e-10 * max(1.0, config.box_half_width)
    for facet in df.facets(bulk_mesh):
        if not facet.exterior():
            continue
        midpoint = facet.midpoint()
        on_square = (
            df.near(abs(midpoint.x()), config.box_half_width, tolerance)
            or df.near(abs(midpoint.y()), config.box_half_width, tolerance)
        )
        facet_markers[facet] = (
            OUTER_BOUNDARY if on_square else COATING_BULK_INTERFACE
        )

    ds = df.Measure("ds", domain=bulk_mesh, subdomain_data=facet_markers)
    one = df.Constant(1.0)
    interface_measure = float(
        df.assemble(one * ds(COATING_BULK_INTERFACE))
    )
    outer_measure = float(df.assemble(one * ds(OUTER_BOUNDARY)))

    if not np.isclose(
        interface_measure,
        config.coating_bulk_interface_perimeter,
        rtol=config.geometry_relative_tolerance,
        atol=0.0,
    ):
        raise RuntimeError(
            "Resolved bulk submesh has an incorrect coating interface "
            f"measure: {interface_measure}."
        )
    if not np.isclose(
        outer_measure,
        config.box_perimeter,
        rtol=config.geometry_relative_tolerance,
        atol=0.0,
    ):
        raise RuntimeError(
            "Resolved bulk submesh has an incorrect outer-boundary "
            f"measure: {outer_measure}."
        )

    context = _harmonic_liftings(
        bulk_mesh,
        facet_markers,
        COATING_BULK_INTERFACE,
        OUTER_BOUNDARY,
        degree,
    )
    context.update({
        "new_solution": df.Function(context["space"]),
        "old_solution": df.Function(context["space"]),
        "requires_interpolation": True,
        "interface_measure": interface_measure,
        "outer_measure": outer_measure,
        "facet_markers": facet_markers,
    })
    return context


def _solutions_on_flux_mesh(new_solution, old_solution, context):
    target_new = context["new_solution"]
    target_old = context["old_solution"]
    if context["requires_interpolation"]:
        df.LagrangeInterpolator.interpolate(target_new, new_solution)
        df.LagrangeInterpolator.interpolate(target_old, old_solution)
    else:
        target_new.assign(new_solution)
        target_old.assign(old_solution)
    return target_new, target_old


def recover_boundary_fluxes(
    new_solution,
    old_solution,
    dt_value,
    diffusivity,
    context,
):
    """Recover signed inner inflow, outer outflow, and mass balance."""
    c_new, c_old = _solutions_on_flux_mesh(
        new_solution,
        old_solution,
        context,
    )
    dx = context["dx"]
    psi_inner = context["psi_inner"]
    psi_outer = context["psi_outer"]
    time_derivative = (c_new - c_old) / dt_value

    inner_inflow = df.assemble(
        time_derivative * psi_inner * dx
        + diffusivity
        * df.dot(df.grad(c_new), df.grad(psi_inner))
        * dx
    )
    outer_signed = df.assemble(
        time_derivative * psi_outer * dx
        + diffusivity
        * df.dot(df.grad(c_new), df.grad(psi_outer))
        * dx
    )
    outer_outflow = -outer_signed
    mass_rate = df.assemble(time_derivative * dx)
    residual = inner_inflow - outer_outflow - mass_rate
    scale = abs(inner_inflow) + abs(outer_outflow) + abs(mass_rate)
    normalized = abs(residual) / max(scale, 1.0e-14)

    return (
        float(inner_inflow),
        float(outer_outflow),
        float(residual),
        float(normalized),
    )


def recover_stationary_boundary_fluxes(solution, diffusivity, context):
    """Recover the two signed boundary fluxes of a stationary state."""
    c_new, _ = _solutions_on_flux_mesh(solution, solution, context)
    dx = context["dx"]
    inner_inflow = df.assemble(
        diffusivity
        * df.dot(df.grad(c_new), df.grad(context["psi_inner"]))
        * dx
    )
    outer_outflow = -df.assemble(
        diffusivity
        * df.dot(df.grad(c_new), df.grad(context["psi_outer"]))
        * dx
    )
    residual = inner_inflow - outer_outflow
    scale = abs(inner_inflow) + abs(outer_outflow)
    normalized = abs(residual) / max(scale, 1.0e-14)
    return (
        float(inner_inflow),
        float(outer_outflow),
        float(residual),
        float(normalized),
    )


def p1_vertex_values(function):
    """Return scalar P1 values in mesh-vertex order."""
    V = function.function_space()
    element = V.ufl_element()
    if element.degree() != 1 or element.value_shape() != ():
        raise ValueError("snapshot export requires a scalar P1 function.")
    vertex_to_dof = df.vertex_to_dof_map(V)
    return np.asarray(function.vector().get_local()[vertex_to_dof], dtype=float)
