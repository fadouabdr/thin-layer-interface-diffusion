"""Matched initial data for the paired circular illustration.

The two exterior finite-element meshes are different, so equality is imposed
at the level of one common continuum datum.  That datum is the stationary
Robin solution on the reduced domain at ``t=0``.  It is interpolated onto both
meshes.  In the resolved annulus, the transformed concentration is then
replaced by an affine radial profile joining the prescribed inner partition
value to the common exterior trace at ``Gamma_h``.
"""

import numpy as np
import dolfin as df

from .circular_config import (
    BULK_CELL,
    COATING_CELL,
    CORE_BOUNDARY,
    OUTER_BOUNDARY,
)
from .circular_mesh_utils import load_dolfin_circular_mesh
from .circular_results import affine_profile_values
from .validation_config import ALPHA, D_F, K_IN, K_OUT, c_int, kappa_h


COMMON_EXTERIOR_INITIALIZATION = "stationary_reduced_robin_common_exterior"


def _stationary_reduced_state(V, ds, facet_markers, config):
    """Solve the common exterior datum on the reduced circular domain."""
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    c0 = df.Function(V)
    dx = df.Measure("dx", domain=V.mesh())
    kappa0 = kappa_h(
        0.0,
        config.coating_thickness,
        scenario=config.scenario,
    )
    cint0 = c_int(0.0, scenario=config.scenario)

    a0 = (
        D_F * df.dot(df.grad(u), df.grad(v)) * dx
        + ALPHA * kappa0 * u * v * ds(CORE_BOUNDARY)
    )
    L0 = kappa0 * cint0 * v * ds(CORE_BOUNDARY)
    bc = df.DirichletBC(
        V,
        df.Constant(0.0),
        facet_markers,
        OUTER_BOUNDARY,
    )
    A = df.assemble(a0)
    b = df.assemble(L0)
    bc.apply(A)
    bc.apply(b)
    df.LUSolver().solve(A, c0.vector(), b)
    return c0


def build_common_exterior_initial_state(mesh_files, config):
    """Return the single exterior datum used by both circular models."""
    config.validate()
    mesh, _, facet_markers = load_dolfin_circular_mesh(mesh_files)
    V = df.FunctionSpace(mesh, "P", config.polynomial_degree)
    ds = df.Measure("ds", domain=mesh, subdomain_data=facet_markers)
    function = _stationary_reduced_state(V, ds, facet_markers, config)
    return {
        "name": COMMON_EXTERIOR_INITIALIZATION,
        "function": function,
        "num_cells": int(mesh.num_cells()),
        "num_dofs": int(V.dim()),
        "definition": (
            "Stationary reduced-domain Robin finite-element solution at t=0; "
            "the same continuum function is interpolated onto both exterior "
            "meshes."
        ),
    }


def _evaluate_at_vertices(source, coordinates, vertex_indices):
    values = []
    for index in np.asarray(vertex_indices, dtype=np.int64):
        point = coordinates[int(index)]
        values.append(float(source(df.Point(float(point[0]), float(point[1])))))
    return np.asarray(values, dtype=float)


def _interpolate_common_function(V, common_state):
    target = df.Function(V)
    df.LagrangeInterpolator.interpolate(target, common_state["function"])
    return target


def _vertex_set(cells, cell_tags, selected_tag):
    selected = np.asarray(cells, dtype=np.int64)[
        np.asarray(cell_tags, dtype=np.int64) == int(selected_tag)
    ]
    if len(selected) == 0:
        raise RuntimeError(f"No cells carry physical tag {selected_tag}.")
    return np.unique(selected)


def initialize_reduced_from_common(V, common_state):
    """Interpolate the common exterior datum onto the reduced space."""
    c0 = _interpolate_common_function(V, common_state)
    coordinates = np.asarray(V.mesh().coordinates(), dtype=float)
    vertices = np.arange(len(coordinates), dtype=np.int64)
    source_values = _evaluate_at_vertices(
        common_state["function"], coordinates, vertices
    )
    vertex_to_dof = df.vertex_to_dof_map(V)
    target_values = c0.vector().get_local()[vertex_to_dof]
    mismatch = float(np.max(np.abs(target_values - source_values)))
    return c0, {
        "common_exterior_vertex_mismatch": mismatch,
    }


def initialize_resolved_from_common(
    V,
    cell_markers,
    config,
    common_state,
):
    """Create resolved data with a common exterior and matched partitions."""
    q0 = _interpolate_common_function(V, common_state)
    mesh = V.mesh()
    coordinates = np.asarray(mesh.coordinates(), dtype=float)
    cells = np.asarray(mesh.cells(), dtype=np.int64)
    tags = np.asarray(cell_markers.array(), dtype=np.int64)
    coating_vertices = _vertex_set(cells, tags, COATING_CELL)
    bulk_vertices = _vertex_set(cells, tags, BULK_CELL)

    coating_points = coordinates[coating_vertices]
    radii = np.linalg.norm(coating_points, axis=1)
    radial_fraction = (
        (radii - config.core_radius) / config.coating_thickness
    )
    tolerance = 5.0e-10
    if np.any(radial_fraction < -tolerance) or np.any(
        radial_fraction > 1.0 + tolerance
    ):
        raise RuntimeError("A coating vertex lies outside the accepted annulus.")
    radial_fraction = np.clip(radial_fraction, 0.0, 1.0)

    outer_points = (
        coating_points
        * (config.outer_coating_radius / radii)[:, np.newaxis]
    )
    outer_trace = np.asarray([
        float(
            common_state["function"](
                df.Point(float(point[0]), float(point[1]))
            )
        )
        for point in outer_points
    ])
    inner_q = (
        (K_IN / K_OUT)
        * c_int(0.0, scenario=config.scenario)
    )
    coating_values = affine_profile_values(
        inner_q,
        outer_trace,
        radial_fraction,
    )

    vertex_to_dof = df.vertex_to_dof_map(V)
    local_values = q0.vector().get_local()
    local_values[vertex_to_dof[coating_vertices]] = coating_values
    q0.vector().set_local(local_values)
    q0.vector().apply("insert")

    source_bulk = _evaluate_at_vertices(
        common_state["function"], coordinates, bulk_vertices
    )
    q_vertex_values = q0.vector().get_local()[vertex_to_dof]
    exterior_mismatch = float(
        np.max(np.abs(q_vertex_values[bulk_vertices] - source_bulk))
    )

    inner_mask = np.isclose(
        radii,
        config.core_radius,
        rtol=0.0,
        atol=5.0e-10,
    )
    outer_mask = np.isclose(
        radii,
        config.outer_coating_radius,
        rtol=0.0,
        atol=5.0e-10,
    )
    if not np.any(inner_mask) or not np.any(outer_mask):
        raise RuntimeError("Could not identify both coating boundary traces.")
    inner_partition_mismatch = float(
        np.max(np.abs(coating_values[inner_mask] - inner_q))
    )
    outer_partition_mismatch = float(
        np.max(
            np.abs(
                coating_values[outer_mask]
                - outer_trace[outer_mask]
            )
        )
    )

    return q0, {
        "common_exterior_vertex_mismatch": exterior_mismatch,
        "inner_partition_vertex_mismatch": inner_partition_mismatch,
        "outer_partition_vertex_mismatch": outer_partition_mismatch,
        "coating_profile": (
            "Affine in radial distance between "
            "K_in*c_int(0)/K_out on Gamma_g and the common exterior "
            "trace on Gamma_h."
        ),
    }
