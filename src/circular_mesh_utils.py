"""Gmsh construction, XDMF conversion, and audits for circular meshes.

Imports of Gmsh, meshio, and DOLFIN are deliberately lazy.  This keeps the
pure configuration tests runnable outside the pinned finite-element image,
while producing a clear error if a mesh command is launched elsewhere.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .circular_config import (
    BULK_CELL,
    COATING_BULK_INTERFACE,
    COATING_CELL,
    CORE_BOUNDARY,
    OUTER_BOUNDARY,
    CircularCaseConfig,
)


@dataclass(frozen=True)
class CircularMeshFiles:
    """Paths belonging to one generated circular mesh."""

    model_name: str
    msh: Path
    cells_xdmf: Path
    facets_xdmf: Path

    def to_manifest(self):
        return {
            "model_name": self.model_name,
            "msh": str(self.msh),
            "cells_xdmf": str(self.cells_xdmf),
            "facets_xdmf": str(self.facets_xdmf),
        }


def _require_gmsh_and_meshio():
    try:
        import gmsh
        import meshio
    except ImportError as exc:
        raise RuntimeError(
            "The circular mesh commands require gmsh and meshio. "
            "Run them inside the pinned fenics-gmsh Docker image."
        ) from exc
    return gmsh, meshio


def _require_dolfin():
    try:
        import dolfin
    except ImportError as exc:
        raise RuntimeError(
            "The circular audit requires legacy DOLFIN. "
            "Run it inside the pinned fenics-gmsh Docker image."
        ) from exc
    return dolfin


def _add_circle(geo, center_tag, radius, config, point_size):
    point_tags = []
    for index in range(config.angular_sectors):
        angle = 2.0 * np.pi * index / config.angular_sectors
        point_tags.append(
            geo.addPoint(
                float(radius * np.cos(angle)),
                float(radius * np.sin(angle)),
                0.0,
                point_size,
            )
        )

    arc_tags = []
    for index, start_tag in enumerate(point_tags):
        end_tag = point_tags[(index + 1) % config.angular_sectors]
        arc_tags.append(
            geo.addCircleArc(start_tag, center_tag, end_tag)
        )

    return point_tags, arc_tags


def _add_square(geo, half_width, point_size):
    coordinates = [
        (-half_width, -half_width),
        (half_width, -half_width),
        (half_width, half_width),
        (-half_width, half_width),
    ]
    point_tags = [
        geo.addPoint(float(x), float(y), 0.0, point_size)
        for x, y in coordinates
    ]
    line_tags = [
        geo.addLine(point_tags[index], point_tags[(index + 1) % 4])
        for index in range(4)
    ]
    return point_tags, line_tags


def _clockwise_loop(curve_tags):
    return [-tag for tag in reversed(curve_tags)]


def _set_physical_group(gmsh, dimension, entity_tags, physical_tag, name):
    gmsh.model.addPhysicalGroup(
        dimension,
        list(entity_tags),
        tag=physical_tag,
    )
    gmsh.model.setPhysicalName(dimension, physical_tag, name)


def _set_bulk_size_field(gmsh, interface_curves, config):
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        distance,
        "CurvesList",
        list(interface_curves),
    )
    gmsh.model.mesh.field.setNumber(distance, "Sampling", 200)

    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(
        threshold,
        "SizeMin",
        config.interface_size,
    )
    gmsh.model.mesh.field.setNumber(
        threshold,
        "SizeMax",
        config.bulk_size,
    )
    gmsh.model.mesh.field.setNumber(
        threshold,
        "DistMin",
        config.transition_start,
    )
    gmsh.model.mesh.field.setNumber(
        threshold,
        "DistMax",
        config.transition_end,
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)


def _configure_gmsh(gmsh, verbosity):
    gmsh.option.setNumber("General.Terminal", 1 if verbosity else 0)
    gmsh.option.setNumber("General.Verbosity", int(verbosity))
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)


def _build_resolved_msh(config, msh_path, verbosity):
    gmsh, _ = _require_gmsh_and_meshio()
    config.validate()
    msh_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize([])
    try:
        _configure_gmsh(gmsh, verbosity)
        gmsh.model.add("circular_resolved")
        geo = gmsh.model.geo

        center_tag = geo.addPoint(0.0, 0.0, 0.0, config.interface_size)
        inner_points, inner_arcs = _add_circle(
            geo,
            center_tag,
            config.core_radius,
            config,
            config.interface_size,
        )
        outer_points, outer_arcs = _add_circle(
            geo,
            center_tag,
            config.outer_coating_radius,
            config,
            config.interface_size,
        )
        _, square_lines = _add_square(
            geo,
            config.box_half_width,
            config.bulk_size,
        )

        radial_lines = [
            geo.addLine(inner_points[index], outer_points[index])
            for index in range(config.angular_sectors)
        ]

        coating_surfaces = []
        coating_corners = []
        for index in range(config.angular_sectors):
            next_index = (index + 1) % config.angular_sectors
            loop = geo.addCurveLoop([
                radial_lines[index],
                outer_arcs[index],
                -radial_lines[next_index],
                -inner_arcs[index],
            ])
            coating_surfaces.append(geo.addPlaneSurface([loop]))
            coating_corners.append([
                inner_points[index],
                outer_points[index],
                outer_points[next_index],
                inner_points[next_index],
            ])

        square_loop = geo.addCurveLoop(square_lines)
        coating_outer_loop = geo.addCurveLoop(
            _clockwise_loop(outer_arcs)
        )
        bulk_surface = geo.addPlaneSurface([
            square_loop,
            coating_outer_loop,
        ])

        geo.synchronize()

        arc_nodes = config.angular_segments_per_sector() + 1
        for curve_tag in inner_arcs + outer_arcs:
            gmsh.model.mesh.setTransfiniteCurve(curve_tag, arc_nodes)
        for curve_tag in radial_lines:
            gmsh.model.mesh.setTransfiniteCurve(
                curve_tag,
                config.radial_layers + 1,
            )
        for surface_tag, corners in zip(coating_surfaces, coating_corners):
            gmsh.model.mesh.setTransfiniteSurface(
                surface_tag,
                "AlternateLeft",
                corners,
            )

        _set_physical_group(
            gmsh,
            2,
            coating_surfaces,
            COATING_CELL,
            "coating",
        )
        _set_physical_group(
            gmsh,
            2,
            [bulk_surface],
            BULK_CELL,
            "bulk",
        )
        _set_physical_group(
            gmsh,
            1,
            inner_arcs,
            CORE_BOUNDARY,
            "core_boundary",
        )
        _set_physical_group(
            gmsh,
            1,
            square_lines,
            OUTER_BOUNDARY,
            "outer_boundary",
        )
        _set_physical_group(
            gmsh,
            1,
            outer_arcs,
            COATING_BULK_INTERFACE,
            "coating_bulk_interface",
        )

        _set_bulk_size_field(gmsh, outer_arcs, config)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()


def _build_reduced_msh(config, msh_path, verbosity):
    gmsh, _ = _require_gmsh_and_meshio()
    config.validate()
    msh_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize([])
    try:
        _configure_gmsh(gmsh, verbosity)
        gmsh.model.add("circular_reduced")
        geo = gmsh.model.geo

        center_tag = geo.addPoint(0.0, 0.0, 0.0, config.interface_size)
        _, core_arcs = _add_circle(
            geo,
            center_tag,
            config.core_radius,
            config,
            config.interface_size,
        )
        _, square_lines = _add_square(
            geo,
            config.box_half_width,
            config.bulk_size,
        )

        square_loop = geo.addCurveLoop(square_lines)
        core_loop = geo.addCurveLoop(_clockwise_loop(core_arcs))
        bulk_surface = geo.addPlaneSurface([square_loop, core_loop])
        geo.synchronize()

        # Use the resolved mesh's angular count so both models have the same
        # polygonal representation of the physical core boundary.
        arc_nodes = config.angular_segments_per_sector() + 1
        for curve_tag in core_arcs:
            gmsh.model.mesh.setTransfiniteCurve(curve_tag, arc_nodes)

        _set_physical_group(
            gmsh,
            2,
            [bulk_surface],
            BULK_CELL,
            "bulk",
        )
        _set_physical_group(
            gmsh,
            1,
            core_arcs,
            CORE_BOUNDARY,
            "core_boundary_robin",
        )
        _set_physical_group(
            gmsh,
            1,
            square_lines,
            OUTER_BOUNDARY,
            "outer_boundary",
        )

        _set_bulk_size_field(gmsh, core_arcs, config)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()


def _extract_meshio_cells(mesh, cell_type):
    cells = mesh.get_cells_type(cell_type)
    if len(cells) == 0:
        raise RuntimeError(f"No {cell_type!r} elements were found in the mesh.")
    try:
        tags = mesh.get_cell_data("gmsh:physical", cell_type)
    except KeyError as exc:
        raise RuntimeError(
            f"The {cell_type!r} elements have no Gmsh physical tags."
        ) from exc
    return cells, np.asarray(tags, dtype=np.int64)


def convert_msh_to_xdmf(msh_path, model_name):
    """Convert one tagged Gmsh mesh to DOLFIN-readable XDMF files."""
    _, meshio = _require_gmsh_and_meshio()
    msh_path = Path(msh_path)
    mesh = meshio.read(str(msh_path))

    triangles, triangle_tags = _extract_meshio_cells(mesh, "triangle")
    lines, line_tags = _extract_meshio_cells(mesh, "line")
    points = np.asarray(mesh.points[:, :2], dtype=float)

    cells_xdmf = msh_path.with_name(f"{model_name}_cells.xdmf")
    facets_xdmf = msh_path.with_name(f"{model_name}_facets.xdmf")

    meshio.write(
        str(cells_xdmf),
        meshio.Mesh(
            points=points,
            cells=[("triangle", triangles)],
            cell_data={"cell_tags": [triangle_tags]},
        ),
    )
    meshio.write(
        str(facets_xdmf),
        meshio.Mesh(
            points=points,
            cells=[("line", lines)],
            cell_data={"facet_tags": [line_tags]},
        ),
    )

    return CircularMeshFiles(
        model_name=model_name,
        msh=msh_path,
        cells_xdmf=cells_xdmf,
        facets_xdmf=facets_xdmf,
    )


def build_resolved_circular_mesh(config, output_dir, verbosity=0):
    output_dir = Path(output_dir)
    msh_path = output_dir / "resolved.msh"
    _build_resolved_msh(config, msh_path, verbosity)
    return convert_msh_to_xdmf(msh_path, "resolved")


def build_reduced_circular_mesh(config, output_dir, verbosity=0):
    output_dir = Path(output_dir)
    msh_path = output_dir / "reduced.msh"
    _build_reduced_msh(config, msh_path, verbosity)
    return convert_msh_to_xdmf(msh_path, "reduced")


def load_dolfin_circular_mesh(mesh_files):
    """Load cells and facet tags from one ``CircularMeshFiles`` bundle."""
    dolfin = _require_dolfin()
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(str(mesh_files.cells_xdmf)) as infile:
        infile.read(mesh)

    cell_values = dolfin.MeshValueCollection(
        "size_t",
        mesh,
        mesh.topology().dim(),
    )
    with dolfin.XDMFFile(str(mesh_files.cells_xdmf)) as infile:
        infile.read(cell_values, "cell_tags")
    cell_markers = dolfin.cpp.mesh.MeshFunctionSizet(mesh, cell_values)

    facet_values = dolfin.MeshValueCollection(
        "size_t",
        mesh,
        mesh.topology().dim() - 1,
    )
    with dolfin.XDMFFile(str(mesh_files.facets_xdmf)) as infile:
        infile.read(facet_values, "facet_tags")
    facet_markers = dolfin.cpp.mesh.MeshFunctionSizet(mesh, facet_values)

    return mesh, cell_markers, facet_markers


def infer_radial_layer_count(coordinates, config):
    """Infer the resolved layer count from nodes on the positive x-axis."""
    coordinates = np.asarray(coordinates, dtype=float)
    tolerance = max(1.0e-10, 1.0e-7 * config.coating_thickness)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    mask = (
        (np.abs(y) <= tolerance)
        & (x >= config.core_radius - tolerance)
        & (x <= config.outer_coating_radius + tolerance)
    )
    axis_coordinates = np.sort(x[mask])
    if len(axis_coordinates) < 2:
        return 0, []

    unique_coordinates = [float(axis_coordinates[0])]
    for value in axis_coordinates[1:]:
        if abs(value - unique_coordinates[-1]) > tolerance:
            unique_coordinates.append(float(value))

    spacings = np.diff(unique_coordinates)
    return len(unique_coordinates) - 1, [float(value) for value in spacings]


def triangle_quality_summary(coordinates, connectivity):
    """Return a scale-free triangle quality summary (one is equilateral)."""
    coordinates = np.asarray(coordinates, dtype=float)
    connectivity = np.asarray(connectivity, dtype=np.int64)
    vertices = coordinates[connectivity]

    edge01 = vertices[:, 1] - vertices[:, 0]
    edge12 = vertices[:, 2] - vertices[:, 1]
    edge20 = vertices[:, 0] - vertices[:, 2]
    edge_sq_sum = (
        np.sum(edge01 ** 2, axis=1)
        + np.sum(edge12 ** 2, axis=1)
        + np.sum(edge20 ** 2, axis=1)
    )
    twice_area = np.abs(
        edge01[:, 0] * (vertices[:, 2, 1] - vertices[:, 0, 1])
        - edge01[:, 1] * (vertices[:, 2, 0] - vertices[:, 0, 0])
    )
    quality = 2.0 * np.sqrt(3.0) * twice_area / edge_sq_sum

    edge_lengths = np.sqrt(np.concatenate([
        np.sum(edge01 ** 2, axis=1),
        np.sum(edge12 ** 2, axis=1),
        np.sum(edge20 ** 2, axis=1),
    ]))

    return {
        "quality_min": float(np.min(quality)),
        "quality_mean": float(np.mean(quality)),
        "quality_median": float(np.median(quality)),
        "edge_length_min": float(np.min(edge_lengths)),
        "edge_length_max": float(np.max(edge_lengths)),
    }


def _tag_counts(values):
    tags, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    result = {}
    for tag, count in zip(tags, counts):
        integer_tag = int(tag)
        # MeshFunctionSizet initializes entities that are absent from the
        # sparse XDMF marker file with the unsigned-size_t maximum.  Casting
        # that sentinel to int64 displays it as -1.  These are ordinary
        # unmarked interior facets, not a Gmsh physical tag.
        label = str(integer_tag) if integer_tag > 0 else "unmarked"
        result[label] = result.get(label, 0) + int(count)
    return result


def physical_tag_set(values):
    """Return only positive Gmsh/DOLFIN physical tags.

    Zero and the DOLFIN unmarked-entity sentinel (which can appear as ``-1``
    after conversion to signed integers) are deliberately excluded.
    """
    return {
        int(tag)
        for tag in np.unique(np.asarray(values, dtype=np.int64))
        if int(tag) > 0
    }


def _measure_check(name, actual, expected, relative_tolerance):
    scale = max(abs(expected), 1.0e-14)
    relative_error = abs(actual - expected) / scale
    return {
        "name": name,
        "actual": float(actual),
        "expected": float(expected),
        "relative_error": float(relative_error),
        "relative_tolerance": float(relative_tolerance),
        "passed": bool(relative_error <= relative_tolerance),
    }


def _absolute_check(name, actual, expected, absolute_tolerance):
    absolute_error = abs(actual - expected)
    return {
        "name": name,
        "actual": float(actual),
        "expected": float(expected),
        "absolute_error": float(absolute_error),
        "absolute_tolerance": float(absolute_tolerance),
        "passed": bool(absolute_error <= absolute_tolerance),
    }


def audit_circular_mesh(mesh_files, config, resolved):
    """Audit areas, perimeters, tags, layer count, and triangle quality."""
    dolfin = _require_dolfin()
    config.validate()
    mesh, cell_markers, facet_markers = load_dolfin_circular_mesh(mesh_files)

    dx = dolfin.Measure("dx", domain=mesh, subdomain_data=cell_markers)
    ds = dolfin.Measure("ds", domain=mesh, subdomain_data=facet_markers)
    dS = dolfin.Measure("dS", domain=mesh, subdomain_data=facet_markers)
    one = dolfin.Constant(1.0)

    checks = []
    if resolved:
        coating_area = dolfin.assemble(one * dx(COATING_CELL))
        checks.append(_measure_check(
            "coating area",
            coating_area,
            config.coating_area,
            config.geometry_relative_tolerance,
        ))
        bulk_area = dolfin.assemble(one * dx(BULK_CELL))
        checks.append(_measure_check(
            "resolved bulk area",
            bulk_area,
            config.resolved_bulk_area,
            config.geometry_relative_tolerance,
        ))
        interface_perimeter = dolfin.assemble(
            one * dS(COATING_BULK_INTERFACE)
        )
        checks.append(_measure_check(
            "coating-bulk interface perimeter",
            interface_perimeter,
            config.coating_bulk_interface_perimeter,
            config.geometry_relative_tolerance,
        ))
    else:
        bulk_area = dolfin.assemble(one * dx(BULK_CELL))
        checks.append(_measure_check(
            "reduced bulk area",
            bulk_area,
            config.reduced_bulk_area,
            config.geometry_relative_tolerance,
        ))

    core_perimeter = dolfin.assemble(one * ds(CORE_BOUNDARY))
    checks.append(_measure_check(
        "core perimeter",
        core_perimeter,
        config.core_perimeter,
        config.geometry_relative_tolerance,
    ))
    outer_perimeter = dolfin.assemble(one * ds(OUTER_BOUNDARY))
    checks.append(_measure_check(
        "square outer perimeter",
        outer_perimeter,
        config.box_perimeter,
        config.geometry_relative_tolerance,
    ))
    # ``ds`` without an id covers the complete exterior boundary even when
    # the MeshFunction uses DOLFIN's nonzero unmarked sentinel.  Comparing it
    # with the two intended exterior groups therefore detects a genuinely
    # untagged boundary without assuming that the sentinel equals zero.
    total_exterior_perimeter = dolfin.assemble(one * ds)
    untagged_boundary = abs(
        total_exterior_perimeter - core_perimeter - outer_perimeter
    )
    checks.append(_absolute_check(
        "untagged exterior-boundary measure",
        untagged_boundary,
        0.0,
        1.0e-10 * max(1.0, total_exterior_perimeter),
    ))

    cell_values = np.asarray(cell_markers.array(), dtype=np.int64)
    facet_values = np.asarray(facet_markers.array(), dtype=np.int64)
    expected_cell_tags = {COATING_CELL, BULK_CELL} if resolved else {BULK_CELL}
    expected_facet_tags = (
        {CORE_BOUNDARY, OUTER_BOUNDARY, COATING_BULK_INTERFACE}
        if resolved
        else {CORE_BOUNDARY, OUTER_BOUNDARY}
    )
    present_cell_tags = physical_tag_set(cell_values)
    present_facet_tags = physical_tag_set(facet_values)

    tag_checks = {
        "cell_tags_exact": present_cell_tags == expected_cell_tags,
        "facet_tags_exact": present_facet_tags == expected_facet_tags,
        "no_untagged_cells": bool(np.all(cell_values > 0)),
    }

    layer_count = 0
    layer_spacings = []
    if resolved:
        layer_count, layer_spacings = infer_radial_layer_count(
            mesh.coordinates(),
            config,
        )
        tag_checks["radial_layer_count"] = (
            layer_count == config.radial_layers
        )

    quality = triangle_quality_summary(mesh.coordinates(), mesh.cells())
    space = dolfin.FunctionSpace(
        mesh,
        "P",
        config.polynomial_degree,
    )

    passed = (
        all(check["passed"] for check in checks)
        and all(tag_checks.values())
    )
    return {
        "model": "resolved" if resolved else "reduced",
        "passed": bool(passed),
        "num_vertices": int(mesh.num_vertices()),
        "num_cells": int(mesh.num_cells()),
        "num_dofs_p1": int(space.dim()),
        "cell_tag_counts": _tag_counts(cell_values),
        "facet_tag_counts": _tag_counts(facet_values),
        "measure_checks": checks,
        "tag_checks": tag_checks,
        "radial_layer_count": int(layer_count),
        "radial_spacing_min": (
            float(min(layer_spacings)) if layer_spacings else None
        ),
        "radial_spacing_max": (
            float(max(layer_spacings)) if layer_spacings else None
        ),
        "triangle_quality": quality,
        "files": mesh_files.to_manifest(),
    }
