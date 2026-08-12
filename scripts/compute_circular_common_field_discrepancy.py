"""Compute circular resolved--reduced field discrepancies on the common exterior.

The comparison domain is the resolved exterior region

    Q \ B_{R+h}(0),

represented by the resolved cells tagged as bulk cells. The reduced P1 field
is interpolated onto the resolved exterior mesh before the L2 norm is
calculated.
"""

import csv
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np


INPUT_FILE = Path(
    "results/circular/production_common_initial/circular_snapshots.npz"
)
OUTPUT_FILE = Path(
    "results/circular/production_common_initial/"
    "circular_common_domain_field_discrepancy.csv"
)

BULK_CELL_TAG = 2


def as_coordinates(values, name):
    values = np.asarray(values)

    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")

    if values.shape[1] == 2:
        return values

    if values.shape[0] == 2:
        return values.T

    raise ValueError(f"Cannot determine coordinate orientation for {name}.")


def as_cells(values, name):
    values = np.asarray(values, dtype=int)

    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")

    if values.shape[1] == 3:
        return values

    if values.shape[0] == 3:
        return values.T

    raise ValueError(f"Cannot determine cell orientation for {name}.")


def as_snapshot_matrix(values, n_snapshots, n_vertices, name):
    values = np.asarray(values, dtype=float)

    if values.shape == (n_snapshots, n_vertices):
        return values

    if values.shape == (n_vertices, n_snapshots):
        return values.T

    raise ValueError(
        f"{name} has shape {values.shape}; expected "
        f"({n_snapshots}, {n_vertices}) or "
        f"({n_vertices}, {n_snapshots})."
    )


def p1_l2_squared(coordinates, cells, nodal_values):
    """Exact integral of the square of a P1 function over triangular cells."""

    points = coordinates[cells]

    twice_area = np.abs(
        (points[:, 1, 0] - points[:, 0, 0])
        * (points[:, 2, 1] - points[:, 0, 1])
        -
        (points[:, 2, 0] - points[:, 0, 0])
        * (points[:, 1, 1] - points[:, 0, 1])
    )
    area = 0.5 * twice_area

    values = nodal_values[cells]
    quadratic_sum = (
        values[:, 0] ** 2
        + values[:, 1] ** 2
        + values[:, 2] ** 2
        + values[:, 0] * values[:, 1]
        + values[:, 1] * values[:, 2]
        + values[:, 2] * values[:, 0]
    )

    return float(np.sum(area * quadratic_sum / 6.0))


def interpolate_reduced_to_resolved(
    reduced_coordinates,
    reduced_cells,
    reduced_values,
    resolved_coordinates,
    used_resolved_vertices,
):
    triangulation = mtri.Triangulation(
        reduced_coordinates[:, 0],
        reduced_coordinates[:, 1],
        triangles=reduced_cells,
    )

    interpolator = mtri.LinearTriInterpolator(
        triangulation,
        reduced_values,
    )

    interpolated = interpolator(
        resolved_coordinates[:, 0],
        resolved_coordinates[:, 1],
    )

    result = np.asarray(
        np.ma.filled(interpolated, np.nan),
        dtype=float,
    )

    # Boundary vertices should normally be handled directly by the
    # triangulation. This coordinate-match fallback protects against
    # round-off masking on a shared outer boundary.
    missing = used_resolved_vertices[~np.isfinite(result[used_resolved_vertices])]

    if missing.size:
        reduced_coordinate_map = {
            tuple(np.round(point, decimals=12)): index
            for index, point in enumerate(reduced_coordinates)
        }

        for vertex in missing:
            key = tuple(np.round(resolved_coordinates[vertex], decimals=12))
            if key in reduced_coordinate_map:
                result[vertex] = reduced_values[reduced_coordinate_map[key]]

    missing = used_resolved_vertices[~np.isfinite(result[used_resolved_vertices])]

    if missing.size:
        raise RuntimeError(
            "Reduced interpolation failed at "
            f"{missing.size} resolved exterior vertices."
        )

    return result


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing snapshot archive: {INPUT_FILE}")

    with np.load(INPUT_FILE, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}

    times = np.asarray(data["snapshot_times"], dtype=float).reshape(-1)
    fractions = np.asarray(
        data["snapshot_fractions"],
        dtype=float,
    ).reshape(-1)

    resolved_coordinates = as_coordinates(
        data["resolved_coordinates"],
        "resolved_coordinates",
    )
    resolved_cells = as_cells(
        data["resolved_cells"],
        "resolved_cells",
    )
    resolved_cell_tags = np.asarray(
        data["resolved_cell_tags"],
        dtype=int,
    ).reshape(-1)

    reduced_coordinates = as_coordinates(
        data["reduced_coordinates"],
        "reduced_coordinates",
    )
    reduced_cells = as_cells(
        data["reduced_cells"],
        "reduced_cells",
    )

    if resolved_cell_tags.size != resolved_cells.shape[0]:
        raise ValueError(
            "The resolved cell-tag array does not match the cell array."
        )

    resolved_fields = as_snapshot_matrix(
        data["resolved_q_vertex"],
        len(times),
        resolved_coordinates.shape[0],
        "resolved_q_vertex",
    )
    reduced_fields = as_snapshot_matrix(
        data["reduced_c_vertex"],
        len(times),
        reduced_coordinates.shape[0],
        "reduced_c_vertex",
    )

    # These cells represent Q \ B_{R+h}(0), the common exterior region.
    common_cells = resolved_cells[resolved_cell_tags == BULK_CELL_TAG]

    if common_cells.size == 0:
        raise RuntimeError("No resolved exterior cells with bulk tag 2 found.")

    used_vertices = np.unique(common_cells)

    rows = []

    for index, time in enumerate(times):
        # On resolved bulk cells, q is the physical exterior concentration c_f.
        resolved_field = resolved_fields[index]

        reduced_on_resolved = interpolate_reduced_to_resolved(
            reduced_coordinates,
            reduced_cells,
            reduced_fields[index],
            resolved_coordinates,
            used_vertices,
        )

        difference = resolved_field - reduced_on_resolved

        difference_squared = p1_l2_squared(
            resolved_coordinates,
            common_cells,
            difference,
        )
        resolved_squared = p1_l2_squared(
            resolved_coordinates,
            common_cells,
            resolved_field,
        )

        if resolved_squared <= 0.0:
            raise RuntimeError(
                f"Non-positive resolved field norm at t={time}."
            )

        absolute_discrepancy = np.sqrt(difference_squared)
        resolved_norm = np.sqrt(resolved_squared)
        relative_discrepancy = absolute_discrepancy / resolved_norm

        rows.append(
            {
                "snapshot_fraction": fractions[index],
                "time": time,
                "absolute_L2_discrepancy": absolute_discrepancy,
                "resolved_L2_norm": resolved_norm,
                "relative_L2_discrepancy": relative_discrepancy,
                "relative_L2_percent": 100.0 * relative_discrepancy,
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Circular common-domain exterior-field discrepancies")
    print("Comparison mesh: resolved exterior bulk mesh")
    print("Domain: Q \\ B_{R+h}(0)")
    print()

    for row in rows:
        print(
            f"t={row['time']:.3f}: "
            f"relative L2 = {row['relative_L2_discrepancy']:.10e} "
            f"({row['relative_L2_percent']:.6f}%)"
        )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
