"""Run the paired circular resolved/reduced transient illustration."""

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.circular_config import DEFAULT_CIRCULAR_CONFIG
from src.circular_initial_data import build_common_exterior_initial_state
from src.circular_full_solver import run_circular_full_case
from src.circular_mesh_utils import CircularMeshFiles
from src.circular_reduced_solver import run_circular_reduced_case
from src.circular_results import relative_max_mismatch
from src.validation_config import c_int
from src.validation_metrics import (
    relative_time_l2_error,
    validate_matching_time_grids,
)


DIAGNOSTIC_TOLERANCE = 1.0e-8
INITIALIZATION_TOLERANCE = 1.0e-10


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-dir",
        default="results/circular/meshes",
        help="Directory produced by scripts.check_circular_setup.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/circular/run",
        help="Directory for paired curves, snapshots, and the run manifest.",
    )
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only two accepted-size time steps (T=0.002, dt=0.001).",
    )
    return parser


def _require_passed_mesh_gate(mesh_dir):
    manifest_path = mesh_dir / "circular_mesh_audit.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Missing {manifest_path}; run scripts.check_circular_setup first."
        )
    with open(manifest_path) as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "passed":
        raise RuntimeError(
            "The circular mesh audit is not passed. Do not run the transient "
            "circular case."
        )
    return manifest_path, manifest


def _mesh_bundle(mesh_dir, model_name):
    return CircularMeshFiles(
        model_name=model_name,
        msh=mesh_dir / f"{model_name}.msh",
        cells_xdmf=mesh_dir / f"{model_name}_cells.xdmf",
        facets_xdmf=mesh_dir / f"{model_name}_facets.xdmf",
    )


def _verify_audited_geometry(audit, config):
    audited = audit.get("config", {})
    expected = config.to_manifest()
    keys = (
        "core_radius",
        "coating_thickness",
        "box_half_width",
        "radial_layers",
        "angular_sectors",
        "interface_size",
        "bulk_size",
        "transition_start",
        "transition_end",
        "polynomial_degree",
    )
    differences = []
    for key in keys:
        if key not in audited or not np.isclose(
            float(audited[key]),
            float(expected[key]),
            rtol=0.0,
            atol=1.0e-14,
        ):
            differences.append(key)
    if differences:
        raise RuntimeError(
            "The passed mesh audit does not match the requested circular "
            "configuration for: " + ", ".join(differences)
        )


def _verify_mesh_files(bundle):
    missing = [
        str(path)
        for path in (bundle.msh, bundle.cells_xdmf, bundle.facets_xdmf)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError("Missing circular mesh files: " + ", ".join(missing))


def _save_curves(path, full, reduced):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "t",
            "J_full",
            "J_red",
            "Q_outer_full",
            "Q_outer_red",
            "M_full",
            "M_red",
            "mass_bulk_full",
            "mass_coating_full",
            "mass_red",
            "balance_normalized_full",
            "balance_normalized_red",
            "robin_flux_mismatch_red",
        ])
        for row in zip(
            full["times"],
            full["J_full"],
            reduced["J_red"],
            full["Q_outer_full"],
            reduced["Q_outer_red"],
            full["M_full"],
            reduced["M_red"],
            full["mass_bulk_full"],
            full["mass_coating_full"],
            reduced["mass_red"],
            full["bulk_balance_normalized"],
            reduced["bulk_balance_normalized"],
            reduced["robin_flux_mismatch"],
        ):
            writer.writerow(row)


def _save_snapshots(path, full, reduced, config):
    snapshot_times = np.asarray(full["snapshot_times"], dtype=float)
    np.savez_compressed(
        path,
        snapshot_fractions=np.asarray(full["snapshot_fractions"], dtype=float),
        snapshot_indices=np.asarray(full["snapshot_indices"], dtype=np.int64),
        snapshot_times=snapshot_times,
        prescribed_core_concentration=np.asarray(
            [c_int(time, scenario=config.scenario) for time in snapshot_times],
            dtype=float,
        ),
        resolved_coordinates=full["coordinates"],
        resolved_cells=full["cells"],
        resolved_cell_tags=full["cell_tags"],
        resolved_q_vertex=full["snapshot_q_vertex"],
        reduced_coordinates=reduced["coordinates"],
        reduced_cells=reduced["cells"],
        reduced_cell_tags=reduced["cell_tags"],
        reduced_c_vertex=reduced["snapshot_c_vertex"],
    )


def _software_versions():
    import dolfin
    import gmsh
    import meshio

    return {
        "dolfin": dolfin.__version__,
        "gmsh": gmsh.__version__,
        "meshio": meshio.__version__,
        "numpy": np.__version__,
    }


def _diagnostics(full, reduced):
    times = validate_matching_time_grids(full["times"], reduced["times"])
    flux_error = relative_time_l2_error(
        times,
        reduced["J_red"],
        full["J_full"],
    )
    release_error = relative_time_l2_error(
        times,
        reduced["M_red"],
        full["M_full"],
    )
    max_balance_full = float(np.max(full["bulk_balance_normalized"]))
    max_balance_reduced = float(np.max(reduced["bulk_balance_normalized"]))
    robin_mismatch_relative = float(
        np.max(np.abs(reduced["robin_flux_mismatch"]))
        / max(float(np.max(np.abs(reduced["J_red"]))), 1.0e-14)
    )
    initialization_checks = {
        "resolved_common_exterior_vertex_mismatch": float(
            full["initialization_checks"][
                "common_exterior_vertex_mismatch"
            ]
        ),
        "reduced_common_exterior_vertex_mismatch": float(
            reduced["initialization_checks"][
                "common_exterior_vertex_mismatch"
            ]
        ),
        "resolved_inner_partition_vertex_mismatch": float(
            full["initialization_checks"][
                "inner_partition_vertex_mismatch"
            ]
        ),
        "resolved_outer_partition_vertex_mismatch": float(
            full["initialization_checks"][
                "outer_partition_vertex_mismatch"
            ]
        ),
    }
    max_initialization_mismatch = max(initialization_checks.values())
    passed = (
        max_balance_full <= DIAGNOSTIC_TOLERANCE
        and max_balance_reduced <= DIAGNOSTIC_TOLERANCE
        and robin_mismatch_relative <= DIAGNOSTIC_TOLERANCE
        and max_initialization_mismatch <= INITIALIZATION_TOLERANCE
    )
    return {
        "passed": bool(passed),
        "diagnostic_tolerance": DIAGNOSTIC_TOLERANCE,
        "max_bulk_balance_normalized_full": max_balance_full,
        "max_bulk_balance_normalized_reduced": max_balance_reduced,
        "max_robin_flux_mismatch_relative": robin_mismatch_relative,
        "initialization_tolerance": INITIALIZATION_TOLERANCE,
        "initialization_checks": initialization_checks,
        "max_initialization_vertex_mismatch": max_initialization_mismatch,
        "relative_time_l2_flux_difference": float(flux_error),
        "relative_time_l2_cumulative_release_difference": float(release_error),
        "relative_max_flux_difference": relative_max_mismatch(
            reduced["J_red"],
            full["J_full"],
        ),
        "relative_max_cumulative_release_difference": relative_max_mismatch(
            reduced["M_red"],
            full["M_full"],
        ),
    }


def main():
    args = _parser().parse_args()
    mesh_dir = Path(args.mesh_dir)
    output_dir = Path(args.output_dir)
    if args.smoke:
        if args.t_final is not None or args.dt is not None:
            raise ValueError("--smoke cannot be combined with --t-final or --dt.")
        config = replace(DEFAULT_CIRCULAR_CONFIG, t_final=0.002, dt=0.001)
    else:
        replacements = {}
        if args.t_final is not None:
            replacements["t_final"] = args.t_final
        if args.dt is not None:
            replacements["dt"] = args.dt
        config = replace(DEFAULT_CIRCULAR_CONFIG, **replacements)
    config.validate()

    audit_path, audit = _require_passed_mesh_gate(mesh_dir)
    _verify_audited_geometry(audit, config)
    resolved_files = _mesh_bundle(mesh_dir, "resolved")
    reduced_files = _mesh_bundle(mesh_dir, "reduced")
    _verify_mesh_files(resolved_files)
    _verify_mesh_files(reduced_files)

    print("Building the common circular exterior initial datum...")
    common_initial_state = build_common_exterior_initial_state(
        reduced_files,
        config,
    )
    print("Running resolved circular model...")
    full = run_circular_full_case(
        resolved_files,
        config,
        common_initial_state,
    )
    print("Running reduced circular model at the same physical times...")
    reduced = run_circular_reduced_case(
        reduced_files,
        config,
        common_initial_state,
        full["snapshot_indices"],
    )
    if not np.allclose(
        full["snapshot_times"],
        reduced["snapshot_times"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Resolved and reduced snapshot times do not match.")

    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = output_dir / "circular_curves.csv"
    snapshots_path = output_dir / "circular_snapshots.npz"
    manifest_path = output_dir / "circular_run_manifest.json"
    _save_curves(curves_path, full, reduced)
    _save_snapshots(snapshots_path, full, reduced, config)
    diagnostics = _diagnostics(full, reduced)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if diagnostics["passed"] else "failed",
        "purpose": (
            "Paired circular resolved/reduced illustration; qualitative curved "
            "comparison, not an asymptotic convergence-rate experiment."
        ),
        "mode": "smoke" if args.smoke else "production",
        "config": config.to_manifest(),
        "mesh_audit": str(audit_path),
        "mesh_audit_created_utc": audit.get("created_utc"),
        "software": _software_versions(),
        "initialization": {
            "name": common_initial_state["name"],
            "common_exterior_initial_datum": True,
            "common_exterior_definition": common_initial_state["definition"],
            "source_mesh_num_cells": common_initial_state["num_cells"],
            "source_mesh_num_dofs": common_initial_state["num_dofs"],
            "resolved_coating_profile": full["initialization_checks"][
                "coating_profile"
            ],
            "flux_compatibility_statement": (
                "The coating profile enforces both partition traces. It is "
                "not forced to satisfy resolved flux continuity at t=0; "
                "displayed snapshots are selected after the coating "
                "normal-equilibration scale."
            ),
        },
        "resolved": {
            "num_cells": full["num_cells"],
            "num_dofs": full["num_dofs"],
            "stationary_balance_normalized": full[
                "stationary_balance_normalized"
            ],
        },
        "reduced": {
            "num_cells": reduced["num_cells"],
            "num_dofs": reduced["num_dofs"],
            "stationary_balance_normalized": reduced[
                "stationary_balance_normalized"
            ],
            "stationary_robin_mismatch": reduced[
                "stationary_robin_mismatch"
            ],
        },
        "snapshot_fractions": [
            float(value) for value in full["snapshot_fractions"]
        ],
        "snapshot_fraction_definition": (
            "Fractions of the resolved cumulative release observed at the "
            "finite production horizon T, not fractions of an eventual "
            "infinite-time total."
        ),
        "snapshot_indices": [int(value) for value in full["snapshot_indices"]],
        "snapshot_times": [float(value) for value in full["snapshot_times"]],
        "diagnostics": diagnostics,
        "files": {
            "curves": str(curves_path),
            "snapshots": str(snapshots_path),
            "manifest": str(manifest_path),
        },
    }
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(f"Resolved cells/dofs: {full['num_cells']}/{full['num_dofs']}")
    print(f"Reduced cells/dofs: {reduced['num_cells']}/{reduced['num_dofs']}")
    print(
        "Selected snapshot times: "
        + ", ".join(f"{time:.6g}" for time in full["snapshot_times"])
    )
    print(
        "Maximum normalized balances (resolved/reduced): "
        f"{diagnostics['max_bulk_balance_normalized_full']:.3e} / "
        f"{diagnostics['max_bulk_balance_normalized_reduced']:.3e}"
    )
    print(
        "Maximum relative Robin-flux mismatch: "
        f"{diagnostics['max_robin_flux_mismatch_relative']:.3e}"
    )
    print(
        "Maximum common-initialization vertex mismatch: "
        f"{diagnostics['max_initialization_vertex_mismatch']:.3e}"
    )
    print(f"Run manifest: {manifest_path}")

    if not diagnostics["passed"]:
        raise RuntimeError(
            "Circular transient diagnostics failed. Do not make paper figures."
        )
    print("Circular transient solver and signed-flux checks passed.")


if __name__ == "__main__":
    main()
