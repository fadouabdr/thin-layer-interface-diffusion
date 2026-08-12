"""Build and audit the resolved and reduced circular meshes.

This is the mandatory gate before either transient circular solver is added.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from src.circular_config import DEFAULT_CIRCULAR_CONFIG
from src.circular_mesh_utils import (
    audit_circular_mesh,
    build_reduced_circular_mesh,
    build_resolved_circular_mesh,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/circular/meshes",
        help="Directory for MSH, XDMF/HDF5, and the audit manifest.",
    )
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--half-width", type=float, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--sectors", type=int, default=None)
    parser.add_argument("--interface-size", type=float, default=None)
    parser.add_argument("--bulk-size", type=float, default=None)
    parser.add_argument("--transition-start", type=float, default=None)
    parser.add_argument("--transition-end", type=float, default=None)
    parser.add_argument("--geometry-rtol", type=float, default=None)
    parser.add_argument(
        "--gmsh-verbosity",
        type=int,
        default=0,
        help="Set to a positive value to show Gmsh diagnostic output.",
    )
    return parser


def _config_from_args(args):
    replacements = {}
    mapping = {
        "radius": "core_radius",
        "h": "coating_thickness",
        "half_width": "box_half_width",
        "layers": "radial_layers",
        "sectors": "angular_sectors",
        "interface_size": "interface_size",
        "bulk_size": "bulk_size",
        "transition_start": "transition_start",
        "transition_end": "transition_end",
        "geometry_rtol": "geometry_relative_tolerance",
    }
    for argument_name, field_name in mapping.items():
        value = getattr(args, argument_name)
        if value is not None:
            replacements[field_name] = value
    return replace(DEFAULT_CIRCULAR_CONFIG, **replacements).validate()


def _software_versions():
    import dolfin
    import gmsh
    import meshio
    import numpy

    return {
        "dolfin": dolfin.__version__,
        "gmsh": gmsh.__version__,
        "meshio": meshio.__version__,
        "numpy": numpy.__version__,
    }


def _print_audit(audit):
    print(f"{audit['model'].capitalize()} mesh")
    print(f"  passed: {audit['passed']}")
    print(f"  cells: {audit['num_cells']}")
    print(f"  P1 dofs: {audit['num_dofs_p1']}")
    if audit["model"] == "resolved":
        print(f"  radial layers: {audit['radial_layer_count']}")
    for check in audit["measure_checks"]:
        if "relative_error" in check:
            print(
                "  {name}: actual={actual:.12g}, expected={expected:.12g}, "
                "relative_error={relative_error:.3e}, passed={passed}".format(
                    **check
                )
            )
        else:
            print(
                "  {name}: actual={actual:.12g}, expected={expected:.12g}, "
                "absolute_error={absolute_error:.3e}, passed={passed}".format(
                    **check
                )
            )
    print(f"  cell tags: {audit['cell_tag_counts']}")
    print(f"  facet tags: {audit['facet_tag_counts']}")


def main():
    args = _parser().parse_args()
    config = _config_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_files = build_resolved_circular_mesh(
        config,
        output_dir,
        verbosity=args.gmsh_verbosity,
    )
    reduced_files = build_reduced_circular_mesh(
        config,
        output_dir,
        verbosity=args.gmsh_verbosity,
    )

    resolved_audit = audit_circular_mesh(
        resolved_files,
        config,
        resolved=True,
    )
    reduced_audit = audit_circular_mesh(
        reduced_files,
        config,
        resolved=False,
    )
    passed = resolved_audit["passed"] and reduced_audit["passed"]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "purpose": (
            "Geometry and physical-tag gate for the circular illustration; "
            "no transient PDE was solved."
        ),
        "config": config.to_manifest(),
        "software": _software_versions(),
        "resolved": resolved_audit,
        "reduced": reduced_audit,
    }
    manifest_path = output_dir / "circular_mesh_audit.json"
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    _print_audit(resolved_audit)
    _print_audit(reduced_audit)
    print(f"Audit manifest: {manifest_path}")

    if not passed:
        raise RuntimeError(
            "Circular mesh audit failed. Do not run a transient circular case."
        )

    print("All circular geometry and physical-tag checks passed.")


if __name__ == "__main__":
    main()
