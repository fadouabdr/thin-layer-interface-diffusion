"""Run the canonical prepared planar thickness study.

This script writes to results/canonical and never reads the June-2026 legacy
tables.  A manifest prevents --resume from mixing incompatible parameters.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import dolfin
import numpy as np
from dolfin import LogLevel, set_log_level

from src.planar_validation import (
    RATE_FIELDS,
    SUMMARY_FIELDS,
    run_planar_comparison,
    save_planar_comparison,
    write_summary,
)
from src.validation_config import (
    BULK_RESOLUTION,
    CANONICAL_DT_CANDIDATE,
    CANONICAL_T_FINAL,
    DEFAULT_INITIALIZATION,
    DEFAULT_SCENARIO,
    INITIALIZATIONS,
    SCENARIOS,
    THICKNESSES,
    scenario_metadata,
)
from src.validation_metrics import empirical_rates


set_log_level(LogLevel.ERROR)


PAPER_FIELDS = [
    "scenario",
    "initialization",
    "h",
    "T",
    "dt",
    "E_L2_T",
    "rel_E_L2_T",
    "rate_E_L2_T",
    "E_B_T",
    "rel_E_B_T",
    "rate_E_B_T",
    "E_J_L2_rel",
    "rate_E_J_L2_rel",
    "E_M_T",
    "rel_E_M_T",
    "rate_E_M_T",
    "max_full_balance_normalized",
    "max_reduced_balance_normalized",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run a scenario-aware full-versus-reduced planar h-study "
            "with prepared initial data and canonical diagnostics."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default=DEFAULT_SCENARIO,
    )
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATIONS,
        default=DEFAULT_INITIALIZATION,
    )
    parser.add_argument(
        "--t-final",
        type=float,
        default=CANONICAL_T_FINAL,
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=CANONICAL_DT_CANDIDATE,
        help=(
            "Candidate production time step. Accept it only after the "
            "resolution audit."
        ),
    )
    parser.add_argument(
        "--thicknesses",
        type=float,
        nargs="+",
        default=THICKNESSES,
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n-bulk",
        type=int,
        default=BULK_RESOLUTION,
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=BULK_RESOLUTION,
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output root. The default is a scenario-specific directory "
            "under results/canonical."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the existing manifest matches exactly.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite requested case files in an existing output root.",
    )
    parser.add_argument(
        "--audit-acceptance",
        default=None,
        help=(
            "Path to the matching resolution-audit acceptance.json. "
            "The default is derived from the requested settings."
        ),
    )
    parser.add_argument(
        "--allow-unaudited",
        action="store_true",
        help=(
            "Permit a pilot run without an accepted audit. Do not use "
            "such output as production manuscript evidence."
        ),
    )

    args = parser.parse_args()

    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive.")

    if any(h <= 0.0 for h in args.thicknesses):
        parser.error("Every coating thickness must be positive.")

    if len(set(args.thicknesses)) != len(args.thicknesses):
        parser.error("The thickness list contains duplicates.")

    return args


def case_tag(h):
    return f"h{int(round(1000.0 * h)):03d}"


def default_output_root(args):
    return Path(
        "results"
    ) / "canonical" / (
        f"{args.scenario}_{args.initialization}"
        f"_T{args.t_final:g}_dt{args.dt:.0e}"
        f"_L{args.n_layer}_B{args.n_bulk}_Y{args.ny}"
    )


def default_audit_acceptance_path(args):
    return Path("results") / "resolution_audit" / (
        f"{args.scenario}_{args.initialization}"
        f"_T{args.t_final:g}_dt{args.dt:.0e}"
        f"_L{args.n_layer}_B{args.n_bulk}_Y{args.ny}"
    ) / "acceptance.json"


def require_matching_audit(args):
    if args.allow_unaudited:
        return

    path = (
        Path(args.audit_acceptance)
        if args.audit_acceptance
        else default_audit_acceptance_path(args)
    )

    if not path.exists():
        raise RuntimeError(
            "The matching resolution audit has not been accepted. "
            f"Expected {path}. Run the audit first, pass "
            "--audit-acceptance explicitly, or use --allow-unaudited "
            "for a non-production pilot."
        )

    with open(path) as acceptance_file:
        acceptance = json.load(acceptance_file)

    expected = {
        "scenario": args.scenario,
        "initialization": args.initialization,
        "T": args.t_final,
        "dt": args.dt,
        "n_layer": args.n_layer,
        "n_bulk": args.n_bulk,
        "ny": args.ny,
    }

    for field, value in expected.items():
        actual = acceptance.get(field)

        if isinstance(value, str):
            matches = actual == value
        else:
            matches = (
                actual is not None
                and np.isclose(
                    float(actual),
                    float(value),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )

        if not matches:
            raise RuntimeError(
                "The audit acceptance file does not match the requested "
                f"{field}: {actual!r} != {value!r}."
            )

    if not acceptance.get("candidate_settings_accepted", False):
        raise RuntimeError(
            f"The audit at {path} did not accept these settings."
        )


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifest_payload(args):
    metadata = scenario_metadata(args.scenario)
    metadata.update({
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "initialization": args.initialization,
        "T": args.t_final,
        "dt": args.dt,
        "thicknesses": list(args.thicknesses),
        "n_layer": args.n_layer,
        "n_bulk": args.n_bulk,
        "ny": args.ny,
        "python_version": platform.python_version(),
        "fenics_dolfin_version": dolfin.__version__,
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "status": (
            "candidate settings; production status requires "
            "resolution-audit acceptance"
        ),
    })
    return metadata


def manifest_comparison_view(payload):
    keys = [
        "schema_version",
        "scenario",
        "initialization",
        "T",
        "dt",
        "thicknesses",
        "n_layer",
        "n_bulk",
        "ny",
    ]
    return {key: payload.get(key) for key in keys}


def prepare_manifest(root, args):
    root_existed = root.exists()
    root_had_content = (
        root_existed
        and any(root.iterdir())
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    requested = manifest_payload(args)

    if root_had_content and not path.exists() and not args.force:
        raise RuntimeError(
            f"{root} is nonempty but has no canonical manifest. "
            "Choose a new --outdir or use --force after verifying it."
        )

    if path.exists():
        with open(path) as manifest_file:
            existing = json.load(manifest_file)

        if args.resume:
            if (
                manifest_comparison_view(existing)
                != manifest_comparison_view(requested)
            ):
                raise RuntimeError(
                    "Cannot resume: the existing manifest does not match "
                    "the requested scenario, time grid, thicknesses, or mesh."
                )
            return existing

        if not args.force:
            raise RuntimeError(
                f"{path} already exists. Use --resume only for an exact "
                "continuation, --force to overwrite requested cases, or "
                "choose a new --outdir."
            )

    with open(path, "w") as manifest_file:
        json.dump(requested, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")

    return requested


def read_single_row(path):
    with open(path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one row in {path}, found {len(rows)}."
        )

    return rows[0]


def validate_resumed_row(row, args, h):
    exact = {
        "scenario": args.scenario,
        "initialization": args.initialization,
    }
    numeric = {
        "h": h,
        "T": args.t_final,
        "dt": args.dt,
        "n_layer": args.n_layer,
        "n_bulk": args.n_bulk,
        "ny": args.ny,
    }

    for field, expected in exact.items():
        if row.get(field) != expected:
            raise RuntimeError(
                f"Resume mismatch in {field}: "
                f"{row.get(field)!r} != {expected!r}."
            )

    for field, expected in numeric.items():
        if not np.isclose(
            float(row[field]),
            float(expected),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"Resume mismatch in {field}: "
                f"{row[field]} != {expected}."
            )


def add_rates(rows):
    h_values = [float(row["h"]) for row in rows]

    for metric, rate_field in RATE_FIELDS.items():
        errors = [float(row[metric]) for row in rows]
        rates = empirical_rates(h_values, errors)

        for row, rate in zip(rows, rates):
            row[rate_field] = (
                ""
                if not np.isfinite(rate)
                else float(rate)
            )


def main():
    args = parse_arguments()
    require_matching_audit(args)
    output_root = (
        Path(args.outdir)
        if args.outdir
        else default_output_root(args)
    )
    prepare_manifest(output_root, args)

    rows = []

    for index, h in enumerate(args.thicknesses, start=1):
        case_root = output_root / "cases" / case_tag(h)
        summary_path = case_root / "case_summary.csv"

        if args.resume and summary_path.exists():
            row = read_single_row(summary_path)
            validate_resumed_row(row, args, h)
            print(
                f"[{index}/{len(args.thicknesses)}] "
                f"Reusing validated case h={h:g}",
                flush=True,
            )
        else:
            print(
                f"[{index}/{len(args.thicknesses)}] "
                f"Running h={h:g}",
                flush=True,
            )
            result = run_planar_comparison(
                h=h,
                t_final=args.t_final,
                dt_value=args.dt,
                scenario=args.scenario,
                initialization=args.initialization,
                n_layer=args.n_layer,
                n_bulk=args.n_bulk,
                ny=args.ny,
            )
            save_planar_comparison(result, case_root)
            row = result["summary"]

        rows.append(row)
        add_rates(rows)
        write_summary(
            output_root / "canonical_summary.csv",
            rows,
            fields=SUMMARY_FIELDS,
        )

    add_rates(rows)
    write_summary(
        output_root / "canonical_summary.csv",
        rows,
        fields=SUMMARY_FIELDS,
    )
    write_summary(
        output_root / "paper_validation_summary.csv",
        rows,
        fields=PAPER_FIELDS,
    )

    print(
        "Canonical candidate study completed. "
        f"Summary: {output_root / 'canonical_summary.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
