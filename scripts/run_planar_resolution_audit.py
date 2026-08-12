"""Audit whether numerical changes are smaller than modeling discrepancies."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from dolfin import LogLevel, set_log_level

from src.planar_validation import (
    SUMMARY_FIELDS,
    run_planar_comparison,
    save_planar_comparison,
)
from src.validation_config import (
    CANONICAL_DT_CANDIDATE,
    CANONICAL_T_FINAL,
    DEFAULT_INITIALIZATION,
    DEFAULT_SCENARIO,
    INITIALIZATIONS,
    SCENARIOS,
)


set_log_level(LogLevel.ERROR)


AUDIT_METRICS = [
    "E_L2_T",
    "E_B_T",
    "E_J_L2_rel",
    "E_M_T",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the paired spatial/temporal production-resolution audit."
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
    )
    parser.add_argument(
        "--h-values",
        type=float,
        nargs="+",
        default=[0.08, 0.01],
    )
    parser.add_argument(
        "--baseline-n-layer",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--baseline-n-bulk",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--baseline-ny",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--outdir",
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args()

    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive.")

    return args


def h_tag(h):
    return f"h{int(round(1000.0 * h)):03d}"


def default_output_root(args):
    return Path("results") / "resolution_audit" / (
        f"{args.scenario}_{args.initialization}"
        f"_T{args.t_final:g}_dt{args.dt:.0e}"
        f"_L{args.baseline_n_layer}"
        f"_B{args.baseline_n_bulk}"
        f"_Y{args.baseline_ny}"
    )


def configurations(args):
    n_layer = args.baseline_n_layer
    n_bulk = args.baseline_n_bulk
    ny = args.baseline_ny
    dt_value = args.dt

    return [
        {
            "case": "baseline",
            "n_layer": n_layer,
            "n_bulk": n_bulk,
            "ny": ny,
            "dt": dt_value,
        },
        {
            "case": "exterior_refined",
            "n_layer": n_layer,
            "n_bulk": 2 * n_bulk,
            "ny": 2 * ny,
            "dt": dt_value,
        },
        {
            "case": "coating_refined",
            "n_layer": 2 * n_layer,
            "n_bulk": n_bulk,
            "ny": ny,
            "dt": dt_value,
        },
        {
            "case": "time_refined",
            "n_layer": n_layer,
            "n_bulk": n_bulk,
            "ny": ny,
            "dt": 0.5 * dt_value,
        },
        {
            "case": "combined_reference",
            "n_layer": 2 * n_layer,
            "n_bulk": 2 * n_bulk,
            "ny": 2 * ny,
            "dt": 0.5 * dt_value,
        },
    ]


def manifest_payload(args):
    return {
        "schema_version": 2,
        "scenario": args.scenario,
        "initialization": args.initialization,
        "T": args.t_final,
        "candidate_dt": args.dt,
        "h_values": list(args.h_values),
        "baseline_n_layer": args.baseline_n_layer,
        "baseline_n_bulk": args.baseline_n_bulk,
        "baseline_ny": args.baseline_ny,
        "configurations": configurations(args),
        "primary_acceptance_threshold": 0.05,
        "all_metric_acceptance_threshold": 0.10,
    }


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
            f"{root} is nonempty but has no audit manifest. "
            "Choose a new --outdir or use --force after verifying it."
        )

    if path.exists():
        with open(path) as manifest_file:
            existing = json.load(manifest_file)

        if args.resume:
            if existing != requested:
                raise RuntimeError(
                    "Cannot resume because the audit manifest differs."
                )
            return

        if not args.force:
            raise RuntimeError(
                f"{path} already exists. Use --resume, --force, or a "
                "different --outdir."
            )

    with open(path, "w") as manifest_file:
        json.dump(requested, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")


def read_single_row(path):
    with open(path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one row in {path}, found {len(rows)}."
        )

    return rows[0]


def validate_resumed_row(row, args, h, configuration):
    exact = {
        "scenario": args.scenario,
        "initialization": args.initialization,
    }
    numeric = {
        "h": h,
        "T": args.t_final,
        "dt": configuration["dt"],
        "n_layer": configuration["n_layer"],
        "n_bulk": configuration["n_bulk"],
        "ny": configuration["ny"],
    }

    for field, expected in exact.items():
        if row.get(field) != expected:
            raise RuntimeError(
                f"Resume mismatch for {field}: "
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
                f"Resume mismatch for {field}: "
                f"{row[field]} != {expected}."
            )


def add_reference_changes(rows):
    reference = next(
        row
        for row in rows
        if row["audit_case"] == "combined_reference"
    )

    for row in rows:
        primary_changes = []
        all_changes = []

        for metric in AUDIT_METRICS:
            current = float(row[metric])
            reference_value = float(reference[metric])
            change = abs(current - reference_value)
            relative_change = (
                change / max(abs(reference_value), 1.0e-14)
            )
            row[f"abs_change_to_reference_{metric}"] = change
            row[f"rel_change_to_reference_{metric}"] = relative_change
            all_changes.append(relative_change)

            if metric in ("E_L2_T", "E_B_T"):
                primary_changes.append(relative_change)

        row["reference_case"] = "combined_reference"
        row["primary_metrics_accepted"] = (
            max(primary_changes) <= 0.05
        )
        row["all_metrics_accepted"] = (
            max(all_changes) <= 0.10
        )


def audit_fields():
    fields = [
        "audit_case",
        "reference_case",
    ] + SUMMARY_FIELDS

    for metric in AUDIT_METRICS:
        fields.extend([
            f"abs_change_to_reference_{metric}",
            f"rel_change_to_reference_{metric}",
        ])

    fields.extend([
        "primary_metrics_accepted",
        "all_metrics_accepted",
    ])
    return fields


def write_audit(path, rows):
    fields = audit_fields()

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    output_root = (
        Path(args.outdir)
        if args.outdir
        else default_output_root(args)
    )
    prepare_manifest(output_root, args)

    all_rows = []

    for h in args.h_values:
        h_rows = []

        for configuration in configurations(args):
            case_name = configuration["case"]
            case_root = (
                output_root
                / "cases"
                / h_tag(h)
                / case_name
            )
            summary_path = case_root / "case_summary.csv"

            if args.resume and summary_path.exists():
                row = read_single_row(summary_path)
                validate_resumed_row(
                    row,
                    args,
                    h,
                    configuration,
                )
                print(
                    f"Reusing h={h:g}, case={case_name}",
                    flush=True,
                )
            else:
                print(
                    f"Running h={h:g}, case={case_name}",
                    flush=True,
                )
                result = run_planar_comparison(
                    h=h,
                    t_final=args.t_final,
                    dt_value=configuration["dt"],
                    scenario=args.scenario,
                    initialization=args.initialization,
                    n_layer=configuration["n_layer"],
                    n_bulk=configuration["n_bulk"],
                    ny=configuration["ny"],
                )
                save_planar_comparison(result, case_root)
                row = result["summary"]

            row["audit_case"] = case_name
            h_rows.append(row)

        add_reference_changes(h_rows)
        all_rows.extend(h_rows)
        write_audit(
            output_root / "resolution_audit.csv",
            all_rows,
        )

    baselines = [
        row
        for row in all_rows
        if row["audit_case"] == "baseline"
    ]
    accepted = all(
        bool(row["primary_metrics_accepted"])
        and bool(row["all_metrics_accepted"])
        for row in baselines
    )

    status = {
        "candidate_settings_accepted": accepted,
        "scenario": args.scenario,
        "initialization": args.initialization,
        "T": args.t_final,
        "dt": args.dt,
        "n_layer": args.baseline_n_layer,
        "n_bulk": args.baseline_n_bulk,
        "ny": args.baseline_ny,
        "interpretation": (
            "Accepted: baseline differs from the combined reference by "
            "at most 5% for field/mass errors and 10% for all audited "
            "model-discrepancy metrics."
            if accepted
            else
            "Not accepted: refine the failing numerical component and "
            "rerun the audit before the thickness study."
        ),
    }

    with open(output_root / "acceptance.json", "w") as status_file:
        json.dump(status, status_file, indent=2, sort_keys=True)
        status_file.write("\n")

    print(status["interpretation"], flush=True)


if __name__ == "__main__":
    main()
