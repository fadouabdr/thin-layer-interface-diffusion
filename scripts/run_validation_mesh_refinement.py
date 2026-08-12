import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dolfin import LogLevel, set_log_level

from scripts.run_validation_time_refinement import (
    SUMMARY_FIELDS,
    read_case_summary,
    run_case,
)
from src.validation_config import DT_VALIDATION


set_log_level(LogLevel.ERROR)


METRICS = [
    "E_L2_T",
    "E_B_T",
    "E_J_T",
    "E_M_T",
]

DERIVED_FIELDS = []

for metric in METRICS:
    DERIVED_FIELDS.extend([
        f"abs_change_previous_{metric}",
        f"rel_change_previous_{metric}",
        f"abs_change_baseline_{metric}",
        f"rel_change_baseline_{metric}",
    ])


OUTPUT_FIELDS = [
    "case",
    "reference_case",
    "source_case_summary",
] + SUMMARY_FIELDS + DERIVED_FIELDS


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the targeted spatial-refinement study for the "
            "full/reduced thin-layer validation."
        )
    )

    parser.add_argument(
        "--h",
        type=float,
        default=0.01,
        help="Coating thickness.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DT_VALIDATION,
        help="Time-step size.",
    )
    parser.add_argument(
        "--baseline-summary",
        default=(
            "results/refinement/time_step_h010/"
            "dt_1p00em04/case_summary.csv"
        ),
        help="Completed baseline case summary.",
    )
    parser.add_argument(
        "--outdir",
        default="results/refinement/mesh_h010",
        help="Root output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip mesh cases that already have a case summary.",
    )

    return parser.parse_args()


def write_status(path, message):
    print(message, flush=True)

    with open(path, "a") as status_file:
        status_file.write(message + "\n")


def as_float(row, field):
    return float(row[field])


def relative_change(change, reference):
    return change / max(abs(reference), 1.0e-14)


def validate_baseline(row, h_value, dt_value):
    expected = {
        "h": h_value,
        "dt": dt_value,
        "n_layer": 8,
        "n_bulk": 64,
        "ny": 64,
    }

    for field, expected_value in expected.items():
        actual_value = float(row[field])

        if not np.isclose(
            actual_value,
            expected_value,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"Baseline mismatch for {field}: "
                f"expected {expected_value}, "
                f"found {actual_value}."
            )


def add_refinement_changes(rows):
    baseline = rows[0]

    for index, row in enumerate(rows):
        if index == 0:
            row["reference_case"] = ""

            for field in DERIVED_FIELDS:
                row[field] = ""

            continue

        previous = rows[index - 1]
        row["reference_case"] = previous["case"]

        for metric in METRICS:
            current_value = as_float(row, metric)
            previous_value = as_float(previous, metric)
            baseline_value = as_float(baseline, metric)

            previous_change = abs(
                current_value - previous_value
            )

            baseline_change = abs(
                current_value - baseline_value
            )

            row[
                f"abs_change_previous_{metric}"
            ] = previous_change

            row[
                f"rel_change_previous_{metric}"
            ] = relative_change(
                previous_change,
                previous_value,
            )

            row[
                f"abs_change_baseline_{metric}"
            ] = baseline_change

            row[
                f"rel_change_baseline_{metric}"
            ] = relative_change(
                baseline_change,
                baseline_value,
            )


def write_summary(path, rows):
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()

    output_root = Path(args.outdir)
    output_root.mkdir(parents=True, exist_ok=True)

    status_path = output_root / "status.txt"
    summary_path = output_root / "mesh_refinement_summary.csv"

    if not args.resume:
        status_path.write_text("")

    baseline_path = Path(args.baseline_summary)

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline summary not found: {baseline_path}"
        )

    baseline = read_case_summary(baseline_path)

    validate_baseline(
        baseline,
        h_value=args.h,
        dt_value=args.dt,
    )

    baseline["case"] = "baseline"
    baseline["source_case_summary"] = str(
        baseline_path
    )

    rows = [baseline]

    configurations = [
        {
            "case": "coating_refined",
            "n_layer": 16,
            "n_bulk": 64,
            "ny": 64,
        },
        {
            "case": "fully_refined",
            "n_layer": 16,
            "n_bulk": 128,
            "ny": 128,
        },
    ]

    for index, configuration in enumerate(
        configurations,
        start=1,
    ):
        case_name = configuration["case"]
        case_root = output_root / case_name
        case_summary = case_root / "case_summary.csv"

        if args.resume and case_summary.exists():
            write_status(
                status_path,
                (
                    f"Skipping completed mesh case "
                    f"{index}/{len(configurations)}: "
                    f"{case_name}"
                ),
            )

            row = read_case_summary(case_summary)

        else:
            write_status(
                status_path,
                (
                    f"Starting mesh case "
                    f"{index}/{len(configurations)}: "
                    f"{case_name}, "
                    f"n_layer={configuration['n_layer']}, "
                    f"n_bulk={configuration['n_bulk']}, "
                    f"ny={configuration['ny']}"
                ),
            )

            case_arguments = SimpleNamespace(
                h=args.h,
                n_layer=configuration["n_layer"],
                n_bulk=configuration["n_bulk"],
                ny=configuration["ny"],
            )

            row = run_case(
                args=case_arguments,
                dt_value=args.dt,
                case_root=case_root,
            )

            write_status(
                status_path,
                (
                    f"Completed {case_name}: "
                    f"E_L2={float(row['E_L2_T']):.6e}, "
                    f"E_B={float(row['E_B_T']):.6e}, "
                    f"E_J={float(row['E_J_T']):.6e}, "
                    f"E_M={float(row['E_M_T']):.6e}, "
                    f"balance="
                    f"{float(row['max_full_balance_residual']):.3e}"
                ),
            )

        row["case"] = case_name
        row["source_case_summary"] = str(
            case_summary
        )

        rows.append(row)

        add_refinement_changes(rows)
        write_summary(summary_path, rows)

    add_refinement_changes(rows)
    write_summary(summary_path, rows)

    write_status(
        status_path,
        f"Saved summary: {summary_path}",
    )


if __name__ == "__main__":
    main()
