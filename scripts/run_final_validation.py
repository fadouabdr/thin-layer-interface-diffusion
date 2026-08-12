import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
from dolfin import LogLevel, set_log_level

from src.full_thin_layer_solver import (
    run_full_validation_case,
    save_full_validation_case,
)
from src.reduced_validation_solver import (
    run_reduced_validation_case,
    save_reduced_validation_case,
)
from src.validation_config import (
    BULK_RESOLUTION,
    DT_VALIDATION,
    THICKNESSES,
)
from src.validation_diagnostics import (
    backward_euler_cumulative,
    compute_L2_bulk_error,
    empirical_rates,
    max_abs_curve_error,
    relative_error,
    validate_matching_time_grids,
)


set_log_level(LogLevel.ERROR)


N_LAYER = 8
N_BULK = BULK_RESOLUTION
N_Y = BULK_RESOLUTION
DT_VALUE = DT_VALIDATION
BALANCE_TOLERANCE = 1.0e-10


SUMMARY_FIELDS = [
    "h",
    "dt",
    "n_layer",
    "n_bulk",
    "ny",
    "full_num_cells",
    "reduced_num_cells",
    "J_full_T",
    "J_red_T",
    "E_J_T",
    "rel_E_J_T",
    "rate_E_J_T",
    "E_J_max",
    "M_full_T",
    "M_red_T",
    "E_M_T",
    "rel_E_M_T",
    "rate_E_M_T",
    "E_M_max",
    "B_full_T",
    "B_red_T",
    "E_B_T",
    "rel_E_B_T",
    "rate_E_B_T",
    "E_L2_T",
    "rel_E_L2_T",
    "rate_E_L2_T",
    "full_field_norm_T",
    "Q_outer_full_T",
    "max_full_balance_residual",
    "global_full_balance_error",
    "full_cumulative_consistency_error",
    "reduced_cumulative_consistency_error",
    "elapsed_seconds",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the final full-versus-reduced validation "
            "for all coating thicknesses."
        )
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases with an existing completed case summary.",
    )

    return parser.parse_args()


def thickness_tag(h):
    return f"h{int(round(h * 1000)):03d}"


def write_status(path, message):
    print(message, flush=True)

    with open(path, "a") as status_file:
        status_file.write(message + "\n")


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def read_single_row(path):
    with open(path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one row in {path}, found {len(rows)}."
        )

    return rows[0]


def ensure_finite(name, values):
    values = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"{name} contains NaN or infinite values."
        )


def run_one_case(h, case_summary_path):
    start_time = time.perf_counter()

    full = run_full_validation_case(
        h=h,
        n_layer=N_LAYER,
        n_bulk=N_BULK,
        ny=N_Y,
        dt_value=DT_VALUE,
    )

    reduced = run_reduced_validation_case(
        h=h,
        resolution=N_BULK,
        dt_value=DT_VALUE,
    )

    validate_matching_time_grids(
        full["times"],
        reduced["times"],
    )

    arrays = {
        "J_full": full["J_full"],
        "M_full": full["M_full"],
        "B_full": full["mass_bulk_full"],
        "Q_outer_full": full["Q_outer_full"],
        "balance_residual": full["bulk_balance_residual"],
        "J_red": reduced["J_red"],
        "M_red": reduced["M_red"],
        "B_red": reduced["mass_red"],
    }

    for name, values in arrays.items():
        ensure_finite(name, values)

    save_full_validation_case(
        full,
        outdir="results/validation_full",
    )

    save_reduced_validation_case(
        reduced,
        outdir="results/validation_reduced",
    )

    full_cumulative_check = backward_euler_cumulative(
        full["times"],
        full["J_full"],
    )

    reduced_cumulative_check = backward_euler_cumulative(
        reduced["times"],
        reduced["J_red"],
    )

    full_cumulative_error = float(
        np.max(
            np.abs(
                full["M_full"] - full_cumulative_check
            )
        )
    )

    reduced_cumulative_error = float(
        np.max(
            np.abs(
                reduced["M_red"] - reduced_cumulative_check
            )
        )
    )

    max_balance_residual = float(
        np.max(
            np.abs(full["bulk_balance_residual"])
        )
    )

    net_flux_cumulative = backward_euler_cumulative(
        full["times"],
        full["J_full"] - full["Q_outer_full"],
    )

    bulk_mass_change = float(
        full["mass_bulk_full"][-1]
        - full["mass_bulk_full"][0]
    )

    global_balance_error = abs(
        bulk_mass_change - net_flux_cumulative[-1]
    )

    checks = {
        "full cumulative consistency":
            full_cumulative_error,
        "reduced cumulative consistency":
            reduced_cumulative_error,
        "stepwise full mass balance":
            max_balance_residual,
        "global full mass balance":
            global_balance_error,
    }

    for name, value in checks.items():
        if value > BALANCE_TOLERANCE:
            raise RuntimeError(
                f"{name} failed for h={h}: {value:.6e}"
            )

    field = compute_L2_bulk_error(
        c_full=full["solution"],
        c_reduced=reduced["solution"],
    )

    J_full_T = float(full["J_full"][-1])
    J_red_T = float(reduced["J_red"][-1])
    E_J_T = abs(J_full_T - J_red_T)

    M_full_T = float(full["M_full"][-1])
    M_red_T = float(reduced["M_red"][-1])
    E_M_T = abs(M_full_T - M_red_T)

    B_full_T = float(full["mass_bulk_full"][-1])
    B_red_T = float(reduced["mass_red"][-1])
    E_B_T = abs(B_full_T - B_red_T)

    row = {
        "h": h,
        "dt": DT_VALUE,
        "n_layer": N_LAYER,
        "n_bulk": N_BULK,
        "ny": N_Y,
        "full_num_cells": full["num_cells"],
        "reduced_num_cells": reduced["num_cells"],
        "J_full_T": J_full_T,
        "J_red_T": J_red_T,
        "E_J_T": E_J_T,
        "rel_E_J_T": relative_error(
            E_J_T,
            J_full_T,
        ),
        "rate_E_J_T": "",
        "E_J_max": max_abs_curve_error(
            full["J_full"],
            reduced["J_red"],
        ),
        "M_full_T": M_full_T,
        "M_red_T": M_red_T,
        "E_M_T": E_M_T,
        "rel_E_M_T": relative_error(
            E_M_T,
            M_full_T,
        ),
        "rate_E_M_T": "",
        "E_M_max": max_abs_curve_error(
            full["M_full"],
            reduced["M_red"],
        ),
        "B_full_T": B_full_T,
        "B_red_T": B_red_T,
        "E_B_T": E_B_T,
        "rel_E_B_T": relative_error(
            E_B_T,
            B_full_T,
        ),
        "rate_E_B_T": "",
        "E_L2_T": field["absolute_error"],
        "rel_E_L2_T": field["relative_error"],
        "rate_E_L2_T": "",
        "full_field_norm_T": field["full_norm"],
        "Q_outer_full_T": float(
            full["Q_outer_full"][-1]
        ),
        "max_full_balance_residual":
            max_balance_residual,
        "global_full_balance_error":
            global_balance_error,
        "full_cumulative_consistency_error":
            full_cumulative_error,
        "reduced_cumulative_consistency_error":
            reduced_cumulative_error,
        "elapsed_seconds":
            time.perf_counter() - start_time,
    }

    write_rows(
        case_summary_path,
        SUMMARY_FIELDS,
        [row],
    )

    del field
    del full
    del reduced
    gc.collect()

    return row


def add_rates(rows):
    h_values = np.array(
        [float(row["h"]) for row in rows],
        dtype=float,
    )

    metric_to_rate = {
        "E_L2_T": "rate_E_L2_T",
        "E_B_T": "rate_E_B_T",
        "E_J_T": "rate_E_J_T",
        "E_M_T": "rate_E_M_T",
    }

    for metric, rate_field in metric_to_rate.items():
        errors = np.array(
            [float(row[metric]) for row in rows],
            dtype=float,
        )

        rates = empirical_rates(
            h_values,
            errors,
        )

        for row, rate in zip(rows, rates):
            row[rate_field] = (
                ""
                if not np.isfinite(rate)
                else float(rate)
            )


def main():
    args = parse_arguments()

    full_dir = Path("results/validation_full")
    reduced_dir = Path("results/validation_reduced")
    comparison_dir = Path(
        "results/validation_comparison"
    )
    case_root = Path("results/validation_cases")

    full_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    case_root.mkdir(parents=True, exist_ok=True)

    status_path = comparison_dir / "status.txt"

    if not args.resume:
        status_path.write_text("")

    rows = []

    for index, h in enumerate(THICKNESSES, start=1):
        tag = thickness_tag(h)
        case_dir = case_root / tag
        case_dir.mkdir(parents=True, exist_ok=True)

        case_summary_path = (
            case_dir / "case_summary.csv"
        )

        if args.resume and case_summary_path.exists():
            write_status(
                status_path,
                (
                    f"Skipping completed case "
                    f"{index}/{len(THICKNESSES)}: h={h}"
                ),
            )

            row = read_single_row(
                case_summary_path
            )

        else:
            write_status(
                status_path,
                (
                    f"Starting case "
                    f"{index}/{len(THICKNESSES)}: h={h}"
                ),
            )

            row = run_one_case(
                h=h,
                case_summary_path=case_summary_path,
            )

            write_status(
                status_path,
                (
                    f"Completed h={h}: "
                    f"E_L2={float(row['E_L2_T']):.6e}, "
                    f"E_B={float(row['E_B_T']):.6e}, "
                    f"E_J={float(row['E_J_T']):.6e}, "
                    f"E_M={float(row['E_M_T']):.6e}"
                ),
            )

        rows.append(row)

    add_rates(rows)

    summary_path = (
        comparison_dir / "validation_summary.csv"
    )

    write_rows(
        summary_path,
        SUMMARY_FIELDS,
        rows,
    )

    field_rows = [
        {
            "h": row["h"],
            "E_L2_T": row["E_L2_T"],
            "rel_E_L2_T": row["rel_E_L2_T"],
            "rate_E_L2_T": row["rate_E_L2_T"],
        }
        for row in rows
    ]

    write_rows(
        comparison_dir / "bulk_field_errors.csv",
        [
            "h",
            "E_L2_T",
            "rel_E_L2_T",
            "rate_E_L2_T",
        ],
        field_rows,
    )

    write_status(
        status_path,
        f"Saved summary: {summary_path}",
    )


if __name__ == "__main__":
    main()
