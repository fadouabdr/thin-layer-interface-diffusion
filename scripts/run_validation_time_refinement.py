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
from src.validation_config import BULK_RESOLUTION
from src.validation_diagnostics import (
    backward_euler_cumulative,
    compute_L2_bulk_error,
    max_abs_curve_error,
    relative_error,
    validate_matching_time_grids,
)


set_log_level(LogLevel.ERROR)


SUMMARY_FIELDS = [
    "h",
    "dt",
    "num_steps",
    "n_layer",
    "n_bulk",
    "ny",
    "full_num_cells",
    "reduced_num_cells",
    "J_full_0",
    "J_full_1",
    "J_full_T",
    "J_red_0",
    "J_red_1",
    "J_red_T",
    "E_J_T",
    "rel_E_J_T",
    "E_J_max",
    "M_full_T",
    "M_red_T",
    "E_M_T",
    "rel_E_M_T",
    "E_M_max",
    "B_full_T",
    "B_red_T",
    "E_B_T",
    "rel_E_B_T",
    "E_L2_T",
    "rel_E_L2_T",
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
            "Run the official full/reduced time-step refinement study."
        )
    )

    parser.add_argument(
        "--h",
        type=float,
        default=0.01,
        help="Coating thickness used in the refinement study.",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=8,
        help="Number of elements across the coating.",
    )
    parser.add_argument(
        "--n-bulk",
        type=int,
        default=BULK_RESOLUTION,
        help="Bulk resolution in the x direction.",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=BULK_RESOLUTION,
        help="Resolution in the y direction.",
    )
    parser.add_argument(
        "--dts",
        type=float,
        nargs="+",
        default=[
            1.0e-4,
            5.0e-5,
            2.5e-5,
        ],
        help="Time-step values to test.",
    )
    parser.add_argument(
        "--outdir",
        default="results/refinement/time_step_h010",
        help="Root output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases that already have a completed case_summary.csv.",
    )

    return parser.parse_args()


def dt_tag(dt_value):
    return (
        f"dt_{dt_value:.2e}"
        .replace(".", "p")
        .replace("+", "")
        .replace("-", "m")
    )


def write_status(path, message):
    print(message, flush=True)

    with open(path, "a") as status_file:
        status_file.write(message + "\n")


def write_summary(path, rows):
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=SUMMARY_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_case_summary(path):
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one row in {path}, found {len(rows)}."
        )

    return rows[0]


def ensure_finite(name, values):
    values = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"{name} contains NaN or infinite values."
        )


def run_case(args, dt_value, case_root):
    start_time = time.perf_counter()

    full_dir = case_root / "full"
    reduced_dir = case_root / "reduced"

    full_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)

    full = run_full_validation_case(
        h=args.h,
        n_layer=args.n_layer,
        n_bulk=args.n_bulk,
        ny=args.ny,
        dt_value=dt_value,
    )

    reduced = run_reduced_validation_case(
        h=args.h,
        resolution=args.n_bulk,
        dt_value=dt_value,
    )

    validate_matching_time_grids(
        full["times"],
        reduced["times"],
    )

    arrays_to_check = {
        "J_full": full["J_full"],
        "M_full": full["M_full"],
        "mass_bulk_full": full["mass_bulk_full"],
        "Q_outer_full": full["Q_outer_full"],
        "bulk_balance_residual": full[
            "bulk_balance_residual"
        ],
        "J_red": reduced["J_red"],
        "M_red": reduced["M_red"],
        "mass_red": reduced["mass_red"],
    }

    for name, values in arrays_to_check.items():
        ensure_finite(name, values)

    save_full_validation_case(
        full,
        outdir=full_dir,
    )

    save_reduced_validation_case(
        reduced,
        outdir=reduced_dir,
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
                full["M_full"]
                - full_cumulative_check
            )
        )
    )

    reduced_cumulative_error = float(
        np.max(
            np.abs(
                reduced["M_red"]
                - reduced_cumulative_check
            )
        )
    )

    max_balance_residual = float(
        np.max(
            np.abs(
                full["bulk_balance_residual"]
            )
        )
    )

    net_flux_cumulative = backward_euler_cumulative(
        full["times"],
        full["J_full"] - full["Q_outer_full"],
    )

    bulk_mass_change = (
        full["mass_bulk_full"][-1]
        - full["mass_bulk_full"][0]
    )

    global_balance_error = float(
        abs(
            bulk_mass_change
            - net_flux_cumulative[-1]
        )
    )

    tolerance = 1.0e-10

    if full_cumulative_error > tolerance:
        raise RuntimeError(
            "Full cumulative-release consistency check failed: "
            f"{full_cumulative_error:.6e}"
        )

    if reduced_cumulative_error > tolerance:
        raise RuntimeError(
            "Reduced cumulative-release consistency check failed: "
            f"{reduced_cumulative_error:.6e}"
        )

    if max_balance_residual > tolerance:
        raise RuntimeError(
            "Full stepwise mass-balance check failed: "
            f"{max_balance_residual:.6e}"
        )

    if global_balance_error > tolerance:
        raise RuntimeError(
            "Full global mass-balance check failed: "
            f"{global_balance_error:.6e}"
        )

    field_diagnostic = compute_L2_bulk_error(
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

    elapsed_seconds = time.perf_counter() - start_time

    row = {
        "h": args.h,
        "dt": dt_value,
        "num_steps": len(full["times"]) - 1,
        "n_layer": args.n_layer,
        "n_bulk": args.n_bulk,
        "ny": args.ny,
        "full_num_cells": full["num_cells"],
        "reduced_num_cells": reduced["num_cells"],
        "J_full_0": full["J_full"][0],
        "J_full_1": full["J_full"][1],
        "J_full_T": J_full_T,
        "J_red_0": reduced["J_red"][0],
        "J_red_1": reduced["J_red"][1],
        "J_red_T": J_red_T,
        "E_J_T": E_J_T,
        "rel_E_J_T": relative_error(
            E_J_T,
            J_full_T,
        ),
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
        "E_L2_T": field_diagnostic[
            "absolute_error"
        ],
        "rel_E_L2_T": field_diagnostic[
            "relative_error"
        ],
        "full_field_norm_T": field_diagnostic[
            "full_norm"
        ],
        "Q_outer_full_T": full[
            "Q_outer_full"
        ][-1],
        "max_full_balance_residual": (
            max_balance_residual
        ),
        "global_full_balance_error": (
            global_balance_error
        ),
        "full_cumulative_consistency_error": (
            full_cumulative_error
        ),
        "reduced_cumulative_consistency_error": (
            reduced_cumulative_error
        ),
        "elapsed_seconds": elapsed_seconds,
    }

    case_summary_path = case_root / "case_summary.csv"

    write_summary(
        case_summary_path,
        [row],
    )

    del field_diagnostic
    del full
    del reduced
    gc.collect()

    return row


def main():
    args = parse_arguments()

    output_root = Path(args.outdir)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "time_step_summary.csv"
    status_path = output_root / "status.txt"

    if not args.resume:
        status_path.write_text("")

    rows = []

    for index, dt_value in enumerate(args.dts, start=1):
        case_root = output_root / dt_tag(dt_value)
        case_summary_path = case_root / "case_summary.csv"

        if args.resume and case_summary_path.exists():
            write_status(
                status_path,
                (
                    f"Skipping completed case "
                    f"{index}/{len(args.dts)}: "
                    f"dt={dt_value:.2e}"
                ),
            )

            row = read_case_summary(
                case_summary_path
            )
            rows.append(row)
            write_summary(summary_path, rows)
            continue

        write_status(
            status_path,
            (
                f"Starting case "
                f"{index}/{len(args.dts)}: "
                f"dt={dt_value:.2e}"
            ),
        )

        row = run_case(
            args=args,
            dt_value=dt_value,
            case_root=case_root,
        )

        rows.append(row)
        write_summary(summary_path, rows)

        write_status(
            status_path,
            (
                f"Completed dt={dt_value:.2e}: "
                f"E_L2={float(row['E_L2_T']):.6e}, "
                f"E_B={float(row['E_B_T']):.6e}, "
                f"E_J={float(row['E_J_T']):.6e}, "
                f"E_M={float(row['E_M_T']):.6e}, "
                f"balance="
                f"{float(row['max_full_balance_residual']):.3e}"
            ),
        )

    write_status(
        status_path,
        f"Saved summary: {summary_path}",
    )


if __name__ == "__main__":
    main()
