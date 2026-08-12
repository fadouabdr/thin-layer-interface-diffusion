import csv
from pathlib import Path

from dolfin import LogLevel, set_log_level

from src.full_thin_layer_solver import run_full_validation_case
from src.reduced_validation_solver import run_reduced_validation_case
from src.validation_config import THICKNESSES
from src.validation_diagnostics import compute_L2_bulk_error


set_log_level(LogLevel.ERROR)


def main():
    """
    Standalone utility for recomputing final-time bulk field errors.

    The final production pipeline will compute the same diagnostic while the
    full and reduced solutions are already in memory. This script remains as
    an independent reproducibility utility.
    """

    outdir = Path("results/validation_comparison")
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "bulk_field_errors.csv"
    rows = []

    for h in THICKNESSES:
        print("=" * 70)
        print(f"Computing bulk field error for h={h}")

        full = run_full_validation_case(h=h)
        reduced = run_reduced_validation_case(h=h)

        diagnostic = compute_L2_bulk_error(
            c_full=full["solution"],
            c_reduced=reduced["solution"],
        )

        absolute_error = diagnostic["absolute_error"]
        relative_error = diagnostic["relative_error"]

        print(f"E_L2(T;h)     = {absolute_error:.8e}")
        print(f"rel_E_L2(T;h) = {relative_error:.8e}")

        rows.append({
            "h": h,
            "E_L2_T": absolute_error,
            "rel_E_L2_T": relative_error,
        })

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "h",
                "E_L2_T",
                "rel_E_L2_T",
            ],
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
