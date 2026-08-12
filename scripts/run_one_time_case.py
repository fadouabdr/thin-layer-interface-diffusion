import sys
import csv
from pathlib import Path

from src.solver import run_reduced_verification


def append_row(csv_path, row):
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_file.exists()

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["resolution", "num_cells", "dt", "L2_error"])
        writer.writerow(row)


if len(sys.argv) != 3:
    raise ValueError("Usage: python scripts/run_one_time_case.py <resolution> <dt>")

resolution = int(sys.argv[1])
dt_value = float(sys.argv[2])

out = run_reduced_verification(resolution=resolution, dt_value=dt_value)

print("resolution =", out["resolution"])
print("num_cells  =", out["num_cells"])
print("dt         =", out["dt"])
print("L2_error   =", out["L2_error"])

append_row(
    "results/time/time_table_manual.csv",
    [out["resolution"], out["num_cells"], out["dt"], out["L2_error"]],
)

print("Saved to results/time/time_table_manual.csv")