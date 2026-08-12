import csv
from pathlib import Path

import numpy as np
import pandas as pd


THICKNESSES = [0.08, 0.04, 0.02, 0.01]


def htag(h):
    return f"h{int(round(h * 1000)):03d}"


def compute_discrete_l2_error(h):
    tag = htag(h)

    full_path = Path(f"results/validation_full/full_bulk_field_{tag}.csv")
    red_path = Path(f"results/validation_reduced/reduced_field_{tag}.csv")

    full = pd.read_csv(full_path)
    red = pd.read_csv(red_path)

    # Round coordinates to avoid floating-point mismatch.
    full["xr"] = full["x"].round(12)
    full["yr"] = full["y"].round(12)

    red["xr"] = red["x"].round(12)
    red["yr"] = red["y"].round(12)

    merged = pd.merge(
        red,
        full,
        on=["xr", "yr"],
        how="inner",
        suffixes=("_red", "_full"),
    )

    if merged.empty:
        raise RuntimeError(f"No matching points found for h={h}")

    diff = merged["c_full"].values - merged["c_red"].values
    cfull = merged["c_full"].values

    # Uniform bulk domain approximation.
    area = 1.0

    E_L2 = np.sqrt(area * np.mean(diff**2))
    norm_full = np.sqrt(area * np.mean(cfull**2))

    rel_E_L2 = E_L2 / max(norm_full, 1.0e-14)

    return E_L2, rel_E_L2, len(merged)


def main():
    outdir = Path("results/validation_comparison")
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / "bulk_field_errors.csv"

    rows = []

    for h in THICKNESSES:
        E_L2, rel_E_L2, npts = compute_discrete_l2_error(h)

        print("=" * 60)
        print(f"h = {h}")
        print(f"matched points   = {npts}")
        print(f"E_L2(T;h)        = {E_L2:.8e}")
        print(f"rel_E_L2(T;h)    = {rel_E_L2:.8e}")

        rows.append({
            "h": h,
            "E_L2_T": E_L2,
            "rel_E_L2_T": rel_E_L2,
            "n_points": npts,
        })

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["h", "E_L2_T", "rel_E_L2_T", "n_points"],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Saved:", outpath)


if __name__ == "__main__":
    main()
