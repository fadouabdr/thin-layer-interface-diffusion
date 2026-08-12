import numpy as np
from src.config import TEMPORAL_DTS, TEMPORAL_RESOLUTION
from src.solver import run_reduced_verification
from src.io_utils import save_csv


def observed_rates(errors, dts):
    rates = [None]
    for i in range(1, len(errors)):
        rate = np.log(errors[i - 1] / errors[i]) / np.log(dts[i - 1] / dts[i])
        rates.append(rate)
    return rates


def main():
    results = []
    for dt in TEMPORAL_DTS:
        out = run_reduced_verification(resolution=TEMPORAL_RESOLUTION, dt_value=dt)
        results.append(out)
        print(f"[TEMPORAL] dt={dt:.6f}, cells={out['num_cells']}, L2={out['L2_error']:.6e}")

    errors = [r["L2_error"] for r in results]
    rates = observed_rates(errors, TEMPORAL_DTS)

    rows = []
    for r, rate in zip(results, rates):
        rows.append([
            r["dt"],
            r["num_cells"],
            r["L2_error"],
            "" if rate is None else rate,
        ])

    save_csv(
        "results/temporal/temporal_table.csv",
        ["dt", "num_cells", "L2_error", "rate"],
        rows,
    )
    print("Saved temporal table.")


if __name__ == "__main__":
    main()