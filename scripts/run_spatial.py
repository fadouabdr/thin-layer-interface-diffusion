import numpy as np

from src.config import DT_SPATIAL, SPATIAL_RESOLUTIONS
from src.io_utils import save_csv
from src.solver import run_reduced_verification


def observed_rates(errors, mesh_sizes):
    """Compute convergence rates between successive mesh refinements."""
    rates = [None]

    for i in range(1, len(errors)):
        rate = np.log(errors[i - 1] / errors[i]) / np.log(
            mesh_sizes[i - 1] / mesh_sizes[i]
        )
        rates.append(rate)

    return rates


def main():
    results = []

    for resolution in SPATIAL_RESOLUTIONS:
        result = run_reduced_verification(
            resolution=resolution,
            dt_value=DT_SPATIAL,
        )
        results.append(result)

        print(
            "[SPATIAL] "
            f"resolution={resolution}, "
            f"cells={result['num_cells']}, "
            f"L2={result['L2_error']:.6e}"
        )

    errors = [result["L2_error"] for result in results]
    mesh_sizes = [1.0 / resolution for resolution in SPATIAL_RESOLUTIONS]
    rates = observed_rates(errors, mesh_sizes)

    rows = []

    for result, mesh_size, rate in zip(results, mesh_sizes, rates):
        rows.append(
            [
                result["resolution"],
                mesh_size,
                result["num_cells"],
                result["L2_error"],
                "" if rate is None else rate,
            ]
        )

    save_csv(
        "results/spatial/spatial_table.csv",
        [
            "resolution",
            "mesh_size",
            "num_cells",
            "L2_error",
            "rate",
        ],
        rows,
    )

    print("Saved spatial convergence table.")


if __name__ == "__main__":
    main()