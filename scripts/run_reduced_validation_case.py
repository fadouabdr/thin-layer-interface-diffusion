import argparse

from src.reduced_validation_solver import (
    run_reduced_validation_case,
    save_reduced_validation_case,
)
from src.validation_config import (
    BULK_RESOLUTION,
    DT_VALIDATION,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run one reduced-interface validation case."
    )

    parser.add_argument(
        "h",
        type=float,
        help="Coating thickness used in the effective coefficient.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=BULK_RESOLUTION,
        help="Bulk mesh resolution.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DT_VALIDATION,
        help="Time-step size.",
    )
    parser.add_argument(
        "--outdir",
        default="results/validation_reduced",
        help="Directory in which the results are saved.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    out = run_reduced_validation_case(
        h=args.h,
        resolution=args.resolution,
        dt_value=args.dt,
    )

    csv_path = save_reduced_validation_case(
        out,
        outdir=args.outdir,
    )

    print("Reduced validation case completed")
    print("h                  =", out["h"])
    print("resolution         =", out["resolution"])
    print("dt                 =", out["dt"])
    print("num_cells          =", out["num_cells"])
    print("final J_red        =", out["J_red"][-1])
    print("final M_red        =", out["M_red"][-1])
    print("final bulk mass    =", out["mass_red"][-1])
    print("final energy       =", out["energy_red"][-1])
    print("saved              =", csv_path)


if __name__ == "__main__":
    main()
