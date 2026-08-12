import argparse

import numpy as np

from src.full_thin_layer_solver import (
    run_full_validation_case,
    save_full_validation_case,
)
from src.validation_config import (
    BULK_RESOLUTION,
    DT_VALIDATION,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run one full thin-layer validation case."
    )

    parser.add_argument(
        "h",
        type=float,
        help="Coating thickness.",
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
        help="Number of bulk elements in the x direction.",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=BULK_RESOLUTION,
        help="Number of elements in the y direction.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DT_VALIDATION,
        help="Time-step size.",
    )
    parser.add_argument(
        "--outdir",
        default="results/validation_full",
        help="Directory in which the results are saved.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    out = run_full_validation_case(
        h=args.h,
        n_layer=args.n_layer,
        n_bulk=args.n_bulk,
        ny=args.ny,
        dt_value=args.dt,
    )

    csv_path = save_full_validation_case(
        out,
        outdir=args.outdir,
    )

    max_balance_residual = np.max(
        np.abs(out["bulk_balance_residual"])
    )

    print("Full thin-layer validation case completed")
    print("h                      =", out["h"])
    print("n_layer                =", out["n_layer"])
    print("n_bulk                 =", out["n_bulk"])
    print("ny                     =", out["ny"])
    print("dt                     =", out["dt"])
    print("num_cells              =", out["num_cells"])
    print("interface length       =", out["interface_length"])
    print("final J_full           =", out["J_full"][-1])
    print("final Q_outer          =", out["Q_outer_full"][-1])
    print("final M_full           =", out["M_full"][-1])
    print("final bulk mass        =", out["mass_bulk_full"][-1])
    print("final coating mass     =", out["mass_coating_full"][-1])
    print("max balance residual   =", max_balance_residual)
    print("saved                  =", csv_path)


if __name__ == "__main__":
    main()
