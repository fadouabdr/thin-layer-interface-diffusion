"""Canonical full-versus-reduced planar comparison pipeline."""

import csv
import time
from pathlib import Path

import numpy as np

from .full_thin_layer_solver import (
    run_full_validation_case,
    save_full_validation_case,
)
from .reduced_validation_solver import (
    run_reduced_validation_case,
    save_reduced_validation_case,
)
from .validation_config import PREPARED_INITIALIZATION
from .validation_diagnostics import compute_L2_bulk_error
from .validation_metrics import (
    backward_euler_cumulative,
    normalized_balance_residual,
    relative_time_l2_error,
    validate_matching_time_grids,
)


SUMMARY_FIELDS = [
    "scenario",
    "initialization",
    "h",
    "T",
    "dt",
    "num_steps",
    "n_layer",
    "n_bulk",
    "ny",
    "full_num_cells",
    "full_num_dofs",
    "reduced_num_cells",
    "reduced_num_dofs",
    "E_L2_T",
    "rel_E_L2_T",
    "rate_E_L2_T",
    "E_B_T",
    "rel_E_B_T",
    "rate_E_B_T",
    "E_J_L2_rel",
    "rate_E_J_L2_rel",
    "E_J_T",
    "rel_E_J_T",
    "rate_E_J_T",
    "E_M_T",
    "rel_E_M_T",
    "rate_E_M_T",
    "J_full_0",
    "J_red_0",
    "initial_flux_mismatch",
    "J_full_T",
    "J_red_T",
    "M_full_T",
    "M_red_T",
    "B_full_T",
    "B_red_T",
    "Q_outer_full_T",
    "Q_outer_red_T",
    "max_full_balance_residual",
    "max_full_balance_normalized",
    "global_full_balance_error",
    "max_reduced_balance_residual",
    "max_reduced_balance_normalized",
    "global_reduced_balance_error",
    "max_reduced_robin_flux_mismatch",
    "full_cumulative_consistency_error",
    "reduced_cumulative_consistency_error",
    "elapsed_seconds",
]

RATE_FIELDS = {
    "E_L2_T": "rate_E_L2_T",
    "E_B_T": "rate_E_B_T",
    "E_J_L2_rel": "rate_E_J_L2_rel",
    "E_J_T": "rate_E_J_T",
    "E_M_T": "rate_E_M_T",
}


def _relative_error(error, reference):
    return float(error) / max(abs(float(reference)), 1.0e-14)


def _ensure_finite(name, values):
    values = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"{name} contains NaN or infinite values."
        )


def _global_balance_error(times, mass, inflow, outflow):
    cumulative_net_flux = backward_euler_cumulative(
        times,
        np.asarray(inflow) - np.asarray(outflow),
    )
    mass_change = float(mass[-1] - mass[0])
    return abs(mass_change - cumulative_net_flux[-1])


def _max_normalized_balance(
    residuals,
    mass,
    inflow,
    outflow,
    dt_value,
):
    normalized = [0.0]

    for index in range(1, len(residuals)):
        mass_rate = (mass[index] - mass[index - 1]) / dt_value
        normalized.append(
            normalized_balance_residual(
                residual=residuals[index],
                mass_rate=mass_rate,
                interface_inflow=inflow[index],
                outer_outflow=outflow[index],
            )
        )

    return float(np.max(normalized))


def run_planar_comparison(
    h,
    t_final,
    dt_value,
    scenario,
    initialization,
    n_layer,
    n_bulk,
    ny,
    invariant_tolerance=1.0e-9,
):
    """Run one paired benchmark and enforce its numerical invariants."""
    start_time = time.perf_counter()

    full = run_full_validation_case(
        h=h,
        n_layer=n_layer,
        n_bulk=n_bulk,
        ny=ny,
        dt_value=dt_value,
        t_final=t_final,
        scenario=scenario,
        initialization=initialization,
    )
    reduced = run_reduced_validation_case(
        h=h,
        resolution=n_bulk,
        dt_value=dt_value,
        t_final=t_final,
        scenario=scenario,
        initialization=initialization,
    )

    times = validate_matching_time_grids(
        full["times"],
        reduced["times"],
    )

    arrays = {
        "J_full": full["J_full"],
        "Q_outer_full": full["Q_outer_full"],
        "balance_full": full["bulk_balance_residual"],
        "M_full": full["M_full"],
        "B_full": full["mass_bulk_full"],
        "J_red": reduced["J_red"],
        "Q_outer_red": reduced["Q_outer_red"],
        "balance_red": reduced["bulk_balance_residual"],
        "robin_flux_mismatch": reduced["robin_flux_mismatch"],
        "M_red": reduced["M_red"],
        "B_red": reduced["mass_red"],
    }

    for name, values in arrays.items():
        _ensure_finite(name, values)

    full_cumulative_check = backward_euler_cumulative(
        times,
        full["J_full"],
    )
    reduced_cumulative_check = backward_euler_cumulative(
        times,
        reduced["J_red"],
    )

    full_cumulative_error = float(
        np.max(np.abs(full["M_full"] - full_cumulative_check))
    )
    reduced_cumulative_error = float(
        np.max(np.abs(reduced["M_red"] - reduced_cumulative_check))
    )

    max_full_balance_residual = float(
        np.max(np.abs(full["bulk_balance_residual"]))
    )
    max_reduced_balance_residual = float(
        np.max(np.abs(reduced["bulk_balance_residual"]))
    )
    max_full_balance_normalized = _max_normalized_balance(
        residuals=full["bulk_balance_residual"],
        mass=full["mass_bulk_full"],
        inflow=full["J_full"],
        outflow=full["Q_outer_full"],
        dt_value=dt_value,
    )
    max_reduced_balance_normalized = _max_normalized_balance(
        residuals=reduced["bulk_balance_residual"],
        mass=reduced["mass_red"],
        inflow=reduced["J_red"],
        outflow=reduced["Q_outer_red"],
        dt_value=dt_value,
    )

    global_full_balance_error = _global_balance_error(
        times=times,
        mass=full["mass_bulk_full"],
        inflow=full["J_full"],
        outflow=full["Q_outer_full"],
    )
    global_reduced_balance_error = _global_balance_error(
        times=times,
        mass=reduced["mass_red"],
        inflow=reduced["J_red"],
        outflow=reduced["Q_outer_red"],
    )
    max_reduced_robin_flux_mismatch = float(
        np.max(np.abs(reduced["robin_flux_mismatch"]))
    )

    initial_flux_mismatch = abs(
        float(full["J_full"][0])
        - float(reduced["J_red"][0])
    )

    invariant_checks = {
        "full cumulative release":
            full_cumulative_error,
        "reduced cumulative release":
            reduced_cumulative_error,
        "full normalized exterior balance":
            max_full_balance_normalized,
        "reduced normalized exterior balance":
            max_reduced_balance_normalized,
        "full global exterior balance":
            global_full_balance_error,
        "reduced global exterior balance":
            global_reduced_balance_error,
        "reduced Robin/residual flux agreement":
            max_reduced_robin_flux_mismatch,
    }

    if initialization == PREPARED_INITIALIZATION:
        invariant_checks[
            "prepared initial full/reduced flux agreement"
        ] = initial_flux_mismatch

    failed = {
        name: value
        for name, value in invariant_checks.items()
        if value > invariant_tolerance
    }

    if failed:
        details = ", ".join(
            f"{name}={value:.6e}"
            for name, value in failed.items()
        )
        raise RuntimeError(
            "Canonical numerical invariant failure: "
            + details
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

    summary = {
        "scenario": scenario,
        "initialization": initialization,
        "h": h,
        "T": t_final,
        "dt": dt_value,
        "num_steps": len(times) - 1,
        "n_layer": n_layer,
        "n_bulk": n_bulk,
        "ny": ny,
        "full_num_cells": full["num_cells"],
        "full_num_dofs": full["num_dofs"],
        "reduced_num_cells": reduced["num_cells"],
        "reduced_num_dofs": reduced["num_dofs"],
        "E_L2_T": field["absolute_error"],
        "rel_E_L2_T": field["relative_error"],
        "rate_E_L2_T": "",
        "E_B_T": E_B_T,
        "rel_E_B_T": _relative_error(E_B_T, B_full_T),
        "rate_E_B_T": "",
        "E_J_L2_rel": relative_time_l2_error(
            times=times,
            first=reduced["J_red"],
            reference=full["J_full"],
        ),
        "rate_E_J_L2_rel": "",
        "E_J_T": E_J_T,
        "rel_E_J_T": _relative_error(E_J_T, J_full_T),
        "rate_E_J_T": "",
        "E_M_T": E_M_T,
        "rel_E_M_T": _relative_error(E_M_T, M_full_T),
        "rate_E_M_T": "",
        "J_full_0": float(full["J_full"][0]),
        "J_red_0": float(reduced["J_red"][0]),
        "initial_flux_mismatch": initial_flux_mismatch,
        "J_full_T": J_full_T,
        "J_red_T": J_red_T,
        "M_full_T": M_full_T,
        "M_red_T": M_red_T,
        "B_full_T": B_full_T,
        "B_red_T": B_red_T,
        "Q_outer_full_T": float(full["Q_outer_full"][-1]),
        "Q_outer_red_T": float(reduced["Q_outer_red"][-1]),
        "max_full_balance_residual":
            max_full_balance_residual,
        "max_full_balance_normalized":
            max_full_balance_normalized,
        "global_full_balance_error":
            global_full_balance_error,
        "max_reduced_balance_residual":
            max_reduced_balance_residual,
        "max_reduced_balance_normalized":
            max_reduced_balance_normalized,
        "global_reduced_balance_error":
            global_reduced_balance_error,
        "max_reduced_robin_flux_mismatch":
            max_reduced_robin_flux_mismatch,
        "full_cumulative_consistency_error":
            full_cumulative_error,
        "reduced_cumulative_consistency_error":
            reduced_cumulative_error,
        "elapsed_seconds": time.perf_counter() - start_time,
    }

    return {
        "summary": summary,
        "full": full,
        "reduced": reduced,
    }


def write_summary(path, rows, fields=SUMMARY_FIELDS):
    """Write a deterministic CSV and reject incomplete rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows):
        missing = [field for field in fields if field not in row]

        if missing:
            raise ValueError(
                f"Row {index} is missing CSV fields {missing}."
            )

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_planar_comparison(result, case_root):
    """Save one paired run without mixing it with legacy outputs."""
    case_root = Path(case_root)
    full_root = case_root / "full"
    reduced_root = case_root / "reduced"

    save_full_validation_case(
        result["full"],
        outdir=full_root,
    )
    save_reduced_validation_case(
        result["reduced"],
        outdir=reduced_root,
    )
    write_summary(
        case_root / "case_summary.csv",
        [result["summary"]],
    )
