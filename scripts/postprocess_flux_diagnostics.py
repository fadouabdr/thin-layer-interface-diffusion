import argparse
import csv
import math
from pathlib import Path


THICKNESSES = [0.08, 0.04, 0.02, 0.01]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process saved full/reduced flux curves without "
            "rerunning the PDE solvers."
        )
    )

    parser.add_argument(
        "--d-min",
        type=float,
        default=1.0,
        help="Lower bound of the coating diffusivity.",
    )
    parser.add_argument(
        "--post-factor",
        type=float,
        default=5.0,
        help="Post-equilibration cutoff factor multiplying t_m.",
    )
    parser.add_argument(
        "--outdir",
        default="results/validation_comparison",
        help="Directory for generated diagnostic CSV files.",
    )

    return parser.parse_args()


def thickness_tag(h):
    return f"h{int(round(1000.0 * h)):03d}"


def read_curve(path, flux_column):
    times = []
    fluxes = []

    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required = {"t", flux_column}

        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(
                f"{path} does not contain columns {sorted(required)}."
            )

        for row in reader:
            times.append(float(row["t"]))
            fluxes.append(float(row[flux_column]))

    if len(times) < 2:
        raise RuntimeError(
            f"Insufficient curve data in {path}."
        )

    return times, fluxes


def read_csv(path):
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    return fields, rows


def read_single_row(path):
    _, rows = read_csv(path)

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one row in {path}, found {len(rows)}."
        )

    return rows[0]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def validate_matching_grids(t_full, t_reduced):
    if len(t_full) != len(t_reduced):
        raise RuntimeError(
            "Full and reduced time grids have different lengths."
        )

    for index, (value_full, value_reduced) in enumerate(
        zip(t_full, t_reduced)
    ):
        if not math.isclose(
            value_full,
            value_reduced,
            rel_tol=1.0e-11,
            abs_tol=1.0e-13,
        ):
            raise RuntimeError(
                "Time-grid mismatch at index "
                f"{index}: {value_full} != {value_reduced}."
            )


def trapezoidal_absolute_integral(times, values):
    total = 0.0

    for index in range(len(times) - 1):
        dt = times[index + 1] - times[index]

        if dt <= 0.0:
            raise RuntimeError(
                "The time grid is not strictly increasing."
            )

        total += 0.5 * dt * (
            abs(values[index])
            + abs(values[index + 1])
        )

    return total


def restrict_from_cutoff(times, values, cutoff):
    if cutoff <= times[0]:
        return list(times), list(values)

    if cutoff > times[-1]:
        raise RuntimeError(
            f"Cutoff {cutoff} lies beyond final time {times[-1]}."
        )

    for index in range(1, len(times)):
        if times[index] >= cutoff:
            if math.isclose(
                times[index],
                cutoff,
                rel_tol=0.0,
                abs_tol=1.0e-14,
            ):
                return (
                    list(times[index:]),
                    list(values[index:]),
                )

            left_time = times[index - 1]
            right_time = times[index]
            left_value = values[index - 1]
            right_value = values[index]

            weight = (
                (cutoff - left_time)
                / (right_time - left_time)
            )

            cutoff_value = (
                left_value
                + weight * (right_value - left_value)
            )

            return (
                [cutoff] + list(times[index:]),
                [cutoff_value] + list(values[index:]),
            )

    raise RuntimeError("Unable to construct cutoff curve.")


def relative_error(error, reference):
    return error / max(abs(reference), 1.0e-14)


def empirical_rates(h_values, errors):
    rates = ["NA"]

    for index in range(1, len(h_values)):
        previous_error = errors[index - 1]
        current_error = errors[index]

        if previous_error <= 0.0 or current_error <= 0.0:
            rates.append("NA")
            continue

        rate = math.log(
            current_error / previous_error
        ) / math.log(
            h_values[index] / h_values[index - 1]
        )

        rates.append(rate)

    return rates


def compute_flux_metrics(
    full_path,
    reduced_path,
    h,
    d_min,
    post_factor,
):
    times_full, flux_full = read_curve(
        full_path,
        "J_full",
    )

    times_reduced, flux_reduced = read_curve(
        reduced_path,
        "J_red",
    )

    validate_matching_grids(
        times_full,
        times_reduced,
    )

    difference = [
        full_value - reduced_value
        for full_value, reduced_value in zip(
            flux_full,
            flux_reduced,
        )
    ]

    t_m = h * h / d_min
    post_cutoff = post_factor * t_m

    post_times, post_difference = restrict_from_cutoff(
        times_full,
        difference,
        post_cutoff,
    )

    _, post_full_flux = restrict_from_cutoff(
        times_full,
        flux_full,
        post_cutoff,
    )

    E_J_L1 = trapezoidal_absolute_integral(
        times_full,
        difference,
    )

    full_L1_norm = trapezoidal_absolute_integral(
        times_full,
        flux_full,
    )

    E_J_post_max = max(
        abs(value)
        for value in post_difference
    )

    full_post_max = max(
        abs(value)
        for value in post_full_flux
    )

    return {
        "h": h,
        "dt": times_full[1] - times_full[0],
        "t_m": t_m,
        "post_cutoff": post_cutoff,
        "dt_over_t_m":
            (times_full[1] - times_full[0]) / t_m,
        "E_J_T": abs(difference[-1]),
        "E_J_L1": E_J_L1,
        "rel_E_J_L1": relative_error(
            E_J_L1,
            full_L1_norm,
        ),
        "E_J_post_max": E_J_post_max,
        "rel_E_J_post_max": relative_error(
            E_J_post_max,
            full_post_max,
        ),
    }


def process_time_refinement(
    root,
    output_path,
    d_min,
    post_factor,
):
    rows = []

    for case_directory in root.glob("dt_*"):
        summary_path = (
            case_directory / "case_summary.csv"
        )

        full_path = (
            case_directory
            / "full"
            / "full_curves_h010.csv"
        )

        reduced_path = (
            case_directory
            / "reduced"
            / "reduced_curves_h010.csv"
        )

        if not (
            summary_path.exists()
            and full_path.exists()
            and reduced_path.exists()
        ):
            continue

        summary = read_single_row(summary_path)

        metrics = compute_flux_metrics(
            full_path=full_path,
            reduced_path=reduced_path,
            h=0.01,
            d_min=d_min,
            post_factor=post_factor,
        )

        metrics["dt"] = float(summary["dt"])
        rows.append(metrics)

    if len(rows) != 3:
        raise RuntimeError(
            "Expected three completed time-refinement cases, "
            f"found {len(rows)}."
        )

    rows.sort(
        key=lambda row: float(row["dt"]),
        reverse=True,
    )

    finest_L1 = float(rows[-1]["E_J_L1"])
    finest_post = float(rows[-1]["E_J_post_max"])

    for row in rows:
        row["rel_change_E_J_L1_to_finest"] = (
            abs(float(row["E_J_L1"]) - finest_L1)
            / max(abs(finest_L1), 1.0e-14)
        )

        row[
            "rel_change_E_J_post_max_to_finest"
        ] = (
            abs(
                float(row["E_J_post_max"])
                - finest_post
            )
            / max(abs(finest_post), 1.0e-14)
        )

    fields = [
        "h",
        "dt",
        "t_m",
        "dt_over_t_m",
        "E_J_T",
        "E_J_L1",
        "rel_E_J_L1",
        "rel_change_E_J_L1_to_finest",
        "post_cutoff",
        "E_J_post_max",
        "rel_E_J_post_max",
        "rel_change_E_J_post_max_to_finest",
    ]

    write_csv(output_path, fields, rows)


def process_thickness_study(
    full_root,
    reduced_root,
    output_path,
    d_min,
    post_factor,
):
    rows = []

    for h in THICKNESSES:
        tag = thickness_tag(h)

        full_path = (
            full_root / f"full_curves_{tag}.csv"
        )

        reduced_path = (
            reduced_root / f"reduced_curves_{tag}.csv"
        )

        if not full_path.exists():
            raise FileNotFoundError(full_path)

        if not reduced_path.exists():
            raise FileNotFoundError(reduced_path)

        rows.append(
            compute_flux_metrics(
                full_path=full_path,
                reduced_path=reduced_path,
                h=h,
                d_min=d_min,
                post_factor=post_factor,
            )
        )

    h_values = [
        float(row["h"])
        for row in rows
    ]

    L1_errors = [
        float(row["E_J_L1"])
        for row in rows
    ]

    post_errors = [
        float(row["E_J_post_max"])
        for row in rows
    ]

    terminal_errors = [
        float(row["E_J_T"])
        for row in rows
    ]

    L1_rates = empirical_rates(
        h_values,
        L1_errors,
    )

    post_rates = empirical_rates(
        h_values,
        post_errors,
    )

    terminal_rates = empirical_rates(
        h_values,
        terminal_errors,
    )

    for row, rate_terminal, rate_L1, rate_post in zip(
        rows,
        terminal_rates,
        L1_rates,
        post_rates,
    ):
        row["rate_E_J_T"] = rate_terminal
        row["rate_E_J_L1"] = rate_L1
        row["rate_E_J_post_max"] = rate_post

    fields = [
        "h",
        "dt",
        "t_m",
        "dt_over_t_m",
        "post_cutoff",
        "E_J_T",
        "rate_E_J_T",
        "E_J_L1",
        "rel_E_J_L1",
        "rate_E_J_L1",
        "E_J_post_max",
        "rel_E_J_post_max",
        "rate_E_J_post_max",
    ]

    write_csv(output_path, fields, rows)

    return {
        round(float(row["h"]), 12): row
        for row in rows
    }


def create_paper_summary(
    raw_summary_path,
    flux_metrics,
    output_path,
):
    _, raw_rows = read_csv(raw_summary_path)

    rows = []

    for raw_row in raw_rows:
        h = round(float(raw_row["h"]), 12)
        flux_row = flux_metrics[h]

        rows.append({
            "h": raw_row["h"],
            "E_L2_T": raw_row["E_L2_T"],
            "rate_E_L2_T":
                raw_row["rate_E_L2_T"] or "NA",
            "E_B_T": raw_row["E_B_T"],
            "rate_E_B_T":
                raw_row["rate_E_B_T"] or "NA",
            "E_J_T": raw_row["E_J_T"],
            "rate_E_J_T":
                raw_row["rate_E_J_T"] or "NA",
            "E_J_L1": flux_row["E_J_L1"],
            "rate_E_J_L1":
                flux_row["rate_E_J_L1"],
            "E_J_post_max":
                flux_row["E_J_post_max"],
            "rate_E_J_post_max":
                flux_row["rate_E_J_post_max"],
            "E_M_T": raw_row["E_M_T"],
            "rate_E_M_T":
                raw_row["rate_E_M_T"] or "NA",
            "max_full_balance_residual":
                raw_row["max_full_balance_residual"],
        })

    fields = [
        "h",
        "E_L2_T",
        "rate_E_L2_T",
        "E_B_T",
        "rate_E_B_T",
        "E_J_T",
        "rate_E_J_T",
        "E_J_L1",
        "rate_E_J_L1",
        "E_J_post_max",
        "rate_E_J_post_max",
        "E_M_T",
        "rate_E_M_T",
        "max_full_balance_residual",
    ]

    write_csv(output_path, fields, rows)


def main():
    args = parse_arguments()

    if args.d_min <= 0.0:
        raise ValueError("--d-min must be positive.")

    if args.post_factor <= 0.0:
        raise ValueError(
            "--post-factor must be positive."
        )

    output_root = Path(args.outdir)
    output_root.mkdir(parents=True, exist_ok=True)

    time_output = (
        output_root
        / "flux_time_refinement_h010.csv"
    )

    flux_output = (
        output_root
        / "flux_diagnostics.csv"
    )

    paper_output = (
        output_root
        / "paper_validation_summary.csv"
    )

    process_time_refinement(
        root=Path(
            "results/refinement/time_step_h010"
        ),
        output_path=time_output,
        d_min=args.d_min,
        post_factor=args.post_factor,
    )

    flux_metrics = process_thickness_study(
        full_root=Path("results/validation_full"),
        reduced_root=Path(
            "results/validation_reduced"
        ),
        output_path=flux_output,
        d_min=args.d_min,
        post_factor=args.post_factor,
    )

    create_paper_summary(
        raw_summary_path=(
            output_root / "validation_summary.csv"
        ),
        flux_metrics=flux_metrics,
        output_path=paper_output,
    )

    print(f"Saved: {time_output}")
    print(f"Saved: {flux_output}")
    print(f"Saved: {paper_output}")


if __name__ == "__main__":
    main()
