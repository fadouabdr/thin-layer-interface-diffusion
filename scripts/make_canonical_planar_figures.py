import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator


# Embed TrueType fonts in vector outputs.
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# Okabe–Ito colors: high contrast on white and distinguishable for
# common forms of red–green color-vision deficiency.
BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "0.35"


def style_axis(axis):
    """Apply the restrained, grid-free manuscript style."""
    axis.grid(False)
    axis.tick_params(
        which="both",
        direction="out",
        width=0.8,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot canonical bulk convergence and representative "
            "flux/release histories."
        )
    )
    parser.add_argument(
        "canonical_root",
        help="Directory containing canonical_summary.csv and cases/.",
    )
    parser.add_argument(
        "--history-thicknesses",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Thicknesses used in the history figure. The default selects "
            "the largest and smallest available values."
        ),
    )
    return parser.parse_args()


def read_rows(path):
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def h_tag(h):
    return f"h{int(round(1000.0 * h)):03d}"


def read_curve(path, fields):
    if not path.exists():
        raise FileNotFoundError(path)

    rows = read_rows(path)

    if not rows:
        raise RuntimeError(f"No data found in {path}.")

    output = {}

    for field in fields:
        if field not in rows[0]:
            raise KeyError(f"Column {field!r} is missing from {path}.")

        output[field] = np.asarray(
            [float(row[field]) for row in rows],
            dtype=float,
        )

    return output


def plot_bulk_convergence(rows, output_path):
    ordered = sorted(rows, key=lambda row: float(row["h"]))

    h_values = np.asarray(
        [float(row["h"]) for row in ordered],
        dtype=float,
    )
    field_errors = np.asarray(
        [float(row["rel_E_L2_T"]) for row in ordered],
        dtype=float,
    )
    mass_errors = np.asarray(
        [float(row["rel_E_B_T"]) for row in ordered],
        dtype=float,
    )

    if (
        np.any(h_values <= 0.0)
        or np.any(field_errors <= 0.0)
        or np.any(mass_errors <= 0.0)
    ):
        raise RuntimeError(
            "Thicknesses and relative errors must be positive "
            "for a log-log plot."
        )

    # Short O(h^2) segment. Its vertical position carries no
    # mathematical meaning; the factor 0.75 keeps it close to,
    # but visually distinct from, the measured curves.
    reference_h = np.geomspace(
        h_values[0],
        h_values[1],
        40,
    )
    reference_anchor = 0.75 * min(
        field_errors[0],
        mass_errors[0],
    )
    reference_scale = reference_anchor / (reference_h[0] ** 2)
    second_order_reference = reference_scale * reference_h**2

    fig, axis = plt.subplots(figsize=(6.2, 4.4))

    mass_line, = axis.loglog(
        h_values,
        mass_errors,
        "--",
        color=VERMILION,
        linewidth=1.35,
        marker="s",
        markerfacecolor="none",
        markeredgecolor=VERMILION,
        markeredgewidth=1.0,
        markersize=6.5,
        zorder=4,
        label="relative exterior-mass discrepancy",
    )

    field_line, = axis.loglog(
        h_values,
        field_errors,
        "-",
        color=BLUE,
        linewidth=1.55,
        marker="o",
        markerfacecolor=BLUE,
        markeredgecolor="white",
        markeredgewidth=0.45,
        markersize=5.0,
        zorder=3,
        label=r"relative exterior-field $L^2$ discrepancy",
    )

    reference_line, = axis.loglog(
        reference_h,
        second_order_reference,
        ":",
        color=GRAY,
        linewidth=1.3,
        zorder=2,
        label=r"$O(h^2)$ guide",
    )

    axis.set_xlabel(r"coating thickness $h$")
    axis.set_ylabel(r"relative discrepancy at $T=1$")
    axis.set_xticks(h_values)
    axis.set_xticklabels([f"{h:g}" for h in h_values])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_minor_formatter(NullFormatter())

    style_axis(axis)

    axis.legend(
        handles=[field_line, mass_line, reference_line],
        frameon=False,
        loc="upper left",
        fontsize=9.5,
        handlelength=2.4,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_histories(root, thicknesses, output_path):
    palette = [BLUE, VERMILION, GREEN, PURPLE]
    colors = [
        palette[i % len(palette)]
        for i in range(len(thicknesses))
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 3.9),
        sharex=True,
    )

    for h, color in zip(thicknesses, colors):
        tag = h_tag(h)
        case_root = root / "cases" / tag

        full_path = (
            case_root
            / "full"
            / f"full_curves_{tag}.csv"
        )
        reduced_path = (
            case_root
            / "reduced"
            / f"reduced_curves_{tag}.csv"
        )

        full = read_curve(
            full_path,
            ["t", "J_full", "M_full"],
        )
        reduced = read_curve(
            reduced_path,
            ["t", "J_red", "M_red"],
        )

        if (
            full["t"].shape != reduced["t"].shape
            or not np.allclose(
                full["t"],
                reduced["t"],
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise RuntimeError(
                f"Full/reduced time-grid mismatch for h={h}."
            )

        marker_spacing = max(
            1,
            len(reduced["t"]) // 10,
        )

        axes[0].plot(
            full["t"],
            full["J_full"],
            color=color,
            linewidth=1.6,
            zorder=2,
        )
        axes[0].plot(
            reduced["t"],
            reduced["J_red"],
            "--",
            color=color,
            linewidth=1.25,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=3.2,
            markevery=marker_spacing,
            zorder=3,
        )

        axes[1].plot(
            full["t"],
            full["M_full"],
            color=color,
            linewidth=1.6,
            zorder=2,
        )
        axes[1].plot(
            reduced["t"],
            reduced["M_red"],
            "--",
            color=color,
            linewidth=1.25,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=3.2,
            markevery=marker_spacing,
            zorder=3,
        )

    axes[0].set_xlabel(r"time $t$")
    axes[0].set_ylabel("signed interface release rate")
    axes[1].set_xlabel(r"time $t$")
    axes[1].set_ylabel("cumulative interface release")

    # Panel labels are placed above the axes, outside the data regions.
    axes[0].set_title(
        "(a)",
        loc="left",
        pad=8,
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_title(
        "(b)",
        loc="left",
        pad=8,
        fontsize=12,
        fontweight="bold",
    )

    for axis in axes:
        style_axis(axis)
        axis.tick_params(labelsize=10.5)
        axis.xaxis.label.set_size(11.5)
        axis.yaxis.label.set_size(11.5)

    thickness_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.8,
            label=fr"$h={h:g}$",
        )
        for h, color in zip(thicknesses, colors)
    ]

    model_handles = [
        Line2D(
            [0],
            [0],
            color="0.15",
            linewidth=1.6,
            label="resolved",
        ),
        Line2D(
            [0],
            [0],
            color="0.15",
            linestyle="--",
            linewidth=1.25,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="0.15",
            markeredgewidth=0.8,
            markersize=3.2,
            label="reduced",
        ),
    ]

    all_handles = thickness_handles + model_handles

    fig.legend(
        handles=all_handles,
        frameon=False,
        fontsize=11,
        ncol=len(all_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=2.2,
        columnspacing=1.5,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_arguments()
    root = Path(args.canonical_root)
    summary_path = root / "canonical_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    rows = read_rows(summary_path)

    if len(rows) < 2:
        raise RuntimeError(
            "At least two thicknesses are needed for the figures."
        )

    available = sorted(
        [float(row["h"]) for row in rows],
        reverse=True,
    )

    selected = (
        list(args.history_thicknesses)
        if args.history_thicknesses
        else [available[0], available[-1]]
    )

    missing = [
        h
        for h in selected
        if not any(
            np.isclose(h, value)
            for value in available
        )
    ]

    if missing:
        raise RuntimeError(
            "Requested history thicknesses are unavailable: "
            f"{missing}."
        )

    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    plot_bulk_convergence(
        rows,
        figure_root / "planar_bulk_convergence.pdf",
    )
    plot_histories(
        root,
        selected,
        figure_root / "planar_release_history.pdf",
    )

    print(f"Saved figures in {figure_root}.")


if __name__ == "__main__":
    main()