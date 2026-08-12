import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


OUTDIR = Path("figures/AMM_final")


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.8,
    "lines.markersize": 7.5,
    "axes.linewidth": 1.25,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def read_columns(path, column_names):
    values = {name: [] for name in column_names}

    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        missing = [
            name
            for name in column_names
            if name not in (reader.fieldnames or [])
        ]

        if missing:
            raise RuntimeError(
                f"{path} is missing columns: {missing}"
            )

        for row in reader:
            for name in column_names:
                values[name].append(float(row[name]))

    return {
        name: np.asarray(column_values, dtype=float)
        for name, column_values in values.items()
    }


def polish_axes(ax):
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        width=1.2,
        length=5,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=0.8,
        length=3,
    )

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    for spine in ax.spines.values():
        spine.set_linewidth(1.25)


def savefig(fig, filename):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTDIR / filename,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_bulk_field_error_plot():
    data = read_columns(
        "results/validation_comparison/"
        "paper_validation_summary.csv",
        ["h", "E_L2_T"],
    )

    h = data["h"]
    error = data["E_L2_T"]

    order = np.argsort(h)
    h = h[order]
    error = error[order]

    fig, ax = plt.subplots(figsize=(6.0, 4.3))

    ax.loglog(
        h,
        error,
        "o-",
        label=r"$E_{L^2}(T;h)$",
    )

    reference = error[0] * (h / h[0]) ** 2

    ax.loglog(
        h,
        reference,
        "--",
        label=r"Reference slope $O(h^2)$",
    )

    ax.set_xlabel(r"Coating thickness $h$")
    ax.set_ylabel(r"$E_{L^2}(T;h)$")

    ax.set_xticks(h)
    ax.set_xticklabels([
        "0.01",
        "0.02",
        "0.04",
        "0.08",
    ])

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )

    polish_axes(ax)
    fig.tight_layout()

    savefig(
        fig,
        "bulk_field_error_vs_h.pdf",
    )


def make_bulk_mass_error_plot():
    data = read_columns(
        "results/validation_comparison/"
        "paper_validation_summary.csv",
        ["h", "E_B_T"],
    )

    h = data["h"]
    error = data["E_B_T"]

    order = np.argsort(h)
    h = h[order]
    error = error[order]

    fig, ax = plt.subplots(figsize=(6.0, 4.3))

    ax.loglog(
        h,
        error,
        "o-",
        label=r"$E_B(T;h)$",
    )

    reference = error[0] * (h / h[0]) ** 2

    ax.loglog(
        h,
        reference,
        "--",
        label=r"Reference slope $O(h^2)$",
    )

    ax.set_xlabel(r"Coating thickness $h$")
    ax.set_ylabel(r"$E_B(T;h)$")

    ax.set_xticks(h)
    ax.set_xticklabels([
        "0.01",
        "0.02",
        "0.04",
        "0.08",
    ])

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )

    polish_axes(ax)
    fig.tight_layout()

    savefig(
        fig,
        "bulk_error_vs_h.pdf",
    )


def make_release_comparison_plot():
    cases = [
        (0.08, "h080"),
        (0.01, "h010"),
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6.0, 6.0),
        sharex=True,
    )

    for ax, (h, tag) in zip(axes, cases):
        full = read_columns(
            f"results/validation_full/"
            f"full_curves_{tag}.csv",
            ["t", "M_full"],
        )

        reduced = read_columns(
            f"results/validation_reduced/"
            f"reduced_curves_{tag}.csv",
            ["t", "M_red"],
        )

        ax.plot(
            full["t"],
            full["M_full"],
            "-",
            label="Full model",
        )

        ax.plot(
            reduced["t"],
            reduced["M_red"],
            "--",
            label="Reduced model",
        )

        ax.set_ylabel("Cumulative release")
        ax.set_title(
            rf"$h={h}$",
            fontsize=12,
            pad=6,
        )

        polish_axes(ax)

    axes[-1].set_xlabel(r"Time $t$")

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    savefig(
        fig,
        "release_comparison.pdf",
    )


def main():
    make_bulk_field_error_plot()
    make_bulk_mass_error_plot()
    make_release_comparison_plot()

    print(
        "Saved final AMM validation plots in: "
        f"{OUTDIR}"
    )


if __name__ == "__main__":
    main()
