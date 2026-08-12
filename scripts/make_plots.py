import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


# -------------------------------------------------------------------------
# Final AMM-style plotting parameters
# -------------------------------------------------------------------------
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


OUTDIR = Path("figures/AMM_final")


def read_csv_table(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def polish_axes(ax):
    ax.grid(False)

    ax.tick_params(axis="both", which="major", width=1.2, length=5)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3)

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    for spine in ax.spines.values():
        spine.set_linewidth(1.25)


def make_spatial_plot():
    rows = read_csv_table("results/spatial/spatial_table.csv")

    resolution = np.array([float(r["resolution"]) for r in rows])
    eta = 1.0 / resolution
    err = np.array([float(r["L2_error"]) for r in rows])

    order = np.argsort(eta)
    eta = eta[order]
    err = err[order]

    fig, ax = plt.subplots(figsize=(6.0, 4.3))

    ax.loglog(
        eta,
        err,
        "o-",
        label=r"$L^2$ error",
    )

    ref = err[0] * (eta / eta[0]) ** 2
    ax.loglog(
        eta,
        ref,
        "--",
        label=r"Reference slope $O(\eta^2)$",
    )

    spatial_offsets = [1.18, 1.23, 1.28]
    for i in range(1, len(eta)):
        rate = np.log(err[i] / err[i - 1]) / np.log(eta[i] / eta[i - 1])
        eta_mid = np.sqrt(eta[i - 1] * eta[i])
        err_mid = np.sqrt(err[i - 1] * err[i])

        ax.text(
            eta_mid,
            err_mid * spatial_offsets[i - 1],
            f"{rate:.2f}",
            fontsize=10,
            ha="center",
            va="bottom",
        )

    ax.set_xlabel(r"Mesh size $\eta$")
    ax.set_ylabel(r"$L^2(\Omega)$ error")

    ax.set_xticks(eta)
    ax.set_xticklabels([r"$1/64$", r"$1/32$", r"$1/16$", r"$1/8$"])

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )

    polish_axes(ax)
    fig.tight_layout()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "spatial_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def make_temporal_plot():
    rows = read_csv_table("results/temporal/temporal_table.csv")

    dt = np.array([float(r["dt"]) for r in rows])
    err = np.array([float(r["L2_error"]) for r in rows])

    order = np.argsort(dt)
    dt = dt[order]
    err = err[order]

    fig, ax = plt.subplots(figsize=(6.0, 4.3))

    ax.loglog(
        dt,
        err,
        "o-",
        label=r"$L^2$ error",
    )

    ref = err[0] * (dt / dt[0])
    ax.loglog(
        dt,
        ref,
        "--",
        label=r"Reference slope $O(\Delta t)$",
    )

    temporal_offsets = [1.18, 1.24, 1.32]
    for i in range(1, len(dt)):
        rate = np.log(err[i] / err[i - 1]) / np.log(dt[i] / dt[i - 1])
        dt_mid = np.sqrt(dt[i - 1] * dt[i])
        err_mid = np.sqrt(err[i - 1] * err[i])

        ax.text(
            dt_mid,
            err_mid * temporal_offsets[i - 1],
            f"{rate:.2f}",
            fontsize=10,
            ha="center",
            va="bottom",
        )

    ax.set_xlabel(r"Time step $\Delta t$")
    ax.set_ylabel(r"$L^2(\Omega)$ error")

    ax.set_xticks(dt)
    ax.set_xticklabels([
        "0.00125",
        "0.0025",
        "0.005",
        "0.01",
    ])

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )

    polish_axes(ax)
    fig.tight_layout()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "temporal_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    make_spatial_plot()
    make_temporal_plot()
    print(f"Saved final convergence plots in: {OUTDIR}")


if __name__ == "__main__":
    main()