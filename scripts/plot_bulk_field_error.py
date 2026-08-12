from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = Path("results/validation_comparison/bulk_field_errors.csv")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing file: {csv_path}. "
            "Run scripts/compute_bulk_field_errors_from_csv.py first."
        )

    df = pd.read_csv(csv_path)

    h = df["h"].values
    E_L2 = df["E_L2_T"].values

    fig_dir = Path("figures/validation")
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = fig_dir / "bulk_field_error_vs_h.pdf"
    out_png = fig_dir / "bulk_field_error_vs_h.png"

    plt.figure(figsize=(5.2, 3.9))

    # Main error curve
    plt.loglog(
        h,
        E_L2,
        marker="o",
        linewidth=2.0,
        markersize=7,
        label=r"$E_{L^2}(T;h)$",
    )

    # Reference slope O(h^2)
    ref = E_L2[0] * (h / h[0]) ** 2

    plt.loglog(
        h,
        ref,
        linestyle="--",
        linewidth=2.0,
        label=r"reference slope $O(h^2)$",
    )

    plt.xlabel(r"coating thickness $h$", fontsize=12)
    plt.ylabel(r"$E_{L^2}(T;h)$", fontsize=12)

    plt.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.5,
    )

    plt.legend(fontsize=9)
    plt.minorticks_off()

    plt.tight_layout()

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    print("Saved:", out_pdf)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()

