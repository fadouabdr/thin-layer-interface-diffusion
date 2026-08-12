import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Thicknesses to plot
hs = [0.08, 0.04, 0.02, 0.01]

# Output directory
outdir = Path("figures/validation")
outdir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(6,4))

for h in hs:

    htag = f"h{int(round(h * 1000)):03d}"

    # Load data
    full_df = pd.read_csv(
        f"results/validation_full/full_curves_{htag}.csv"
    )

    red_df = pd.read_csv(
        f"results/validation_reduced/reduced_curves_{htag}.csv"
    )

    # Compute absolute flux difference
    err = abs(full_df["J_full"] - red_df["J_red"])

    # Plot
    plt.plot(
        full_df["t"],
        err,
        label=f"h={h}"
    )

plt.xlabel("time t")
plt.ylabel(r"$|J_{\mathrm{full}}-J_{\mathrm{red}}|$")
plt.legend()
plt.grid(True)

plt.tight_layout()

outfile = outdir / "flux_difference_time.pdf"
plt.savefig(outfile)

print(f"Saved: {outfile}")
