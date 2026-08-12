import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load comparison table
df = pd.read_csv("results/validation_comparison/validation_summary.csv")

# Create figures directory
outdir = Path("figures/validation")
outdir.mkdir(parents=True, exist_ok=True)

# Plot bulk mass error vs h
plt.figure(figsize=(5,4))

plt.loglog(
    df["h"],
    df["E_B_T"],
    marker="o",
    linewidth=2,
)

plt.xlabel("coating thickness h")
plt.ylabel(r"$E_B(T;h)$")
plt.grid(True, which="both")

plt.tight_layout()

outfile = outdir / "bulk_error_vs_h.pdf"
plt.savefig(outfile)

print(f"Saved: {outfile}")
