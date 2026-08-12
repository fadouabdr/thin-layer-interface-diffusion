import pandas as pd
import numpy as np
from pathlib import Path

hs = [0.08, 0.04, 0.02, 0.01]
htags = ["h080", "h040", "h020", "h010"]

rows = []

for h, htag in zip(hs, htags):
    full = pd.read_csv(f"results/validation_full/full_curves_{htag}.csv")
    red = pd.read_csv(f"results/validation_reduced/reduced_curves_{htag}.csv")

    J_full = full["J_full"].values
    M_full = full["M_full"].values
    B_full = full["mass_bulk_full"].values
    E_full = full["energy_bulk_full"].values

    J_red = red["J_red"].values
    M_red = red["M_red"].values
    B_red = red["mass_red"].values
    E_red = red["energy_red"].values

    E_J_T = abs(J_full[-1] - J_red[-1])
    E_M_T = abs(M_full[-1] - M_red[-1])
    E_J_max = np.max(np.abs(J_full - J_red))
    E_M_max = np.max(np.abs(M_full - M_red))
    rel_M_T = E_M_T / abs(M_full[-1])
    E_B_T = abs(B_full[-1] - B_red[-1])
    E_E_T = abs(E_full[-1] - E_red[-1])

    rel_B_T = E_B_T / max(abs(B_full[-1]), 1e-14)
    rel_E_T = E_E_T / max(abs(E_full[-1]), 1e-14)

    rows.append({
        "h": h,
        "J_full_T": J_full[-1],
        "J_red_T": J_red[-1],
        "E_J_T": E_J_T,
        "M_full_T": M_full[-1],
        "M_red_T": M_red[-1],
        "E_M_T": E_M_T,
        "rel_E_M_T": rel_M_T,
        "E_J_max": E_J_max,
        "E_M_max": E_M_max,
        ##
        "B_full_T": B_full[-1],
        "B_red_T": B_red[-1],
        "E_B_T": E_B_T,
        "rel_E_B_T": rel_B_T,

        "E_full_T": E_full[-1],
        "E_red_T": E_red[-1],
        "E_E_T": E_E_T,
        "rel_E_E_T": rel_E_T,
    })

df = pd.DataFrame(rows)

Path("results/validation_comparison").mkdir(parents=True, exist_ok=True)
df.to_csv("results/validation_comparison/validation_summary.csv", index=False)

print(df.to_string(index=False))
print()
print("Saved: results/validation_comparison/validation_summary.csv")
