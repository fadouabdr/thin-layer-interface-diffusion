import pandas as pd
import numpy as np

h_tag = "h040"

full = pd.read_csv(f"results/validation_full/full_curves_{h_tag}.csv")
red = pd.read_csv(f"results/validation_reduced/reduced_curves_{h_tag}.csv")

# assumes same time grid
t = full["t"].values

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
##
E_B_T = abs(B_full[-1] - B_red[-1])
E_E_T = abs(E_full[-1] - E_red[-1])

rel_B_T = E_B_T / max(abs(B_full[-1]), 1e-14)
rel_E_T = E_E_T / max(abs(E_full[-1]), 1e-14)
##
E_J_max = np.max(np.abs(J_full - J_red))
E_M_max = np.max(np.abs(M_full - M_red))

rel_M_T = E_M_T / abs(M_full[-1])

print("Comparison for h = 0.04")
print("-----------------------")
print("J_full(T) =", J_full[-1])
print("J_red(T)  =", J_red[-1])
print("E_J(T)    =", E_J_T)
print()
print("M_full(T) =", M_full[-1])
print("M_red(T)  =", M_red[-1])
print("E_M(T)    =", E_M_T)
print("rel E_M(T)=", rel_M_T)
print()
print("max E_J(t)=", E_J_max)
print("max E_M(t)=", E_M_max)
##
print()
print("B_full(T) =", B_full[-1])
print("B_red(T)  =", B_red[-1])
print("E_B(T)    =", E_B_T)
print("rel E_B(T)=", rel_B_T)

print()
print("E_full(T) =", E_full[-1])
print("E_red(T)  =", E_red[-1])
print("E_E(T)    =", E_E_T)
print("rel E_E(T)=", rel_E_T)
