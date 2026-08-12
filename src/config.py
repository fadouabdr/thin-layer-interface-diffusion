import numpy as np

# Physical parameters
D_F = 1.0
ALPHA = 1.3

# Final time
T_FINAL = 0.1

# Mesh resolutions for spatial study
SPATIAL_RESOLUTIONS = [8, 16, 32, 64]

# Time step used for the spatial convergence study
DT_SPATIAL = 1.0e-4

# Time steps for temporal study
TEMPORAL_DTS = [1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3]

# Fine mesh resolution for temporal study
TEMPORAL_RESOLUTION = 128


def kappa(t):
    return 1.0 + 0.5 * np.sin(t)