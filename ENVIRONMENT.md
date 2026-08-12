# Reproducibility environment

The production environment archived for reproducing the manuscript
computations is the Docker image

```text
ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
```

For an immutable reference, the verified image digest is

```text
ghcr.io/scientificcomputing/fenics-gmsh@sha256:0bf22f477ef594ac5df53bc0b62bee426ad9b1bb184604dd6442718c6f898851
```

The runtime versions verified directly from this image are:

- Python 3.10.12
- DOLFIN 2019.2.0.dev0
- Gmsh 4.12.2
- meshio 5.3.5
- NumPy 1.21.5
- Matplotlib 3.5.1

These versions correspond to the environment used for the archived numerical
workflow. In particular, the DOLFIN version agrees with the implementation
version reported in the manuscript.

## Verify the environment

The recorded runtime versions can be checked with:

```bash
docker run --rm \
  ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30 \
  python3 -c "
import sys, dolfin, gmsh, meshio, numpy, matplotlib
print('Python     :', sys.version.split()[0])
print('DOLFIN     :', dolfin.__version__)
print('Gmsh       :', gmsh.__version__)
print('meshio     :', meshio.__version__)
print('NumPy      :', numpy.__version__)
print('Matplotlib :', matplotlib.__version__)
"
```

To open an interactive shell with the repository mounted at `/workspace`,
run from the repository root:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30 \
  bash
```

The numerical reproduction commands are documented in `README.md`.

