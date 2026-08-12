# Thin-layer asymptotics for diffusion through coatings with time-dependent diffusivity

This repository contains the numerical code, verification data, and
resolved–reduced computational results supporting the manuscript

> **Thin-layer asymptotics for diffusion through coatings with time-dependent
> diffusivity: derivation and numerical assessment of an effective interface
> model**

by Fadoua Boudrari, Abdessamad Belfakir, Abderrahmane Habbal, and Ahmed Ratnani.

The manuscript derives and numerically assesses an effective interface model
for transient diffusion through a thin coating surrounding a well-mixed core,
with controlled-release fertilizer granules as a motivating application.

Under the thin-coating and rapid-normal-equilibration regime considered in the
manuscript, the resolved coating is replaced by the Robin law

$$
-D_f \nabla c_f \cdot n
=
\kappa_h(t)
\left(c_{\mathrm{int}}(t)-\alpha c_f\right),
$$

where

$$
\kappa_h(t)=\frac{K_{\mathrm{in}}D_m(t)}{h},
\qquad
\alpha=\frac{K_{\mathrm{out}}}{K_{\mathrm{in}}}.
$$

The transfer coefficient is determined directly from the coating thickness,
the prescribed time-dependent diffusivity, and the partition parameters; it
is not fitted from the resolved simulations.

The manuscript first retains finite transfer kinetics at the two coating
interfaces through a resistance-in-series formulation. The partition-controlled
Robin law above is recovered in the fast-transfer limit. The numerical studies
archived in this repository assess this partition-controlled reduced model.

## Purpose of the repository

The repository is a computational companion to the manuscript. It contains:

- manufactured-solution verification of the spatial and temporal
  discretizations;
- the final planar resolved–reduced thickness study;
- the numerical-resolution audits used to validate the planar production
  discretization;
- the final circular-granule resolved–reduced comparison;
- the circular geometry and mesh audit;
- the circular time-step sensitivity calculation;
- the data and figure files associated with the numerical section of the
  manuscript.

The numerical experiments are nondimensional benchmark calculations. They
assess the asymptotic model reduction and its numerical implementation; they
are not an experimental validation or calibration of a particular fertilizer
formulation.

## Relation to the manuscript

The numerical evidence has three distinct roles.

1. **Manufactured-solution verification** checks the finite-element and
   backward-Euler implementation independently of the thin-layer reduction.

2. **The planar decreasing-thickness study** provides the principal
   quantitative assessment of the asymptotic interface replacement.

3. **The circular-granule study** tests transfer of the leading interface law
   to a curved two-dimensional geometry at one value of \(h/R\).

The near-quadratic decrease observed in the planar field discrepancy is a
numerical observation for the smooth prepared-data benchmark and is not a
general \(O(h^2)\) error theorem.

Likewise, the circular experiment is not a curvature-convergence study and is
not used to derive or validate a curvature-corrected interface law.

## Final manuscript datasets

The `results/` directory contains the numerical datasets corresponding to the
manuscript version associated with this repository.

### Manufactured-solution verification — Table 1

The verification data are

```text
results/spatial/spatial_table.csv
results/temporal/temporal_table.csv
```

The spatial study confirms approximately second-order convergence in the
\(L^2\)-norm for continuous piecewise affine finite elements.

The temporal study confirms approximately first-order convergence for the
backward-Euler discretization.

These tests verify the numerical discretization independently of the
resolved–reduced comparisons.

### Planar resolved–reduced assessment — Tables 2–3 and Figures 2–3

The final time-dependent planar production study is stored in

```text
results/canonical/time_dependent_prepared_T1_dt1e-03_L8_B64_Y64/
```

The benchmark uses

$$
D_f=1,
\qquad
K_{\mathrm{in}}=1,
\qquad
K_{\mathrm{out}}=1.3,
$$

$$
D_m(t)=1+0.5\sin t,
\qquad
c_{\mathrm{int}}(t)=e^{-t},
\qquad
T=1,
$$

with coating thicknesses

$$
h\in\{0.08,0.04,0.02,0.01\}.
$$

The accepted production discretization is

$$
\Delta t=10^{-3},
\qquad
n_{\mathrm{layer}}=8,
\qquad
n_{\mathrm{bulk}}=n_y=64.
$$

The manuscript summary data are stored in

```text
results/canonical/time_dependent_prepared_T1_dt1e-03_L8_B64_Y64/
    paper_validation_summary.csv
```

The corresponding manuscript figures are

```text
results/canonical/time_dependent_prepared_T1_dt1e-03_L8_B64_Y64/
    figures/planar_bulk_convergence.pdf

results/canonical/time_dependent_prepared_T1_dt1e-03_L8_B64_Y64/
    figures/planar_release_history.pdf
```

The relative final-time exterior-field discrepancy decreases from

$$
2.251\times10^{-3}
$$

at \(h=0.08\) to

$$
3.576\times10^{-5}
$$

at \(h=0.01\).

The field and bulk-mass discrepancies exhibit approximately quadratic decrease
for this smooth prepared-data benchmark. This empirical behavior is specific to
the tested configuration and is not asserted as a general convergence theorem
for the thin-layer reduction.

### Planar resolution audit

Before the four-thickness production study, the endpoint thicknesses

$$
h=0.08
\qquad\text{and}\qquad
h=0.01
$$

were tested using exterior-mesh refinement, coating refinement, time-step
refinement, and combined refinement.

The archived audit results are

```text
results/resolution_audit/
    constant_prepared_T1_dt1e-03_L8_B64_Y64/

results/resolution_audit/
    time_dependent_prepared_T1_dt1e-03_L8_B64_Y64/
```

Each directory contains an `acceptance.json` file recording the audit decision.

For both endpoint thicknesses and both diffusivity scenarios, the exterior-field
and bulk-mass discrepancies changed by at most approximately \(0.03\%\)
relative to the combined-refinement calculation, and none of the audited
model-discrepancy metrics changed by more than approximately \(1.40\%\).

The constant-diffusivity case is retained as a numerical control. The
resolved–reduced results reported in Tables 2–3 use the accepted
time-dependent setting.

### Circular-granule comparison — Figures 4–5

The circular experiment uses

$$
R=0.25,
\qquad
h=0.01,
\qquad
\frac{h}{R}=0.04,
$$

in the square

$$
Q=(-0.75,0.75)^2.
$$

The annular coating is discretized using eight finite-element layers across
its thickness.

The physical data are

$$
D_f=1,
\qquad
K_{\mathrm{in}}=1,
\qquad
K_{\mathrm{out}}=1.3,
$$

$$
D_m(t)=1+0.5\sin t,
\qquad
c_{\mathrm{int}}(t)=e^{-t}.
$$

Both models are integrated to

$$
T=1
$$

using

$$
\Delta t=10^{-3}.
$$

The final production data are stored in

```text
results/circular/production_common_initial/
```

The mesh files and automated geometry/tag audit are stored in

```text
results/circular/meshes/
```

The final meshes contain

```text
resolved:  14816 cells, 7636 P1 degrees of freedom
reduced:    9216 cells, 4836 P1 degrees of freedom
```

The final production calculation gives the relative time-\(L^2\) history
discrepancies

$$
\mathcal E^{\mathrm{circ}}_{J,\mathrm{rel}}=4.18\%,
\qquad
\mathcal E^{\mathrm{circ}}_{M,\mathrm{rel}}=3.82\%.
$$

The concentration snapshots are selected at the first times at which the
resolved cumulative release reaches 10%, 50%, and 90% of its value at the
finite production horizon \(T=1\):

$$
t=0.063,
\qquad
t=0.379,
\qquad
t=0.842.
$$

The figure outputs generated from the final circular calculation are stored in

```text
results/circular/figures_common_initial/
```

The manuscript uses

```text
circular_geometry_mesh.pdf
circular_concentration_snapshots.pdf
```

as Figures 4 and 5, respectively.

PNG versions of these figures are included for convenience.

The files

```text
circular_release_curves.pdf
circular_release_curves.png
```

are retained as auxiliary diagnostics and are not separate figures in the
manuscript.

Because the resolved flux is integrated on \(\Gamma_h\) while the reduced flux
is integrated on \(\Gamma_g\), the circular flux discrepancy also includes the
finite-\(h\) difference between the two integration surfaces. It should
therefore be interpreted as a complementary global diagnostic rather than as a
local flux error or a curvature-convergence rate.

### Circular time-step sensitivity check

The time-step sensitivity calculation with

$$
\Delta t=5\times10^{-4}
$$

is archived in

```text
results/circular/time_step_check_dt5e-4/
```

Halving the production time step changes the relative flux-history discrepancy
from approximately \(4.18\%\) to \(4.21\%\), while the cumulative-release
history discrepancy remains approximately \(3.82\%\).

## Initial data

### Planar benchmark

The planar production study uses prepared initial data to avoid introducing an
artificial resolved–reduced flux mismatch at \(t=0\).

Define

$$
A_h=
\frac{\kappa_h(0)c_{\mathrm{int}}(0)}
     {D_f+\alpha\kappa_h(0)}.
$$

The exterior initial concentration is

$$
c_{f,\mathrm{in}}^h(x)=A_h(1-x_1),
$$

and the coating concentration is affine between the inner and outer partition
traces.

These data make the resolved coating flux, exterior gradient, and reduced
Robin flux agree at the initial time.

### Circular benchmark

For the circular comparison, the stationary reduced-domain Robin solution at
\(t=0\) defines a single continuum exterior initial datum. The same function is
interpolated onto the reduced and resolved exterior meshes.

The resolved coating is initialized by an affine radial profile joining the two
partition traces.

These coating data satisfy the partition conditions but are not forced to
satisfy exact resolved flux compatibility at \(t=0\). The associated normal
adjustment occurs on the short coating equilibration time scale and is
completed before the first displayed snapshot.

## Diagnostics

The planar resolved–reduced calculations report:

- absolute and relative final-time exterior-field discrepancy;
- absolute and relative final-time exterior-mass discrepancy;
- relative time-\(L^2\) interface-flux discrepancy;
- absolute and relative cumulative-release discrepancy;
- resolved and reduced outer-boundary outflow;
- exterior-domain balance residuals;
- reduced Robin-versus-variational flux consistency;
- initial resolved–reduced flux compatibility.

The cumulative release is advanced consistently with backward Euler:

$$
M^{n+1}=M^n+\Delta t\,J^{n+1}.
$$

The exterior-domain balance contains both the interface influx and the
outer-boundary outflow.

Because \(c_{\mathrm{int}}(t)\) is prescribed, this balance is not a
conservation law for the complete granule–coating–exterior system.

## Reproducibility environment

The production environment archived for reproducing the manuscript
computations is the Docker image

```text
ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
```

The runtime verified from this image is

```text
Python      3.10.12
DOLFIN      2019.2.0.dev0
Gmsh        4.12.2
meshio      5.3.5
NumPy       1.21.5
Matplotlib  3.5.1
```

The manuscript calculations use the DOLFIN interface of the FEniCS Project.

Additional environment information is recorded in

```text
ENVIRONMENT.md
```

### Start the environment

From the repository root:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30 \
  bash
```

The reproduction commands below are then run inside the container.

## Reproduction workflow

### 1. Unit tests

```bash
python3 -m unittest discover -s tests -v
```

### 2. Planar geometry and flux-sign checks

```bash
python3 scripts/check_planar_setup.py
```

For the test profile \(c=1-x_1\), the recovered interface influx and
outer-boundary outflow should have positive unit magnitude.

### 3. Planar resolution audit

Constant-diffusivity control:

```bash
python3 scripts/run_planar_resolution_audit.py \
  --scenario constant \
  --initialization prepared \
  --t-final 1.0 \
  --dt 1e-3
```

Time-dependent case:

```bash
python3 scripts/run_planar_resolution_audit.py \
  --scenario time_dependent \
  --initialization prepared \
  --t-final 1.0 \
  --dt 1e-3
```

The manuscript audit is performed at the endpoint thicknesses
\(h=0.08\) and \(h=0.01\).

The audit compares the production calculation with exterior-refined,
coating-refined, time-refined, and combined-reference calculations.

### 4. Final planar thickness study

After the resolution audit has passed:

```bash
python3 scripts/run_canonical_planar_validation.py \
  --scenario time_dependent \
  --initialization prepared \
  --t-final 1.0 \
  --dt 1e-3
```

This produces the four-thickness study for

```text
h = 0.08, 0.04, 0.02, 0.01
```

using the accepted production settings.

### 5. Planar manuscript figures

```bash
python3 scripts/make_canonical_planar_figures.py \
  results/canonical/time_dependent_prepared_T1_dt1e-03_L8_B64_Y64
```

### 6. Circular geometry and mesh audit

```bash
python3 -m scripts.check_circular_setup
```

The generated file

```text
results/circular/meshes/circular_mesh_audit.json
```

must report a passed status before the transient circular calculation is used.

### 7. Circular production calculation

```bash
python3 -m scripts.run_circular_illustration \
  --output-dir results/circular/production_common_initial
```

The run produces the paired release histories, snapshot data, and a manifest
containing the configuration, diagnostics, initialization checks, mesh counts,
software versions, and output locations.

### 8. Circular manuscript figures

The circular figure-generation script is

```bash
python3 -m scripts.make_circular_publication_figures
```

The archived outputs corresponding to the manuscript are stored in

```text
results/circular/figures_common_initial/
```

## Publication-facing repository structure

```text
.
├── src/
│   ├── full_thin_layer_solver.py
│   ├── reduced_validation_solver.py
│   ├── planar_validation.py
│   ├── initial_data.py
│   ├── validation_config.py
│   ├── validation_diagnostics.py
│   ├── validation_metrics.py
│   ├── validation_audit.py
│   ├── circular_config.py
│   ├── circular_mesh_utils.py
│   ├── circular_full_solver.py
│   ├── circular_reduced_solver.py
│   ├── circular_initial_data.py
│   ├── circular_results.py
│   └── circular_solver_utils.py
│
├── scripts/
│   ├── check_planar_setup.py
│   ├── run_planar_resolution_audit.py
│   ├── run_canonical_planar_validation.py
│   ├── make_canonical_planar_figures.py
│   ├── check_circular_setup.py
│   ├── run_circular_illustration.py
│   └── make_circular_publication_figures.py
│
├── tests/
│
├── results/
│   ├── spatial/
│   ├── temporal/
│   ├── canonical/
│   ├── resolution_audit/
│   └── circular/
│       ├── meshes/
│       ├── production_common_initial/
│       ├── figures_common_initial/
│       └── time_step_check_dt5e-4/
│
├── ENVIRONMENT.md
├── CITATION.cff
├── LICENSE
└── README.md
```

The paths above identify the publication-facing code and data associated with
the manuscript. Development-only files and obsolete numerical outputs are not
part of the archived release.

## Scientific scope

The derived Robin law is intended for thin coatings satisfying the
geometric-thinness, rapid-normal-equilibration, and controlled-tangential-
variation assumptions stated in the manuscript.

The core concentration \(c_{\mathrm{int}}(t)\) and coating diffusivity
\(D_m(t)\) are prescribed inputs.

The present model does not separately resolve water uptake, swelling,
moisture-dependent transport, polymer degradation, or feedback from a dynamic
core mass balance.

The planar study provides the principal thickness-dependent asymptotic
assessment.

The circular calculation provides a complementary curved-domain comparison at
a single value \(h/R=0.04\). It should not be interpreted as evidence for a
curvature correction or as a curvature-convergence study.

All reported numerical differences between the two models are
resolved–reduced model discrepancies, not experimental errors.

## Citation

Citation metadata for the archived computational release are provided in

```text
CITATION.cff
```

When referring to the mathematical model and scientific conclusions, please
cite the associated manuscript.

When referring specifically to the computational implementation or archived
numerical dataset, please cite the versioned repository release and its
archival DOI.