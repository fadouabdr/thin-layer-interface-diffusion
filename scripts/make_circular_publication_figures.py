import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np

from src.validation_config import K_OUT


COATING_CELL = 1
BULK_CELL = 2
EXPECTED_SNAPSHOT_FRACTIONS = np.asarray([0.1, 0.5, 0.9])
RESOLVED_COLOR = "#1f4e79"
REDUCED_COLOR = "#c45a2d"
COATING_COLOR = "#e8a36a"
BULK_COLOR = "#dbe9f4"
CORE_COLOR = "#d8d8d8"


def _snapshot_title(time):
    """Use the physical time without implying eventual-release fractions."""
    return rf"$t={float(time):.3f}$"


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="results/circular/production",
        help="Directory containing the accepted manifest, CSV, and NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/circular/figures",
        help="Directory for PDF/PNG figures and the figure manifest.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PNG resolution; PDFs retain vector text and lines.",
    )
    return parser


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(input_dir):
    manifest_path = input_dir / "circular_run_manifest.json"
    curves_path = input_dir / "circular_curves.csv"
    snapshots_path = input_dir / "circular_snapshots.npz"
    missing = [
        str(path)
        for path in (manifest_path, curves_path, snapshots_path)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError("Missing circular production files: " + ", ".join(missing))

    with open(manifest_path) as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "passed":
        raise RuntimeError("The circular run manifest is not passed.")
    if manifest.get("mode") != "production":
        raise RuntimeError("Publication figures require the full production run.")
    if not manifest.get("diagnostics", {}).get("passed", False):
        raise RuntimeError("The circular production diagnostics are not passed.")
    initialization = manifest.get("initialization", {})
    if not initialization.get("common_exterior_initial_datum", False):
        raise RuntimeError(
            "Publication figures require the corrected circular run with one "
            "common exterior initial datum."
        )

    curves = np.genfromtxt(curves_path, delimiter=",", names=True)
    if curves.ndim != 1 or curves.size < 2:
        raise RuntimeError("The circular curve file is empty or malformed.")
    required_columns = {
        "t", "J_full", "J_red", "M_full", "M_red",
        "balance_normalized_full", "balance_normalized_red",
        "robin_flux_mismatch_red",
    }
    if not required_columns.issubset(curves.dtype.names or ()):
        missing_columns = sorted(required_columns - set(curves.dtype.names or ()))
        raise RuntimeError("Missing curve columns: " + ", ".join(missing_columns))
    for name in curves.dtype.names:
        if not np.all(np.isfinite(curves[name])):
            raise RuntimeError(f"Curve column {name} contains non-finite values.")
    if not np.all(np.diff(curves["t"]) > 0.0):
        raise RuntimeError("Curve times must be strictly increasing.")

    with np.load(snapshots_path, allow_pickle=False) as archive:
        snapshots = {name: archive[name].copy() for name in archive.files}
    required_arrays = {
        "snapshot_fractions", "snapshot_indices", "snapshot_times",
        "prescribed_core_concentration", "resolved_coordinates",
        "resolved_cells", "resolved_cell_tags", "resolved_q_vertex",
        "reduced_coordinates", "reduced_cells", "reduced_cell_tags",
        "reduced_c_vertex",
    }
    missing_arrays = sorted(required_arrays - set(snapshots))
    if missing_arrays:
        raise RuntimeError("Missing snapshot arrays: " + ", ".join(missing_arrays))
    for name, values in snapshots.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Snapshot array {name} contains non-finite values.")
    if not np.allclose(
        snapshots["snapshot_fractions"],
        EXPECTED_SNAPSHOT_FRACTIONS,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Expected 10%, 50%, and 90% release snapshots.")

    n_snapshots = len(EXPECTED_SNAPSHOT_FRACTIONS)
    if snapshots["resolved_q_vertex"].shape != (
        n_snapshots,
        len(snapshots["resolved_coordinates"]),
    ):
        raise RuntimeError("Resolved snapshot dimensions are inconsistent.")
    if snapshots["reduced_c_vertex"].shape != (
        n_snapshots,
        len(snapshots["reduced_coordinates"]),
    ):
        raise RuntimeError("Reduced snapshot dimensions are inconsistent.")
    if len(snapshots["resolved_cells"]) != len(
        snapshots["resolved_cell_tags"]
    ):
        raise RuntimeError("Resolved cell tags do not match resolved cells.")
    if len(snapshots["reduced_cells"]) != len(
        snapshots["reduced_cell_tags"]
    ):
        raise RuntimeError("Reduced cell tags do not match reduced cells.")

    return manifest, curves, snapshots, {
        "manifest": manifest_path,
        "curves": curves_path,
        "snapshots": snapshots_path,
    }


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_figure(fig, stem, dpi):
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    signatures = {pdf: b"%PDF-", png: b"\x89PNG\r\n\x1a\n"}
    for path, signature in signatures.items():
        if not path.exists() or path.stat().st_size < 1024:
            raise RuntimeError(f"Figure output is missing or truncated: {path}")
        with open(path, "rb") as stream:
            if stream.read(len(signature)) != signature:
                raise RuntimeError(f"Figure output has an invalid header: {path}")
    return [pdf, png]


def _cell_polygons(coordinates, cells, mask):
    return coordinates[cells[mask]]


def _make_mesh_figure(snapshots, config, output_dir, dpi):
    r_core = float(config["core_radius"])
    r_outer = float(config["outer_coating_radius"])
    half_width = float(config["box_half_width"])
    zoom_radius = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.45), constrained_layout=True)
    specs = [
        (
            axes[0],
            snapshots["resolved_coordinates"],
            snapshots["resolved_cells"],
            snapshots["resolved_cell_tags"],
            "Resolved thin coating",
        ),
        (
            axes[1],
            snapshots["reduced_coordinates"],
            snapshots["reduced_cells"],
            snapshots["reduced_cell_tags"],
            "Effective interface",
        ),
    ]

    for panel, (ax, coordinates, cells, tags, title) in enumerate(specs):
        facecolors = [
            COATING_COLOR if tag == COATING_CELL else BULK_COLOR
            for tag in tags
        ]
        ax.add_collection(PolyCollection(
            coordinates[cells],
            facecolors=facecolors,
            edgecolors="#75808a",
            linewidths=0.055,
            alpha=0.96,
            rasterized=True,
            zorder=1,
        ))
        ax.add_patch(Circle(
            (0.0, 0.0),
            r_core,
            facecolor=CORE_COLOR,
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        ))
        if panel == 0:
            ax.add_patch(Circle(
                (0.0, 0.0),
                r_outer,
                fill=False,
                edgecolor="#8c2d04",
                linewidth=0.8,
                zorder=5,
            ))
        ax.set_title(title)
        ax.text(
            0.02,
            0.98,
            f"({chr(97 + panel)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_width, half_width)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_xticks([-0.5, 0.0, 0.5])
        ax.set_yticks([-0.5, 0.0, 0.5])

        zoom = inset_axes(
            ax,
            width="45%",
            height="45%",
            loc="upper right",
            borderpad=0.75,
        )
        centroids = coordinates[cells].mean(axis=1)
        keep = np.linalg.norm(centroids, axis=1) <= zoom_radius
        kept_cells = cells[keep]
        kept_tags = tags[keep]
        zoom_facecolors = [
            COATING_COLOR if tag == COATING_CELL else BULK_COLOR
            for tag in kept_tags
        ]
        collection = PolyCollection(
            coordinates[kept_cells],
            facecolors=zoom_facecolors,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        zoom.add_collection(collection)
        edges = LineCollection(
            coordinates[kept_cells],
            colors="#52606b",
            linewidths=0.20,
            alpha=0.78,
            rasterized=True,
            zorder=2,
        )
        zoom.add_collection(edges)
        zoom.add_patch(Circle(
            (0.0, 0.0),
            r_core,
            facecolor=CORE_COLOR,
            edgecolor="black",
            linewidth=0.85,
            zorder=4,
        ))
        if panel == 0:
            zoom.add_patch(Circle(
                (0.0, 0.0),
                r_outer,
                fill=False,
                edgecolor="#8c2d04",
                linewidth=1.0,
                zorder=5,
            ))
            zoom.text(
                0.0,
                0.0,
                r"$\Omega_g$",
                ha="center",
                va="center",
                zorder=7,
            )
            zoom.annotate(
                r"$\Gamma_g$",
                xy=(-0.177, 0.177),
                xytext=(-0.30, 0.30),
                arrowprops={"arrowstyle": "-", "lw": 0.7},
                ha="left",
            )
            zoom.annotate(
                r"$\Gamma_h$",
                xy=(0.184, 0.184),
                xytext=(0.27, 0.30),
                arrowprops={"arrowstyle": "-", "lw": 0.7},
                ha="right",
            )
            zoom.text(
                0.0,
                -0.315,
                f"{int(config['radial_layers'])} radial layers",
                ha="center",
                va="center",
                fontsize=6.5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7},
            )
        else:
            zoom.text(
                0.0,
                0.0,
                r"$\Omega_g$",
                ha="center",
                va="center",
                zorder=7,
            )
            zoom.annotate(
                r"$\Gamma_g$ (Robin)",
                xy=(0.177, 0.177),
                xytext=(0.0, 0.282),
                arrowprops={"arrowstyle": "-", "lw": 0.7},
                ha="center",
                fontsize=6.7,
                zorder=7,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            )
            zoom.text(
                0.0,
                -0.315,
                "no coating subdomain",
                ha="center",
                va="center",
                fontsize=6.5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7},
            )
        zoom.set_xlim(-zoom_radius, zoom_radius)
        zoom.set_ylim(-zoom_radius, zoom_radius)
        zoom.set_aspect("equal")
        zoom.set_xticks([])
        zoom.set_yticks([])
        mark_inset(
            ax,
            zoom,
            loc1=2,
            loc2=4,
            fc="none",
            ec="#555555",
            lw=0.55,
        )

    return _save_figure(
        fig,
        output_dir / "circular_geometry_mesh",
        dpi,
    )


def _triangulation(coordinates, cells, cell_tags, selected_tag):
    return mtri.Triangulation(
        coordinates[:, 0],
        coordinates[:, 1],
        cells,
        mask=np.asarray(cell_tags) != selected_tag,
    )


def _selected_contour_levels(values, cells, tags, selected_tag, levels):
    selected_cells = cells[np.asarray(tags) == selected_tag]
    selected_vertices = np.unique(selected_cells)
    lower = float(np.min(values[selected_vertices]))
    upper = float(np.max(values[selected_vertices]))
    return np.asarray([level for level in levels if lower < level < upper])


def _plot_resolved_field(ax, snapshots, index, norm, cmap, levels, config):
    coordinates = snapshots["resolved_coordinates"]
    cells = snapshots["resolved_cells"]
    tags = snapshots["resolved_cell_tags"]
    q = snapshots["resolved_q_vertex"][index]
    bulk_tri = _triangulation(coordinates, cells, tags, BULK_CELL)
    coating_tri = _triangulation(coordinates, cells, tags, COATING_CELL)

    bulk_plot = ax.tripcolor(
        bulk_tri,
        q,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    ax.tripcolor(
        coating_tri,
        K_OUT * q,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    for tri, values, tag in (
        (bulk_tri, q, BULK_CELL),
        (coating_tri, K_OUT * q, COATING_CELL),
    ):
        active_levels = _selected_contour_levels(values, cells, tags, tag, levels)
        if len(active_levels):
            ax.tricontour(
                tri,
                values,
                levels=active_levels,
                colors="white",
                linewidths=0.32,
                alpha=0.72,
            )
    ax.add_patch(Circle(
        (0.0, 0.0),
        float(config["core_radius"]),
        facecolor=cmap(norm(snapshots["prescribed_core_concentration"][index])),
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
    ))
    ax.add_patch(Circle(
        (0.0, 0.0),
        float(config["outer_coating_radius"]),
        facecolor="none",
        edgecolor="#333333",
        linewidth=0.38,
        zorder=6,
    ))
    return bulk_plot


def _plot_reduced_field(ax, snapshots, index, norm, cmap, levels, config):
    coordinates = snapshots["reduced_coordinates"]
    cells = snapshots["reduced_cells"]
    tags = snapshots["reduced_cell_tags"]
    concentration = snapshots["reduced_c_vertex"][index]
    tri = _triangulation(coordinates, cells, tags, BULK_CELL)
    field_plot = ax.tripcolor(
        tri,
        concentration,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    active_levels = _selected_contour_levels(
        concentration, cells, tags, BULK_CELL, levels
    )
    if len(active_levels):
        ax.tricontour(
            tri,
            concentration,
            levels=active_levels,
            colors="white",
            linewidths=0.32,
            alpha=0.72,
        )
    ax.add_patch(Circle(
        (0.0, 0.0),
        float(config["core_radius"]),
        facecolor=cmap(norm(snapshots["prescribed_core_concentration"][index])),
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
    ))
    return field_plot


def _make_snapshot_figure(snapshots, config, output_dir, dpi):
    resolved_physical_max = max(
        float(np.max(snapshots["resolved_q_vertex"])),
        float(K_OUT * np.max(snapshots["resolved_q_vertex"])),
    )
    concentration_max = max(
        resolved_physical_max,
        float(np.max(snapshots["reduced_c_vertex"])),
        float(np.max(snapshots["prescribed_core_concentration"])),
    )
    norm = Normalize(vmin=0.0, vmax=concentration_max)
    cmap = plt.get_cmap("viridis")
    levels = np.linspace(0.0, concentration_max, 10)

    # Use an explicit GridSpec instead of constrained_layout.  This reserves
    # independent, deterministic gutters for
    # row name | y-label | tick labels | panels | colorbar.
    fig = plt.figure(figsize=(7.15, 4.75))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        left=0.115,
        right=0.92,
        bottom=0.10,
        top=0.94,
        wspace=0.10,
        hspace=0.18,
    )
    axes = np.empty((2, 3), dtype=object)
    for row in range(2):
        for column in range(3):
            shared_axis = axes[0, 0] if (row, column) != (0, 0) else None
            axes[row, column] = fig.add_subplot(
                grid[row, column],
                sharex=shared_axis,
                sharey=shared_axis,
            )
    colorbar_axis = fig.add_subplot(grid[:, 3])
    mappable = None
    for index in range(3):
        mappable = _plot_resolved_field(
            axes[0, index], snapshots, index, norm, cmap, levels, config
        )
        _plot_reduced_field(
            axes[1, index], snapshots, index, norm, cmap, levels, config
        )
        time = snapshots["snapshot_times"][index]
        axes[0, index].set_title(_snapshot_title(time))

    half_width = float(config["box_half_width"])
    for row in range(2):
        for column in range(3):
            ax = axes[row, column]
            ax.set_xlim(-half_width, half_width)
            ax.set_ylim(-half_width, half_width)
            ax.set_aspect("equal")
            ax.set_xticks([-0.5, 0.0, 0.5])
            ax.set_yticks([-0.5, 0.0, 0.5])
            ax.tick_params(
                labelbottom=(row == 1),
                labelleft=(column == 0),
            )
            if row == 1:
                ax.set_xlabel(r"$x$")
            if column == 0:
                ax.set_ylabel(r"$y$", labelpad=5)

    colorbar = fig.colorbar(
        mappable,
        cax=colorbar_axis,
    )
    colorbar.set_label("Physical concentration")

    # Put the model names in the reserved far-left gutter.  Their positions
    # are computed from the fixed GridSpec, so they cannot collide with y.
    for row, label in enumerate(("Resolved", "Reduced")):
        position = axes[row, 0].get_position()
        row_center = 0.5 * (position.y0 + position.y1)

        fig.text(
            0.018,
            row_center,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    paths = _save_figure(
        fig,
        output_dir / "circular_concentration_snapshots",
        dpi,
    )
    return paths, concentration_max


def _make_curve_figure(curves, manifest, snapshots, output_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    time = curves["t"]

    axes[0].plot(
        time,
        curves["J_full"],
        color=RESOLVED_COLOR,
        label=r"Resolved ($\Gamma_h$)",
    )
    axes[0].plot(
        time,
        curves["J_red"],
        color=REDUCED_COLOR,
        linestyle="--",
        label=r"Reduced ($\Gamma_g$)",
    )
    axes[0].set_xlabel(r"Time $t$")
    axes[0].set_ylabel(r"Total release flux $J(t)$")
    axes[0].set_title("Release flux")

    axes[1].plot(
        time,
        curves["M_full"],
        color=RESOLVED_COLOR,
        label=r"Resolved ($\Gamma_h$)",
    )
    axes[1].plot(
        time,
        curves["M_red"],
        color=REDUCED_COLOR,
        linestyle="--",
        label=r"Reduced ($\Gamma_g$)",
    )
    axes[1].scatter(
        snapshots["snapshot_times"],
        curves["M_full"][snapshots["snapshot_indices"]],
        s=15,
        facecolors="white",
        edgecolors=RESOLVED_COLOR,
        linewidths=0.8,
        zorder=4,
    )
    for fraction, snapshot_time, index in zip(
        snapshots["snapshot_fractions"],
        snapshots["snapshot_times"],
        snapshots["snapshot_indices"],
    ):
        axes[1].annotate(
            f"{int(round(100 * fraction))}%",
            xy=(snapshot_time, curves["M_full"][index]),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set_xlabel(r"Time $t$")
    axes[1].set_ylabel(r"Cumulative release $M(t)$")
    axes[1].set_title("Cumulative release")

    diagnostics = manifest["diagnostics"]
    annotations = [
        100.0 * float(diagnostics["relative_time_l2_flux_difference"]),
        100.0 * float(
            diagnostics["relative_time_l2_cumulative_release_difference"]
        ),
    ]
    for panel, (ax, discrepancy) in enumerate(zip(axes, annotations)):
        ax.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.8)
        ax.legend(frameon=False, loc="best")
        annotation_x = 0.02 if panel == 0 else 0.98
        annotation_alignment = "left" if panel == 0 else "right"
        ax.text(
            annotation_x,
            0.04,
            rf"relative time-$L^2$: {discrepancy:.2f}\%",
            transform=ax.transAxes,
            ha=annotation_alignment,
            va="bottom",
            fontsize=7.2,
        )
        ax.text(
            0.02,
            0.98,
            f"({chr(97 + panel)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    return _save_figure(fig, output_dir / "circular_release_curves", dpi)


def main():
    args = _parser().parse_args()
    if args.dpi < 300:
        raise ValueError("Publication PNG output requires dpi >= 300.")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest, curves, snapshots, input_paths = _load_inputs(input_dir)
    config = manifest["config"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()

    generated = []
    generated.extend(
        _make_mesh_figure(snapshots, config, output_dir, args.dpi)
    )
    snapshot_paths, concentration_max = _make_snapshot_figure(
        snapshots, config, output_dir, args.dpi
    )
    generated.extend(snapshot_paths)
    generated.extend(
        _make_curve_figure(curves, manifest, snapshots, output_dir, args.dpi)
    )

    figure_manifest_path = output_dir / "circular_figure_manifest.json"
    figure_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "purpose": (
            "Publication figures for the accepted qualitative circular "
            "resolved-versus-reduced comparison."
        ),
        "production_manifest_created_utc": manifest.get("created_utc"),
        "input_sha256": {
            name: _sha256(path) for name, path in input_paths.items()
        },
        "physical_concentration_reconstruction": {
            "coating": f"c_m = {K_OUT:g} q",
            "exterior": "c_f = q",
            "subdomains_rendered_separately": True,
        },
        "core_rendering": {
            "value": "prescribed c_int(t)",
            "computed_finite_element_field_inside_core": False,
        },
        "geometry_figure": {
            "complete_square_domain": True,
            "annulus_insets": True,
            "resolved_radial_layers_shown": int(config["radial_layers"]),
        },
        "shared_snapshot_color_limits": [0.0, concentration_max],
        "snapshot_fractions": [
            float(value) for value in snapshots["snapshot_fractions"]
        ],
        "snapshot_times": [
            float(value) for value in snapshots["snapshot_times"]
        ],
        "snapshot_title_rule": "actual physical times only",
        "snapshot_selection_rule": manifest.get(
            "snapshot_fraction_definition"
        ),
        "flux_integration_surfaces": {
            "resolved": "Gamma_h",
            "reduced": "Gamma_g",
        },
        "diagnostics": manifest["diagnostics"],
        "figures": [str(path) for path in generated],
        "output_sha256": {
            path.name: _sha256(path) for path in generated
        },
        "png_dpi": args.dpi,
    }
    with open(figure_manifest_path, "w") as stream:
        json.dump(figure_manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print("Circular publication figures generated:")
    for path in generated:
        print(f"  {path}")
    print(f"Figure manifest: {figure_manifest_path}")
    print("Circular publication-figure checks passed.")


if __name__ == "__main__":
    main()