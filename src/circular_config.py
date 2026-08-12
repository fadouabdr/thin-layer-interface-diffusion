"""Shared configuration for the circular coated-granule illustration.

The planar physical laws remain authoritative in ``validation_config``.  This
module adds only the geometry, mesh, and run settings that are specific to the
circular illustration, so the two experiments cannot silently drift apart.
"""

from dataclasses import asdict, dataclass
import math

from .validation_config import (
    CANONICAL_DT_CANDIDATE,
    CANONICAL_T_FINAL,
    DEFAULT_SCENARIO,
    scenario_metadata,
    validate_scenario,
)


# Cell tags shared by the Gmsh files and the DOLFIN formulations.
COATING_CELL = 1
BULK_CELL = 2

# Facet tags.
CORE_BOUNDARY = 10
OUTER_BOUNDARY = 20
COATING_BULK_INTERFACE = 40

PHYSICAL_TAGS = {
    "coating_cell": COATING_CELL,
    "bulk_cell": BULK_CELL,
    "core_boundary": CORE_BOUNDARY,
    "outer_boundary": OUTER_BOUNDARY,
    "coating_bulk_interface": COATING_BULK_INTERFACE,
}


@dataclass(frozen=True)
class CircularCaseConfig:
    """Single source of truth for the circular mesh and paired run."""

    core_radius: float = 0.25
    coating_thickness: float = 0.01
    box_half_width: float = 0.75
    radial_layers: int = 8
    angular_sectors: int = 16
    interface_size: float = 0.005
    bulk_size: float = 0.05
    transition_start: float = 0.02
    transition_end: float = 0.25
    geometry_relative_tolerance: float = 5.0e-3
    polynomial_degree: int = 1
    scenario: str = DEFAULT_SCENARIO
    t_final: float = CANONICAL_T_FINAL
    dt: float = CANONICAL_DT_CANDIDATE

    @property
    def outer_coating_radius(self):
        return self.core_radius + self.coating_thickness

    @property
    def box_side_length(self):
        return 2.0 * self.box_half_width

    @property
    def box_area(self):
        return self.box_side_length ** 2

    @property
    def box_perimeter(self):
        return 4.0 * self.box_side_length

    @property
    def coating_area(self):
        return math.pi * (
            self.outer_coating_radius ** 2 - self.core_radius ** 2
        )

    @property
    def resolved_bulk_area(self):
        return self.box_area - math.pi * self.outer_coating_radius ** 2

    @property
    def reduced_bulk_area(self):
        return self.box_area - math.pi * self.core_radius ** 2

    @property
    def core_perimeter(self):
        return 2.0 * math.pi * self.core_radius

    @property
    def coating_bulk_interface_perimeter(self):
        return 2.0 * math.pi * self.outer_coating_radius

    def angular_segments_per_sector(self, radius=None):
        """Return a common integer edge count for each circular sector."""
        if radius is None:
            radius = self.outer_coating_radius
        arc_length = 2.0 * math.pi * radius / self.angular_sectors
        return max(1, int(math.ceil(arc_length / self.interface_size)))

    def validate(self):
        validate_scenario(self.scenario)

        positive_values = {
            "core_radius": self.core_radius,
            "coating_thickness": self.coating_thickness,
            "box_half_width": self.box_half_width,
            "interface_size": self.interface_size,
            "bulk_size": self.bulk_size,
            "transition_start": self.transition_start,
            "transition_end": self.transition_end,
            "geometry_relative_tolerance": self.geometry_relative_tolerance,
            "t_final": self.t_final,
            "dt": self.dt,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive; found {value}.")

        if self.outer_coating_radius >= self.box_half_width:
            raise ValueError(
                "The coated disk must lie strictly inside the square box."
            )
        if self.radial_layers < 8:
            raise ValueError(
                "The resolved coating requires at least eight radial layers."
            )
        if self.angular_sectors < 4:
            raise ValueError("angular_sectors must be at least four.")
        if self.interface_size > self.bulk_size:
            raise ValueError(
                "interface_size must not exceed the far-field bulk_size."
            )
        if self.transition_end <= self.transition_start:
            raise ValueError(
                "transition_end must be greater than transition_start."
            )
        if self.polynomial_degree != 1:
            raise ValueError(
                "Paper 1 uses continuous piecewise-affine elements (degree 1)."
            )
        if self.dt > self.t_final:
            raise ValueError("dt must not exceed t_final.")

        ratio = self.t_final / self.dt
        if abs(ratio - round(ratio)) > 1.0e-10 * max(1.0, abs(ratio)):
            raise ValueError("t_final must be an integer multiple of dt.")

        return self

    def to_manifest(self):
        """Return JSON-safe settings with derived geometric information."""
        self.validate()
        data = asdict(self)
        data.update({
            "outer_coating_radius": self.outer_coating_radius,
            "box_side_length": self.box_side_length,
            "h_over_R": self.coating_thickness / self.core_radius,
            "angular_segments_per_sector":
                self.angular_segments_per_sector(),
            "total_circle_segments": (
                self.angular_sectors
                * self.angular_segments_per_sector()
            ),
            "physical_tags": dict(PHYSICAL_TAGS),
            "scenario_metadata": scenario_metadata(self.scenario),
        })
        return data


DEFAULT_CIRCULAR_CONFIG = CircularCaseConfig()
