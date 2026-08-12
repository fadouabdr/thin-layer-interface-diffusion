import math
import unittest
from dataclasses import replace

import numpy as np

from src.circular_config import DEFAULT_CIRCULAR_CONFIG
from src.circular_mesh_utils import (
    _tag_counts,
    infer_radial_layer_count,
    physical_tag_set,
    triangle_quality_summary,
)


class CircularConfigTests(unittest.TestCase):
    def test_default_geometry_matches_specification(self):
        config = DEFAULT_CIRCULAR_CONFIG.validate()
        self.assertAlmostEqual(config.core_radius, 0.25)
        self.assertAlmostEqual(config.coating_thickness, 0.01)
        self.assertAlmostEqual(
            config.coating_area,
            math.pi * (0.26 ** 2 - 0.25 ** 2),
        )
        self.assertAlmostEqual(config.core_perimeter, 0.5 * math.pi)
        self.assertEqual(config.radial_layers, 8)
        self.assertAlmostEqual(
            config.coating_thickness / config.core_radius,
            0.04,
        )

    def test_too_few_radial_layers_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(DEFAULT_CIRCULAR_CONFIG, radial_layers=7).validate()

    def test_invalid_box_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(DEFAULT_CIRCULAR_CONFIG, box_half_width=0.25).validate()

    def test_radial_layer_inference(self):
        config = DEFAULT_CIRCULAR_CONFIG
        axis_x = np.linspace(
            config.core_radius,
            config.outer_coating_radius,
            config.radial_layers + 1,
        )
        coordinates = np.column_stack([axis_x, np.zeros_like(axis_x)])
        layer_count, spacings = infer_radial_layer_count(
            coordinates,
            config,
        )
        self.assertEqual(layer_count, config.radial_layers)
        self.assertTrue(np.allclose(
            spacings,
            config.coating_thickness / config.radial_layers,
        ))

    def test_equilateral_triangle_has_unit_quality(self):
        coordinates = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0],
        ])
        connectivity = np.array([[0, 1, 2]])
        summary = triangle_quality_summary(coordinates, connectivity)
        self.assertAlmostEqual(summary["quality_min"], 1.0)
        self.assertAlmostEqual(summary["quality_mean"], 1.0)

    def test_unmarked_facet_sentinel_is_not_a_physical_tag(self):
        values = np.array([-1, -1, 0, 10, 20, 40], dtype=np.int64)
        self.assertEqual(physical_tag_set(values), {10, 20, 40})

    def test_unmarked_facet_counts_are_reported_explicitly(self):
        values = np.array([-1, -1, 10, 20, 20], dtype=np.int64)
        self.assertEqual(
            _tag_counts(values),
            {"unmarked": 2, "10": 1, "20": 2},
        )


if __name__ == "__main__":
    unittest.main()
