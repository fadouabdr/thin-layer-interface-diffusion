"""Environment-independent checks for circular figure reconstruction rules."""

import unittest

import numpy as np

from scripts.make_circular_publication_figures import (
    BULK_CELL,
    COATING_CELL,
    _selected_contour_levels,
    _snapshot_title,
)


class CircularFigureTests(unittest.TestCase):
    def test_contour_levels_are_restricted_to_selected_subdomain(self):
        values = np.asarray([0.0, 0.2, 0.8, 1.0])
        cells = np.asarray([[0, 1, 2], [1, 2, 3]])
        tags = np.asarray([COATING_CELL, BULK_CELL])
        levels = np.asarray([0.1, 0.5, 0.9])

        coating = _selected_contour_levels(
            values, cells, tags, COATING_CELL, levels
        )
        bulk = _selected_contour_levels(values, cells, tags, BULK_CELL, levels)

        np.testing.assert_allclose(coating, [0.1, 0.5])
        np.testing.assert_allclose(bulk, [0.5, 0.9])

    def test_expected_physical_cell_tags_are_distinct(self):
        self.assertNotEqual(COATING_CELL, BULK_CELL)

    def test_snapshot_title_reports_time_without_release_percentage(self):
        title = _snapshot_title(0.065)
        self.assertEqual(title, r"$t=0.065$")
        self.assertNotIn("%", title)


if __name__ == "__main__":
    unittest.main()
