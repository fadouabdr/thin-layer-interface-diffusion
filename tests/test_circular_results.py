import unittest

import numpy as np

from src.circular_results import (
    affine_profile_values,
    release_fraction_indices,
    relative_max_mismatch,
    validate_snapshot_indices,
)


class CircularResultTests(unittest.TestCase):
    def test_affine_coating_profile_matches_both_traces(self):
        values = affine_profile_values(
            2.0,
            np.asarray([0.5, 1.0, 1.5]),
            np.asarray([0.0, 0.5, 1.0]),
        )
        np.testing.assert_allclose(values, [2.0, 1.5, 1.5])

    def test_affine_coating_profile_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            affine_profile_values(1.0, [0.0], [1.01])

    def test_release_fraction_crossings(self):
        times = np.arange(6, dtype=float)
        cumulative = np.array([0.0, 0.05, 0.12, 0.51, 0.91, 1.0])
        indices = release_fraction_indices(times, cumulative)
        np.testing.assert_array_equal(indices, [2, 3, 4])

    def test_signed_negative_release_crossings(self):
        times = np.arange(5, dtype=float)
        cumulative = np.array([0.0, -0.1, -0.55, -0.8, -1.0])
        indices = release_fraction_indices(times, cumulative)
        np.testing.assert_array_equal(indices, [1, 2, 4])

    def test_zero_terminal_release_is_rejected(self):
        with self.assertRaises(ValueError):
            release_fraction_indices([0.0, 1.0], [0.0, 0.0])

    def test_snapshot_indices_allow_repeated_smoke_samples(self):
        indices = validate_snapshot_indices([1, 1, 2], 3)
        np.testing.assert_array_equal(indices, [1, 1, 2])

    def test_relative_max_mismatch(self):
        mismatch = relative_max_mismatch([1.0, 2.0], [1.0, 1.0])
        self.assertAlmostEqual(mismatch, 0.5)


if __name__ == "__main__":
    unittest.main()
