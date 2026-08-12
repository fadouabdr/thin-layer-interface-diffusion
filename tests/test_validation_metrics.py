import unittest

import numpy as np

from src.validation_metrics import (
    backward_euler_cumulative,
    empirical_rates,
    normalized_balance_residual,
    number_of_time_steps,
    relative_time_l2_error,
    validate_matching_time_grids,
)


class ValidationMetricTests(unittest.TestCase):
    def test_exact_time_grid(self):
        self.assertEqual(
            number_of_time_steps(1.0, 1.0e-3),
            1000,
        )

        with self.assertRaises(ValueError):
            number_of_time_steps(1.0, 0.3)

    def test_backward_euler_cumulative(self):
        times = np.array([0.0, 0.1, 0.2])
        flux = np.array([99.0, 2.0, 3.0])
        expected = np.array([0.0, 0.2, 0.5])
        np.testing.assert_allclose(
            backward_euler_cumulative(times, flux),
            expected,
        )

    def test_relative_time_l2_error(self):
        times = np.array([0.0, 0.5, 1.0])
        reference = np.array([0.0, 2.0, 2.0])
        first = np.array([0.0, 1.0, 1.0])
        self.assertAlmostEqual(
            relative_time_l2_error(
                times,
                first,
                reference,
            ),
            0.5,
        )

    def test_normalized_balance(self):
        self.assertAlmostEqual(
            normalized_balance_residual(
                residual=1.0e-6,
                mass_rate=1.0,
                interface_inflow=2.0,
                outer_outflow=1.0,
            ),
            2.5e-7,
        )

    def test_matching_grids(self):
        grid = np.array([0.0, 0.1, 0.2])
        np.testing.assert_allclose(
            validate_matching_time_grids(grid, grid.copy()),
            grid,
        )

        with self.assertRaises(ValueError):
            validate_matching_time_grids(
                grid,
                np.array([0.0, 0.2, 0.1]),
            )

    def test_empirical_rates(self):
        h = [0.08, 0.04, 0.02, 0.01]
        errors = [value * value for value in h]
        rates = empirical_rates(h, errors)
        self.assertTrue(np.isnan(rates[0]))
        np.testing.assert_allclose(rates[1:], 2.0)


if __name__ == "__main__":
    unittest.main()
