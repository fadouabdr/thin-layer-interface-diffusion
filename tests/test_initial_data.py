import unittest

from src.initial_data import (
    prepared_bulk_value,
    prepared_coating_flux,
    prepared_full_transformed_value,
    prepared_interface_concentration,
    prepared_release_flux,
    prepared_robin_flux,
)
from src.validation_config import (
    CONSTANT_SCENARIO,
    K_IN,
    K_OUT,
    TIME_DEPENDENT_SCENARIO,
    c_int,
)


class PreparedInitialDataTests(unittest.TestCase):
    def test_flux_identities(self):
        for scenario in (
            CONSTANT_SCENARIO,
            TIME_DEPENDENT_SCENARIO,
        ):
            for h in (0.08, 0.04, 0.02, 0.01):
                expected = prepared_release_flux(h, scenario)
                self.assertAlmostEqual(
                    prepared_coating_flux(h, scenario),
                    expected,
                    places=12,
                )
                self.assertAlmostEqual(
                    prepared_robin_flux(h, scenario),
                    expected,
                    places=12,
                )

    def test_partition_values_and_continuity(self):
        h = 0.02
        scenario = TIME_DEPENDENT_SCENARIO
        interface = prepared_interface_concentration(
            h,
            scenario,
        )
        q_inner = prepared_full_transformed_value(
            -h,
            h,
            scenario,
        )
        q_coating_interface = prepared_full_transformed_value(
            -1.0e-15,
            h,
            scenario,
        )
        q_bulk_interface = prepared_bulk_value(
            0.0,
            h,
            scenario,
        )

        self.assertAlmostEqual(
            K_OUT * q_inner,
            K_IN * c_int(0.0, scenario),
            places=12,
        )
        self.assertAlmostEqual(
            q_coating_interface,
            interface,
            places=12,
        )
        self.assertAlmostEqual(
            q_bulk_interface,
            interface,
            places=12,
        )
        self.assertAlmostEqual(
            prepared_bulk_value(1.0, h, scenario),
            0.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
