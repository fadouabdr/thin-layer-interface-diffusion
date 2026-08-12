import unittest

from src.validation_audit import (
    audit_covers_thicknesses,
    is_valid_coverage_extension,
    normalized_thicknesses,
)


class ValidationAuditTests(unittest.TestCase):
    def test_all_four_thicknesses_are_required_for_canonical_coverage(self):
        required = [0.08, 0.04, 0.02, 0.01]
        self.assertFalse(audit_covers_thicknesses([0.08, 0.01], required))
        self.assertTrue(audit_covers_thicknesses(required, required))

    def test_extension_may_add_but_not_remove_thicknesses(self):
        self.assertTrue(is_valid_coverage_extension(
            [0.08, 0.01],
            [0.08, 0.04, 0.02, 0.01],
        ))
        self.assertFalse(is_valid_coverage_extension(
            [0.08, 0.04, 0.02, 0.01],
            [0.08, 0.01],
        ))

    def test_duplicate_thickness_is_rejected(self):
        with self.assertRaises(ValueError):
            normalized_thicknesses([0.08, 0.08])


if __name__ == "__main__":
    unittest.main()
