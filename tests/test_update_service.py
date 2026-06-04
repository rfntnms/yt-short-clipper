import unittest

from services.update_service import build_update_url, compare_versions


class UpdateServiceParityTests(unittest.TestCase):
    def test_compare_versions_preserves_padding_and_invalid_fallback(self):
        self.assertEqual(compare_versions("1.2.1", "1.2.0"), 1)
        self.assertEqual(compare_versions("1.2", "1.2.0"), 0)
        self.assertEqual(compare_versions("1.1.9", "1.2.0"), -1)
        self.assertEqual(compare_versions("invalid", "1.2.0"), 0)

    def test_build_update_url_preserves_query_shape(self):
        self.assertEqual(
            build_update_url("https://example.test/update", "install-1", "1.2.3"),
            "https://example.test/update?installation_id=install-1&app_version=1.2.3",
        )


if __name__ == "__main__":
    unittest.main()
