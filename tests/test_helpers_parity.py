import unittest

from utils.helpers import extract_video_id, hex_to_rgb, parse_timestamp


class HelpersParityTests(unittest.TestCase):
    def test_parse_timestamp_accepts_comma_and_dot_milliseconds(self):
        self.assertEqual(parse_timestamp("00:01:02,500"), 62.5)
        self.assertEqual(parse_timestamp("01:02:03.250"), 3723.25)

    def test_extract_video_id_keeps_existing_url_patterns(self):
        self.assertEqual(
            extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )
        self.assertEqual(
            extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=12"),
            "dQw4w9WgXcQ",
        )
        self.assertIsNone(extract_video_id("https://example.com/nope"))

    def test_hex_to_rgb_preserves_fallback_behavior(self):
        self.assertEqual(hex_to_rgb("#abc"), (170, 187, 204))
        self.assertEqual(hex_to_rgb("#00ff7f"), (0, 255, 127))
        self.assertEqual(hex_to_rgb("not-a-color"), (255, 255, 255))
        self.assertEqual(hex_to_rgb(None), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
