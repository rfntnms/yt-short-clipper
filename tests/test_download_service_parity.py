import tempfile
import unittest
import os
from pathlib import Path

from services.download_service import DownloadService


SAMPLE_SRT = """1
00:00:01,000 --> 00:00:03,500
Hello
world.

2
00:00:04,000 --> 00:00:05,000
Second line.

3
00:00:06,000 --> 00:00:07,000
Outside range.
"""


class DownloadServiceParityTests(unittest.TestCase):
    def make_service(self, root: Path) -> DownloadService:
        return DownloadService(
            temp_dir=root / "temp",
            output_dir=root / "output",
            cookies_file="",
            subtitle_language="id",
            ytdlp_path="yt-dlp",
            log_callback=None,
            progress_callback=None,
            is_cancelled_callback=lambda: False,
            performance_settings={},
        )

    def test_parse_srt_preserves_timestamped_transcript_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt_path = root / "sample.srt"
            srt_path.write_text(SAMPLE_SRT, encoding="utf-8")

            transcript = self.make_service(root).parse_srt(str(srt_path))

            self.assertEqual(
                transcript,
                "\n".join(
                    [
                        "[00:00:01,000 - 00:00:03,500] Hello world.",
                        "[00:00:04,000 - 00:00:05,000] Second line.",
                        "[00:00:06,000 - 00:00:07,000] Outside range.",
                    ]
                ),
            )

    def test_extract_transcript_for_highlight_includes_overlapping_subtitles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt_path = root / "sample.srt"
            srt_path.write_text(SAMPLE_SRT, encoding="utf-8")
            highlight = {
                "start_time": "00:00:03,000",
                "end_time": "00:00:04,250",
            }

            text = self.make_service(root).extract_transcript_for_highlight(str(srt_path), highlight)

            self.assertEqual(text, "Hello world. Second line.")

    def test_download_option_helpers_keep_expected_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = self.make_service(root)

            self.assertEqual(
                service._format_selector(),
                "bestvideo[height>=720][height<=2160]+bestaudio/best[height>=720][height<=2160]/bestvideo+bestaudio/best",
            )

            options = {}
            service._apply_js_runtime_options(options, str(root / "missing-deno"))
            self.assertEqual(options, {})

            deno = root / "deno"
            deno.write_text("", encoding="utf-8")
            service._apply_js_runtime_options(options, str(deno))
            self.assertEqual(options["js_runtimes"], {"deno": {"path": str(deno)}})
            self.assertEqual(options["remote_components"], ["ejs:github"])

    def test_cookie_lookup_prefers_current_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                (root / "cookies.txt").write_text("# cookies", encoding="utf-8")
                service = self.make_service(root)

                self.assertEqual(service._find_cookies_path(), Path("cookies.txt"))
                self.assertEqual(service._require_cookies_path(), Path("cookies.txt"))
            finally:
                os.chdir(old_cwd)

    def test_find_downloaded_srt_falls_back_to_available_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = self.make_service(root)
            service.temp_dir.mkdir()
            fallback = service.temp_dir / "source.en.srt"
            fallback.write_text(SAMPLE_SRT, encoding="utf-8")

            self.assertEqual(service._find_downloaded_srt(), fallback)


if __name__ == "__main__":
    unittest.main()
