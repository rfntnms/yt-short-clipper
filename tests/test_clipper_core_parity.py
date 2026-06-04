import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

for dependency in ("cv2", "numpy", "openai"):
    if importlib.util.find_spec(dependency) is None:
        raise unittest.SkipTest(f"{dependency} is not installed")

from clipper_core import AutoClipperCore


class AutoClipperCoreHelperParityTests(unittest.TestCase):
    def make_core(self) -> AutoClipperCore:
        core = AutoClipperCore.__new__(AutoClipperCore)
        core.watermark_settings = {"enabled": True}
        core.credit_watermark_settings = {"enabled": True}
        core.channel_name = "Source Channel"
        core.progress_updates = []
        core.log_messages = []
        core.set_progress = lambda status, progress: core.progress_updates.append((status, progress))
        core.log = lambda message: core.log_messages.append(message)
        core.ffmpeg_path = "ffmpeg"
        core.get_hwaccel_args = lambda: ["-hwaccel", "auto"]
        core.get_video_encoder_args = lambda: ["-c:v", "libx264", "-preset", "fast"]
        core.report_tokens = lambda *_args: None
        core.log_ffmpeg_command = lambda *_args: None
        return core

    def test_clip_total_steps_preserves_existing_options(self):
        core = self.make_core()

        self.assertEqual(core._get_clip_total_steps(add_captions=False, add_hook=False), 2)
        self.assertEqual(core._get_clip_total_steps(add_captions=True, add_hook=False), 3)
        self.assertEqual(core._get_clip_total_steps(add_captions=False, add_hook=True), 3)
        self.assertEqual(core._get_clip_total_steps(add_captions=True, add_hook=True), 4)

    def test_report_clip_progress_preserves_existing_scale(self):
        core = self.make_core()

        core._report_clip_progress(
            index=2,
            total_clips=4,
            total_steps=4,
            step_name="Adding captions...",
            step_num=2,
            sub_progress=0.5,
        )

        self.assertEqual(core.progress_updates[0][0], "Clip 2/4: Adding captions... (50%)")
        self.assertAlmostEqual(core.progress_updates[0][1], 0.54375)

    def test_landscape_clip_command_preserves_cut_and_precut_shapes(self):
        core = self.make_core()

        cut_cmd = core._build_landscape_clip_command(
            "source.mp4",
            "00:00:01.000",
            "00:01:01.000",
            Path("temp_landscape.mp4"),
            pre_cut=False,
        )
        precut_cmd = core._build_landscape_clip_command(
            "section.mp4",
            "00:00:01.000",
            "00:01:01.000",
            Path("temp_landscape.mp4"),
            pre_cut=True,
        )

        self.assertIn("-ss", cut_cmd)
        self.assertIn("-to", cut_cmd)
        self.assertNotIn("-ss", precut_cmd)
        self.assertNotIn("-to", precut_cmd)
        self.assertEqual(cut_cmd[-3:], ["-progress", "pipe:1", "temp_landscape.mp4"])
        self.assertEqual(precut_cmd[-3:], ["-progress", "pipe:1", "temp_landscape.mp4"])

    def test_write_clip_metadata_preserves_existing_fields(self):
        core = self.make_core()
        highlight = {
            "title": "Clip title",
            "hook_text": "Hook title",
            "start_time": "00:00:01,000",
            "end_time": "00:01:01,000",
            "duration_seconds": 60,
        }

        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp)
            core._write_clip_metadata(clip_dir, highlight, add_hook=True, add_captions=False)

            metadata = json.loads((clip_dir / "data.json").read_text(encoding="utf-8"))

        self.assertEqual(
            metadata,
            {
                "title": "Clip title",
                "hook_text": "Hook title",
                "start_time": "00:00:01,000",
                "end_time": "00:01:01,000",
                "duration_seconds": 60,
                "has_hook": True,
                "has_captions": False,
                "has_watermark": True,
                "has_credit": True,
                "channel_name": "Source Channel",
            },
        )


if __name__ == "__main__":
    unittest.main()
