import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def install_fake_video_dependencies():
    cv2 = types.ModuleType("cv2")
    cv2.__spec__ = ModuleSpec("cv2", loader=None)
    numpy = types.ModuleType("numpy")
    numpy.__spec__ = ModuleSpec("numpy", loader=None)
    openai = types.ModuleType("openai")
    openai.__spec__ = ModuleSpec("openai", loader=None)
    openai.OpenAI = object
    openai.APIConnectionError = Exception
    openai.RateLimitError = Exception
    openai.APIStatusError = Exception

    sys.modules.setdefault("cv2", cv2)
    sys.modules.setdefault("numpy", numpy)
    sys.modules.setdefault("openai", openai)


install_fake_video_dependencies()

from clipper_core import AutoClipperCore


class CaptionFallbackParityTests(unittest.TestCase):
    def make_core(self) -> AutoClipperCore:
        core = AutoClipperCore.__new__(AutoClipperCore)
        core.ffmpeg_path = "ffmpeg"
        core.log_messages = []
        core.log = lambda message: core.log_messages.append(message)
        core.get_hwaccel_args = lambda: ["-hwaccel", "auto"]
        core.get_video_encoder_args = lambda: ["-c:v", "libx264", "-preset", "fast"]
        core.report_tokens = lambda *_args: None
        core.log_ffmpeg_command = lambda *_args: None
        return core

    def test_audio_extraction_failure_copies_without_captions(self):
        core = self.make_core()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            input_path.write_bytes(b"original video")

            with patch("clipper_core.subprocess.run", return_value=SimpleNamespace(returncode=1, stderr="failed")):
                core.add_captions_api_with_progress(str(input_path), str(output_path))

            self.assertEqual(output_path.read_bytes(), b"original video")
            self.assertIn("  Warning: Audio extraction failed", core.log_messages)

    def test_tiny_audio_file_copies_without_captions(self):
        core = self.make_core()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            input_path.write_bytes(b"original video")

            with patch("clipper_core.subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")):
                core.add_captions_api_with_progress(str(input_path), str(output_path))

            self.assertEqual(output_path.read_bytes(), b"original video")
            self.assertIn("  Warning: Audio file too small or missing", core.log_messages)

    def test_whisper_failure_copies_without_captions(self):
        core = self.make_core()
        core._whisper_transcribe_words_api = lambda _audio_file: (_ for _ in ()).throw(Exception("api down"))

        def fake_run(cmd, **_kwargs):
            if cmd[-1] == "-":
                return SimpleNamespace(returncode=0, stderr="Duration: 00:00:10.00")
            Path(cmd[-1]).write_bytes(b"x" * 2048)
            return SimpleNamespace(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            input_path.write_bytes(b"original video")

            with patch("clipper_core.subprocess.run", side_effect=fake_run):
                core.add_captions_api_with_progress(str(input_path), str(output_path))

            self.assertEqual(output_path.read_bytes(), b"original video")
            self.assertIn("  Warning: Whisper API error: api down", core.log_messages)

    def test_caption_success_reports_expected_progress_range(self):
        core = self.make_core()
        core._whisper_transcribe_words_api = lambda _audio_file: [{"word": "hello", "start": 0.0, "end": 0.5}]
        core.create_ass_subtitle_capcut = (
            lambda _transcript, ass_file, _offset=0: Path(ass_file).write_text("[ASS]", encoding="utf-8")
        )

        def fake_run(cmd, **_kwargs):
            if cmd[-1] == "-":
                return SimpleNamespace(returncode=0, stderr="Duration: 00:00:10.00")
            Path(cmd[-1]).write_bytes(b"x" * 2048)
            return SimpleNamespace(returncode=0, stderr="")

        def fake_ffmpeg(_cmd, _duration, progress_callback):
            progress_callback(1.0)

        core.run_ffmpeg_with_progress = fake_ffmpeg

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            input_path.write_bytes(b"original video")
            progress_updates = []

            with patch("clipper_core.subprocess.run", side_effect=fake_run):
                core.add_captions_api_with_progress(
                    str(input_path),
                    str(output_path),
                    progress_callback=progress_updates.append,
                )

            self.assertEqual(progress_updates[:4], [0.1, 0.2, 0.3, 0.5])
            self.assertEqual(progress_updates[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
