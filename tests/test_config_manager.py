import json
import tempfile
import unittest
from pathlib import Path

from config.ai_provider_config import DEFAULT_HIGHLIGHT_PROMPT
from config.config_manager import ConfigManager


class ConfigManagerParityTests(unittest.TestCase):
    def test_default_config_contains_current_runtime_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ConfigManager(root / "config.json", root / "output")

            self.assertEqual(manager.get("output_dir"), str(root / "output"))
            self.assertEqual(manager.get("system_prompt"), DEFAULT_HIGHLIGHT_PROMPT)
            self.assertEqual(manager.get("temperature"), 1.0)
            self.assertEqual(manager.get("tts_model"), "tts-1")
            self.assertIn("installation_id", manager.config)
            self.assertIn("watermark", manager.config)
            self.assertIn("ai_providers", manager.config)
            self.assertIn("performance", manager.config)
            self.assertEqual(manager.config["performance"]["profile"], "balanced")
            self.assertEqual(manager.config["performance"]["codec"], "h264")
            self.assertEqual(manager.config["performance"]["speaker_framing_mode"], "center_speaker")

    def test_legacy_single_provider_config_migrates_without_dropping_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_file = root / "config.json"
            legacy_config = {
                "api_key": "sk-legacy",
                "base_url": "https://example.test/v1",
                "model": "gpt-legacy",
                "tts_model": "tts-legacy",
                "output_dir": "/already/set",
                "custom_legacy_key": "keep-me",
                "gpu_acceleration": {"enabled": True},
                "face_tracking_mode": "opencv_fast",
            }
            config_file.write_text(json.dumps(legacy_config), encoding="utf-8")

            manager = ConfigManager(config_file, root / "output")

            self.assertEqual(manager.get("custom_legacy_key"), "keep-me")
            self.assertEqual(manager.get("output_dir"), "/already/set")
            providers = manager.get("ai_providers")
            self.assertEqual(providers["highlight_finder"]["api_key"], "sk-legacy")
            self.assertEqual(providers["highlight_finder"]["base_url"], "https://example.test/v1")
            self.assertEqual(providers["highlight_finder"]["model"], "gpt-legacy")
            self.assertEqual(providers["caption_maker"]["model"], "whisper-1")
            self.assertEqual(providers["hook_maker"]["model"], "tts-legacy")
            self.assertTrue(manager.config["performance"]["prefer_gpu"])
            self.assertEqual(manager.config["performance"]["detection_engine"], "opencv_fast")

    def test_current_multi_provider_config_gets_missing_runtime_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_file = root / "config.json"
            current_config = {
                "output_dir": "/already/set",
                "ai_providers": {
                    "highlight_finder": {
                        "base_url": "https://example.test/v1",
                        "api_key": "sk-current",
                        "model": "gpt-current",
                    }
                },
                "performance": {
                    "profile": "fast",
                    "prefer_gpu": True,
                },
                "custom_key": "keep-me",
            }
            config_file.write_text(json.dumps(current_config), encoding="utf-8")

            manager = ConfigManager(config_file, root / "output")

            self.assertEqual(manager.get("custom_key"), "keep-me")
            self.assertEqual(manager.get("output_dir"), "/already/set")
            self.assertEqual(manager.get("ai_providers")["highlight_finder"]["model"], "gpt-current")
            self.assertEqual(manager.get("system_prompt"), DEFAULT_HIGHLIGHT_PROMPT)
            self.assertEqual(manager.config["performance"]["profile"], "fast")
            self.assertEqual(manager.config["performance"]["codec"], "h264")
            self.assertEqual(manager.config["performance"]["speaker_framing_mode"], "center_speaker")
            self.assertTrue(manager.config["performance"]["prefer_gpu"])
            self.assertIn("installation_id", manager.config)

    def test_missing_performance_uses_legacy_gpu_and_face_tracking_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_file = root / "config.json"
            config_file.write_text(
                json.dumps(
                    {
                        "output_dir": "/already/set",
                        "gpu_acceleration": {"enabled": True},
                        "face_tracking_mode": "mediapipe",
                        "ai_providers": {},
                    }
                ),
                encoding="utf-8",
            )

            manager = ConfigManager(config_file, root / "output")

            self.assertEqual(manager.config["performance"]["detection_engine"], "mediapipe")
            self.assertTrue(manager.config["performance"]["prefer_gpu"])
            self.assertEqual(manager.config["performance"]["codec"], "h264")
            self.assertIn("cached_encoder", manager.config["gpu_acceleration"])

    def test_set_persists_config_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_file = root / "config.json"
            manager = ConfigManager(config_file, root / "output")

            manager.set("temperature", 0.5)

            saved = json.loads(config_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["temperature"], 0.5)


if __name__ == "__main__":
    unittest.main()
