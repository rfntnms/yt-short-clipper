import os
import unittest
from unittest.mock import patch, MagicMock
from pipeline.caption_generator import generate_ass_content, generate_and_burn, CaptioningError

class TestCaptionGenerator(unittest.TestCase):

    def setUp(self):
        self.sample_word_json = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
            {"word": "this", "start": 1.0, "end": 1.2},
            {"word": "is", "start": 1.2, "end": 1.4},
            {"word": "a", "start": 1.4, "end": 1.5},
            {"word": "test", "start": 1.5, "end": 2.0},
        ]

    def test_ass_file_generation_matches_schema(self):
        config = {
            "caption_style": {
                "font_name": "Roboto",
                "font_size": 20,
                "highlight_color": "&H0000FFFF"
            }
        }
        ass_text = generate_ass_content(self.sample_word_json, config)
        
        self.assertIn("[Script Info]", ass_text)
        self.assertIn("[V4+ Styles]", ass_text)
        self.assertIn("Roboto,20", ass_text)
        self.assertIn("[Events]", ass_text)
        
        # Check that it breaks into groups of 5 words
        self.assertIn("Dialogue: 0,0:00:00.00,0:00:00.50,Default,,0,0,0,,{\\c&H0000FFFF&}Hello{\\c&H00FFFFFF&} world this is a", ass_text)
        # Check second group
        self.assertIn("Dialogue: 0,0:00:01.50,0:00:02.00,Default,,0,0,0,,{\\c&H0000FFFF&}test{\\c&H00FFFFFF&}", ass_text)

    @patch("pipeline.caption_generator.subprocess.run")
    @patch("pipeline.caption_generator.os.path.exists")
    def test_ffmpeg_subtitle_burn_command(self, mock_exists, mock_run):
        mock_exists.return_value = True # Pretend clip_path exists
        
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc
        
        output = generate_and_burn("fake_clip.mp4", self.sample_word_json, {})
        
        self.assertEqual(output, "fake_clip_captioned.mp4")
        mock_run.assert_called_once()
        
        # Check command arguments
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-i", cmd)
        self.assertIn("fake_clip.mp4", cmd)
        self.assertIn("-vf", cmd)
        
        # Subtitles filter check
        vf_arg = cmd[cmd.index("-vf") + 1]
        self.assertTrue(vf_arg.startswith("ass="))

    @patch("pipeline.caption_generator.subprocess.run")
    @patch("pipeline.caption_generator.os.path.exists")
    def test_ffmpeg_failure_raises_error(self, mock_exists, mock_run):
        mock_exists.side_effect = lambda p: True if p.endswith(".mp4") else False
        
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "FFmpeg fatal error"
        mock_run.return_value = mock_proc
        
        with self.assertRaises(CaptioningError) as context:
            generate_and_burn("fake_clip.mp4", self.sample_word_json, {})
            
        self.assertIn("FFmpeg captioning", str(context.exception))

    @patch("pipeline.caption_generator.subprocess.run")
    @patch("pipeline.caption_generator.os.path.exists")
    def test_ffmpeg_timeout_raises_error(self, mock_exists, mock_run):
        import subprocess
        mock_exists.side_effect = lambda p: True if p.endswith(".mp4") else False
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1800)
        
        with self.assertRaises(CaptioningError) as context:
            generate_and_burn("fake_clip.mp4", self.sample_word_json, {"ffmpeg_timeout_sec": 1800})
            
        self.assertIn("FFmpeg captioning", str(context.exception))

if __name__ == '__main__':
    unittest.main()
