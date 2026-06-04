import unittest

from services.portrait_service import DetectionResult, EngineManager


class PortraitTrackingTests(unittest.TestCase):
    def test_center_speaker_initial_pick_prefers_centered_subject(self):
        manager = EngineManager(settings={"speaker_framing_mode": "center_speaker"})
        detections = [
            DetectionResult(x_center=100, confidence=1.0, width=200, height=400),
            DetectionResult(x_center=520, confidence=0.8, width=120, height=240),
        ]

        selected, pending_x, switch_count, switched = manager._select_candidate(
            detections, None, None, 0, orig_w=1000, orig_h=600
        )

        self.assertEqual(selected.x_center, 520)
        self.assertIsNone(pending_x)
        self.assertEqual(switch_count, 0)
        self.assertFalse(switched)

    def test_center_speaker_does_not_switch_on_one_brief_better_candidate(self):
        manager = EngineManager(settings={"speaker_framing_mode": "center_speaker"})
        detections = [
            DetectionResult(x_center=220, confidence=0.8, width=120, height=240),
            DetectionResult(x_center=760, confidence=1.0, width=220, height=440),
        ]

        selected, pending_x, switch_count, switched = manager._select_candidate(
            detections, locked_x=220, pending_x=None, stable_switch_count=0, orig_w=1000, orig_h=600
        )

        self.assertEqual(selected.x_center, 220)
        self.assertEqual(pending_x, 760)
        self.assertEqual(switch_count, 1)
        self.assertFalse(switched)

    def test_center_speaker_switches_after_repeated_better_candidate(self):
        manager = EngineManager(settings={"speaker_framing_mode": "center_speaker"})
        detections = [
            DetectionResult(x_center=220, confidence=0.8, width=120, height=240),
            DetectionResult(x_center=760, confidence=1.0, width=220, height=440),
        ]

        selected, pending_x, switch_count, switched = manager._select_candidate(
            detections, locked_x=220, pending_x=760, stable_switch_count=2, orig_w=1000, orig_h=600
        )

        self.assertEqual(selected.x_center, 760)
        self.assertIsNone(pending_x)
        self.assertEqual(switch_count, 0)
        self.assertTrue(switched)

    def test_invalid_framing_mode_defaults_to_center_speaker(self):
        manager = EngineManager(settings={"speaker_framing_mode": "wide_group"})

        self.assertEqual(manager.framing_mode, "center_speaker")


if __name__ == "__main__":
    unittest.main()
