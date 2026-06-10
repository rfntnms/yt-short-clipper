"""Smoke test: run orchestrator through a mocked pipeline end-to-end.

Verifies: run_job returns COMPLETED, data.json exists and is valid JSON with
correct status, and run_job_streaming yields the expected events.

Usage:
    cd /path/to/yt-short-clipper
    . venv/bin/activate
    python tests/smoke_orchestrator.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is importable
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.orchestrator import (
    JobConfig,
    JobStatus,
    PipelineError,
    run_job,
    run_job_streaming,
)

def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    output_dir = tmp / "output"
    output_dir.mkdir()

    config = {
        "output_dir": str(output_dir),
        "llm": {"model": "gpt-4"},
        "transcription": {"model": "whisper-1"},
    }

    hl = MagicMock()
    hl.start = 0.0
    hl.end = 3.0
    hl.hook_text = "smoke hook"
    hl.score = 5

    words = [{"word": "smoke", "start": 0.0, "end": 1.0}]

    errors: list[str] = []

    # ── Smoke 1: run_job completes and writes data.json ────────────────────
    print("[SMOKE 1] run_job with mocked pipeline…")
    job = JobConfig(id="smoke_1", url="https://youtu.be/smoke", config=config)

    with patch("pipeline.orchestrator.downloader.download") as mock_dl, \
         patch("pipeline.orchestrator.transcriber.transcribe") as mock_ts, \
         patch("pipeline.orchestrator.highlight_detector.find_highlights") as mock_hd, \
         patch("pipeline.orchestrator.video_processor.cut") as mock_cut, \
         patch("pipeline.orchestrator.video_processor.convert_to_portrait") as mock_cv, \
         patch("pipeline.orchestrator.caption_generator.generate_and_burn") as mock_cg:

        mock_dl.return_value = (Path("/tmp/fake.mp4"), None)
        mock_ts.return_value = words
        mock_hd.return_value = [hl]
        mock_cut.return_value = str(output_dir / "clip_raw.mp4")
        mock_cv.return_value = str(output_dir / "clip_portrait.mp4")
        mock_cg.return_value = str(output_dir / "clip_final.mp4")

        result = run_job(job)

    if result.status != JobStatus.COMPLETED:
        errors.append(f"Expected COMPLETED, got {result.status}")
    data_file = output_dir / "smoke_1" / "data.json"
    if not data_file.exists():
        errors.append("data.json not found")
    else:
        data = json.loads(data_file.read_text())
        if data["status"] != "COMPLETED":
            errors.append(f"data.json status={data['status']}")
        if len(data.get("clips", [])) != 1:
            errors.append(f"Expected 1 clip, got {len(data.get('clips', []))}")

    if not errors:
        print("  PASS")
    else:
        print(f"  FAIL: {'; '.join(errors)}")

    # ── Smoke 2: run_job_streaming yields events ──────────────────────────
    print("[SMOKE 2] run_job_streaming …")
    stream_errors: list[str] = []
    stream_output = Path(tempfile.mkdtemp()) / "output"
    stream_output.mkdir()
    stream_cfg = {**config, "output_dir": str(stream_output)}

    with patch("pipeline.orchestrator.downloader.download") as mock_dl, \
         patch("pipeline.orchestrator.transcriber.transcribe") as mock_ts, \
         patch("pipeline.orchestrator.highlight_detector.find_highlights") as mock_hd, \
         patch("pipeline.orchestrator.video_processor.cut") as mock_cut, \
         patch("pipeline.orchestrator.video_processor.convert_to_portrait") as mock_cv, \
         patch("pipeline.orchestrator.caption_generator.generate_and_burn") as mock_cg:

        mock_dl.return_value = (Path("/tmp/fake.mp4"), None)
        mock_ts.return_value = words
        mock_hd.return_value = [hl]
        mock_cut.return_value = str(stream_output / "clip_raw.mp4")
        mock_cv.return_value = str(stream_output / "clip_portrait.mp4")
        mock_cg.return_value = str(stream_output / "clip_final.mp4")

        events = list(run_job_streaming("https://youtu.be/stream", stream_cfg))

    if events[0]["status"] != "PENDING":
        stream_errors.append(f"First event not PENDING: {events[0]}")
    if events[-1]["status"] != "COMPLETED":
        stream_errors.append(f"Last event not COMPLETED: {events[-1]}")
    if not stream_errors:
        print("  PASS")
    else:
        print(f"  FAIL: {'; '.join(stream_errors)}")

    # ── Smoke 3: error writes FAILED data.json ────────────────────────────
    print("[SMOKE 3] run_job error path writes FAILED data.json …")
    from pipeline.downloader import DownloadError
    fail_cfg = {**config}
    fail_job = JobConfig(id="smoke_fail", url="bad_url", config=fail_cfg)
    fail_errors: list[str] = []

    with patch("pipeline.orchestrator.downloader.download") as mock_dl:
        mock_dl.side_effect = DownloadError("bad url")
        run_job(fail_job)

    fail_data_file = output_dir / "smoke_fail" / "data.json"
    if not fail_data_file.exists():
        fail_errors.append("FAILED data.json not written")
    else:
        fdata = json.loads(fail_data_file.read_text())
        if fdata["status"] != "FAILED":
            fail_errors.append(f"data.json status={fdata['status']}")

    if not fail_errors:
        print("  PASS")
    else:
        print(f"  FAIL: {'; '.join(fail_errors)}")

    # ── Summary ────────────────────────────────────────────────────────────
    all_errors = errors + stream_errors + fail_errors
    print()
    if not all_errors:
        print("ALL SMOKE TESTS PASSED")
        return 0
    else:
        print(f"SMOKE TESTS FAILED: {len(all_errors)} error(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
