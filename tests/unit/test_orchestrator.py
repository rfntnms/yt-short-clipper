from pathlib import Path

from pipeline.highlight_detector import Highlight
from pipeline.orchestrator import JobConfig, run_job, run_job_streaming


def _config() -> dict:
    return {
        "llm": {"api_key": "test", "base_url": "http://example.test/v1", "model": "test-model"},
        "transcription": {"api_key": "test", "base_url": "http://example.test/v1", "model": "whisper-1"},
        "portrait": {"face_backend": "opencv"},
    }


def test_run_job_streaming_uses_existing_subtitle(monkeypatch, tmp_path):
    subtitle_path = tmp_path / "source.en.srt"
    subtitle_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello world\n", encoding="utf-8")

    calls = {"transcribe": 0}

    (tmp_path / "source.mp4").touch()
    monkeypatch.setattr(
        "pipeline.orchestrator.download",
        lambda url, output_dir, cookies_path=None: (tmp_path / "source.mp4", subtitle_path),
    )
    monkeypatch.setattr("pipeline.orchestrator.transcribe", lambda video_path, config: calls.__setitem__("transcribe", calls["transcribe"] + 1) or [])
    monkeypatch.setattr("pipeline.orchestrator.find_highlights", lambda srt_text, config: [Highlight(0.0, 2.0, "hello", 9)])
    # monkeypatch.setattr("pipeline.orchestrator.cut", lambda video_path, highlight, output_path: Path(output_path).write_text("raw") or output_path)
    # monkeypatch.setattr("pipeline.orchestrator.convert_to_portrait", lambda clip_path, config, output_path: Path(output_path).write_text("portrait") or output_path)
    monkeypatch.setattr("pipeline.orchestrator._generate_and_burn", lambda clip_path, word_json, config: f"{clip_path}_captioned.mp4")

    job = JobConfig(url="https://youtu.be/test", job_id="job1", output_dir=tmp_path, config=_config())
    statuses = list(run_job_streaming(job))

    assert statuses[-1].status == "DONE"
    assert statuses[-1].output_clips == [str(tmp_path / "job1" / "clip_01_portrait.mp4_captioned.mp4")]
    assert calls["transcribe"] == 1  # called once for captions because existing SRT has no word JSON
    assert (tmp_path / "job1" / "data.json").exists()


def test_run_job_streaming_force_transcribes(monkeypatch, tmp_path):
    subtitle_path = tmp_path / "source.en.srt"
    subtitle_path.write_text("subtitle", encoding="utf-8")
    words = [{"word": "hello", "start": 0.0, "end": 0.4}]

    (tmp_path / "source.mp4").touch()
    (tmp_path / "source.srt").touch()
    monkeypatch.setattr("pipeline.orchestrator.download", lambda url, output_dir, cookies_path=None: (tmp_path / "source.mp4", tmp_path / "source.srt"))
    monkeypatch.setattr("pipeline.orchestrator.transcribe", lambda video_path, config: words)
    monkeypatch.setattr("pipeline.orchestrator.find_highlights", lambda text, config: [Highlight(0.0, 1.0, "hello", 10)])
    # monkeypatch.setattr("pipeline.orchestrator.cut", lambda video_path, highlight, output_path: Path(output_path).write_text("raw") or output_path)
    # monkeypatch.setattr("pipeline.orchestrator.convert_to_portrait", lambda clip_path, config, output_path: Path(output_path).write_text("portrait") or output_path)
    monkeypatch.setattr("pipeline.orchestrator._generate_and_burn", lambda clip_path, word_json, config: clip_path)

    job = JobConfig(url="https://youtu.be/test", job_id="job2", output_dir=tmp_path, config=_config(), force_transcribe=True)
    status = run_job(job)

    assert status.status == "DONE"


def test_run_job_streaming_reports_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.orchestrator.download", lambda url, output_dir, cookies_path=None: (_ for _ in ()).throw(RuntimeError("download failed")))

    job = JobConfig(url="https://youtu.be/test", job_id="job3", output_dir=tmp_path, config=_config())
    statuses = list(run_job_streaming(job))

    assert statuses[-1].status == "FAILED"
    assert statuses[-1].error is not None
    assert "download failed" in statuses[-1].error


def test_multi_highlight_slices_word_data_per_clip(monkeypatch, tmp_path):
    """Regression: clips 2+ must NOT reuse clip 1's word-level data."""
    words = [
        {"word": "one", "start": 0.0, "end": 0.5},
        {"word": "two", "start": 0.5, "end": 1.0},
        {"word": "three", "start": 1.0, "end": 1.5},
        {"word": "four", "start": 2.5, "end": 3.0},
        {"word": "five", "start": 3.0, "end": 3.5},
        {"word": "six", "start": 3.5, "end": 4.0},
    ]
    highlights = [
        Highlight(0.0, 2.0, "segment-1", 9),
        Highlight(2.5, 4.0, "segment-2", 8),
    ]
    captured = {"call_index": 0, "clip_word_data": []}

    def fake_generate_and_burn(clip_path, word_json, config):
        captured["clip_word_data"].append(word_json)
        return f"{clip_path}_captioned.mp4"

    (tmp_path / "source.mp4").touch()
    (tmp_path / "source.srt").touch()
    monkeypatch.setattr("pipeline.orchestrator.download", lambda url, output_dir, cookies_path=None: (tmp_path / "source.mp4", tmp_path / "source.srt"))
    monkeypatch.setattr("pipeline.orchestrator.transcribe", lambda video_path, config: words)
    monkeypatch.setattr("pipeline.orchestrator.find_highlights", lambda text, config: highlights)
    # monkeypatch.setattr("pipeline.orchestrator.cut", lambda video_path, highlight, output_path: Path(output_path).write_text("raw") or output_path)
    # monkeypatch.setattr("pipeline.orchestrator.convert_to_portrait", lambda clip_path, config, output_path: Path(output_path).write_text("portrait") or output_path)
    monkeypatch.setattr("pipeline.orchestrator._generate_and_burn", fake_generate_and_burn)

    job = JobConfig(url="https://youtu.be/multi", job_id="multi-job", output_dir=tmp_path, config=_config())
    statuses = list(run_job_streaming(job))

    assert statuses[-1].status == "DONE"
    assert len(statuses[-1].output_clips) == 2
    assert len(captured["clip_word_data"]) == 2

    # Clip 1 (0.0-2.0): words at 0.0-0.5, 0.5-1.0, 1.0-1.5 → offset by 0
    clip1 = captured["clip_word_data"][0]
    assert clip1 == [
        {"word": "one", "start": 0.0, "end": 0.5},
        {"word": "two", "start": 0.5, "end": 1.0},
        {"word": "three", "start": 1.0, "end": 1.5},
    ], f"Clip 1 word data mismatch: {clip1}"

    # Clip 2 (2.5-4.0): words at 2.5-3.0, 3.0-3.5, 3.5-4.0 → offset by 2.5
    clip2 = captured["clip_word_data"][1]
    assert clip2 == [
        {"word": "four", "start": 0.0, "end": 0.5},
        {"word": "five", "start": 0.5, "end": 1.0},
        {"word": "six", "start": 1.0, "end": 1.5},
    ], f"Clip 2 word data mismatch: {clip2}"
