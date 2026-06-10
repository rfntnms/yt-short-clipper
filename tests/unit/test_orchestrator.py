"""Tests for pipeline/orchestrator.py — full mocked pipeline + smoke test."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.orchestrator import (
    JobConfig,
    JobResult,
    JobStatus,
    PipelineError,
    run_job,
    run_job_streaming,
)
from pipeline.downloader import DownloadError
from pipeline.transcriber import TranscriptionError
from pipeline.highlight_detector import HighlightDetectionError
from pipeline.video_processor import VideoProcessingError


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture
def base_config(temp_output_dir: Path) -> dict:
    return {
        "output_dir": str(temp_output_dir),
        "llm": {"model": "gpt-4"},
        "transcription": {"model": "whisper-1"},
    }


@pytest.fixture
def sample_highlight() -> MagicMock:
    hl = MagicMock()
    hl.start = 1.0
    hl.end = 5.0
    hl.hook_text = "test hook"
    hl.score = 8
    return hl


@pytest.fixture
def sample_words() -> list[dict]:
    return [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
    ]


# ── RED: run_job success path ────────────────────────────────────────────────


class TestRunJobSuccess:
    @patch("pipeline.orchestrator.downloader.download")
    @patch("pipeline.orchestrator.transcriber.transcribe")
    @patch("pipeline.orchestrator.highlight_detector.find_highlights")
    @patch("pipeline.orchestrator.video_processor.cut")
    @patch("pipeline.orchestrator.video_processor.convert_to_portrait")
    def test_completes_and_writes_data_json(
        self,
        mock_convert, mock_cut, mock_find, mock_transcribe, mock_download,
        base_config: dict,
        temp_output_dir: Path,
        sample_highlight: MagicMock,
        sample_words: list[dict],
    ) -> None:
        job = JobConfig(id="job_ok", url="https://youtu.be/ok", config=base_config)

        mock_download.return_value = (Path("/tmp/fake.mp4"), None)
        mock_transcribe.return_value = sample_words
        mock_find.return_value = [sample_highlight]
        mock_cut.return_value = str(temp_output_dir / "job_ok" / "clip_01_raw.mp4")
        mock_convert.return_value = str(
            temp_output_dir / "job_ok" / "clip_01_portrait.mp4"
        )

        result = run_job(job)

        assert result.status == JobStatus.COMPLETED
        assert len(result.clips) == 1
        assert result.clips[0]["start"] == 1.0
        assert result.clips[0]["end"] == 5.0
        assert result.clips[0]["hook_text"] == "test hook"
        assert result.clips[0]["score"] == 8

        data_file = temp_output_dir / "job_ok" / "data.json"
        assert data_file.exists()
        data = json.loads(data_file.read_text())
        assert data["id"] == "job_ok"
        assert data["status"] == "COMPLETED"
        assert data["url"] == "https://youtu.be/ok"
        assert len(data["clips"]) == 1

    @patch("pipeline.orchestrator.downloader.download")
    @patch("pipeline.orchestrator.transcriber.transcribe")
    @patch("pipeline.orchestrator.highlight_detector.find_highlights")
    @patch("pipeline.orchestrator.video_processor.cut")
    @patch("pipeline.orchestrator.video_processor.convert_to_portrait")
    def test_no_highlights_still_completes(
        self,
        mock_convert, mock_cut, mock_find, mock_transcribe, mock_download,
        base_config: dict,
        temp_output_dir: Path,
        sample_words: list[dict],
    ) -> None:
        job = JobConfig(id="job_empty", url="https://youtu.be/empty", config=base_config)

        mock_download.return_value = (Path("/tmp/fake.mp4"), None)
        mock_transcribe.return_value = sample_words
        mock_find.return_value = []

        result = run_job(job)

        assert result.status == JobStatus.COMPLETED
        assert result.clips == []
        data = json.loads(
            (temp_output_dir / "job_empty" / "data.json").read_text()
        )
        assert data["status"] == "COMPLETED"
        assert data["clips"] == []

    @patch("pipeline.orchestrator.downloader.download")
    @patch("pipeline.orchestrator.transcriber.transcribe")
    @patch("pipeline.orchestrator.highlight_detector.find_highlights")
    @patch("pipeline.orchestrator.video_processor.cut")
    @patch("pipeline.orchestrator.video_processor.convert_to_portrait")
    def test_multiple_highlights(
        self,
        mock_convert, mock_cut, mock_find, mock_transcribe, mock_download,
        base_config: dict,
        temp_output_dir: Path,
        sample_words: list[dict],
    ) -> None:
        job = JobConfig(id="job_multi", url="https://youtu.be/multi", config=base_config)

        hl1 = MagicMock()
        hl1.start = 1.0
        hl1.end = 3.0
        hl1.hook_text = "first"
        hl1.score = 9

        hl2 = MagicMock()
        hl2.start = 5.0
        hl2.end = 8.0
        hl2.hook_text = "second"
        hl2.score = 7

        mock_download.return_value = (Path("/tmp/fake.mp4"), None)
        mock_transcribe.return_value = sample_words
        mock_find.return_value = [hl1, hl2]
        mock_cut.side_effect = [
            str(temp_output_dir / "job_multi" / "clip_01_raw.mp4"),
            str(temp_output_dir / "job_multi" / "clip_02_raw.mp4"),
        ]
        mock_convert.side_effect = [
            str(temp_output_dir / "job_multi" / "clip_01_portrait.mp4"),
            str(temp_output_dir / "job_multi" / "clip_02_portrait.mp4"),
        ]

        result = run_job(job)

        assert result.status == JobStatus.COMPLETED
        assert len(result.clips) == 2
        assert result.clips[0]["index"] == 1
        assert result.clips[1]["index"] == 2


# ── RED: exception mapping ────────────────────────────────────────────────────


class TestExceptionMapping:
    @patch("pipeline.orchestrator.downloader.download")
    def test_download_error_mapped(
        self, mock_download, base_config: dict, temp_output_dir: Path
    ) -> None:
        job = JobConfig(id="job_dl_err", url="bad", config=base_config)
        mock_download.side_effect = DownloadError("invalid url")

        with pytest.raises(PipelineError) as exc_info:
            run_job(job)

        assert "download failed" in str(exc_info.value)
        assert exc_info.value.step == "download"

        data = json.loads(
            (temp_output_dir / "job_dl_err" / "data.json").read_text()
        )
        assert data["status"] == "FAILED"
        assert "download failed" in data["error"]

    @patch("pipeline.orchestrator.downloader.download")
    @patch("pipeline.orchestrator.transcriber.transcribe")
    def test_transcription_error_mapped(
        self, mock_transcribe, mock_download, base_config: dict, temp_output_dir: Path
    ) -> None:
        job = JobConfig(id="job_ts_err", url="https://youtu.be/x", config=base_config)
        mock_download.return_value = (Path("/tmp/fake.mp4"), None)
        mock_transcribe.side_effect = TranscriptionError("api down")

        with pytest.raises(PipelineError) as exc_info:
            run_job(job)

        assert "transcription failed" in str(exc_info.value)
        assert exc_info.value.step == "transcription"

    @patch("pipeline.orchestrator.downloader.download")
    @patch("pipeline.orchestrator.transcriber.transcribe")
    @patch("pipeline.orchestrator.highlight_detector.find_highlights")
    def test_highlight_error_mapped(
        self, mock_find, mock_transcribe, mock_download, 
        base_config: dict, temp_output_dir: Path, sample_words: list[dict]
    ) -> None:
        job = JobConfig(id="job_hl_err", url="https://youtu.be/x", config=base_config)
        mock_download.return_value = (Path("/tmp/fake.mp4"), None)
        mock_transcribe.return_value = sample_words
        mock_find.side_effect = HighlightDetectionError("bad json")

        with pytest.raises(PipelineError) as exc_info:
            run_job(job)

        assert "highlight_detection failed" in str(exc_info.value)
        assert exc_info.value.step == "highlight_detection"

    @patch("pipeline.orchestrator.downloader.download")
    @patch("pipeline.orchestrator.transcriber.transcribe")
    @patch("pipeline.orchestrator.highlight_detector.find_highlights")
    @patch("pipeline.orchestrator.video_processor.cut")
    def test_video_processing_error_mapped(
        self,
        mock_cut, mock_find, mock_transcribe, mock_download,
        base_config: dict,
        temp_output_dir: Path,
        sample_highlight: MagicMock,
        sample_words: list[dict],
    ) -> None:
        job = JobConfig(id="job_vp_err", url="https://youtu.be/x", config=base_config)
        mock_download.return_value = (Path("/tmp/fake.mp4"), None)
        mock_transcribe.return_value = sample_words
        mock_find.return_value = [sample_highlight]
        mock_cut.side_effect = VideoProcessingError("ffmpeg crash")

        with pytest.raises(PipelineError) as exc_info:
            run_job(job)

        assert "video_processing failed" in str(exc_info.value)
        assert exc_info.value.step == "video_processing"

    @patch("pipeline.orchestrator.downloader.download")
    def test_unknown_error_mapped_to_pipeline(
        self, mock_download, base_config: dict, temp_output_dir: Path
    ) -> None:
        job = JobConfig(id="job_unk_err", url="https://youtu.be/x", config=base_config)
        mock_download.side_effect = RuntimeError("unexpected")

        with pytest.raises(PipelineError) as exc_info:
            run_job(job)

        assert "pipeline failed" in str(exc_info.value)
        assert exc_info.value.step == "pipeline"


# ── RED: data.json lifecycle ─────────────────────────────────────────────────


class TestDataJsonLifecycle:
    @patch("pipeline.orchestrator.downloader.download")
    @patch("pipeline.orchestrator.transcriber.transcribe")
    @patch("pipeline.orchestrator.highlight_detector.find_highlights")
    @patch("pipeline.orchestrator.video_processor.cut")
    @patch("pipeline.orchestrator.video_processor.convert_to_portrait")
    def test_data_json_written_on_each_status_change(
        self,
        mock_convert, mock_cut, mock_find, mock_transcribe, mock_download,
        base_config: dict,
        temp_output_dir: Path,
        sample_highlight: MagicMock,
        sample_words: list[dict],
    ) -> None:
        job = JobConfig(id="job_lc", url="https://youtu.be/lc", config=base_config)

        mock_download.return_value = (Path("/tmp/fake.mp4"), None)
        mock_transcribe.return_value = sample_words
        mock_find.return_value = [sample_highlight]
        mock_cut.return_value = str(temp_output_dir / "job_lc" / "clip_01_raw.mp4")
        mock_convert.return_value = str(
            temp_output_dir / "job_lc" / "clip_01_portrait.mp4"
        )

        run_job(job)

        data_file = temp_output_dir / "job_lc" / "data.json"
        assert data_file.exists()
        data = json.loads(data_file.read_text())
        assert data["status"] == "COMPLETED"
        assert "created_at" in data
        assert "updated_at" in data

    @patch("pipeline.orchestrator.downloader.download")
    def test_data_json_written_even_on_failure(
        self, mock_download, base_config: dict, temp_output_dir: Path
    ) -> None:
        job = JobConfig(id="job_fail_w", url="bad", config=base_config)
        mock_download.side_effect = DownloadError("fail")

        with pytest.raises(PipelineError):
            run_job(job)

        data_file = temp_output_dir / "job_fail_w" / "data.json"
        assert data_file.exists()
        data = json.loads(data_file.read_text())
        assert data["status"] == "FAILED"
        assert data["error"] is not None


# ── RED: run_job_streaming ────────────────────────────────────────────────────


class TestRunJobStreaming:
    @patch("pipeline.orchestrator.downloader.download")
    @patch("pipeline.orchestrator.transcriber.transcribe")
    @patch("pipeline.orchestrator.highlight_detector.find_highlights")
    @patch("pipeline.orchestrator.video_processor.cut")
    @patch("pipeline.orchestrator.video_processor.convert_to_portrait")
    def test_yields_pending_then_completed(
        self,
        mock_convert, mock_cut, mock_find, mock_transcribe, mock_download,
        base_config: dict,
        temp_output_dir: Path,
        sample_highlight: MagicMock,
        sample_words: list[dict],
    ) -> None:
        url = "https://youtu.be/stream"

        mock_download.return_value = (Path("/tmp/fake.mp4"), None)
        mock_transcribe.return_value = sample_words
        mock_find.return_value = [sample_highlight]
        mock_cut.return_value = str(temp_output_dir / "clip_01_raw.mp4")
        mock_convert.return_value = str(
            temp_output_dir / "clip_01_portrait.mp4"
        )

        events = list(run_job_streaming(url, base_config))

        assert events[0]["status"] == "PENDING"
        assert "job_id" in events[0]
        assert events[-1]["status"] == "COMPLETED"
        assert "clips" in events[-1]

    @patch("pipeline.orchestrator.downloader.download")
    def test_yields_failed_on_error(
        self, mock_download, base_config: dict, temp_output_dir: Path
    ) -> None:
        url = "https://youtu.be/stream_fail"
        mock_download.side_effect = DownloadError("bad url")

        gen = run_job_streaming(url, base_config)
        events = []
        with pytest.raises(PipelineError):
            for evt in gen:
                events.append(evt)

        assert events[0]["status"] == "PENDING"
        assert events[-1]["status"] == "FAILED"
        assert "error" in events[-1]


# ── RED: JobResult serialization ──────────────────────────────────────────────


class TestJobResult:
    def test_to_dict_includes_status_value(self) -> None:
        result = JobResult(id="x", url="y", status=JobStatus.COMPLETED)
        d = result.to_dict()
        assert d["status"] == "COMPLETED"
        assert isinstance(d["clips"], list)

    def test_default_timestamps_present(self) -> None:
        result = JobResult(id="x", url="y", status=JobStatus.PENDING)
        d = result.to_dict()
        assert d["created_at"] is not None
        assert d["updated_at"] is not None


# ── RED: PipelineError ────────────────────────────────────────────────────────


class TestPipelineError:
    def test_stores_step_and_original(self) -> None:
        original = ValueError("root cause")
        err = PipelineError("mapped", step="download", original=original)
        assert err.step == "download"
        assert err.original is original
        assert str(err) == "mapped"

    def test_re_pipelineerror_passthrough(self) -> None:
        existing = PipelineError("already mapped", step="download")
        mapped = PipelineError("wrapper", step="other", original=existing)
        assert mapped.step == "other"
        assert mapped.original is existing
