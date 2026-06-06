"""Unit tests for pipeline/highlight_detector.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.highlight_detector import (
    Highlight,
    HighlightDetectionError,
    find_highlights,
)


# ── fixture: write a fake SYSTEM_PROMPT.md to a temp dir ──────────────────
@pytest.fixture
def fake_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the module's _PROMPT_PATH to a tmp file and return the text."""
    prompt_text = "You are a highlight detection assistant."
    p = tmp_path / "SYSTEM_PROMPT.md"
    p.write_text(prompt_text, encoding="utf-8")

    import pipeline.highlight_detector as hd

    monkeypatch.setattr(hd, "_PROMPT_PATH", p)
    return prompt_text


# ── test 1: valid JSON response → sorted by score descending ─────────────
@patch("pipeline.highlight_detector.get_client")
def test_find_highlights_valid_response(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    mock_client = mock_get_client.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        [
                            {"start": 0.0, "end": 30.0, "hook_text": "weak one", "score": 3},
                            {"start": 60.0, "end": 90.0, "hook_text": "great one", "score": 9},
                            {"start": 100.0, "end": 130.0, "hook_text": "okay one", "score": 6},
                        ]
                    )
                )
            )
        ]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}
    result = find_highlights("1\n00:00:00,000 --> 00:00:05,000\nhello world", config)

    assert len(result) == 3
    assert all(isinstance(h, Highlight) for h in result)
    # Sorted by score descending
    assert [h.score for h in result] == [9, 6, 3]
    assert result[0].hook_text == "great one"
    assert result[0].start == 60.0
    assert result[0].end == 90.0


# ── test 2: empty list response → empty list, no raise ───────────────────
@patch("pipeline.highlight_detector.get_client")
def test_empty_highlights(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    mock_client = mock_get_client.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="[]"))]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}
    result = find_highlights("transcript", config)

    assert result == []


# ── test 3: retry on malformed JSON ──────────────────────────────────────
@patch("pipeline.highlight_detector.get_client")
def test_retry_on_malformed_json(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    """All 3 attempts (1 initial + 2 retries) return malformed JSON → raise."""
    mock_client = mock_get_client.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="not json at all"))]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}

    with pytest.raises(HighlightDetectionError, match="Malformed LLM response"):
        find_highlights("transcript", config)

    # Initial + 2 retries = 3 total calls.
    assert mock_client.chat.completions.create.call_count == 3


# ── test 4: retry succeeds on second attempt ─────────────────────────────
@patch("pipeline.highlight_detector.get_client")
def test_retry_succeeds_on_second_attempt(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    """First call malformed, second call valid → return valid highlights."""
    mock_client = mock_get_client.return_value
    valid_json = json.dumps(
        [{"start": 0.0, "end": 30.0, "hook_text": "the good bit", "score": 8}]
    )
    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="garbage {"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content=valid_json))]),
    ]

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}
    result = find_highlights("transcript", config)

    assert len(result) == 1
    assert result[0].hook_text == "the good bit"
    assert mock_client.chat.completions.create.call_count == 2


# ── test 5: JSON missing required field raises ───────────────────────────
@patch("pipeline.highlight_detector.get_client")
def test_retry_on_missing_required_field(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    mock_client = mock_get_client.return_value
    bad_json = json.dumps([{"start": 0.0, "end": 30.0, "hook_text": "no score field"}])
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=bad_json))]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}

    with pytest.raises(HighlightDetectionError, match="missing fields"):
        find_highlights("transcript", config)

    # All retries exhausted before raise.
    assert mock_client.chat.completions.create.call_count == 3


# ── test 6: raw LLM response is logged ───────────────────────────────────
@patch("pipeline.highlight_detector.get_client")
def test_raw_llm_response_logged(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    mock_client = mock_get_client.return_value
    raw = json.dumps([{"start": 0.0, "end": 30.0, "hook_text": "x", "score": 5}])
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=raw))]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}
    with patch("pipeline.highlight_detector.logger") as mock_logger:
        find_highlights("transcript", config)

    # The raw response should have been logged at DEBUG level.
    debug_calls = [c for c in mock_logger.debug.call_args_list]
    assert any(raw in str(call) for call in debug_calls), (
        f"Expected raw LLM response in logger.debug calls, got: {debug_calls}"
    )


# ── test 7: LLM API error raises HighlightDetectionError ─────────────────
@patch("pipeline.highlight_detector.get_client")
def test_raises_on_api_error(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    mock_client = mock_get_client.return_value
    mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized")

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}

    with pytest.raises(HighlightDetectionError, match="LLM API call failed"):
        find_highlights("transcript", config)


# ── test 8: missing system prompt file raises ────────────────────────────
@patch("pipeline.highlight_detector.get_client")
def test_raises_when_prompt_missing(
    mock_get_client: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If SYSTEM_PROMPT.md is missing, raise HighlightDetectionError."""
    import pipeline.highlight_detector as hd

    monkeypatch.setattr(hd, "_PROMPT_PATH", tmp_path / "does_not_exist.md")

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}
    with pytest.raises(HighlightDetectionError, match="System prompt not found"):
        find_highlights("transcript", config)

    # Should not even reach the LLM.
    mock_get_client.assert_not_called()


# ── test 9: markdown code-fence in response is stripped ──────────────────
@patch("pipeline.highlight_detector.get_client")
def test_strips_markdown_code_fences(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    mock_client = mock_get_client.return_value
    fenced = "```json\n" + json.dumps(
        [{"start": 0.0, "end": 30.0, "hook_text": "fenced", "score": 7}]
    ) + "\n```"
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=fenced))]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}
    result = find_highlights("transcript", config)

    assert len(result) == 1
    assert result[0].hook_text == "fenced"


# ── test 11: single-line code fence like ```[]``` ────────────────────────
@patch("pipeline.highlight_detector.get_client")
def test_handles_single_line_fence(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    """Single-line fence like ```[]``` should not raise ValueError."""
    mock_client = mock_get_client.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="```[]```"))]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}
    result = find_highlights("transcript", config)

    assert result == []
    # Must have been parsed on first attempt — no retries.
    assert mock_client.chat.completions.create.call_count == 1


# ── test 12: non-numeric field values trigger retry ──────────────────────
@patch("pipeline.highlight_detector.get_client")
def test_retry_on_invalid_field_types(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    """start=\"abc\" raises ValueError in float() → retried, not raw exception."""
    mock_client = mock_get_client.return_value
    bad_json = json.dumps([{"start": "abc", "end": 30.0, "hook_text": "x", "score": 5}])
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=bad_json))]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}

    with pytest.raises(HighlightDetectionError, match="Malformed LLM response"):
        find_highlights("transcript", config)

    # Retries exhausted.
    assert mock_client.chat.completions.create.call_count == 3


# ── test 13: negative start raises HighlightDetectionError → retry ──────
@patch("pipeline.highlight_detector.get_client")
def test_retry_on_negative_start(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    mock_client = mock_get_client.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=json.dumps(
                    [{"start": -5.0, "end": 30.0, "hook_text": "x", "score": 5}]
                )
            )
        )]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}

    with pytest.raises(HighlightDetectionError, match="invalid time range"):
        find_highlights("transcript", config)
    assert mock_client.chat.completions.create.call_count == 3


# ── test 14: end <= start raises HighlightDetectionError → retry ─────────
@patch("pipeline.highlight_detector.get_client")
def test_retry_on_end_before_start(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    mock_client = mock_get_client.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=json.dumps(
                    [{"start": 30.0, "end": 0.0, "hook_text": "x", "score": 5}]
                )
            )
        )]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}
    with pytest.raises(HighlightDetectionError, match="invalid time range"):
        find_highlights("transcript", config)
    assert mock_client.chat.completions.create.call_count == 3


# ── test 15: score outside 1-10 raises HighlightDetectionError → retry ───
@patch("pipeline.highlight_detector.get_client")
def test_retry_on_invalid_score(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    mock_client = mock_get_client.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=json.dumps(
                    [{"start": 0.0, "end": 30.0, "hook_text": "x", "score": 999}]
                )
            )
        )]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}
    with pytest.raises(HighlightDetectionError, match="invalid score"):
        find_highlights("transcript", config)
    assert mock_client.chat.completions.create.call_count == 3


# ── test 16: NaN timestamp passes float() but not math.isfinite → retry ──
@patch("pipeline.highlight_detector.get_client")
def test_retry_on_nan_timestamp(
    mock_get_client: MagicMock, fake_prompt: str
) -> None:
    """NaN passes float() and NaN<0 is False, so must catch via isfinite."""
    mock_client = mock_get_client.return_value
    # json.loads accepts NaN literal
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content='[{"start": NaN, "end": 30.0, "hook_text": "x", "score": 5}]'
            )
        )]
    )

    config = {"llm": {"base_url": "http://localhost:9999/v1", "model": "gpt-4", "api_key": "***"}}
    with pytest.raises(HighlightDetectionError, match="invalid time range"):
        find_highlights("transcript", config)
    assert mock_client.chat.completions.create.call_count == 3


# ── test 17: no direct openai import (uses get_client only) ─────────────
def test_no_direct_openai_import() -> None:
    """Static guard: the module must not import openai directly."""
    import pipeline.highlight_detector as hd

    source = Path(hd.__file__).read_text(encoding="utf-8")
    assert "import openai" not in source
    assert "from openai" not in source
