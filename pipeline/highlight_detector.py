"""LLM-based highlight detection from transcript.

Third step in the orchestrator pipeline.
Sends transcript + system prompt to LLM, parses JSON response into
List[Highlight] sorted by score. ADR-005 compliant.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from providers.ai_client import get_client
from utils.logger import logger

# Path to the default system prompt for highlight detection.
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "SYSTEM_PROMPT.md"

# Maximum number of corrective retries on malformed JSON (ADR-005).
_MAX_RETRIES = 2

# Schema fields required in each highlight object.
_REQUIRED_FIELDS = {"start", "end", "hook_text", "score"}


class HighlightDetectionError(Exception):
    """Raised when highlight detection fails after retries."""


@dataclass
class Highlight:
    """A single detected highlight segment."""

    start: float
    end: float
    hook_text: str
    score: int


def _load_system_prompt() -> str:
    """Load the system prompt from the prompts directory."""
    if not _PROMPT_PATH.exists():
        raise HighlightDetectionError(
            f"System prompt not found: {_PROMPT_PATH}"
        )
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _parse_highlights(raw: Any) -> list[Highlight]:
    """Parse LLM JSON response into a list of Highlight dataclasses.

    Returns an empty list if the response is an empty array.
    Raises HighlightDetectionError if the JSON is malformed or missing fields.
    """
    # Strip markdown code fences if present — some models emit ```json ... ```
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            # Remove opening fence (optionally with language tag)
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        data = json.loads(text)
    else:
        data = raw

    if not isinstance(data, list):
        raise HighlightDetectionError(
            f"Expected JSON array from LLM, got {type(data).__name__}"
        )

    highlights: list[Highlight] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise HighlightDetectionError(
                f"Highlight entry {i} is not a dict: {type(entry).__name__}"
            )
        missing = _REQUIRED_FIELDS - set(entry.keys())
        if missing:
            raise HighlightDetectionError(
                f"Highlight entry {i} missing fields: {sorted(missing)}"
            )
        highlights.append(
            Highlight(
                start=float(entry["start"]),
                end=float(entry["end"]),
                hook_text=str(entry["hook_text"]),
                score=int(entry["score"]),
            )
        )

    return highlights


def find_highlights(
    srt_text: str,
    config: dict[str, Any],
) -> list[Highlight]:
    """Detect highlight segments from a transcript using an LLM.

    Args:
        srt_text: The full transcript text (SRT or plain) from the transcriber.
        config: Full application config dict. Uses config["llm"].

    Returns:
        List of Highlight objects sorted by score descending.
        An empty list is valid (no highlights found).

    Raises:
        HighlightDetectionError: After 2 retries on malformed JSON, or if
            the system prompt file is missing.
    """
    system_prompt = _load_system_prompt()

    llm_cfg: dict[str, Any] = config.get("llm", {})
    model: str = llm_cfg.get("model", "gpt-4")
    client = get_client({"llm": llm_cfg})

    user_message = f"Here is the transcript:\n\n{srt_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    last_error: HighlightDetectionError | None = None

    for attempt in range(1 + _MAX_RETRIES):
        logger.info(
            "Highlight detection attempt %d/%d (model=%s)",
            attempt + 1,
            1 + _MAX_RETRIES,
            model,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
            )
        except Exception as exc:
            raise HighlightDetectionError(
                f"LLM API call failed: {exc}"
            ) from exc

        raw_content = response.choices[0].message.content
        logger.debug("Raw LLM response (attempt %d): %s", attempt + 1, raw_content)

        try:
            highlights = _parse_highlights(raw_content)
        except (json.JSONDecodeError, HighlightDetectionError) as exc:
            last_error = HighlightDetectionError(
                f"Malformed LLM response (attempt {attempt + 1}): {exc}"
            )
            logger.warning("JSON parse failed, attempt %d: %s", attempt + 1, exc)

            if attempt < _MAX_RETRIES:
                # Append a corrective follow-up for the next retry.
                messages.append({
                    "role": "assistant",
                    "content": raw_content,
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response was not valid JSON. "
                        "Return ONLY a valid JSON array of highlight objects. "
                        "Each object must have: start, end, hook_text, score. "
                        "No markdown, no code blocks, no explanation."
                    ),
                })
            continue

        # Success — sort by score descending and return.
        highlights.sort(key=lambda h: h.score, reverse=True)
        logger.info("Detected %d highlights", len(highlights))
        return highlights

    raise last_error  # type: ignore[misc]


__all__ = ["find_highlights", "Highlight", "HighlightDetectionError"]
