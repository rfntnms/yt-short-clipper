### Session Update — 2026-06-07

**Branch:** main
**Status:** In Review / Done

**What I did:**
- Implemented `pipeline/caption_generator.py` mapping word-level JSON timestamps to an Advanced SubStation Alpha (`.ass`) file format with karaoke-style highlight coloring.
- Implemented FFmpeg `subprocess` execution securely invoking the `subtitles` filter with escaped paths to burn `.ass` directly into the video stream.
- Added comprehensive unit tests in `tests/unit/test_caption_generator.py` testing `.ass` generation, FFmpeg invocation, and typed exception `CaptioningError` handling.

**Decisions made:**
- Skipped audio re-extraction natively here because the issue defines taking `word_json` directly as input which is already handled upstream in `transcriber.py`. 
- Overlap styling is handled via inline override formatting `{\c&H0000FFFF&}` for efficiency rather than multi-layered subtitles.

**Tests run:**
- `python3 -m unittest tests/unit/test_caption_generator.py`
- *Result:* Passed (3/3 tests passed, covering ass generation, FFmpeg execution arguments, and failure handling).

**Blockers:**
- None.

**Next session should start with:**
- RFN-26 — `pipeline/orchestrator.py — Sequential Flow & Generator`
