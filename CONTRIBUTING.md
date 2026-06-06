# Contributing to YT-Short-Clipper (v2 Migration)

> **Important:** This repository is currently undergoing a major architecture rewrite (v1 CustomTkinter → v2 Gradio/Docker). All development must follow the strict rules defined in the `.hermes/` directory and `AGENTS.md`.

## 🚨 Migration Safety Rules

To prevent regressions and ensure a clean v2 architecture:
1. **No Code Changes Without an Issue:** Every commit must be tied to a Linear issue (RFN-XXX).
2. **No ADR Overrides:** Decisions logged in `DECISIONS.md` are final. Do not introduce new frameworks (e.g., FastAPI, Celery, asyncio) without an accepted ADR update.
3. **Isolate v1 Code:** Do not refactor legacy v1 files (`app.py`, `clipper_core.py`, `pages/`, etc.). Write new code in the new v2 structure (`pipeline/`, `providers/`, `batch/`).
4. **Test Before Merge:** All new pipeline modules must have unit tests covering success and failure paths.

## 🌿 Branching Strategy

We use a strict feature-branching model tied to Linear:

* **Format:** `<issue-id>-<short-description>` (lowercase, hyphenated)
* **Example:** `rfn-17-structured-logging`
* **Main Branch:** `main` is protected. No direct commits. All changes must go through a Pull Request.

### Pull Request Rules
1. Title must start with the Linear Issue ID (e.g., `RFN-17: Add structured logger`).
2. PR description must include a checklist of the issue's Acceptance Criteria.
3. CI/CD (if applicable) must pass before merge.
4. Squash and merge is preferred to keep the `main` history clean.

## 🤖 AI Agent Workflow

If you are an AI Agent (Hermes, Claude, etc.) working on this repo:
1. Always read `CONTEXT.md` first.
2. Read `.hermes/LINEAR_WORKFLOW.md` for project management rules.
3. Strictly follow `AGENTS.md` for architectural constraints.

## 📚 Technical Standards

* **Language:** Python 3.11+
* **Style:** PEP 8 + Type Hints (strictly required for all public functions in `pipeline/`).
* **Docs:** Google-style docstrings.
* **Exceptions:** No bare `Exception`. Use typed exceptions (e.g., `DownloadError`, `TranscriptionError`).
* **Environment:** No secrets in code. Use `providers/config_manager.py` to access API keys.