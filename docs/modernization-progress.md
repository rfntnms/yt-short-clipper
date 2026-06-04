# Modernization Progress

This document tracks the behavior-preserving refactor work so future passes can stay reviewable.

## Completed Passes

- Parity contract expanded in `docs/refactor-parity.md` with session data, clip metadata, command construction, manual smoke checks, and migration boundaries.
- Shared page navigation helpers added in `utils/ui_helpers.py`.
- Repeated `open_github`, `open_discord`, and parent `show_page` methods removed from CustomTkinter page classes.
- `ConfigManager.load()` split into default-shape and normalization helpers while keeping `load/get/set/save` stable.
- `DownloadService` now uses shared helpers for yt-dlp module selection, format selection, cookies lookup, JS runtime options, FFmpeg options, and downloaded subtitle discovery.
- `AutoClipperCore.add_captions_api(...)` delegates to the progress-aware caption path to remove a duplicate workflow while keeping the public method.
- `process_clip(...)` now delegates cut/re-encode, portrait, hook, caption, watermark, final-copy, and credit steps to private helpers.
- Update-check request parsing and version comparison moved to `services/update_service.py`.
- Clearly unused imports were removed from high-risk modules.
- Portrait framing now evaluates multiple detections but locks the crop to one centered speaker candidate at a time, avoiding wide multi-person framing while preserving stable tracking.

## Added Parity Coverage

- Config migration coverage for legacy single-provider config, current multi-provider config, missing performance config, and unknown custom keys.
- Download helper coverage for format selector, Deno options, cookies lookup, and subtitle fallback discovery.
- Clip command coverage for full-video cut versus pre-cut section re-encode shapes.
- Caption fallback coverage for audio extraction failure, tiny audio, Whisper failure, and mocked caption-burn success.
- Update helper coverage for version comparison and update URL shape.
- Portrait candidate selection coverage for centered initial selection, stable speaker lock, delayed speaker switching, and invalid framing-mode fallback.

## Remaining Refactor Passes

- Consolidate old direct portrait methods only after fixture-video parity checks exist; their legacy OpenCV/MediaPipe behavior may differ from the active progress route.
- Extract app subtitle loading, thumbnail loading, processing runner construction, and session resume orchestration from `app.py`.
- Add fixture-video tests for portrait output dimensions, duration, and audio presence.
- Refresh README architecture once the remaining app and clipper extractions land.

## Still Separate Migration Tasks

- CustomTkinter-to-web migration.
- Replacing `threading.Thread` with another concurrency model.
- Dependency upgrades.
- AI provider API changes.
- Output/session schema changes.
- Removing legacy session or config support.
- Packaging/build-system replacement.
