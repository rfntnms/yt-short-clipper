# Refactor Parity Checklist

This checklist protects behavior while the codebase is modernized in small passes.

## Public APIs To Keep Stable

- `AutoClipperCore.process(...)`
- `AutoClipperCore.find_highlights_only(...)`
- `AutoClipperCore.find_highlights_with_transcription(...)`
- `AutoClipperCore.process_selected_highlights(...)`
- `AutoClipperCore.process_clip(...)`
- `DownloadService.download_video(...)`
- `DownloadService.download_subtitle_only(...)`
- `DownloadService.download_video_section(...)`
- `DownloadService.parse_srt(...)`
- `ConfigManager.load()`, `ConfigManager.get(...)`, `ConfigManager.set(...)`, `ConfigManager.save()`

## Automated Checks

Run these after each refactor pass:

```bash
python3 -m unittest discover -s tests
python3 -m py_compile app.py clipper_core.py services/download_service.py services/portrait_service.py config/config_manager.py
```

## Current CLI Entrypoints

- `python3 app.py` launches the desktop CustomTkinter app.
- `python3 -m unittest discover -s tests` runs the current parity suite.
- `python3 -m py_compile ...` checks syntax for high-risk modules.
- `pyinstaller build.spec` and `pyinstaller build_macos.spec` are packaging paths and should be treated as separate validation from ordinary refactors.

## Session Data Contract

`AutoClipperCore.find_highlights_only(...)` creates:

- `output/sessions/<YYYYMMDD_HHMMSS>/session_data.json`
- `output/sessions/<YYYYMMDD_HHMMSS>/_temp/`

`session_data.json` must keep these keys stable:

- `session_dir`: string path to the session directory.
- `url`: original YouTube URL used for later section downloads.
- `srt_path`: string path to the downloaded subtitle file.
- `highlights`: list of highlight dicts, each enriched with `transcript_text`.
- `video_info`: metadata from yt-dlp.
- `created_at`: ISO timestamp string.
- `status`: `"highlights_found"` before clip processing, then completed status after selected clips finish.

Legacy sessions may contain `video_path` instead of `url`; refactors must keep that resume path working until a separate migration removes it.

## Clip Output Contract

Each processed clip folder keeps:

- `clip_###/master.mp4`: final rendered clip.
- `clip_###/data.json`: metadata used by Browse, Results, and upload dialogs.

`data.json` must keep these fields:

- `title`
- `hook_text`
- `start_time`
- `end_time`
- `duration_seconds`
- `has_hook`
- `has_captions`
- `has_watermark`
- `has_credit`
- `channel_name`

Temporary files such as `temp_landscape.mp4`, `temp_portrait.mp4`, `temp_hooked.mp4`, `temp_captioned.mp4`, and `temp_before_credit.mp4` are implementation details and should be removed after successful processing.

## Command Construction Parity

FFmpeg command refactors should preserve:

- CPU fallback encoder behavior through `AutoClipperCore._run_ffmpeg_subprocess(...)`.
- `get_hwaccel_args()` placement before FFmpeg inputs.
- `get_video_encoder_args()` output for configured CPU/GPU modes.
- Audio settings used in clip cutting, portrait merge, captions, watermarks, and credits.
- The `-progress pipe:1` usage where progress callbacks depend on it.

yt-dlp command and option refactors should preserve:

- The 720p+ format selector fallback string.
- Current-directory `cookies.txt` preference before app-directory `cookies.txt`.
- Deno `js_runtimes` / `remote_components` behavior when Deno exists.
- FFmpeg subtitle conversion postprocessor for subtitle downloads.
- Module and subprocess branches for video, subtitle-only, and section downloads.

## Manual Smoke Checks

- Launch the desktop app with `python3 app.py`.
- Open Settings, visit every settings card, save one harmless value, and confirm `config.json` updates.
- Load a YouTube URL with subtitles and confirm subtitle language selection still appears.
- Run highlight detection with captions and hooks disabled on a short known video.
- Process one selected highlight and confirm `master.mp4` and `data.json` are created.
- Cancel during processing and confirm the GUI returns to a stable state.
- Resume a current session from Session Browser and confirm selected highlights process.
- Resume a legacy session with `video_path` if a fixture is available.
- Open Browse and Results pages and confirm clip cards still upload/open/play as before.

## Behavior Snapshots To Add Before Larger Refactors

- Golden config migrations for old single-provider config, current multi-provider config, missing performance config, and unknown custom keys.
- Golden SRT parsing and highlight transcript extraction.
- Golden FFmpeg command construction for CPU and GPU fallback paths.
- Golden yt-dlp options for module and subprocess downloads.
- Mocked Whisper/caption tests for audio extraction failure, tiny audio, API failure, and caption burn fallback.
- A tiny fixture video for portrait output dimensions, duration, and audio presence.

## Separate Migration Tasks

Do not mix these with behavior-preserving refactors:

- CustomTkinter-to-web or other frontend architecture migrations.
- Replacing `threading.Thread` with asyncio or another concurrency model.
- Dependency upgrades beyond minimal compatibility fixes.
- AI provider API contract changes.
- Output/session schema changes.
- Removing legacy config or legacy session support.
- Packaging/build-system replacement.
