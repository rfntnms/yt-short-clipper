# 📊 PROGRESS.md — Status Implementasi YT-Short-Clipper v2

> **Update file ini setiap kali ada perubahan signifikan.**
> Legend: `✅ DONE` · `🔨 WIP` · `⬜ TODO` · `❌ BLOCKED` · `⚠️ NEEDS REVIEW`

---

## 🏗️ Status Per Modul

### Pipeline (`pipeline/`)

| Modul                   | Status | Catatan |
| ----------------------- | ------ | ------- |
| `orchestrator.py`       | ⬜ TODO | —       |
| `downloader.py`         | ✅ DONE | —       |
| `transcriber.py`        | ✅ DONE | —       |
| `highlight_detector.py` | ✅ DONE | —       |
| `video_processor.py`    | ✅ DONE | —       |
| `speaker_layout.py`     | ✅ DONE | Smart crop layout analysis, body-safe crops, SPLIT decision helpers |
| `caption_generator.py`  | ✅ DONE | —       |

### Providers (`providers/`)

| Modul               | Status | Catatan |
| ------------------- | ------ | ------- |
| `ai_client.py`      | ✅ DONE | —       |
| `config_manager.py` | ✅ DONE | —       |

### Utils (`utils/`)

| Modul                 | Status | Catatan |
| --------------------- | ------ | ------- |
| `logger.py`           | ✅ DONE | Structured logger dengan rotating file |
| `gpu_detector.py`     | ✅ DONE | Deteksi CUDA dan HWAccel flags |
| `dependency_check.py` | ✅ DONE | Validasi ffmpeg dan yt-dlp |

### Batch (`batch/`)

| Modul             | Status | Catatan |
| ----------------- | ------ | ------- |
| `job_queue.py`    | ⬜ TODO | —       |
| `batch_runner.py` | ⬜ TODO | —       |

### Root Files

| File                       | Status | Catatan |
| -------------------------- | ------ | ------- |
| `server.py` (Gradio UI)    | ⬜ TODO | —       |
| `scheduler.py`             | ⬜ TODO | —       |
| `Dockerfile`               | ⬜ TODO | —       |
| `docker-compose.yml`       | ⬜ TODO | —       |
| `prompts/SYSTEM_PROMPT.md` | ⬜ TODO | —       |
| `requirements.txt`         | ⬜ TODO | —       |
| `config.json` (template)   | ⬜ TODO | —       |
| `MIGRATION_MAP.md`         | ✅ DONE | Mapping dari audit repo v1   |
| `LINEAR_WORKFLOW.md`       | ✅ DONE | Workflow rules + branching    |

---

## 🐛 Known Bugs & Issues

> Catat bug yang ditemukan di sini, beri nomor urut, dan hapus jika sudah fix.

| # | Deskripsi                  | Modul | Severity | Status |
| - | -------------------------- | ----- | -------- | ------ |
| — | *(belum ada bug tercatat)* | —     | —        | —      |

---

## 🔀 Migrasi dari v1

> Pantau status migrasi komponen dari v1 ke v2.

| Komponen v1                      | Status Migrasi | Diganti oleh                             |
| -------------------------------- | -------------- | ---------------------------------------- |
| `app.py` (CustomTkinter)         | ⬜ TODO         | `server.py` (Gradio)                     |
| `clipper_core.py`                | ⬜ TODO         | `pipeline/` package (5 modul)            |
| `pages/`                         | ⬜ TODO         | Gradio Tabs di `server.py`               |
| `components/ai_provider_card.py` | ⬜ TODO         | `providers/ai_client.py` + Settings Tab  |
| `utils/dependency_manager.py`    | ⬜ TODO         | `utils/dependency_check.py` + Dockerfile |
| `build.spec`                     | ⬜ TODO         | `Dockerfile` + `docker-compose.yml`      |

---

## 📈 Coverage Test

| Area                             | Coverage | Target |
| -------------------------------- | -------- | ------ |
| `pipeline/highlight_detector.py` | 95%      | 80%    |
| `pipeline/speaker_layout.py`     | 90%      | 80%    |
| `pipeline/video_processor.py`    | 94%      | 60%    |
| `batch/job_queue.py`             | 0%       | 90%    |
| `providers/`                     | 0%       | 70%    |

---

## 📅 Changelog Progress

> Catat update besar per tanggal di sini (bukan changelog kode — itu di git).

```
[2026-06-05] — RFN-16 Selesai. Tambah .hermes/LINEAR_WORKFLOW.md (workflow + branching rules). Update CONTRIBUTING.md v2 safety rules. Link di AGENTS.md header.
[2026-06-05] — RFN-15 Selesai. MIGRATION_MAP.md dibuat (audit v1 ke v2).
[2026-06-05] — Inisialisasi PROGRESS.md, semua modul masih TODO
[2026-06-07] — RFN-17 Selesai. Utils foundation (logger, gpu_detector, dependency_check) dibuat.
```

[2026-06-07] — Sync state with Linear
[2026-06-07] — RFN-25 Selesai. pipeline/caption_generator.py (ASS Subtitle Burn-in) diimplementasi.: RFN-18, RFN-19, RFN-20, RFN-21, RFN-22, RFN-23, RFN-24 are DONE.
