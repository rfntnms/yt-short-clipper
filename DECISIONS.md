# 📐 DECISIONS.md — Architecture Decision Records (ADR)

> **Tujuan:** Dokumen ini mencatat *kenapa* keputusan desain dibuat, bukan hanya *apa*-nya.
> Agent AI harus membaca file ini sebelum mengusulkan perubahan arsitektur.
> Jangan ubah keputusan yang sudah `ACCEPTED` tanpa diskusi eksplisit dengan user.
>
> **Status:** `ACCEPTED` · `DEPRECATED` · `SUPERSEDED BY ADR-XXX` · `PROPOSED`

---

## ADR-001: Gradio sebagai UI Framework (bukan Flask/FastAPI/Streamlit)

**Status:** ACCEPTED
**Tanggal:** *(isi tanggal)*

**Konteks:**
v1 menggunakan CustomTkinter (desktop app). v2 butuh web UI untuk self-hosted deployment.

**Keputusan:**
Pakai Gradio, bukan Flask/FastAPI + custom frontend atau Streamlit.

**Alasan:**

* Gradio punya built-in support untuk file upload, video gallery, progress update, dan live log — semua dibutuhkan tanpa menulis frontend custom
* Generator pattern (`yield`) di Gradio cocok untuk streaming progress pipeline yang panjang
* Deployment lebih simpel: satu server, satu port (7860)
* Streamlit dipertimbangkan tapi kalah di: session isolation per-user dan UI customization untuk gallery video

**Trade-off yang diterima:**

* Tampilan UI lebih terbatas dibanding custom frontend
* Gradio state management (`gr.State`) lebih terbatas — heavy state harus di `batch/job_queue.py`

**Implikasi untuk agent:**
Jangan mengusulkan migrasi ke FastAPI/Flask tanpa alasan yang sangat kuat. Jika ada UI limitation, cari solusi dalam constraint Gradio dulu.

---

## ADR-002: Threading, Bukan Asyncio

**Status:** ACCEPTED
**Tanggal:** *(isi tanggal)*

**Konteks:**
Pipeline melibatkan banyak I/O blocking: download video, API call ke Whisper/LLM, FFmpeg subprocess.

**Keputusan:**
Gunakan `threading.Thread` dan `concurrent.futures.ThreadPoolExecutor`. **Tidak pakai `asyncio`.**

**Alasan:**

* FFmpeg subprocess dan OpenCV tidak punya async interface — wrapping dengan asyncio menambah kompleksitas tanpa manfaat nyata
* Gradio's generator pattern bekerja natively dengan thread, bukan coroutine
* `queue.Queue` Python standard library cukup untuk job queue — tidak perlu async queue
* Asyncio lebih tepat untuk high-concurrency network I/O (web server), bukan batch video processing

**Trade-off yang diterima:**

* Scaling horizontal lebih sulit dibanding async architecture
* GIL Python membatasi true parallelism — tapi pipeline ini I/O-bound, bukan CPU-bound (FFmpeg jalan di subprocess, bukan Python)

**Implikasi untuk agent:**
Jika ada blocking I/O baru, bungkus dengan `ThreadPoolExecutor`, bukan `asyncio`. Exception: jika Gradio di versi baru *mengharuskan* async untuk pattern tertentu, itu boleh — dokumentasikan di sini.

---

## ADR-003: OpenAI-Compatible Generic Endpoint (Bukan Provider-Specific SDK)

**Status:** ACCEPTED
**Tanggal:** *(isi tanggal)*

**Konteks:**
User ingin support LLM lokal (Ollama, LM Studio) dan cloud (OpenAI, dll.) tanpa perlu kode berbeda per provider.

**Keputusan:**
Semua AI call lewat satu abstraksi di `providers/ai_client.py` menggunakan `openai` SDK Python dengan `base_url` yang dikonfigurasi user.

**Alasan:**

* Ollama, LM Studio, vLLM, dan banyak provider lokal sudah support `/v1/chat/completions` dan `/v1/audio/transcriptions` (OpenAI-compatible spec)
* Tidak ada vendor lock-in — user bisa ganti provider hanya dengan ganti `base_url` + `model` di `config.json`
* Satu SDK, satu interface, tidak ada if-else provider di business logic

**Trade-off yang diterima:**

* Provider yang tidak support OpenAI spec tidak bisa dipakai (tapi ini desain yang disengaja)
* Fitur provider-specific (misalnya Anthropic tool use, OpenAI function calling format baru) tidak diekspos langsung

**Implikasi untuk agent:**
**JANGAN** tambah provider-specific branching di luar `providers/ai_client.py`. Jika ada provider baru yang perlu special handling, tambahkan di `ai_client.py` saja, bukan di modul pipeline.

---

## ADR-004: In-Memory Job Queue (Bukan Redis/Celery by Default)

**Status:** ACCEPTED
**Tanggal:** *(isi tanggal)*

**Konteks:**
Butuh job queue untuk batch processing dan scheduling.

**Keputusan:**
Default pakai `queue.Queue` Python in-process. Redis adalah **opsional** (disebutkan di AGENTS.md tapi bukan requirement).

**Alasan:**

* Target deployment adalah **single-server self-hosted** — bukan distributed system
* `queue.Queue` zero-dependency, zero-configuration
* Untuk use case personal/small team, in-memory queue sudah cukup
* Redis menambah dependency Docker dan kompleksitas setup

**Trade-off yang diterima:**

* Job `PENDING` hilang saat server restart → **mitigasi:** re-queue dari `output/jobs.json` saat startup (sudah ada di spec `batch_runner.py`)
* Tidak bisa scale ke multiple workers/machines tanpa refactor ke Redis/Celery

**Implikasi untuk agent:**
Implement dengan `queue.Queue` dulu. Jika user secara eksplisit meminta Redis support, buat sebagai optional backend — jangan replace default.

---

## ADR-005: Modular Pipeline, Orchestrator sebagai Satu-satunya Cross-Importer

**Status:** ACCEPTED
**Tanggal:** *(isi tanggal)*

**Konteks:**
v1 memiliki `clipper_core.py` monolith yang sulit di-test dan di-maintain.

**Keputusan:**
Setiap modul `pipeline/` hanya export satu fungsi/class publik utama. Hanya `orchestrator.py` yang boleh import lintas modul pipeline.

**Alasan:**

* Setiap modul bisa di-unit-test secara independen dengan mock sederhana
* Perubahan di satu modul tidak cascade ke modul lain
* Memudahkan agent AI untuk bekerja pada satu modul tanpa memahami keseluruhan system

**Implikasi untuk agent:**
Jika kamu butuh memanggil fungsi dari modul pipeline lain, **bukan** dengan import langsung di modul tersebut — refactor ke `orchestrator.py` atau buat utility di `utils/`.

---

## ADR-006: Output Video Selalu 1080×1920 (9:16)

**Status:** ACCEPTED
**Tanggal:** *(isi tanggal)*

**Keputusan:**
Semua output video dari `video_processor.py` harus `1080×1920` tanpa exception, baik SINGLE maupun SPLIT mode.

**Alasan:**

* Platform target (TikTok, Reels, YouTube Shorts) semua menggunakan 9:16
* Consistency memudahkan thumbnail generation dan metadata handling
* SPLIT mode: masing-masing panel `1080×960` yang di-`vstack` menjadi `1080×1920`

**Implikasi untuk agent:**
Jika ada edge case di mana resolusi output bisa berbeda (misalnya source video landscape sangat lebar), **tetap scale/crop ke `1080×1920`**. Jangan ada output resolution lain.

---

## Template ADR Baru

> Copy template ini untuk menambah keputusan baru.

```markdown
## ADR-XXX: [Judul Keputusan]

**Status:** PROPOSED / ACCEPTED / DEPRECATED
**Tanggal:** [YYYY-MM-DD]

**Konteks:**
[Masalah apa yang sedang dihadapi?]

**Keputusan:**
[Apa yang diputuskan?]

**Alasan:**
[Kenapa pilihan ini? Apa alternatif yang dipertimbangkan?]

**Trade-off yang diterima:**
[Apa kekurangannya dan kenapa masih diterima?]

**Implikasi untuk agent:**
[Apa yang harus / tidak boleh dilakukan agent terkait keputusan ini?]
```
