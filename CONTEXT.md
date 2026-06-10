# 🧠 CONTEXT.md — Session Handoff Memory

> **INSTRUKSI UNTUK AI AGENT:**
> Baca file ini PERTAMA KALI di setiap sesi baru sebelum melakukan apapun.
> Setelah sesi selesai, **update bagian "Sesi Terakhir" dan "Next Steps"** sebelum menutup sesi.
> Jangan hapus bagian "Riwayat Sesi" — tambahkan entri baru saja.

---

## 📍 Status Saat Ini (Update Setiap Sesi)

**Tanggal update terakhir:** 2026-06-10
**Fase development aktif:** Milestone 6 (M6) — Portrait Smart Crop & Split Mode

### Apa yang sedang dikerjakan sekarang:

> RFN-31 — Portrait Smart Crop & SPLIT Mode implemented in branch `feat/RFN-31-portrait-crop`.
> Menunggu review sebelum merge.

### File yang terakhir dimodifikasi:

> `pipeline/speaker_layout.py` (layout analysis, hysteresis, body-safe crop)
> `pipeline/video_processor.py` (SINGLE/SPLIT portrait rendering, segment concat)
> `tests/unit/test_speaker_layout.py`, `tests/unit/test_video_processor.py`
> `TASKS.md`, `PROGRESS.md` (Update tracking)

### Keputusan pending yang belum dikonfirmasi:

> Per-frame real face detection masih placeholder `_compute_active_scores`; unit tests mock hook ini. Butuh task lanjutan kalau ingin OpenCV DNN/MediaPipe detection real-time.

---

## ⏭️ Next Steps (Prioritas Sesi Berikutnya)

> Agent harus mulai dari sini di sesi berikutnya, kecuali ada perubahan prioritas.

1. Review branch `feat/RFN-31-portrait-crop` and validate RFN-31 behavior in real sample video.
2. If approved, merge RFN-31; create follow-up for real OpenCV DNN/MediaPipe face detection if needed.

---

## 🗂️ Sesi Terakhir

**Tanggal:** 2026-06-10
**Yang dikerjakan:**
> * Implementasi `pipeline/speaker_layout.py` untuk RFN-31: LayoutMode, FaceInfo, LayoutSegment, body-safe crop, active threshold, sliding window, hysteresis.
> * Integrasi `pipeline/video_processor.py` dengan speaker_layout untuk SINGLE/SPLIT crop, FFmpeg vstack, segment render, concat.
> * Tambah unit tests untuk speaker layout, body-safe crop, split mode, multi-segment concat, dan output 1080×1920.

**Blocker yang ditemukan:**
> Real per-frame face detection masih placeholder; `_compute_active_scores` siap diisi OpenCV DNN/MediaPipe pada task lanjutan.

**Keputusan yang dibuat sesi ini:**
> Multi-segment layout dirender per segmen lalu concat agar SINGLE/SPLIT transition tidak memakai `segments[0]` untuk seluruh clip.

---

## 📚 Riwayat Sesi

> Tambahkan entri baru di ATAS setiap selesai sesi. Format singkat saja.

```
[2026-06-05] — RFN-15 Selesai. MIGRATION_MAP.md dibuat. Audit struktur codebase v1 komplit.
[2026-06-05] — Inisialisasi project documents (TASKS, PROGRESS, CONTEXT, DECISIONS, TEST_PLAN)
```

---

## 🔒 Constraint Aktif yang Harus Selalu Diingat

> Ini adalah aturan keras dari AGENTS.md yang TIDAK BOLEH dilanggar di sesi manapun.

1. **Modul `pipeline/`** tidak boleh import satu sama lain — hanya `orchestrator.py` yang boleh cross-import
2. **`providers/ai_client.py`** adalah satu-satunya tempat instansiasi `openai.OpenAI`
3. **Semua config** masuk lewat `providers/config_manager.py` — tidak ada global state
4. **API keys** tidak boleh muncul di log apapun
5. **Tidak boleh `asyncio`** — gunakan `threading.Thread` atau `concurrent.futures`
6. **Output video selalu `1080×1920`** — tidak ada exception
7. **`HEAD_PAD_RATIO` ≥ 0.2, `BODY_PAD_RATIO` ≥ 1.0** — tidak boleh dikurangi
8. **LLM JSON parse failure** → retry maks 2x, baru raise exception
9. **Gradio event loop** tidak boleh diblok — semua pipeline di thread terpisah

---

## 🤝 Cara Agent Harus Update File Ini

Di akhir setiap sesi, agent harus:

```markdown
## Sesi Terakhir
Tanggal: [isi tanggal]
Yang dikerjakan:
- [list modul/file yang dibuat atau diubah]
- [keputusan teknis yang dibuat]

Blocker: [ada/tidak ada, deskripsi jika ada]

## Next Steps
1. [task pertama untuk sesi berikutnya]
2. [task kedua]

## Riwayat Sesi
[TANGGAL] — [ringkasan 1 baris]
```