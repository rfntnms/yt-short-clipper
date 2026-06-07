# 🧠 CONTEXT.md — Session Handoff Memory

> **INSTRUKSI UNTUK AI AGENT:**
> Baca file ini PERTAMA KALI di setiap sesi baru sebelum melakukan apapun.
> Setelah sesi selesai, **update bagian "Sesi Terakhir" dan "Next Steps"** sebelum menutup sesi.
> Jangan hapus bagian "Riwayat Sesi" — tambahkan entri baru saja.

---

## 📍 Status Saat Ini (Update Setiap Sesi)

**Tanggal update terakhir:** 2026-06-05
**Fase development aktif:** Milestone 0 (M0) — Migration Preparation

### Apa yang sedang dikerjakan sekarang:

> RFN-16 — Development Safety Rules & Branching Strategy (DONE, in review)
> Menunggu approval & merge. Setelah itu pindah ke M1 (RFN-17 first code).

### File yang terakhir dimodifikasi:

> `AGENTS.md` (Link banner ke .hermes/LINEAR_WORKFLOW.md)
> `CONTRIBUTING.md` (Rewrite ke v2 migration safety rules)
> `.hermes/LINEAR_WORKFLOW.md` (Sudah ada dari sesi sebelumnya)
> `TASKS.md`, `PROGRESS.md` (Update tracking)

### Keputusan pending yang belum dikonfirmasi:

> *(Belum ada)*

---

## ⏭️ Next Steps (Prioritas Sesi Berikutnya)

> Agent harus mulai dari sini di sesi berikutnya, kecuali ada perubahan prioritas.

1. Lanjut ke `RFN-26` (pipeline/orchestrator.py — Sequential Flow & Generator)
2. Mulai mengerjakan Gradio UI MVP (`RFN-28`)

---

## 🗂️ Sesi Terakhir

**Tanggal:** 2026-06-07
**Yang dikerjakan:**
> * Sinkronisasi state lokal dengan Linear MCP
> * Implementasi `pipeline/caption_generator.py` (ASS Subtitle Burn-in) untuk RFN-25
> * Membuat unit test test_caption_generator.py untuk validasi render .ass dan FFmpeg args

**Blocker yang ditemukan:**
> *(none)*

**Keputusan yang dibuat sesi ini:**
> Menggunakan override tags untuk efek karaoke di .ass file dan mengabaikan extract audio ulang karena whisper word-level input sudah tersedia dari module sebelumnya.

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