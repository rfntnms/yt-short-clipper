"""Modular processing pipeline.

Each module is a standalone unit responsible for one step:
- downloader       → fetch video + optional SRT
- transcriber      → Whisper transcription
- highlight_detector → LLM-driven highlight selection
- video_processor  → FFmpeg cut + portrait crop (basic in M2, smart in M6)
- caption_generator → ASS subtitle + burn-in
- orchestrator     → the only module that cross-imports pipeline steps
"""
