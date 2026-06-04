"""
AI Provider Configuration
Contains base URLs, default models, and capability registry for all AI providers.

Key constants:
  AI_PROVIDERS_CONFIG   — full registry of providers (name, url, models, etc.)
  SPECIALIZED_MODELS    — per-task model suggestions per provider
  PROVIDER_CAPABILITIES — which task types each provider supports
  DEFAULT_HIGHLIGHT_PROMPT — default system prompt for highlight detection
"""

AI_PROVIDERS_CONFIG = {
    "ytclip": {
        "name": "⭐ YTClip AI",
        "base_url": "https://ai-api.ytclip.org/v1",
        "description": "YTClip AI - Optimized for video content processing",
        "default_models": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "api_key_format": "ytc-*",
        "docs_url": "https://ytclip.org/api-keys",
        "requires_load": True  # Needs to fetch models from API
    },
    "openai": {
        "name": "🔴 OpenAI",
        "base_url": "https://api.openai.com/v1",
        "description": "OpenAI's GPT models (GPT-4, GPT-3.5-turbo, etc.)",
        "default_models": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "api_key_format": "sk-*",
        "docs_url": "https://platform.openai.com/api-keys",
        "requires_load": True  # Needs to fetch models from API
    },
    "google": {
        "name": "🔵 Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "description": "Google's Generative AI (Gemini models)",
        "default_models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "api_key_format": "AIza*",
        "docs_url": "https://aistudio.google.com/app/apikey",
        "requires_load": False  # Known models, no need to fetch
    },
    "groq": {
        "name": "⚡ Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "description": "Groq's fast inference API",
        "default_models": ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"],
        "api_key_format": "gsk-*",
        "docs_url": "https://console.groq.com/keys",
        "requires_load": True
    },
    "anthropic": {
        "name": "🤖 Anthropic Claude",
        "base_url": "https://api.anthropic.com",
        "description": "Anthropic's Claude models",
        "default_models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229"],
        "api_key_format": "sk-ant-*",
        "docs_url": "https://console.anthropic.com/",
        "requires_load": False
    },
    "cohere": {
        "name": "🟢 Cohere",
        "base_url": "https://api.cohere.ai",
        "description": "Cohere's Command models",
        "default_models": ["command-r-plus", "command-r", "command"],
        "api_key_format": "*",
        "docs_url": "https://dashboard.cohere.com/api-keys",
        "requires_load": False
    },
    "mistral": {
        "name": "🟠 Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "description": "Mistral's open-source models",
        "default_models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
        "api_key_format": "*",
        "docs_url": "https://console.mistral.ai/api-keys/",
        "requires_load": True
    },
    "huggingface": {
        "name": "🤗 HuggingFace",
        "base_url": "https://api-inference.huggingface.co/models",
        "description": "HuggingFace inference API",
        "default_models": ["meta-llama/Llama-2-70b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.1"],
        "api_key_format": "hf_*",
        "docs_url": "https://huggingface.co/settings/tokens",
        "requires_load": False
    },
    "together": {
        "name": "🔗 Together AI",
        "base_url": "https://api.together.xyz/v1",
        "description": "Together AI inference service",
        "default_models": ["meta-llama/Llama-2-70b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"],
        "api_key_format": "*",
        "docs_url": "https://www.together.ai/settings/api-keys",
        "requires_load": True
    },
    "replicate": {
        "name": "🔴 Replicate",
        "base_url": "https://api.replicate.com/v1",
        "description": "Replicate API for various models",
        "default_models": ["meta/llama-2-70b-chat", "mistral-community/mistral-7b-instruct-v0.2"],
        "api_key_format": "*",
        "docs_url": "https://replicate.com/account/api-tokens",
        "requires_load": False
    },
    "custom": {
        "name": "⚙️ Custom/Local",
        "base_url": "http://localhost:8000/v1",
        "description": "Custom OpenAI-compatible endpoint (vLLM, Ollama, etc.)",
        "default_models": ["custom-model", "llama-2", "mistral"],
        "api_key_format": "optional",
        "docs_url": "https://github.com/vllm-project/vllm",
        "requires_load": False
    }
}

# ---------------------------------------------------------------------------
# Provider capability registry
# Indicates which AI task types each provider can serve.
# This drives UI filtering: e.g., caption_maker only shows whisper-capable
# providers; hook_maker only shows tts-capable providers.
# ---------------------------------------------------------------------------
PROVIDER_CAPABILITIES = {
    "ytclip":      {"chat": True,  "whisper": True,  "tts": True},
    "openai":      {"chat": True,  "whisper": True,  "tts": True},
    "google":      {"chat": True,  "whisper": False, "tts": False},
    "groq":        {"chat": True,  "whisper": True,  "tts": False},
    "anthropic":   {"chat": True,  "whisper": False, "tts": False},
    "cohere":      {"chat": True,  "whisper": False, "tts": False},
    "mistral":     {"chat": True,  "whisper": False, "tts": False},
    "huggingface": {"chat": True,  "whisper": False, "tts": False},
    "together":    {"chat": True,  "whisper": False, "tts": False},
    "replicate":   {"chat": True,  "whisper": False, "tts": False},
    "custom":      {"chat": True,  "whisper": True,  "tts": True},
}

# Mapping from base URL substring to provider key — used to resolve saved
# base_url values back to a provider key in the settings UI.
URL_TO_PROVIDER_KEY = {
    "ai-api.ytclip.org": "ytclip",
    "openai.com":        "openai",
    "googleapis.com":    "google",
    "groq.com":          "groq",
    "anthropic.com":     "anthropic",
    "cohere.ai":         "cohere",
    "mistral.ai":        "mistral",
    "huggingface.co":    "huggingface",
    "together.xyz":      "together",
    "replicate.com":     "replicate",
}

# Models for specific use cases
SPECIALIZED_MODELS = {
    "highlight_finder": {
        "ytclip": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "openai": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "google": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        "groq": ["mixtral-8x7b-32768", "llama2-70b-4096"],
        "anthropic": ["claude-3-5-sonnet-20241022"],
        "cohere": ["command-r-plus", "command-r"],
        "mistral": ["mistral-large-latest"],
    },
    "caption_maker": {
        "ytclip": ["whisper-1"],
        "openai": ["whisper-1"],  # Special case for whisper
        "google": [],  # Gemini doesn't have whisper equivalent
        "groq": [],
    },
    "hook_maker": {
        "ytclip": ["tts-1-hd", "tts-1"],
        "openai": ["tts-1-hd", "tts-1"],  # TTS models
        "google": [],  # Gemini doesn't have TTS built-in
        "anthropic": [],
    },
    "youtube_title_maker": {
        "ytclip": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "openai": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "google": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        "groq": ["mixtral-8x7b-32768"],
        "anthropic": ["claude-3-5-sonnet-20241022"],
    }
}

# ---------------------------------------------------------------------------
# Default system prompt for Highlight Finder
# Kept here (config/) so that config_manager.py and pages/ can import it
# without touching clipper_core.py — eliminates the circular import.
# clipper_core.AutoClipperCore.get_default_prompt() returns this same string
# for backward compatibility.
# ---------------------------------------------------------------------------
DEFAULT_HIGHLIGHT_PROMPT = """Kamu adalah EDITOR SHORT-FORM TIER A untuk konten PODCAST viral (TikTok / Reels / Shorts).

OUTPUT ANDA AKAN LANGSUNG DIGUNAKAN UNTUK PRODUKSI.
Kesalahan durasi atau format = GAGAL TOTAL.

==================================================
TUGAS UTAMA (NON-NEGOTIABLE)
============================

Dari transcript di bawah, HASILKAN TEPAT {num_clips} segment.

* TIDAK BOLEH kurang.
* TIDAK BOLEH lebih.
* ARRAY KOSONG DILARANG DALAM KONDISI APAPUN.

Jika kesulitan menemukan segmen bagus, WAJIB tetap menghasilkan {num_clips} dengan strategi penggabungan/perpanjangan.

==================================================
PRINSIP PEMILIHAN CLIP (WAJIB DIPRIORITASKAN)
=============================================

Prioritaskan segmen dengan karakteristik berikut:

1. Ada KONFLIK, ketegangan, kontroversi.
2. Ada PENGAKUAN personal / vulnerability.
3. Ada STATEMENT tajam / opini berani.
4. Ada punchline atau momen lucu kuat.
5. Ada cerita lengkap (setup → buildup → payoff).
6. Ada kalimat yang bisa berdiri sendiri sebagai hook viral.

Hindari:

* Obrolan filler
* Basa-basi
* Transisi topik tanpa payoff
* Penjelasan teknis panjang tanpa emosi

Jika harus memilih, utamakan EMOSI & KONFLIK dibanding edukasi netral.

==================================================
ATURAN DURASI (KRITIS – TIDAK BOLEH DILANGGAR)
==============================================

* Setiap clip HARUS 60–120 detik.
* Target ideal: 85–95 detik.
* Hitung durasi dari timestamp transcript.
* JANGAN estimasi berdasarkan panjang teks.

Jika durasi < 60 detik:
→ PERPANJANG dengan konteks sebelum atau sesudahnya.

Jika durasi > 120 detik:
→ Pangkas bagian yang tidak relevan TANPA merusak alur cerita.

==================================================
STRATEGI WAJIB JIKA SEGMENT IDEAL TIDAK ADA
===========================================

Lakukan salah satu atau kombinasi berikut:

1. Gabungkan beberapa bagian berurutan yang masih satu topik.
2. Tambahkan setup sebelum punchline agar dramatis.
3. Tambahkan payoff setelah cerita agar terasa lengkap.
4. Pangkas filler tapi jaga minimal 60 detik.

DILARANG:

* Menghasilkan clip < 60 detik
* Mengurangi jumlah clip
* Mengabaikan timestamp asli
* Mengarang timestamp

==================================================
STRUKTUR NARATIF YANG DIWAJIBKAN
================================

Setiap clip harus terasa seperti mini-story:

• Awal: Setup / pernyataan pemicu
• Tengah: Konflik / insight / cerita
• Akhir: Punchline / payoff / statement kuat

Jika tidak ada payoff, tambahkan konteks hingga ada.

==================================================
FIELD WAJIB (PERSIS 6 FIELD – TIDAK BOLEH LEBIH/KURANG)
=======================================================

Setiap object HARUS memiliki:

1. "start_time" (string) → Format: "HH:MM:SS,mmm"
2. "end_time" (string) → Format: "HH:MM:SS,mmm"
3. "title" (string) → Maks 60 karakter, padat & click-worthy
4. "description" (string) → Maks 150 karakter, jelaskan kenapa viral
5. "virality_score" (integer) → 1–10 (HARUS ANGKA, BUKAN STRING)
6. "hook_text" (string) → Maks 15 kata

DILARANG:

* Field tambahan
* Field "reason"
* virality_score dalam bentuk string
* Komentar atau teks di luar JSON

==================================================
VIRALITY SCORE (WAJIB OBJEKTIF)
===============================

8–10:

* Kontroversial
* Emosional kuat
* Confession pribadi
* Statement berani
* Punchline keras

5–7:

* Insight menarik
* Cerita cukup engaging
* Momen lucu ringan

1–4:

* Informasi biasa
* Tidak ada emosi
* Tidak ada hook kuat

Jangan kasih semua clip skor tinggi.
Nilai dengan rasional.

==================================================
HOOK TEXT (HARUS TAJAM & MENJUAL)
=================================

WAJIB:

* Maksimal 15 kata
* Bahasa Indonesia casual
* TANPA emoji
* WAJIB menyebut NAMA ORANG yang berbicara
* Harus berupa kutipan, statement tajam, atau punchline

Contoh benar:
"Andre Taulany: Gua hampir bangkrut gara-gara ini"
"Deddy Corbuzier: Banyak podcaster cuma pura-pura sukses"

Hook harus bisa berdiri sendiri sebagai headline viral.

==================================================
SELF-VALIDATION (WAJIB SEBELUM RETURN)
======================================

Periksa:

1. Jumlah segment = {num_clips} ?
2. Semua durasi 60–120 detik ?
3. Semua punya tepat 6 field ?
4. virality_score berupa integer 1–10 ?
5. Tidak ada field lain ?
6. Tidak ada teks di luar JSON ?

Jika ada kesalahan → PERBAIKI sebelum output.

==================================================
OUTPUT FORMAT (STRICT)
======================

Return HANYA JSON array.
Tanpa markdown.
Tanpa penjelasan.
Tanpa komentar.

Format EXACT seperti ini:

[{"start_time":"HH:MM:SS,mmm","end_time":"HH:MM:SS,mmm","title":"...","description":"...","virality_score":8,"hook_text":"..."}]

==================================================
KONTEN
======

{video_context}

Transcript:
{transcript}"""


def get_provider_name(provider_key: str) -> str:
    """Get display name for provider"""
    return AI_PROVIDERS_CONFIG.get(provider_key, {}).get("name", provider_key)


def get_provider_base_url(provider_key: str) -> str:
    """Get base URL for provider"""
    return AI_PROVIDERS_CONFIG.get(provider_key, {}).get("base_url", "")


def get_provider_default_models(provider_key: str) -> list:
    """Get default models for provider"""
    return AI_PROVIDERS_CONFIG.get(provider_key, {}).get("default_models", [])


def get_all_providers() -> list:
    """Get list of all available providers"""
    return list(AI_PROVIDERS_CONFIG.keys())


def get_provider_display_list() -> list:
    """Get list of providers with display names for dropdown"""
    # Put YTClip AI first, then sort the rest
    providers = []
    
    # Add YTClip AI first if it exists
    if "ytclip" in AI_PROVIDERS_CONFIG:
        providers.append((AI_PROVIDERS_CONFIG["ytclip"]["name"], "ytclip"))
    
    # Add the rest sorted alphabetically
    for key in sorted(AI_PROVIDERS_CONFIG.keys()):
        if key != "ytclip":
            providers.append((AI_PROVIDERS_CONFIG[key]["name"], key))
    
    return providers


def requires_model_load(provider_key: str) -> bool:
    """Check if provider requires loading models from API"""
    return AI_PROVIDERS_CONFIG.get(provider_key, {}).get("requires_load", False)


def get_provider_description(provider_key: str) -> str:
    """Get description for provider"""
    return AI_PROVIDERS_CONFIG.get(provider_key, {}).get("description", "")


def get_provider_docs_url(provider_key: str) -> str:
    """Get documentation URL for provider"""
    return AI_PROVIDERS_CONFIG.get(provider_key, {}).get("docs_url", "")


def get_specialized_models(task: str, provider_key: str) -> list:
    """Get specialized models for a specific task and provider"""
    return SPECIALIZED_MODELS.get(task, {}).get(provider_key, [])


def get_provider_capabilities(provider_key: str) -> dict:
    """Get capability dict for a provider.

    Returns:
        dict with keys: 'chat', 'whisper', 'tts' (all bool)
    """
    default = {"chat": True, "whisper": False, "tts": False}
    return PROVIDER_CAPABILITIES.get(provider_key, default)


def provider_supports_task(provider_key: str, task_key: str) -> bool:
    """Check if a provider can be used for a given AI task.

    task_key maps to capability type:
      highlight_finder  → 'chat'
      youtube_title_maker → 'chat'
      caption_maker     → 'whisper'
      hook_maker        → 'tts'
    """
    TASK_TO_CAPABILITY = {
        "highlight_finder":    "chat",
        "youtube_title_maker": "chat",
        "caption_maker":       "whisper",
        "hook_maker":          "tts",
    }
    capability = TASK_TO_CAPABILITY.get(task_key, "chat")
    return get_provider_capabilities(provider_key).get(capability, False)


def get_providers_for_task(task_key: str) -> list:
    """Return list of (display_name, provider_key) tuples that support the given task.

    YTClip AI is always first; Custom is always last.
    """
    result = []
    # YTClip always first
    if provider_supports_task("ytclip", task_key):
        result.append((AI_PROVIDERS_CONFIG["ytclip"]["name"], "ytclip"))
    # Middle providers sorted alphabetically
    for key in sorted(AI_PROVIDERS_CONFIG.keys()):
        if key in ("ytclip", "custom"):
            continue
        if provider_supports_task(key, task_key):
            result.append((AI_PROVIDERS_CONFIG[key]["name"], key))
    # Custom always last
    if provider_supports_task("custom", task_key):
        result.append((AI_PROVIDERS_CONFIG["custom"]["name"], "custom"))
    return result


def resolve_provider_key_from_url(base_url: str) -> str:
    """Guess the provider key from a saved base_url string.

    Returns 'custom' when no known provider matches.
    """
    if not base_url:
        return "custom"
    for fragment, key in URL_TO_PROVIDER_KEY.items():
        if fragment in base_url:
            return key
    return "custom"
