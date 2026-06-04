"""
Base class for AI Provider settings pages
"""

import threading
import customtkinter as ctk
from tkinter import messagebox

from pages.settings.base_dialog import BaseSettingsSubPage
from config.ai_provider_config import (
    AI_PROVIDERS_CONFIG,
    get_providers_for_task,
    resolve_provider_key_from_url,
    get_provider_base_url,
    get_specialized_models,
)


class BaseProviderSettingsPage(BaseSettingsSubPage):
    """Base class for AI provider settings pages.

    Child classes must pass `provider_key` matching one of the task keys:
      highlight_finder | caption_maker | hook_maker | youtube_title_maker

    Class-level overrides:
      FIXED_MODELS    — list[str] | None.  None = load dynamically from API.
      USE_MANUAL_INPUT — bool.  True = show a text entry instead of dropdown.
      DEFAULT_MODEL   — str.   Placeholder used when USE_MANUAL_INPUT is True.
    """

    # Override in child class for fixed model list (None = load from API)
    FIXED_MODELS = None
    # Override in child class to use manual input instead of dropdown
    USE_MANUAL_INPUT = False
    # Default model value when using manual input
    DEFAULT_MODEL = ""

    def __init__(self, parent, title, provider_key, config, on_save_callback, on_back_callback):
        self.config = config
        self.provider_key = provider_key
        self.on_save_callback = on_save_callback
        self.models_list = []

        # Build ordered list of (display_name, key) for this task
        self._provider_list = get_providers_for_task(provider_key)
        # Parallel list of just the keys (used for lookups)
        self._provider_keys = [k for _, k in self._provider_list]
        # Display names only (for CTkOptionMenu values)
        self._provider_names = [name for name, _ in self._provider_list]

        # Currently selected provider key — updated by dropdown
        self._current_provider_key = "ytclip"

        super().__init__(parent, title, on_back_callback)

        self.create_provider_content()
        self.load_config()

    def create_provider_content(self):
        """Create provider settings content"""
        # Provider Type Section
        type_section = self.create_section("Provider Type")

        type_frame = ctk.CTkFrame(type_section, fg_color="transparent")
        type_frame.pack(fill="x", padx=15, pady=(0, 12))

        ctk.CTkLabel(type_frame, text="Select API Provider", font=ctk.CTkFont(size=11)).pack(anchor="w")

        self.provider_type_var = ctk.StringVar(value=self._provider_names[0] if self._provider_names else "")
        self.provider_dropdown = ctk.CTkOptionMenu(
            type_frame,
            values=self._provider_names,
            variable=self.provider_type_var,
            height=36,
            command=self._on_provider_type_changed,
        )
        self.provider_dropdown.pack(fill="x", pady=(5, 0))

        # System Message Section (optional, can be overridden by child)
        self.system_message_textbox = None

        # URL Section (only visible for custom providers)
        self.url_section = self.create_section("Base URL")
        self.url_section.pack_forget()  # Hidden by default

        url_frame = ctk.CTkFrame(self.url_section, fg_color="transparent")
        url_frame.pack(fill="x", padx=15, pady=(0, 12))

        ctk.CTkLabel(url_frame, text="API Base URL", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.url_entry = ctk.CTkEntry(url_frame, placeholder_text="https://api.openai.com/v1", height=36)
        self.url_entry.pack(fill="x", pady=(5, 0))

        # API Key Section
        key_section = self.create_section("API Key")

        key_frame = ctk.CTkFrame(key_section, fg_color="transparent")
        key_frame.pack(fill="x", padx=15, pady=(0, 12))

        ctk.CTkLabel(key_frame, text="API Key", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.key_entry = ctk.CTkEntry(key_frame, placeholder_text="sk-...", show="•", height=36)
        self.key_entry.pack(fill="x", pady=(5, 0))

        # Model Section
        self.model_section = self.create_section("Model")

        model_frame = ctk.CTkFrame(self.model_section, fg_color="transparent")
        model_frame.pack(fill="x", padx=15, pady=(0, 12))

        ctk.CTkLabel(model_frame, text="Model Name", font=ctk.CTkFont(size=11)).pack(anchor="w")

        model_row = ctk.CTkFrame(model_frame, fg_color="transparent")
        model_row.pack(fill="x", pady=(5, 0))

        # Check if using manual input mode
        if self.USE_MANUAL_INPUT:
            # Manual input mode - use CTkEntry
            self.model_entry = ctk.CTkEntry(
                model_row, placeholder_text=f"e.g., {self.DEFAULT_MODEL}", height=36
            )
            self.model_entry.pack(fill="x")
            self.model_dropdown = None
            self.model_var = None
            self.load_btn = None
        else:
            # Dropdown mode
            self.model_var = ctk.StringVar(value="")
            self.model_entry = None

            # Check if using fixed models or load from API
            if self.FIXED_MODELS:
                # Fixed dropdown - no load button needed
                self.model_dropdown = ctk.CTkOptionMenu(
                    model_row, values=self.FIXED_MODELS, variable=self.model_var, height=36
                )
                self.model_dropdown.pack(fill="x")
                self.load_btn = None
            else:
                # Dynamic dropdown with load button
                self.model_dropdown = ctk.CTkOptionMenu(
                    model_row,
                    values=["-- Click Load to fetch models --"],
                    variable=self.model_var,
                    height=36,
                    width=200,
                )
                self.model_dropdown.pack(side="left", fill="x", expand=True, padx=(0, 5))

                self.load_btn = ctk.CTkButton(
                    model_row, text="🔄 Load", width=80, height=36, command=self.load_models
                )
                self.load_btn.pack(side="right")

        # Actions
        actions_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        actions_frame.pack(fill="x", pady=(10, 0))

        # Validate button — shows spinner label while running in thread
        validate_row = ctk.CTkFrame(actions_frame, fg_color="transparent")
        validate_row.pack(fill="x", pady=(0, 10))

        self.validate_btn = ctk.CTkButton(
            validate_row,
            text="🔍 Validate Configuration",
            height=40,
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray40"),
            command=self.validate_config,
        )
        self.validate_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.validate_status_label = ctk.CTkLabel(
            validate_row, text="", font=ctk.CTkFont(size=11), width=90, anchor="w"
        )
        self.validate_status_label.pack(side="right")

        # Save button
        self.create_save_button(self.save_settings)

    # ------------------------------------------------------------------
    # Provider type handling
    # ------------------------------------------------------------------

    def _on_provider_type_changed(self, display_name: str):
        """Handle provider type dropdown change."""
        # Resolve the selected display name back to a provider key
        key = self._display_name_to_key(display_name)
        self._current_provider_key = key

        is_custom = (key == "custom")
        if is_custom:
            self.url_section.pack(fill="x", pady=(0, 10), after=self.content.winfo_children()[1])
        else:
            self.url_section.pack_forget()
            # Pre-fill URL entry with the known base URL (useful if user later
            # switches to custom and wants the last known value)
            known_url = get_provider_base_url(key)
            if known_url:
                self.url_entry.delete(0, "end")
                self.url_entry.insert(0, known_url)

        # Pre-populate model dropdown with known specialized models when available
        if not self.USE_MANUAL_INPUT and not self.FIXED_MODELS and self.model_dropdown:
            specialized = get_specialized_models(self.provider_key, key)
            if specialized:
                current = self.model_var.get() if self.model_var else ""
                self.model_dropdown.configure(values=specialized)
                if current not in specialized:
                    self.model_var.set(specialized[0])
            else:
                # No known models — reset to prompt the user to load
                self.model_dropdown.configure(values=["-- Click Load to fetch models --"])
                if self.model_var:
                    self.model_var.set("")

    def _display_name_to_key(self, display_name: str) -> str:
        """Convert a display name string back to a provider key."""
        for name, key in self._provider_list:
            if name == display_name:
                return key
        return "custom"

    def _key_to_display_name(self, provider_key: str) -> str:
        """Convert a provider key to its display name."""
        for name, key in self._provider_list:
            if key == provider_key:
                return name
        # Fallback to custom if the key isn't in our list for this task
        for name, key in self._provider_list:
            if key == "custom":
                return name
        return self._provider_names[0] if self._provider_names else ""

    def get_base_url(self) -> str:
        """Get base URL based on currently selected provider."""
        key = self._current_provider_key
        if key == "custom":
            return self.url_entry.get().strip() or "https://api.openai.com/v1"
        known_url = get_provider_base_url(key)
        return known_url or "https://api.openai.com/v1"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self):
        """Load available models from API (runs in background thread)."""
        if self.FIXED_MODELS:
            return  # No need to load for fixed models

        api_key = self.key_entry.get().strip()

        if not api_key:
            messagebox.showerror("Error", "Please enter API Key first")
            return

        url = self.get_base_url()
        self.load_btn.configure(state="disabled", text="Loading...")

        def do_load():
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key, base_url=url)
                models_response = client.models.list()
                models = sorted([m.id for m in models_response.data])

                self.after(0, lambda: self._on_models_loaded(models))
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self._on_models_error(msg))

        threading.Thread(target=do_load, daemon=True).start()

    def _on_models_loaded(self, models):
        """Handle models loaded (called on main thread via .after())."""
        self.load_btn.configure(state="normal", text="🔄 Load")
        self.models_list = models

        if models:
            self.model_dropdown.configure(values=models)
            current = self.model_var.get()
            if current not in models:
                self.model_var.set(models[0])
            messagebox.showinfo("Success", f"Loaded {len(models)} models")
        else:
            messagebox.showwarning("Warning", "No models found")

    def _on_models_error(self, error):
        """Handle models load error (called on main thread via .after())."""
        self.load_btn.configure(state="normal", text="🔄 Load")
        messagebox.showerror("Error", f"Failed to load models:\n{error}")

    # ------------------------------------------------------------------
    # Validate (FIXED: now runs in background thread — ADR-005 compliance)
    # ------------------------------------------------------------------

    def validate_config(self):
        """Validate provider configuration.

        Runs the API call in a daemon thread to avoid blocking the tkinter
        main thread (fixes the ADR-005 violation that previously caused a
        5-10 second UI freeze on every Validate click).
        """
        api_key = self.key_entry.get().strip()
        url = self.get_base_url()

        if not api_key:
            messagebox.showerror("Error", "API Key is required")
            return

        if not self.USE_MANUAL_INPUT and self.model_var:
            model = self.model_var.get().strip()
            if not model or model.startswith("--"):
                messagebox.showerror("Error", "Please select a model first")
                return

        # Disable button and show spinner while validating
        self.validate_btn.configure(state="disabled", text="🔍 Validating...")
        self.validate_status_label.configure(text="⏳ Checking...", text_color="gray")

        def do_validate():
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key, base_url=url)
                client.models.list()
                self.after(0, self._on_validate_success)
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self._on_validate_error(msg))

        threading.Thread(target=do_validate, daemon=True).start()

    def _on_validate_success(self):
        """Handle successful validation (main thread)."""
        self.validate_btn.configure(
            state="normal",
            text="🔍 Validate Configuration",
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray40"),
        )
        self.validate_status_label.configure(text="✅ Valid", text_color=("green", "lightgreen"))
        model = ""
        if not self.USE_MANUAL_INPUT and self.model_var:
            model = self.model_var.get()
        messagebox.showinfo(
            "Success",
            f"✓ Configuration valid!\n\nModel: {model}\nURL: {self.get_base_url()}",
        )

    def _on_validate_error(self, error: str):
        """Handle validation error (main thread)."""
        self.validate_btn.configure(
            state="normal",
            text="🔍 Validate Configuration",
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray40"),
        )
        self.validate_status_label.configure(text="❌ Failed", text_color=("red", "tomato"))
        messagebox.showerror("Validation Failed", f"API check failed:\n{error}")

    # ------------------------------------------------------------------
    # Config load / save
    # ------------------------------------------------------------------

    def load_config(self):
        """Load config into UI widgets."""
        # Handle both ConfigManager and dict
        if hasattr(self.config, "config"):
            config_dict = self.config.config
        else:
            config_dict = self.config

        ai_providers = config_dict.get("ai_providers", {})
        provider = ai_providers.get(self.provider_key, {})

        # Determine provider key from saved base_url
        base_url = provider.get("base_url", "")
        resolved_key = resolve_provider_key_from_url(base_url)

        # Verify the resolved key is actually in our task-filtered list
        if resolved_key not in self._provider_keys:
            resolved_key = "custom"

        self._current_provider_key = resolved_key
        display_name = self._key_to_display_name(resolved_key)
        self.provider_type_var.set(display_name)

        # Show/hide URL section based on provider type
        if resolved_key == "custom":
            self.url_section.pack(fill="x", pady=(0, 10), after=self.content.winfo_children()[1])
        else:
            self.url_section.pack_forget()

        # Always populate url_entry with the stored value (useful for custom)
        self.url_entry.delete(0, "end")
        self.url_entry.insert(0, base_url)

        # API key
        self.key_entry.delete(0, "end")
        self.key_entry.insert(0, provider.get("api_key", ""))

        saved_model = provider.get("model", "")

        # Load model based on input type
        if self.USE_MANUAL_INPUT:
            self.model_entry.delete(0, "end")
            self.model_entry.insert(0, saved_model if saved_model else self.DEFAULT_MODEL)
        else:
            if self.FIXED_MODELS:
                if saved_model in self.FIXED_MODELS:
                    self.model_var.set(saved_model)
                else:
                    self.model_var.set(self.FIXED_MODELS[0])
            else:
                # Dynamic dropdown: populate with specialized models for this provider,
                # then add the saved model so it's always selectable.
                specialized = get_specialized_models(self.provider_key, resolved_key)
                dropdown_values = list(specialized) if specialized else ["-- Click Load to fetch models --"]

                if saved_model and saved_model not in dropdown_values:
                    dropdown_values = [saved_model] + dropdown_values

                self.model_dropdown.configure(values=dropdown_values)
                if saved_model:
                    self.model_var.set(saved_model)
                elif specialized:
                    self.model_var.set(specialized[0])

        # Load system message if textbox exists (Highlight Finder only)
        if self.system_message_textbox:
            system_message = provider.get("system_message", "")
            if not system_message:
                system_message = config_dict.get("system_prompt", "")
            self.system_message_textbox.delete("1.0", "end")
            self.system_message_textbox.insert("1.0", system_message)

    def save_settings(self):
        """Save settings and call the on_save_callback."""
        api_key = self.key_entry.get().strip()

        # Get model from entry or dropdown
        if self.USE_MANUAL_INPUT:
            model = self.model_entry.get().strip()
            if not model:
                model = self.DEFAULT_MODEL
        else:
            model = self.model_var.get().strip() if self.model_var else ""

        url = self.get_base_url()

        if not api_key:
            messagebox.showerror("Error", "API Key is required")
            return

        if not model or model.startswith("--"):
            messagebox.showerror("Error", "Please select a model")
            return

        # Handle both ConfigManager and dict
        if hasattr(self.config, "config"):
            config_dict = self.config.config
        else:
            config_dict = self.config

        # Update config
        if "ai_providers" not in config_dict:
            config_dict["ai_providers"] = {}

        provider_config = {"base_url": url, "api_key": api_key, "model": model}

        # Save system message if textbox exists
        if self.system_message_textbox:
            system_message = self.system_message_textbox.get("1.0", "end").strip()
            if system_message:
                provider_config["system_message"] = system_message

        config_dict["ai_providers"][self.provider_key] = provider_config

        # Call save callback with the full config dict
        if self.on_save_callback:
            self.on_save_callback(config_dict)

        messagebox.showinfo("Success", f"{self.title} settings saved!")
        self.on_back()
