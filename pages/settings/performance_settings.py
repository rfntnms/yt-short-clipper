"""
Performance Settings Sub-Page with GPU Detection
"""

import threading
import customtkinter as ctk
from tkinter import messagebox

from pages.settings.base_dialog import BaseSettingsSubPage


class PerformanceSettingsSubPage(BaseSettingsSubPage):
    """Sub-page for configuring performance settings with GPU detection"""
    
    def __init__(self, parent, config, on_save_callback, on_back_callback):
        self.config = config
        self.on_save_callback = on_save_callback
        
        super().__init__(parent, "Performance Settings", on_back_callback)
        
        self.create_content()
        self.load_config()
        
        # Auto-detect GPU on load
        self.after(500, self.detect_gpu)
    
    def create_content(self):
        """Create page content"""
        # GPU Detection Section
        detection_section = self.create_section("GPU Detection")
        
        detection_frame = ctk.CTkFrame(detection_section, fg_color="transparent")
        detection_frame.pack(fill="x", padx=15, pady=(0, 12))
        
        # GPU info display
        self.gpu_info_frame = ctk.CTkFrame(detection_frame, fg_color=("gray90", "gray15"), corner_radius=8)
        self.gpu_info_frame.pack(fill="x", pady=(0, 10))
        
        self.gpu_status_label = ctk.CTkLabel(self.gpu_info_frame, text="Detecting GPU...", 
            font=ctk.CTkFont(size=11), anchor="w", justify="left")
        self.gpu_status_label.pack(fill="x", padx=12, pady=12)
        
        # Detect button
        self.detect_gpu_btn = ctk.CTkButton(detection_frame, text="🔄 Detect GPU", height=36,
            fg_color=("#3B8ED0", "#1F6AA5"), command=lambda: self.detect_gpu(force=True))
        self.detect_gpu_btn.pack(fill="x")
        
        # GPU Acceleration Section
        accel_section = self.create_section("GPU Acceleration")
        
        accel_frame = ctk.CTkFrame(accel_section, fg_color="transparent")
        accel_frame.pack(fill="x", padx=15, pady=(0, 12))
        
        self.gpu_enabled_var = ctk.BooleanVar(value=False)
        self.gpu_switch = ctk.CTkSwitch(accel_frame, text="Enable GPU Acceleration", 
            variable=self.gpu_enabled_var, font=ctk.CTkFont(size=12),
            command=self.toggle_gpu_acceleration, state="disabled")
        self.gpu_switch.pack(anchor="w", pady=(0, 10))
        
        ctk.CTkLabel(accel_frame, 
            text="GPU encoding is 3-5x faster than CPU. Requires compatible hardware.",
            font=ctk.CTkFont(size=10), text_color="gray", anchor="w", justify="left").pack(fill="x")

        # Processing Profile Section
        profile_section = self.create_section("Processing Profile")
        profile_frame = ctk.CTkFrame(profile_section, fg_color="transparent")
        profile_frame.pack(fill="x", padx=15, pady=(0, 12))

        self.profile_var = ctk.StringVar(value="balanced")
        ctk.CTkLabel(profile_frame, text="Portrait Detection Profile", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.profile_menu = ctk.CTkOptionMenu(
            profile_frame,
            values=["quality", "balanced", "fast"],
            variable=self.profile_var,
            command=self._on_profile_changed
        )
        self.profile_menu.pack(fill="x", pady=(5, 10))

        self.detection_engine_var = ctk.StringVar(value="hybrid_auto")
        ctk.CTkLabel(profile_frame, text="Detection Engine", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.detection_engine_menu = ctk.CTkOptionMenu(
            profile_frame,
            values=["hybrid_auto", "opencv_fast", "mediapipe_quality", "mediapipe", "yolo_fast"],
            variable=self.detection_engine_var
        )
        self.detection_engine_menu.pack(fill="x", pady=(5, 10))

        self.speaker_framing_var = ctk.StringVar(value="center_speaker")
        ctk.CTkLabel(profile_frame, text="Speaker Framing", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.speaker_framing_menu = ctk.CTkOptionMenu(
            profile_frame,
            values=["center_speaker", "active_speaker"],
            variable=self.speaker_framing_var
        )
        self.speaker_framing_menu.pack(fill="x", pady=(5, 10))

        ctk.CTkLabel(profile_frame, text="Detection Interval (frames)", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.detection_interval_entry = ctk.CTkEntry(profile_frame, height=34)
        self.detection_interval_entry.pack(fill="x", pady=(5, 0))

        # Encoder Section
        encoder_section = self.create_section("Encoder Options")
        encoder_frame = ctk.CTkFrame(encoder_section, fg_color="transparent")
        encoder_frame.pack(fill="x", padx=15, pady=(0, 12))

        self.encoder_var = ctk.StringVar(value="auto")
        ctk.CTkLabel(encoder_frame, text="Encoder Preference", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.encoder_menu = ctk.CTkOptionMenu(
            encoder_frame,
            values=["auto", "cpu", "h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf", "h264_qsv", "hevc_qsv", "h264_videotoolbox", "hevc_videotoolbox"],
            variable=self.encoder_var
        )
        self.encoder_menu.pack(fill="x", pady=(5, 10))

        self.codec_var = ctk.StringVar(value="h264")
        ctk.CTkLabel(encoder_frame, text="CPU Fallback Codec", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.codec_menu = ctk.CTkOptionMenu(encoder_frame, values=["h264", "hevc"], variable=self.codec_var)
        self.codec_menu.pack(fill="x", pady=(5, 10))

        self.decode_enabled_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(encoder_frame, text="Enable hardware decode when GPU encoding is enabled",
            variable=self.decode_enabled_var, font=ctk.CTkFont(size=11)).pack(anchor="w", pady=(0, 8))

        self.test_encoder_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(encoder_frame, text="Test selected encoder before processing",
            variable=self.test_encoder_var, font=ctk.CTkFont(size=11)).pack(anchor="w")

        # Optional YOLO Section
        yolo_section = self.create_section("Optional YOLO")
        yolo_frame = ctk.CTkFrame(yolo_section, fg_color="transparent")
        yolo_frame.pack(fill="x", padx=15, pady=(0, 12))

        ctk.CTkLabel(yolo_frame, text="YOLO Model Path", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.yolo_model_entry = ctk.CTkEntry(yolo_frame, height=34, placeholder_text="Optional local .pt model path")
        self.yolo_model_entry.pack(fill="x", pady=(5, 10))

        self.allow_yolo_download_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(yolo_frame, text="Allow ultralytics to download yolov8n.pt when needed",
            variable=self.allow_yolo_download_var, font=ctk.CTkFont(size=11)).pack(anchor="w")
        
        # Technical Details Section
        details_section = self.create_section("Technical Details")
        
        details_frame = ctk.CTkFrame(details_section, fg_color="transparent")
        details_frame.pack(fill="x", padx=15, pady=(0, 12))
        
        self.encoder_info_label = ctk.CTkLabel(details_frame, 
            text="Encoder: Not detected\nPreset: N/A\nStatus: Click 'Detect GPU' to check",
            font=ctk.CTkFont(size=10), text_color="gray", anchor="w", justify="left")
        self.encoder_info_label.pack(fill="x")
        
        # Save button
        self.create_save_button(self.save_settings)

    def _on_profile_changed(self, value):
        intervals = {"quality": "5", "balanced": "10", "fast": "30"}
        self.detection_interval_entry.delete(0, "end")
        self.detection_interval_entry.insert(0, intervals.get(value, "10"))
    
    def detect_gpu(self, force=False):
        """Detect GPU and update UI"""
        if not force:
            if hasattr(self.config, 'config'):
                config_dict = self.config.config
            else:
                config_dict = self.config
                
            gpu_config = config_dict.get("gpu_acceleration", {})
            cached_encoder = gpu_config.get("cached_encoder")
            cached_gpu = gpu_config.get("cached_gpu_name")
            
            if cached_encoder and cached_gpu:
                gpu_info = {'available': True, 'name': cached_gpu, 'type': 'unknown'}
                recommendation = {'available': True, 'encoder': cached_encoder, 'preset': 'auto', 'reason': 'Loaded from cache'}
                
                name_lower = cached_gpu.lower()
                if 'nvidia' in name_lower or 'geforce' in name_lower or 'rtx' in name_lower:
                    gpu_info['type'] = 'nvidia'
                elif 'amd' in name_lower or 'radeon' in name_lower:
                    gpu_info['type'] = 'amd'
                elif 'intel' in name_lower or 'arc' in name_lower or 'iris' in name_lower:
                    gpu_info['type'] = 'intel'
                elif 'apple' in name_lower or name_lower.startswith('m'):
                    gpu_info['type'] = 'apple'
                    
                self._on_gpu_detected(gpu_info, recommendation, from_cache=True)
                return

        self.detect_gpu_btn.configure(state="disabled", text="Detecting...")
        
        def do_detect():
            try:
                from utils.gpu_detector import GPUDetector
                detector = GPUDetector()
                
                gpu_info = detector.detect_gpu()
                recommendation = detector.get_recommended_encoder()
                
                self.after(0, lambda g=gpu_info, r=recommendation: self._on_gpu_detected(g, r))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda err=error_msg: self._on_gpu_detect_error(err))
        
        threading.Thread(target=do_detect, daemon=True).start()
    
    def _on_gpu_detected(self, gpu_info, recommendation, from_cache=False):
        """Handle GPU detection result"""
        self.detect_gpu_btn.configure(state="normal", text="🔄 Detect GPU")
        
        self.latest_gpu_info = gpu_info
        self.latest_recommendation = recommendation
        self.is_from_cache = from_cache
        
        if gpu_info['available']:
            gpu_type_emoji = {'nvidia': '🟢', 'amd': '🔴', 'intel': '🔵'}
            emoji = gpu_type_emoji.get(gpu_info['type'], '⚪')
            
            status_text = f"{emoji} GPU Detected\n"
            status_text += f"Name: {gpu_info['name']}\n"
            status_text += f"Type: {gpu_info['type'].upper()}"
            
            self.gpu_status_label.configure(text=status_text, text_color=("green", "lightgreen"))
            
            if recommendation['available']:
                encoder_text = f"Encoder: {recommendation['encoder']}\n"
                encoder_text += f"Preset: {recommendation['preset']}\n"
                encoder_text += f"Status: ✓ Ready to use"
                self.encoder_info_label.configure(text=encoder_text, text_color=("green", "lightgreen"))
                self.gpu_switch.configure(state="normal")
            else:
                encoder_text = f"Encoder: Not available\n"
                encoder_text += f"Reason: {recommendation.get('reason', 'Unknown')}"
                self.encoder_info_label.configure(text=encoder_text, text_color=("orange", "yellow"))
                self.gpu_switch.configure(state="disabled")
                self.gpu_enabled_var.set(False)
        else:
            status_text = "⚪ No GPU Detected\n"
            status_text += "Video processing will use CPU."
            
            self.gpu_status_label.configure(text=status_text, text_color="gray")
            
            encoder_text = "Encoder: libx264 (CPU)\n"
            encoder_text += "Preset: fast\n"
            encoder_text += "Status: Using CPU encoding"
            self.encoder_info_label.configure(text=encoder_text, text_color="gray")
            
            self.gpu_switch.configure(state="disabled")
            self.gpu_enabled_var.set(False)
    
    def _on_gpu_detect_error(self, error):
        """Handle GPU detection error"""
        self.detect_gpu_btn.configure(state="normal", text="🔄 Detect GPU")
        
        status_text = f"❌ Detection Error\nError: {error}"
        self.gpu_status_label.configure(text=status_text, text_color=("red", "orange"))
        
        self.gpu_switch.configure(state="disabled")
        self.gpu_enabled_var.set(False)
    
    def toggle_gpu_acceleration(self):
        """Handle GPU acceleration toggle"""
        if self.gpu_enabled_var.get():
            messagebox.showinfo("GPU Enabled", 
                "GPU acceleration enabled.\nDon't forget to save settings.")
        else:
            messagebox.showinfo("GPU Disabled", 
                "GPU acceleration disabled.\nDon't forget to save settings.")
    
    def load_config(self):
        """Load config into UI"""
        # Handle both ConfigManager and dict
        if hasattr(self.config, 'config'):
            config_dict = self.config.config
        else:
            config_dict = self.config
            
        gpu_config = config_dict.get("gpu_acceleration", {})
        self.gpu_enabled_var.set(gpu_config.get("enabled", False))

        performance = config_dict.get("performance", {})
        self.profile_var.set(performance.get("profile", "balanced"))
        self.detection_engine_var.set(performance.get("detection_engine", config_dict.get("face_tracking_mode", "hybrid_auto")))
        self.speaker_framing_var.set(performance.get("speaker_framing_mode", "center_speaker"))
        self.detection_interval_entry.delete(0, "end")
        self.detection_interval_entry.insert(0, str(performance.get("detection_interval", 10)))
        self.encoder_var.set(performance.get("encoder", "auto"))
        self.codec_var.set(performance.get("codec", "h264"))
        self.decode_enabled_var.set(performance.get("decode_enabled", True))
        self.test_encoder_var.set(performance.get("test_encoder", False))
        self.yolo_model_entry.delete(0, "end")
        self.yolo_model_entry.insert(0, performance.get("yolo_model_path", ""))
        self.allow_yolo_download_var.set(performance.get("allow_yolo_download", False))
    
    def save_settings(self):
        # Handle both ConfigManager and dict
        if hasattr(self.config, 'config'):
            config_dict = self.config.config
        else:
            config_dict = self.config
            
        old_gpu_config = config_dict.get("gpu_acceleration", {})
        
        # Determine cache values to save
        cached_encoder = old_gpu_config.get("cached_encoder")
        cached_gpu_name = old_gpu_config.get("cached_gpu_name")
        cache_timestamp = old_gpu_config.get("cache_timestamp")
        
        if hasattr(self, 'latest_gpu_info') and hasattr(self, 'latest_recommendation') and not getattr(self, 'is_from_cache', False):
            if self.latest_gpu_info.get('available') and self.latest_recommendation.get('available'):
                import datetime
                cached_encoder = self.latest_recommendation.get('encoder')
                cached_gpu_name = self.latest_gpu_info.get('name')
                cache_timestamp = datetime.datetime.now().isoformat()
        
        config_dict["gpu_acceleration"] = {
            "enabled": self.gpu_enabled_var.get(),
            "cached_encoder": cached_encoder,
            "cached_gpu_name": cached_gpu_name,
            "cache_timestamp": cache_timestamp
        }

        try:
            detection_interval = int(self.detection_interval_entry.get().strip())
            if detection_interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Setting", "Detection interval must be a positive whole number.")
            return

        old_performance = config_dict.get("performance", {})
        config_dict["performance"] = {
            **old_performance,
            "profile": self.profile_var.get(),
            "encoder": self.encoder_var.get(),
            "codec": self.codec_var.get(),
            "detection_engine": self.detection_engine_var.get(),
            "speaker_framing_mode": self.speaker_framing_var.get(),
            "detection_interval": detection_interval,
            "prefer_gpu": self.gpu_enabled_var.get(),
            "fallback_to_cpu": True,
            "decode_enabled": self.decode_enabled_var.get(),
            "test_encoder": self.test_encoder_var.get(),
            "yolo_model_path": self.yolo_model_entry.get().strip(),
            "allow_yolo_download": self.allow_yolo_download_var.get(),
        }
        
        if self.on_save_callback:
            self.on_save_callback(config_dict)
        
        messagebox.showinfo("Success", "Performance settings saved!")
        self.on_back()
