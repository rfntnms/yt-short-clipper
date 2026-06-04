"""
Small UI helpers shared by CustomTkinter pages.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from tkinter import messagebox


GITHUB_URL = "https://github.com/jipraks/yt-short-clipper"
DISCORD_URL = "https://s.id/ytsdiscord"


class PageNavigationMixin:
    """Common page actions expected by PageHeader and PageFooter."""

    def open_github(self):
        webbrowser.open(GITHUB_URL)

    def open_discord(self):
        webbrowser.open(DISCORD_URL)

    def show_page(self, page_name):
        """Delegate page navigation to the nearest parent app."""
        try:
            parent = self.master
            while parent and not hasattr(parent, "show_page"):
                parent = parent.master
            if parent and hasattr(parent, "show_page"):
                parent.show_page(page_name)
        except Exception:
            pass


def open_folder(folder, missing_message="Folder not found"):
    """Open a folder in the host file manager."""
    path = Path(folder)
    if not path.exists():
        messagebox.showerror("Error", missing_message)
        return

    if os.name == "nt":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)])
    else:
        subprocess.run(["xdg-open", str(path)])
