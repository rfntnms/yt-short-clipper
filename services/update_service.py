"""
Update-checking helpers for the desktop app.
"""

import json
import urllib.request


USER_AGENT = "YT-Short-Clipper"


def build_update_url(base_url: str, installation_id: str, app_version: str) -> str:
    return f"{base_url}?installation_id={installation_id}&app_version={app_version}"


def fetch_update_info(base_url: str, installation_id: str, app_version: str, timeout: int = 5) -> dict:
    url = build_update_url(base_url, installation_id, app_version)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode())
    return {
        "version": data.get("version", ""),
        "download_url": data.get("download_url", ""),
        "changelog": data.get("changelog", ""),
    }


def compare_versions(v1: str, v2: str) -> int:
    """Compare version strings. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal."""
    try:
        parts1 = [int(x) for x in v1.split(".")]
        parts2 = [int(x) for x in v2.split(".")]

        max_len = max(len(parts1), len(parts2))
        parts1 += [0] * (max_len - len(parts1))
        parts2 += [0] * (max_len - len(parts2))

        for p1, p2 in zip(parts1, parts2):
            if p1 > p2:
                return 1
            if p1 < p2:
                return -1
        return 0
    except Exception:
        return 0
