# Codebase Recon Report

Generated: 2026-06-04

## Repo Vitals

Age: 2026-01-15 to 2026-06-04 | Commits: 64 | Branches: 3 | Analysis window: all time

## 1. Code Hotspots

Most-changed files:

```text
26  clipper_core.py
22  app.py
17  version.py
15  pages/settings_page.py
13  requirements.txt
11  README.md
10  .gitignore
 9  config/config_manager.py
 7  pages/processing_page.py
 6  utils/helpers.py
```

## 2. Bug Magnets

Files most associated with fix, bug, or broken commits:

```text
13  clipper_core.py
 8  version.py
 6  app.py
 3  requirements.txt
 3  pages/processing_page.py
 3  README.md
 3  .gitignore
 2  utils/gpu_detector.py
 2  utils/dependency_manager.py
 2  pages/status_pages.py
```

## 3. High-Risk Files

Files appearing in both hotspots and bug magnets:

```text
clipper_core.py            hotspot #1, bug magnet #1, owner: jipraks
app.py                     hotspot #2, bug magnet #3, owner: jipraks
version.py                 hotspot #3, bug magnet #2, owner: jipraks
requirements.txt           hotspot #5, bug magnet #4, owner: jipraks
README.md                  hotspot #6, bug magnet #6, owner: jipraks
.gitignore                 hotspot #7, bug magnet #7, owner: jipraks
pages/processing_page.py   hotspot #9, bug magnet #5, owner: jipraks
```

## 4. Bus Factor

Top contributors:

```text
53  jipraks
 3  arihidayatm
 3  rfntnms
```

Active last 3 months: 2 of 3 total contributors.

No low active-contributor warning, though ownership is heavily concentrated around `jipraks`.

## 5. Team Momentum

Monthly commit counts:

```text
37  2026-01
11  2026-02
 2  2026-03
 3  2026-04
 8  2026-05
 3  2026-06
```

Trend: declining overall. Last 3 months average is approximately 3.7 commits/month versus approximately 16.7 in the prior 3 months.

## 6. Firefighting Frequency

None found.

Rate: 0 emergency, revert, or hotfix commits out of 64 total commits, 0%.

## 7. Recently Added Files

```text
1  webview_app.py
1  web/index.html
1  web/css/layout.css
1  web/css/components.css
1  web/css/base.css
1  web/components/home.js
1  web/components/header.js
1  web/components/ai-settings.js
1  web/app.js
1  version.py
```

## 8. Recommendations

- Start reading: `clipper_core.py`, `app.py`, `version.py`.
- Talk to: `jipraks`, especially for pipeline behavior in `clipper_core.py`.
- Watch out: `clipper_core.py` is both the top hotspot and top bug magnet, so changes there deserve extra testing. Momentum has slowed since the initial January push, and most code knowledge appears concentrated in one contributor.
