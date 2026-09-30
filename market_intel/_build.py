"""Build stamp, rewritten by publish.bat on every push.

Streamlit Cloud gives no notification when a rebuild finishes, and the browser
caches aggressively enough that a stale page looks exactly like a successful
deploy. So the app prints this stamp in the sidebar: push, wait a minute,
refresh, and if the timestamp matches the one publish.bat printed, your
changes are live. If it does not, they are not — check the Cloud logs.

This file is generated. Edit scripts/stamp_build.py instead.
"""

from __future__ import annotations

#: UTC timestamp of the last publish, as "YYYY-MM-DD HH:MM".
BUILD = "2026-09-30 07:44"
