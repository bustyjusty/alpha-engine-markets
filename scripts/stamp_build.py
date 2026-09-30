"""Write the build stamp that the sidebar displays.

Called by publish.bat immediately before the commit, so the timestamp that
ships is the moment of the push. Printed to stdout too, so the terminal and
the live site can be compared without digging.
"""

from __future__ import annotations

import datetime as dt
import pathlib

TARGET = pathlib.Path(__file__).resolve().parent.parent / "market_intel" / "_build.py"

TEMPLATE = '''"""Build stamp, rewritten by publish.bat on every push.

Streamlit Cloud gives no notification when a rebuild finishes, and the browser
caches aggressively enough that a stale page looks exactly like a successful
deploy. So the app prints this stamp in the sidebar: push, wait a minute,
refresh, and if the timestamp matches the one publish.bat printed, your
changes are live. If it does not, they are not — check the Cloud logs.

This file is generated. Edit scripts/stamp_build.py instead.
"""

from __future__ import annotations

#: UTC timestamp of the last publish, as "YYYY-MM-DD HH:MM".
BUILD = "{stamp}"
'''


def main() -> int:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M")
    TARGET.write_text(TEMPLATE.format(stamp=stamp), encoding="utf-8")
    print(f"Build stamp: {stamp} UTC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
