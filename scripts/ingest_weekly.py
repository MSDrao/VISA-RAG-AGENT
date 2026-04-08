"""
scripts/ingest_weekly.py

Convenience wrapper for slower-changing policy and operational sources.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    cmd = [sys.executable, "scripts/refresh_sources.py", "--group", "weekly"]
    raise SystemExit(subprocess.run(cmd, cwd=ROOT).returncode)
