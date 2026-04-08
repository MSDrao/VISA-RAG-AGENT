"""
scripts/ingest_daily.py

Convenience wrapper for volatile sources that should be refreshed daily.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    cmd = [sys.executable, "scripts/refresh_sources.py", "--group", "daily"]
    raise SystemExit(subprocess.run(cmd, cwd=ROOT).returncode)
