"""
scripts/refresh_sources.py

Run ingestion for a selected refresh group based on source metadata in config/sources.yaml.

Usage:
  python scripts/refresh_sources.py --group daily
  python scripts/refresh_sources.py --group weekly
  python scripts/refresh_sources.py --group monthly
  python scripts/refresh_sources.py --time-sensitive
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "sources.yaml"


def load_sources() -> list[dict]:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)["sources"]


def select_sources(
    group: str | None = None,
    time_sensitive: bool = False,
) -> list[dict]:
    sources = [s for s in load_sources() if s.get("enabled", True)]
    selected: list[dict] = []
    for source in sources:
        if group and source.get("refresh_group") != group:
            continue
        if time_sensitive and not source.get("time_sensitive", False):
            continue
        selected.append(source)
    return sorted(selected, key=lambda s: s.get("priority", 0), reverse=True)


def run_ingest(source_id: str) -> int:
    cmd = [sys.executable, "scripts/ingest.py", "--source", source_id]
    completed = subprocess.run(cmd, cwd=ROOT)
    return completed.returncode


def main():
    ap = argparse.ArgumentParser(description="Refresh sources by group")
    ap.add_argument("--group", choices=["daily", "weekly", "monthly"])
    ap.add_argument("--time-sensitive", action="store_true")
    args = ap.parse_args()

    if not args.group and not args.time_sensitive:
        ap.error("Specify --group or --time-sensitive")

    selected = select_sources(group=args.group, time_sensitive=args.time_sensitive)
    if not selected:
        print("No sources matched the requested refresh selection.")
        return

    print(f"Refreshing {len(selected)} source(s):")
    for source in selected:
        print(f"- {source['id']} ({source['name']})")

    failures = 0
    for source in selected:
        print(f"\n=== Refreshing {source['id']} ===")
        rc = run_ingest(source["id"])
        if rc != 0:
            failures += 1
            print(f"FAILED: {source['id']} (exit {rc})")

    print("\n" + "=" * 50)
    print(f"Refresh complete. failures={failures} total={len(selected)}")
    print("=" * 50)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
