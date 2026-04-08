"""
scripts/audit_sources.py

Audit source and topic coverage inside the Chroma collection.

Usage:
  python scripts/audit_sources.py
"""

import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import yaml
from chromadb.config import Settings


COLLECTION_NAME = "visa_rag_chunks"
TOPIC_EXPECTATIONS = {
    "employment_based_green_card": [
        "i-140",
        "permanent-workers",
        "processing-times",
        "green-card",
    ],
    "f1_opt_stem": [
        "optional-practical-training",
        "stem-opt",
        "curricular-practical-training",
        "students-and-employment",
    ],
    "travel_reentry": [
        "travel",
        "i-94",
        "advance-parole",
    ],
    "h1b": [
        "h-1b-specialty-occupations",
        "how-apply-h-1b",
        "how-apply-perm",
    ],
    "h4_ead": [
        "h-4-eda",
        "i-765",
    ],
    "processing_times": [
        "processing-times",
        "visa-bulletin",
    ],
}


def load_sources() -> list[dict]:
    with open("config/sources.yaml") as f:
        return yaml.safe_load(f)["sources"]


def get_collection():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def main():
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    total = len(metadatas)
    print(f"Total chunks in collection: {total}")
    print("=" * 60)

    by_source = defaultdict(lambda: {"chunks": 0, "urls": set()})
    by_form = defaultdict(int)
    by_group = defaultdict(list)

    for meta in metadatas:
        source_id = meta.get("source_id", "unknown")
        by_source[source_id]["chunks"] += 1
        url = meta.get("source_url", "")
        if url:
            by_source[source_id]["urls"].add(url)

        for form in str(meta.get("form_numbers", "")).split(","):
            form = form.strip()
            if form:
                by_form[form] += 1

    print("Source Coverage")
    for source in load_sources():
        source_id = source["id"]
        info = by_source.get(source_id, {"chunks": 0, "urls": set()})
        refresh_group = source.get("refresh_group", "unassigned")
        by_group[refresh_group].append(source_id)
        print(
            f"- {source_id:30s} "
            f"tier={source.get('tier')} "
            f"group={refresh_group:8s} "
            f"chunks={info['chunks']:4d} "
            f"urls={len(info['urls']):3d}"
        )

    print("\nRefresh Groups")
    for group in sorted(by_group):
        print(f"- {group}: {', '.join(by_group[group])}")

    print("\nForm Coverage")
    for form in sorted(by_form):
        print(f"- {form:8s} chunks={by_form[form]}")

    print("\nTopic Expectations")
    all_urls = [meta.get("source_url", "") for meta in metadatas]
    for topic, needles in TOPIC_EXPECTATIONS.items():
        matched = [needle for needle in needles if any(needle in url for url in all_urls)]
        missing = [needle for needle in needles if needle not in matched]
        status = "OK" if not missing else "GAP"
        print(f"- {topic}: {status}")
        print(f"  matched: {', '.join(matched) if matched else 'none'}")
        print(f"  missing: {', '.join(missing) if missing else 'none'}")


if __name__ == "__main__":
    main()
