"""Streaming I/O for the candidate pool. Handles plain .jsonl and gzipped .jsonl.gz."""

import gzip
import json


def _open_any(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_candidates(path):
    """Yield candidate dicts one at a time (memory-friendly for the 465 MB pool)."""
    with _open_any(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_candidates(path):
    """Load all candidates into a list. ~100K records fit comfortably in 16 GB."""
    return list(stream_candidates(path))
