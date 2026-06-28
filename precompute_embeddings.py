#!/usr/bin/env python3
"""
Offline embedding precomputation.

submission_spec.md allows pre-computation outside the 5-minute ranking budget, as
long as the *ranking step* (rank.py) stays within it. This script embeds every
candidate's career narrative once with sentence-transformers and caches the matrix
to artifacts/embeddings.npz, keyed by candidate_id. rank.py then loads it instantly.

Usage:
    python precompute_embeddings.py --candidates ./candidates.jsonl
    python precompute_embeddings.py --candidates ./candidates.jsonl --model sentence-transformers/all-MiniLM-L6-v2

Needs network only to download the model the first time; the resulting cache and the
ranking step are fully offline.
"""

import argparse
import os
import time

from ranker import embeddings as E
from ranker import features as F
from ranker import io_utils

DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "artifacts", "embeddings.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default=DEFAULT_CACHE)
    ap.add_argument("--model", default=E.DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()

    embedder = E.Embedder(model_name=args.model)
    print(f"[precompute] backend = {embedder.method}, dim = {embedder.dim}")

    ids, texts = [], []
    for c in io_utils.stream_candidates(args.candidates):
        ids.append(c["candidate_id"])
        texts.append(F.career_text(c))
    print(f"[precompute] prepared {len(texts)} narratives in {time.time()-t0:.1f}s")

    matrix = embedder.encode(texts, batch_size=args.batch_size, show_progress=True)
    print(f"[precompute] embedded in {time.time()-t0:.1f}s -> matrix {matrix.shape}")

    E.save_cache(args.out, ids, matrix, embedder.method)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[precompute] saved {args.out} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
