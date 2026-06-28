#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Discovery & Ranking — main ranking step.

Reproduce command (Stage-3):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

This is the <=5-minute, CPU-only, no-network step. It loads the candidate pool and
the *precomputed* embedding cache (built offline by precompute_embeddings.py), scores
every candidate with the hybrid model in ranker/, and writes the top-100 CSV in the
exact format submission_spec.md Sections 2-3 require.

If the embedding cache is missing or does not match the active backend (e.g. running
on a small sandbox sample with no precompute), embeddings are computed inline — fine
for small inputs, and the hashing fallback keeps it dependency-light. For the full
100K pool, run precompute_embeddings.py first so this step stays well within 5 minutes.
"""

import argparse
import csv
import os
import time

import numpy as np

from ranker import embeddings as E
from ranker import features as F
from ranker import io_utils, scoring, reasoning
from ranker.jd import JD_QUERY_TEXT

DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "artifacts", "embeddings.npz")


def compute_semantic_fit(candidates, cache_path):
    """Return a numpy array of semantic_fit in [0,1], one per candidate, in order."""
    ids = [c["candidate_id"] for c in candidates]
    embedder = E.Embedder()  # prefers sentence-transformers, falls back to hashing
    query_vec = embedder.encode([JD_QUERY_TEXT])[0]

    cache = E.load_cache(cache_path)
    use_cache = False
    if cache is not None:
        id_index, matrix, method = cache
        if method == embedder.method and all(cid in id_index for cid in ids):
            use_cache = True

    if use_cache:
        rows = np.array([id_index[cid] for cid in ids])
        cand_matrix = matrix[rows]
        print(f"[rank] using cached embeddings ({method}) for {len(ids)} candidates")
    else:
        if cache is not None:
            print("[rank] cache missing/mismatched; embedding inline "
                  f"(backend={embedder.method}, n={len(ids)})")
        texts = [F.career_text(c) for c in candidates]
        cand_matrix = embedder.encode(texts, show_progress=len(texts) > 5000)

    return E.cosine_to_query(cand_matrix, query_vec)


def main():
    ap = argparse.ArgumentParser(description="Redrob hybrid candidate ranker")
    ap.add_argument("--candidates", required=True, help="candidates.jsonl(.gz)")
    ap.add_argument("--out", required=True, help="output submission CSV path")
    ap.add_argument("--embeddings", default=DEFAULT_CACHE, help="precomputed embedding cache (.npz)")
    ap.add_argument("--top", type=int, default=100, help="how many to rank (spec: 100)")
    args = ap.parse_args()

    t0 = time.time()
    candidates = io_utils.load_candidates(args.candidates)
    print(f"[rank] loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")

    semantic = compute_semantic_fit(candidates, args.embeddings)
    print(f"[rank] semantic fit computed at {time.time()-t0:.1f}s")

    scored = []
    for c, sem in zip(candidates, semantic):
        final, comp = scoring.score_candidate(c, float(sem))
        scored.append((c, final, comp))
    print(f"[rank] scored all candidates at {time.time()-t0:.1f}s")

    # Round first, then sort, so the on-disk scores satisfy the validator's
    # "non-increasing by rank, ties broken by candidate_id ascending" rule exactly.
    rounded = [(c, round(final, 4), comp) for c, final, comp in scored]
    rounded.sort(key=lambda x: (-x[1], x[0]["candidate_id"]))
    top = rounded[: args.top]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (c, score, comp) in enumerate(top, start=1):
            text = reasoning.build_reasoning(c, comp)
            w.writerow([c["candidate_id"], i, f"{score:.4f}", text])

    print(f"[rank] wrote {len(top)} rows to {args.out} in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
