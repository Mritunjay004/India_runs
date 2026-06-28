"""
Semantic embedding layer (the "dense" half of the hybrid ranker).

Primary path: sentence-transformers (all-MiniLM-L6-v2 by default) — a compact,
CPU-friendly bi-encoder. Embeddings for the 100K pool are produced *offline* by
precompute_embeddings.py (this can exceed the 5-minute ranking budget; the spec
explicitly allows pre-computation). rank.py then just loads the cached matrix and
does cosine similarity, which is milliseconds.

Fallback path: if sentence-transformers/torch are unavailable (e.g. a minimal
Stage-3 sandbox), we degrade gracefully to a stateless hashing term-frequency
vectorizer. It is deterministic, needs only numpy, and keeps the JD query and the
candidates in the *same* space so cosine is meaningful. Quality is lower than the
transformer, but the rule-based components carry the ranking and the pipeline
never breaks. This graceful degradation is intentional engineering, not an accident.

All vectors are L2-normalized so similarity == dot product.
"""

import hashlib
import os

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HASH_DIM = 4096


def _l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Stateless hashing TF vectorizer (fallback backend)
# ---------------------------------------------------------------------------
def _hash_token(tok):
    h = hashlib.md5(tok.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % _HASH_DIM


def _hash_embed(texts):
    out = np.zeros((len(texts), _HASH_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        toks = "".join(c if c.isalnum() else " " for c in (text or "").lower()).split()
        for tok in toks:
            if len(tok) < 2:
                continue
            out[i, _hash_token(tok)] += 1.0
    # sublinear tf damping, then L2 normalize
    np.log1p(out, out=out)
    return _l2_normalize(out)


# ---------------------------------------------------------------------------
# Embedder: picks the best available backend
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL, prefer_transformer=True):
        self.model_name = model_name
        self.method = "hash"
        self._model = None
        if prefer_transformer:
            try:
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name, device="cpu")
                self.method = "st:" + model_name
            except Exception as exc:  # noqa: BLE001 - any failure => fallback
                print(f"[embeddings] sentence-transformers unavailable ({exc}); "
                      f"falling back to hashing vectorizer.")
                self._model = None
                self.method = "hash"

    @property
    def dim(self):
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return _HASH_DIM

    def encode(self, texts, batch_size=256, show_progress=False):
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is not None:
            vecs = self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            return vecs.astype(np.float32)
        return _hash_embed(texts)


# ---------------------------------------------------------------------------
# Cache I/O for the precomputed candidate matrix
# ---------------------------------------------------------------------------
def save_cache(path, ids, matrix, method):
    np.savez_compressed(
        path,
        ids=np.array(ids),
        matrix=matrix.astype(np.float32),
        method=np.array(method),
    )


def load_cache(path):
    """Return (id->row_index dict, matrix, method) or None if absent/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        ids = [str(x) for x in data["ids"].tolist()]
        matrix = data["matrix"]
        method = str(data["method"])
        return {cid: i for i, cid in enumerate(ids)}, matrix, method
    except Exception as exc:  # noqa: BLE001
        print(f"[embeddings] could not load cache {path}: {exc}")
        return None


def cosine_to_query(matrix, query_vec):
    """Cosine of each row against the (already normalized) query vector -> [0,1]."""
    sims = matrix @ query_vec.astype(np.float32)
    return np.clip((sims + 1.0) / 2.0, 0.0, 1.0)  # map [-1,1] -> [0,1]
