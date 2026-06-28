# Methodology

## ≤200-word summary

Hybrid ranker combining a precomputed semantic bi-encoder with an explicit,
auditable rule layer. Each candidate gets an **additive fit_base** — semantic
similarity (career narrative vs. a JD-intent query), career-substance (does the
job-history text actually describe building ranking/search/recsys systems),
trust-weighted skills (relevance × endorsements × duration × Redrob assessment,
so keyword-stuffing earns little), experience-band fit, evaluation-rigor mentions,
and education tier. That base is then passed through **multiplicative gates**:
role-credibility (the decisive anti-keyword-stuffer signal from title + career),
a behavioral availability multiplier (recency, recruiter response rate, open-to-work),
location preference (Pune/Noida → Tier-1 India → relocation), product-vs-services,
lifelong-consulting and job-hopping penalties, a vision/speech-without-NLP penalty,
notice-period, and a **honeypot kill factor** for internally impossible profiles.
The additive part rates how good a real engineer is; the gates decide whether we
believe it and can act on it. Embeddings are precomputed offline (CPU MiniLM) so the
ranking step is cosine + feature math — comfortably within the 5-min, no-network,
CPU-only budget. Reasoning strings are generated only from facts in the record.

## Why this design

The JD's own "note to hackathon participants" states the trap explicitly: the right
answer is reasoning about the gap between what the JD says and what it means. A
pure-embedding ranker ranks keyword stuffers and honeypots highly because their text
is full of the right words; a pure-keyword ranker misses the plain-language Tier-5 who
built a recommender at a product company but never wrote "RAG". The additive/gated
hybrid is built precisely to separate those cases.

## Honeypot handling

We do not maintain a blocklist. We detect three fully-disjoint impossibility signatures
that, on the full 100K pool, flag 70 candidates with no false positives among real
ML/AI profiles:

1. **career tenure ≫ stated experience** — job durations sum to far more than the
   profile's `years_of_experience` (e.g. 3.0 yrs claimed, 61 career-months).
2. **claims more experience than the career spans** — `years_of_experience` exceeds the
   months elapsed since the earliest role (e.g. 13.7 yrs claimed, career began 11 months ago).
3. **"expert" proficiency in a skill used ~0 months.**

These are exactly the self-consistency checks a careful recruiter performs. We keep the
rules tight on purpose: a false positive demotes a *real* strong candidate (costly for
NDCG), and honeypots are far from the top-10 anyway. A flagged profile is multiplied by
0.02, sinking it well below rank 100.

## Known tradeoffs / honest limitations

- The detector catches ~70 of ~80 honeypots; the remaining few rely on signals not
  present in the schema (e.g. company founding dates). They are still pushed down by the
  rule layer because their *roles* and *substance* are usually weak.
- Semantic quality is bounded by a small CPU model (MiniLM); we deliberately trade some
  retrieval quality for reproducibility within the compute budget, and lean on the rule
  layer to carry precision.
- Weights are hand-set from data analysis and the JD, not learned (no labels are
  available pre-scoring). They are centralized in `ranker/jd.py` for easy auditing.
