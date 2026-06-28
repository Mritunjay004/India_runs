"""
Sandbox demo (submission_spec Section 10.5).

A small hosted environment where the ranker can be run on a candidate sample:
upload a .jsonl/.json of <=100 candidates (or use the bundled sample) and get the
ranked CSV the same code path produces for the full pool. Runs on CPU well within
the 5-minute budget. Deploy to Streamlit Cloud or HuggingFace Spaces.

    streamlit run app.py
"""

import io
import json

import pandas as pd
import streamlit as st

from ranker import embeddings as E
from ranker import features as F
from ranker import scoring, reasoning
from ranker.jd import JD_QUERY_TEXT


@st.cache_resource
def get_embedder():
    return E.Embedder()


def rank_candidates(candidates):
    embedder = get_embedder()
    query_vec = embedder.encode([JD_QUERY_TEXT])[0]
    matrix = embedder.encode([F.career_text(c) for c in candidates])
    sem = E.cosine_to_query(matrix, query_vec)

    rows = []
    for c, s in zip(candidates, sem):
        final, comp = scoring.score_candidate(c, float(s))
        rows.append((c, round(final, 4), comp))
    rows.sort(key=lambda x: (-x[1], x[0]["candidate_id"]))

    out = []
    for i, (c, score, comp) in enumerate(rows, start=1):
        out.append({
            "candidate_id": c["candidate_id"], "rank": i, "score": f"{score:.4f}",
            "reasoning": reasoning.build_reasoning(c, comp),
        })
    return pd.DataFrame(out)


def parse_upload(raw):
    text = raw.decode("utf-8")
    try:                                   # try JSON array first
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:           # fall back to JSONL
        return [json.loads(line) for line in text.splitlines() if line.strip()]


st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob — Intelligent Candidate Ranker")
st.caption("Hybrid semantic + rule-based ranker for the Senior AI Engineer JD. "
           "Upload up to 100 candidates (JSONL or JSON array) to see the ranking.")

with st.expander("Backend / how it works"):
    st.write(f"Embedding backend: **{get_embedder().method}**")
    st.markdown(
        "Score = additive fit (semantic, career-substance, trust-weighted skills, "
        "experience, eval rigor, education) × gates (role-credibility, availability, "
        "location, product-vs-services, consulting/job-hop/vision penalties, honeypot kill)."
    )

upload = st.file_uploader("Candidate file (.jsonl or .json)", type=["jsonl", "json", "txt"])
top_n = st.slider("Show top N", 1, 100, 20)

if upload is not None:
    candidates = parse_upload(upload.read())[:100]
    st.write(f"Ranking **{len(candidates)}** candidates…")
    df = rank_candidates(candidates)
    st.dataframe(df.head(top_n), use_container_width=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download ranked CSV", buf.getvalue(),
                       file_name="submission.csv", mime="text/csv")
else:
    st.info("Upload a candidate sample to begin. "
            "The repo's artifacts/sample.jsonl (50 candidates) works as a demo input.")
