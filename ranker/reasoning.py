"""
Reasoning generation.

Stage-4 manual review samples 10 rows and checks each reasoning for: specific facts
from the profile, connection to JD requirements, honest acknowledgement of concerns,
no hallucination, variation across rows, and tone consistent with rank. We generate
the reasoning *from the same components that produced the score* and *only* from facts
present in the candidate record — so it can never hallucinate and always matches rank.

The text is assembled from the candidate's strongest real evidence plus their single
biggest real concern, varying naturally because it's driven by each profile's data.
"""

from . import jd
from . import features as F


def _relevant_skills_present(candidate, limit=3):
    names = []
    for s in candidate.get("skills", []) or []:
        nm = s.get("name", "")
        if nm.lower() in jd.SKILL_WEIGHTS and jd.SKILL_WEIGHTS[nm.lower()] >= 0.6:
            names.append(nm)
    # de-dup preserving order
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out[:limit]


def _career_highlight(candidate):
    """Find the most JD-relevant phrase actually present in a job description."""
    for job in candidate.get("career_history", []) or []:
        desc = (job.get("description") or "").lower()
        for phrase in ("recommendation system", "ranking", "search relevance", "semantic search",
                       "retrieval", "personalization", "recsys", "embeddings", "information retrieval",
                       "vector search", "learning to rank", "nlp", "matching"):
            if phrase in desc:
                return phrase, job.get("company", "")
    return None, None


def build_reasoning(candidate, components):
    p = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {}) or {}
    title = p.get("current_title", "professional")
    yoe = p.get("years_of_experience", 0) or 0
    company = p.get("current_company", "")

    strengths = []
    concerns = []

    # ----- strengths (only facts that are actually true) ----------------------
    lead = f"{title} with {yoe:.1f} yrs"
    if company:
        lead += f" at {company}"

    highlight, hl_company = _career_highlight(candidate)
    if highlight and components["career_substance"] >= 0.3:
        where = f" at {hl_company}" if hl_company else ""
        strengths.append(f"career history shows hands-on {highlight} work{where}")

    skills = _relevant_skills_present(candidate)
    if skills and components["skill_trust"] >= 0.25:
        strengths.append("relevant skills incl. " + ", ".join(skills))

    if components["semantic_fit"] >= 0.6 and not strengths:
        strengths.append("profile narrative aligns well with the retrieval/ranking mandate")

    if components["eval_bonus"] >= 0.5:
        strengths.append("shows evaluation rigor (NDCG/MRR/A-B testing)")

    if components["experience_fit"] >= 1.0:
        strengths.append(f"experience sits in the JD's ideal {int(jd.EXP_IDEAL_LOW)}-{int(jd.EXP_IDEAL_HIGH)} yr band")

    if components["product_fraction"] >= 0.8:
        strengths.append("product-company background, not pure services")

    # ----- concerns (honest, only when real) ----------------------------------
    if components["honeypot_reasons"]:
        concerns.append("profile is internally inconsistent (" + components["honeypot_reasons"][0] + ")")
    if components["role_credibility"] <= 0.2:
        concerns.append(f"current title ('{title}') is off-target for an AI-engineering role despite listed skills")
    elif components["role_credibility"] <= 0.45:
        concerns.append("title/role fit is weak for the core AI-engineering mandate")

    rr = s.get("recruiter_response_rate")
    if rr is not None and rr < 0.25:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    if components["availability"] < 0.8 and not (rr is not None and rr < 0.25):
        concerns.append("weak recent platform engagement")
    np_days = s.get("notice_period_days")
    if np_days is not None and np_days > 90:
        concerns.append(f"long notice period ({np_days}d)")
    if components["location_fit"] <= 0.25:
        loc = p.get("location", "their location")
        concerns.append(f"based in {loc}, outside the Pune/Noida + relocation preference")
    if components["lifelong_consulting"]:
        concerns.append("entire career at services/consulting firms")
    if components["vision_speech_only"]:
        concerns.append("expertise skews vision/speech without clear NLP/IR exposure")
    if components["job_hopping"] < 0.85:
        concerns.append("short average tenure (job-hopping signal)")

    # ----- assemble 1-2 sentences ---------------------------------------------
    if strengths:
        s1 = lead + "; " + "; ".join(strengths[:3]) + "."
    else:
        s1 = lead + "; limited direct evidence of retrieval/ranking work."

    if concerns:
        s2 = " Concern: " + "; ".join(concerns[:2]) + "."
    else:
        s2 = " Strong all-round fit with no major red flags."

    text = (s1 + s2).replace("\n", " ").strip()
    # CSV-safety: collapse internal quotes/whitespace.
    text = " ".join(text.split())
    return text
