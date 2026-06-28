"""
Feature extraction. Turns a raw candidate record into the structured signals the
scorer consumes. Pure functions, no model dependencies — so this layer is fast,
testable, and fully explainable in the interview.
"""

from datetime import date

from . import jd

REFERENCE_DATE = date(2026, 6, 1)


def _lower(s):
    return (s or "").lower()


def _title_class(title):
    """Classify a single title string into core / adjacent / off / leadership / other."""
    t = _lower(title)
    if not t:
        return "other"
    # Off-target check first, but a title like "data analyst" shouldn't be caught by
    # "analyst" in business analyst — we match full phrases from the lists.
    for phrase in jd.CORE_AI_TITLES:
        if phrase in t:
            return "core"
    for phrase in jd.NON_CODING_LEADERSHIP_TITLES:
        if phrase in t:
            return "leadership"
    for phrase in jd.OFF_TARGET_TITLES:
        if phrase in t:
            return "off"
    for phrase in jd.ADJACENT_ENG_TITLES:
        if phrase in t:
            return "adjacent"
    return "other"


def career_title_classes(candidate):
    """Return the set of title classes across the full career history + current title."""
    classes = []
    profile = candidate.get("profile", {})
    classes.append(_title_class(profile.get("current_title")))
    for job in candidate.get("career_history", []) or []:
        classes.append(_title_class(job.get("title")))
    return classes


def career_text(candidate):
    """Concatenate the free-text narrative used for semantic matching."""
    p = candidate.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for job in candidate.get("career_history", []) or []:
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    parts.append(" ".join(s.get("name", "") for s in candidate.get("skills", []) or []))
    return " ".join(x for x in parts if x)


def skill_trust_score(candidate):
    """
    Relevance-weighted, trust-discounted skill score in [0, 1].

    A skill only counts to the extent it is (a) relevant to the JD ontology and
    (b) *credible* — backed by endorsements, real duration of use, and a Redrob
    skill-assessment score. This is the core defence against keyword stuffers:
    listing 'Pinecone' as a skill with 0 endorsements, 0 months and no assessment
    earns almost nothing.
    """
    signals = candidate.get("redrob_signals", {}) or {}
    assessments = {k.lower(): v for k, v in (signals.get("skill_assessment_scores") or {}).items()}

    total = 0.0
    for s in candidate.get("skills", []) or []:
        name = _lower(s.get("name"))
        weight = jd.SKILL_WEIGHTS.get(name)
        if weight is None:
            continue
        # Trust multiplier in ~[0.2, 1.2]: proficiency + endorsements + tenure + assessment.
        prof = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}.get(
            s.get("proficiency", "intermediate"), 0.6
        )
        endorse = min(s.get("endorsements", 0) or 0, 50) / 50.0      # saturate at 50
        months = min(s.get("duration_months", 0) or 0, 60) / 60.0    # saturate at 5 yrs
        assess = assessments.get(name)
        assess_factor = (assess / 100.0) if assess is not None else 0.5  # neutral if untested
        trust = 0.25 + 0.30 * prof + 0.20 * endorse + 0.15 * months + 0.10 * assess_factor
        total += weight * min(trust, 1.0)

    # Normalise: ~6 strong, well-backed relevant skills should approach 1.0.
    return min(total / 5.0, 1.0)


def career_substance_score(candidate):
    """How much the career *descriptions* show real ranking/search/recsys/ML work."""
    text = " " + _lower(career_text(candidate)) + " "
    hits = sum(1 for phrase in jd.CAREER_SUBSTANCE_PHRASES if phrase in text)
    return min(hits / 6.0, 1.0)


def eval_bonus(candidate):
    text = " " + _lower(career_text(candidate)) + " "
    return min(sum(1 for p in jd.EVAL_PHRASES if p in text) / 2.0, 1.0)


def experience_fit(candidate):
    """Gaussian-ish band around the JD's ideal 6-8 years, soft edges to 5-9."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    if jd.EXP_IDEAL_LOW <= yoe <= jd.EXP_IDEAL_HIGH:
        return 1.0
    if jd.EXP_SOFT_LOW <= yoe <= jd.EXP_SOFT_HIGH:
        return 0.85
    # Decay outside the band; the JD says it will "seriously consider" out-of-band.
    if yoe < jd.EXP_SOFT_LOW:
        return max(0.0, 0.85 - 0.18 * (jd.EXP_SOFT_LOW - yoe))
    return max(0.2, 0.85 - 0.10 * (yoe - jd.EXP_SOFT_HIGH))


def location_fit(candidate):
    """JD: Pune/Noida preferred; Tier-1 India or willing-to-relocate ok; else low."""
    p = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {}) or {}
    loc = _lower(p.get("location"))
    country = _lower(p.get("country"))
    city = loc.split(",")[0].strip()
    relocate = bool(signals.get("willing_to_relocate"))

    if city in jd.PREFERRED_CITIES:
        return 1.0
    if city in jd.TIER1_INDIA_CITIES:
        return 0.9
    if country == "india":
        return 0.7 if relocate else 0.5
    # Outside India: JD is "case-by-case, no visa sponsorship" — only viable if relocating.
    return 0.4 if relocate else 0.2


def is_product_company_career(candidate):
    """Fraction of career spent at product companies vs services/consulting."""
    history = candidate.get("career_history", []) or []
    if not history:
        ind = _lower(candidate.get("profile", {}).get("current_industry"))
        return 1.0 if ind in jd.PRODUCT_INDUSTRIES else 0.5
    product_months = services_months = other_months = 0
    for job in history:
        dur = job.get("duration_months", 0) or 0
        ind = _lower(job.get("industry"))
        company = _lower(job.get("company"))
        is_consult = any(f in company for f in jd.CONSULTING_FIRMS)
        if is_consult or ind in jd.SERVICES_INDUSTRIES:
            services_months += dur
        elif ind in jd.PRODUCT_INDUSTRIES:
            product_months += dur
        else:
            other_months += dur
    total = product_months + services_months + other_months
    if total == 0:
        return 0.5
    return product_months / total


def lifelong_consulting(candidate):
    """JD hard penalty: entire career at named consulting firms with no product exp."""
    history = candidate.get("career_history", []) or []
    if not history:
        return False
    consult = 0
    for job in history:
        company = _lower(job.get("company"))
        ind = _lower(job.get("industry"))
        if any(f in company for f in jd.CONSULTING_FIRMS) or ind in jd.SERVICES_INDUSTRIES:
            consult += 1
    return consult == len(history)


def job_hopping_score(candidate):
    """
    Title-chaser detector. JD explicitly does not want people who switch companies
    every ~1.5 years optimizing for title. Returns a penalty factor in [0, 1] where
    1 = no concern. Only flags when there are several short stints.
    """
    history = candidate.get("career_history", []) or []
    durations = [j.get("duration_months", 0) or 0 for j in history if (j.get("duration_months") or 0) > 0]
    if len(durations) < 3:
        return 1.0
    avg = sum(durations) / len(durations)
    if avg >= 24:
        return 1.0
    if avg >= 18:
        return 0.9
    return 0.75  # average tenure under 18 months across 3+ jobs => title-chaser signal


def vision_speech_only(candidate):
    """
    JD does not want primary CV/speech/robotics expertise *without* NLP/IR exposure.
    Returns True only when vision/speech dominates and there is no NLP/IR signal.
    """
    text = " " + _lower(career_text(candidate)) + " "
    vision = any(k in text for k in ["computer vision", "image segmentation", "object detection",
                                     "opencv", "robotics", "speech recognition", "tts ", "asr ",
                                     "image classification", "lidar", "slam "])
    nlp_ir = any(k in text for k in ["nlp", "natural language", "retrieval", "search", "ranking",
                                     "recommendation", "embeddings", "information retrieval", "text "])
    return vision and not nlp_ir


def availability_multiplier(candidate):
    """
    Behavioral multiplier on availability (submission_spec + redrob_signals_doc).
    A perfect-on-paper candidate who is inactive / unresponsive is, for hiring
    purposes, not actually available — down-weight, don't eliminate. Range ~[0.5, 1.08].
    """
    s = candidate.get("redrob_signals", {}) or {}
    m = 1.0

    # Recency of last activity (strongest availability signal).
    last = s.get("last_active_date")
    try:
        days = (REFERENCE_DATE - date.fromisoformat(last)).days if last else 9999
    except (ValueError, TypeError):
        days = 9999
    if days <= 14:
        m *= 1.05
    elif days <= 45:
        m *= 1.0
    elif days <= 120:
        m *= 0.9
    elif days <= 180:
        m *= 0.78
    else:
        m *= 0.6                       # ~6+ months dark

    # Recruiter responsiveness.
    rr = s.get("recruiter_response_rate", 0.5)
    if rr is not None:
        m *= 0.7 + 0.4 * max(0.0, min(rr, 1.0))   # 0.7 .. 1.1

    # Open to work / actively in market.
    if s.get("open_to_work_flag"):
        m *= 1.04

    # Interview reliability and offer history (mild).
    icr = s.get("interview_completion_rate")
    if icr is not None:
        m *= 0.9 + 0.1 * max(0.0, min(icr, 1.0))

    # Recruiter demand pull (mild positive).
    saved = s.get("saved_by_recruiters_30d", 0) or 0
    if saved >= 5:
        m *= 1.03

    # Trust / verification (mild).
    if s.get("verified_email") and s.get("verified_phone"):
        m *= 1.02

    return max(0.45, min(m, 1.08))


def notice_period_factor(candidate):
    """JD prefers sub-30-day notice; 30+ raises the bar. Mild multiplier."""
    np_days = (candidate.get("redrob_signals", {}) or {}).get("notice_period_days", 60)
    if np_days is None:
        return 1.0
    if np_days <= 30:
        return 1.03
    if np_days <= 60:
        return 1.0
    if np_days <= 90:
        return 0.97
    return 0.93
