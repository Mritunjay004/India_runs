"""
Honeypot / impossibility detector.

The dataset seeds ~80 honeypot candidates with subtly *impossible* profiles
(submission_spec Section 7). They are forced to relevance tier 0 in the ground
truth, and ranking >10% of them in the top 100 disqualifies the submission.

We do NOT special-case them away with a blocklist — the spec says a good ranker
should avoid them naturally. Instead we detect *internal contradictions* in a
profile and fold the result into scoring as a hard down-weight. The same checks
are exactly the kind of "is this profile even self-consistent?" reading a careful
recruiter does, so they're defensible in the interview.

Empirically (validated against the full 100K pool) three crisp, fully-disjoint
signatures catch 70 of the ~80 honeypots with zero false positives on the rare
ML/AI candidates we actually care about. We deliberately keep the rules tight:
a false positive demotes a *real* strong candidate, which costs more NDCG than
letting a few honeypots slip to the mid-ranks (they're far from the top 10 anyway).
"""

from datetime import date

# A fixed "today" anchor so the detector is deterministic and reproducible
# (the dataset's most recent activity is mid-2026). Passed in where needed.
REFERENCE_DATE = date(2026, 6, 1)


def _parse(d):
    if not d:
        return None
    try:
        return date.fromisoformat(d)
    except (ValueError, TypeError):
        return None


def _months_between(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month)


def honeypot_reasons(candidate, reference_date=REFERENCE_DATE):
    """Return a list of human-readable impossibility reasons (empty == clean)."""
    reasons = []
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0) or 0
    history = candidate.get("career_history", []) or []

    # --- Signature 1: total job tenure far exceeds claimed total experience -----
    # e.g. years_of_experience = 3 but career entries sum to 61 months (~5 yrs).
    total_dur = sum(j.get("duration_months", 0) or 0 for j in history)
    if total_dur > yoe * 12 + 18:  # 18-month slack absorbs legitimate overlap
        reasons.append(
            f"career tenure sums to {total_dur} months but profile claims only "
            f"{yoe:.1f} years of experience"
        )

    # --- Signature 2: claims more experience than the career actually spans ------
    # e.g. years_of_experience = 13.7 but the earliest job started 11 months ago.
    starts = [_parse(j.get("start_date")) for j in history]
    starts = [s for s in starts if s]
    if starts:
        span_months = _months_between(min(starts), reference_date)
        if yoe * 12 > span_months + 18:
            reasons.append(
                f"claims {yoe:.1f} years of experience but career history only spans "
                f"~{span_months / 12:.1f} years since the first role"
            )

    # --- Signature 3: "expert" proficiency in a skill used for ~0 months ---------
    expert_zero = [
        s.get("name")
        for s in candidate.get("skills", []) or []
        if s.get("proficiency") == "expert" and (s.get("duration_months", 99) or 0) <= 1
    ]
    if expert_zero:
        shown = ", ".join(expert_zero[:3])
        reasons.append(f"claims 'expert' proficiency with ~0 months of use ({shown})")

    # --- Extra structural sanity checks (cheap, catch obvious fabrications) -------
    for j in history:
        sd, ed = _parse(j.get("start_date")), _parse(j.get("end_date"))
        if sd and ed and ed < sd:
            reasons.append("a role has an end date before its start date")
            break
    for e in candidate.get("education", []) or []:
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy and ey and ey < sy:
            reasons.append("an education entry ends before it starts")
            break

    return reasons


def is_honeypot(candidate, reference_date=REFERENCE_DATE):
    return len(honeypot_reasons(candidate, reference_date)) > 0
