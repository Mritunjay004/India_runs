"""
The scoring engine — where the hybrid comes together.

Design (defensible, auditable, interview-ready):

  fit_base  = additive blend of evidence the candidate *can do the job*
              (semantic match, career substance, trust-weighted skills,
               experience band, evaluation-thinking, education tier)

  final     = fit_base
              * role_credibility   <- decisive anti-keyword-stuffer gate
              * availability       <- behavioral / "actually hireable" multiplier
              * location           <- JD location preference
              * product_vs_services
              * consulting_penalty
              * job_hopping
              * vision_speech_penalty
              * notice_period
              * honeypot_kill       <- impossible profiles sink to the bottom

The additive part decides *how good* a real engineer is; the multiplicative gates
decide *whether we believe it and can act on it*. Keyword stuffers die on
role_credibility, behavioral twins separate on availability, plain-language Tier-5s
survive because semantic + career_substance don't need the buzzwords, and honeypots
are driven below rank 100 by the kill factor.
"""

from . import features as F
from . import honeypot as HP

TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4, "unknown": 0.5}


def education_tier_score(candidate):
    edus = candidate.get("education", []) or []
    if not edus:
        return 0.5
    return max(TIER_SCORE.get(e.get("tier", "unknown"), 0.5) for e in edus)


def role_credibility(candidate):
    """
    How much we believe this person is (or was) really an AI/ML/software engineer.
    The single most important defence against keyword-stuffer traps.
    """
    classes = F.career_title_classes(candidate)
    cur = classes[0] if classes else "other"
    history_classes = set(classes[1:])
    has_core = "core" in classes
    has_adjacent = "adjacent" in classes

    if cur == "core":
        return 1.0
    if cur == "adjacent":
        return 0.88
    if cur == "leadership":               # JD: "moved into architecture, doesn't code"
        return 0.58 if has_core else 0.42
    if cur == "off":
        # Current title is something the JD doesn't want (HR/Marketing/Sales/Mechanical...).
        if has_core:
            return 0.6                     # genuinely was an ML engineer; unusual pivot
        if has_adjacent:
            return 0.4
        return 0.12                        # pure keyword stuffer: off-target + no eng career
    # "other" / unclassified current title
    if has_core:
        return 0.75
    if has_adjacent:
        return 0.6
    return 0.4


def score_candidate(candidate, semantic_fit, reference_date=HP.REFERENCE_DATE):
    """
    Returns (final_score, components) where components is a dict used both for
    debugging and for generating honest, grounded reasoning strings.
    """
    # --- evidence the candidate can do the job (additive) ----------------------
    career_sub = F.career_substance_score(candidate)
    skill_trust = F.skill_trust_score(candidate)
    exp_fit = F.experience_fit(candidate)
    ev_bonus = F.eval_bonus(candidate)
    edu = education_tier_score(candidate)

    fit_base = (
        0.30 * semantic_fit
        + 0.24 * career_sub
        + 0.20 * skill_trust
        + 0.12 * exp_fit
        + 0.09 * ev_bonus
        + 0.05 * edu
    )

    # --- gates (multiplicative) -----------------------------------------------
    cred = role_credibility(candidate)
    avail = F.availability_multiplier(candidate)
    loc_fit = F.location_fit(candidate)
    loc_mult = 0.5 + 0.5 * loc_fit                          # 0.6 .. 1.0
    product_frac = F.is_product_company_career(candidate)
    product_mult = 0.7 + 0.3 * product_frac                 # 0.7 .. 1.0
    consulting_mult = 0.5 if F.lifelong_consulting(candidate) else 1.0
    job_hop = F.job_hopping_score(candidate)
    vision_mult = 0.55 if F.vision_speech_only(candidate) else 1.0
    notice = F.notice_period_factor(candidate)

    hp_reasons = HP.honeypot_reasons(candidate, reference_date)
    honeypot_mult = 0.02 if hp_reasons else 1.0             # drive impossibles below rank 100

    final = (
        fit_base
        * cred
        * avail
        * loc_mult
        * product_mult
        * consulting_mult
        * job_hop
        * vision_mult
        * notice
        * honeypot_mult
    )
    final = max(0.0, min(final, 1.0))

    components = {
        "semantic_fit": semantic_fit,
        "career_substance": career_sub,
        "skill_trust": skill_trust,
        "experience_fit": exp_fit,
        "eval_bonus": ev_bonus,
        "education_tier": edu,
        "fit_base": fit_base,
        "role_credibility": cred,
        "availability": avail,
        "location_fit": loc_fit,
        "product_fraction": product_frac,
        "lifelong_consulting": consulting_mult < 1.0,
        "job_hopping": job_hop,
        "vision_speech_only": vision_mult < 1.0,
        "notice_factor": notice,
        "honeypot_reasons": hp_reasons,
        "final": final,
    }
    return final, components
