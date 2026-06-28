"""
Unit tests for the parts of the ranker that encode the challenge's hard traps.
Run with:  python -m pytest -q   (or: python tests/test_ranker.py)

These tests are intentionally about *behavior the JD cares about*, not line coverage:
honeypots must be caught, keyword stuffers must lose to real engineers, plain-language
Tier-5s must survive, and behavioral twins must separate on availability.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranker import honeypot, scoring, features  # noqa: E402


def _base_signals(**over):
    s = {
        "profile_completeness_score": 90, "signup_date": "2022-01-01",
        "last_active_date": "2026-05-20", "open_to_work_flag": True,
        "profile_views_received_30d": 20, "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.8, "avg_response_time_hours": 5,
        "skill_assessment_scores": {}, "connection_count": 300,
        "endorsements_received": 50, "notice_period_days": 20,
        "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 60, "search_appearance_30d": 30,
        "saved_by_recruiters_30d": 6, "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.5, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    }
    s.update(over)
    return s


def _cand(title, skills, descs, yoe=7.0, location="Pune, Maharashtra", country="India",
          industry="Software", company="Acme", signals=None, history=None):
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User", "headline": title, "summary": " ".join(descs),
            "location": location, "country": country, "years_of_experience": yoe,
            "current_title": title, "current_company": company,
            "current_company_size": "201-500", "current_industry": industry,
        },
        "career_history": history or [{
            "company": company, "title": title, "start_date": "2019-01-01",
            "end_date": None, "duration_months": int(yoe * 12), "is_current": True,
            "industry": industry, "company_size": "201-500", "description": " ".join(descs),
        }],
        "education": [{"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
                       "start_year": 2012, "end_year": 2016, "grade": "8.5", "tier": "tier_1"}],
        "skills": [{"name": n, "proficiency": "advanced", "endorsements": 20,
                    "duration_months": 36} for n in skills],
        "redrob_signals": signals or _base_signals(),
    }


def _score(c, sem=0.7):
    return scoring.score_candidate(c, sem)[0]


# --- honeypots --------------------------------------------------------------
def test_honeypot_career_exceeds_experience():
    c = _cand("ML Engineer", ["Python"], ["built ranking"], yoe=3.0,
              history=[{"company": "X", "title": "ML Engineer", "start_date": "2023-11-09",
                        "end_date": None, "duration_months": 31, "is_current": True,
                        "industry": "Software", "company_size": "201-500", "description": "ml"},
                       {"company": "Y", "title": "Data Scientist", "start_date": "2022-02-17",
                        "end_date": "2023-11-09", "duration_months": 21, "is_current": False,
                        "industry": "Software", "company_size": "201-500", "description": "ds"},
                       {"company": "Z", "title": "ML Engineer", "start_date": "2021-05-23",
                        "end_date": "2022-02-17", "duration_months": 9, "is_current": False,
                        "industry": "Software", "company_size": "201-500", "description": "ml"}])
    assert honeypot.is_honeypot(c)


def test_honeypot_expert_zero_duration():
    c = _cand("ML Engineer", ["Python"], ["work"])
    c["skills"].append({"name": "Pinecone", "proficiency": "expert",
                        "endorsements": 0, "duration_months": 0})
    assert honeypot.is_honeypot(c)


def test_real_strong_candidate_is_not_a_honeypot():
    c = _cand("ML Engineer", ["FAISS", "Embeddings", "Python"],
              ["built a recommendation system and ranking model"])
    assert not honeypot.is_honeypot(c)


def test_honeypot_sinks_below_a_real_candidate():
    real = _cand("ML Engineer", ["FAISS", "Embeddings", "Ranking"],
                 ["built ranking and retrieval systems, NDCG evaluation"])
    hp = _cand("ML Engineer", ["FAISS", "Embeddings", "Ranking"],
               ["built ranking and retrieval systems"], yoe=2.0,
               history=[{"company": "X", "title": "ML Engineer", "start_date": "2024-01-01",
                         "end_date": None, "duration_months": 80, "is_current": True,
                         "industry": "Software", "company_size": "201-500", "description": "x"}])
    assert _score(real) > _score(hp) * 5


# --- keyword stuffer vs real engineer --------------------------------------
def test_keyword_stuffer_loses_to_real_engineer():
    stuffer = _cand("Marketing Manager",
                    ["FAISS", "Pinecone", "Embeddings", "RAG", "LLM", "Ranking", "NLP"],
                    ["managed marketing campaigns and brand strategy"])
    engineer = _cand("ML Engineer", ["FAISS", "Embeddings", "Ranking"],
                     ["built and shipped a ranking and retrieval system at scale"])
    assert _score(engineer, 0.8) > _score(stuffer, 0.5)


# --- plain-language Tier-5 survives -----------------------------------------
def test_plain_language_engineer_beats_offtitle_stuffer():
    plain = _cand("Backend Engineer", ["Python", "Elasticsearch"],
                  ["built the recommendation system and search relevance pipeline serving "
                   "real users; ran A/B tests and tracked NDCG"])
    stuffer = _cand("Sales Executive",
                    ["Pinecone", "FAISS", "Embeddings", "RAG", "LLM", "Ranking"],
                    ["hit sales quota, managed accounts"])
    assert _score(plain, 0.7) > _score(stuffer, 0.55)


# --- behavioral twins separate on availability ------------------------------
def test_behavioral_twin_inactive_loses():
    active = _cand("ML Engineer", ["FAISS", "Ranking"], ["built ranking"],
                   signals=_base_signals(last_active_date="2026-05-25",
                                         recruiter_response_rate=0.9))
    ghost = _cand("ML Engineer", ["FAISS", "Ranking"], ["built ranking"],
                  signals=_base_signals(last_active_date="2025-10-01",
                                        recruiter_response_rate=0.05))
    assert _score(active) > _score(ghost)


# --- location preference ----------------------------------------------------
def test_pune_beats_overseas_non_relocator():
    pune = _cand("ML Engineer", ["FAISS"], ["ranking"], location="Pune, Maharashtra",
                 country="India")
    overseas = _cand("ML Engineer", ["FAISS"], ["ranking"], location="Berlin",
                     country="Germany", signals=_base_signals(willing_to_relocate=False))
    assert _score(pune) > _score(overseas)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
