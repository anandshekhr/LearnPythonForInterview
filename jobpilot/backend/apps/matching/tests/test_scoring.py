from apps.matching.scoring import CandidateFacts, JobFacts, score_match

WEIGHTS = {"required_skills": 30, "preferred_skills": 10, "experience": 15, "role_alignment": 15, "semantic": 10,
           "location": 10, "salary": 5, "employment_type": 5}
THRESHOLDS = {"excellent": 80, "good": 65, "possible": 45}


def candidate(**kw):
    base = dict(
        verified_skills={"Python", "Django", "PostgreSQL", "Redis", "Docker", "Angular"}, unverified_skills={"Kubernetes"},
        years_experience=7, target_roles=["Senior Python Developer", "Technical Lead"],
        preferred_locations=["Noida", "Delhi NCR", "India"], work_models=["hybrid", "remote"], employment_types=["full_time"],
        salary_min=3_000_000, salary_max=4_000_000, salary_currency="INR", open_to_relocation=False,
        excluded_companies=[], excluded_keywords=[],
        experiences=[{"kind": "employment", "label": "Senior Engineer at Nimbus", "title": "Senior Engineer",
                      "skills": {"Python", "Django"}, "text": ""}],
    )
    base.update(kw)
    return CandidateFacts(**base)


def job(**kw):
    base = dict(title="Senior Python Developer", company="Acme", seniority="senior",
                required_skills=["Python", "Django", "PostgreSQL"], preferred_skills=["Kubernetes"], min_years=5, max_years=None,
                work_model="hybrid", locations=["Noida"], location_raw="Noida", employment_type="full_time",
                salary_min=3_200_000, salary_max=4_200_000, salary_currency="INR")
    base.update(kw)
    return JobFacts(**base)


def run(j=None, c=None, sim=0.4):
    return score_match(j or job(), c or candidate(), WEIGHTS, THRESHOLDS, sim)


def test_strong_match_is_excellent_and_explained():
    r = run()
    assert r.category == "excellent" and r.score >= 85
    assert r.skills_matched == ["Python", "Django", "PostgreSQL"]
    assert r.skills_unverified == ["Kubernetes"]  # listed but unverified: no credit, flagged
    assert r.signals["preferred_skills"]["score"] == 0
    assert "Strengths" in r.explanation and r.recommended_action.startswith("Strong fit")
    assert r.relevant_experience[0]["label"] == "Senior Engineer at Nimbus"


def test_missing_salary_is_not_a_penalty():
    with_salary = run()
    without = run(job(salary_min=None, salary_max=None))
    assert without.signals["salary"]["score"] is None
    assert abs(without.score - with_salary.score) <= 1
    assert "salary" in without.explanation.lower()


def test_unknown_information_lowers_coverage_not_score():
    r = run(job(required_skills=[], preferred_skills=[], min_years=None, salary_min=None, salary_max=None, employment_type=""))
    assert r.coverage < 0.6 and "indicative only" in r.explanation


def test_missing_required_skills_capped():
    r = run(job(required_skills=["Go", "Kafka", "Rust", "Python"], preferred_skills=[]), sim=0.5)
    assert r.score < THRESHOLDS["good"]
    assert "Go" in r.skills_missing and any("Required skills not in your profile" in g for g in r.gaps)


def test_experience_gap_reported():
    r = run(job(min_years=10))
    assert r.signals["experience"]["score"] < 1 and any("Experience gap" in g for g in r.gaps)


def test_location_outside_preferences():
    r = run(job(locations=["Berlin"], location_raw="Berlin, Germany", work_model="onsite"))
    assert r.signals["location"]["score"] == 0.1
    r2 = run(job(locations=["Berlin"], location_raw="Berlin, Germany", work_model="onsite"), candidate(open_to_relocation=True))
    assert r2.signals["location"]["score"] == 0.4


def test_remote_in_india_accepted():
    r = run(job(work_model="remote", locations=["Remote - India"], location_raw="Remote - India"))
    assert r.signals["location"]["score"] == 1.0


def test_salary_below_expectation_partial():
    r = run(job(salary_min=1_800_000, salary_max=2_400_000))
    assert 0 < r.signals["salary"]["score"] < 1 and "Below expectation" in r.salary_fit


def test_different_currency_not_compared():
    r = run(job(salary_min=120_000, salary_max=150_000, salary_currency="USD"))
    assert r.signals["salary"]["score"] is None


def test_excluded_company_capped():
    r = run(c=candidate(excluded_companies=["acme"]))
    assert r.category in {"low"} and r.excluded_reason and "Excluded" in r.recommended_action


def test_excluded_keyword():
    r = run(job(description="Must be willing to work night shifts"), candidate(excluded_keywords=["night shifts"]))
    assert "night shifts" in r.excluded_reason


def test_seniority_mismatch_reduces_role_alignment():
    junior = run(job(title="Junior Python Developer", seniority="junior"))
    assert junior.signals["role_alignment"]["score"] < run().signals["role_alignment"]["score"]


def test_weights_are_configurable():
    only_skills = {k: 0 for k in WEIGHTS} | {"required_skills": 1}
    r = score_match(job(required_skills=["Python", "Go"]), candidate(), only_skills, THRESHOLDS, 0.9)
    assert r.score == 50
