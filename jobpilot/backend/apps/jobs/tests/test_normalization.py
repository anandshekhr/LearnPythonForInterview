import pytest

from apps.jobs.normalization import (
    canonical_url,
    detect_seniority,
    detect_work_model,
    location_tags,
    normalize_title,
    parse_salary_text,
)
from apps.jobs.services.requirements import extract_requirements_rules


@pytest.mark.parametrize("title,expected", [
    ("Sr. Backend Dev", "senior backend developer"),
    ("Software Engineer II", "software engineer"),
    ("Full-Stack Engineer (Python/Angular)", "full stack engineer"),
])
def test_normalize_title(title, expected):
    assert normalize_title(title) == expected


@pytest.mark.parametrize("title,level", [
    ("Senior Python Developer", "senior"), ("Staff Engineer", "lead"), ("Engineering Manager", "manager"),
    ("Software Engineering Intern", "intern"), ("Backend Engineer", "mid"), ("Python + Angular Lead", "lead"),
])
def test_seniority(title, level):
    assert detect_seniority(title) == level


def test_work_model_detection():
    assert detect_work_model("Remote - India") == "remote"
    assert detect_work_model("Noida (Hybrid)") == "hybrid"
    assert detect_work_model("Pune", "This role requires working on-site 5 days.") == "onsite"
    assert detect_work_model("Pune") == "unknown"


@pytest.mark.parametrize("text,expected", [
    ("CTC: ₹30 - 40 LPA", (3_000_000, 4_000_000, "INR", "year")),
    ("Salary INR 25,00,000 - 32,00,000 per annum", (2_500_000, 3_200_000, "INR", "year")),
    ("$120k - $150k", (120_000, 150_000, "USD", "year")),
    ("3 - 5 years of experience", (None, None, "", "")),
])
def test_salary_parsing(text, expected):
    s = parse_salary_text(text)
    assert (s.minimum, s.maximum, s.currency, s.period) == expected


def test_location_tags_ncr():
    tags = location_tags(["Gurugram"])
    assert {"gurugram", "delhi ncr", "india"} <= tags
    assert "delhi ncr" in location_tags(["Delhi NCR"])


def test_canonical_url():
    assert canonical_url("HTTPS://Jobs.Lever.co/acme/123/?utm_source=x#apply") == "https://jobs.lever.co/acme/123"


def test_requirements_split_required_vs_preferred():
    text = "About\nGreat team\nRequirements\n- 5+ years of experience with Python\n- Django, PostgreSQL\nNice to have\n- Kubernetes\n- React"
    req = extract_requirements_rules("Senior Python Developer", text)
    assert req["required_skills"][:3] == ["Python", "Django", "PostgreSQL"]
    assert set(req["preferred_skills"]) == {"Kubernetes", "React"}
    assert req["min_years"] == 5


def test_requirements_without_sections_treats_all_as_required():
    req = extract_requirements_rules("Engineer", "You will use Go and Kafka daily. 3-5 years experience.")
    assert set(req["required_skills"]) == {"Go", "Kafka"}
    assert req["min_years"] == 3 and req["max_years"] == 5
