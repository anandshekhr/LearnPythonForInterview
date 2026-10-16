from datetime import date

import pytest

from apps.core.exceptions import DomainError
from apps.core.testing import make_docx, make_pdf, sample_resume_text
from apps.profiles.services import parser
from apps.profiles.services.validation import validate_resume_upload
from django.core.files.uploadedfile import SimpleUploadedFile


def test_parse_dates():
    assert parser.parse_date("Jan 2021") == date(2021, 1, 1)
    assert parser.parse_date("Sept 2019") == date(2019, 9, 1)
    assert parser.parse_date("03/2018") == date(2018, 3, 1)
    assert parser.parse_date("2017") == date(2017, 1, 1)
    assert parser.parse_date("nonsense") is None


def test_structured_extraction_from_text():
    result = parser.parse_resume_text(sample_resume_text(), use_llm=False)
    assert result["contact"]["full_name"] == "Asha Verma"
    assert result["contact"]["email"] == "asha.verma@example.com"
    assert result["contact"]["linkedin_url"].startswith("https://linkedin.com/in/")
    jobs = result["employment"]
    assert [j["company"] for j in jobs] == ["Nimbus Analytics Pvt Ltd", "Orbit Software Solutions", "Pixel Labs"]
    assert jobs[0]["title"] == "Senior Software Engineer"
    assert jobs[0]["is_current"] is True and jobs[0]["end_date"] is None
    assert jobs[1]["start_date"] == "2018-07-01" and jobs[1]["end_date"] == "2020-12-01"
    assert jobs[2]["title"] == "Associate Developer"
    assert len(jobs[0]["achievements"]) == 3
    assert "Celery" in jobs[0]["technologies"]
    skills = {s["name"] for s in result["skills"]}
    assert {"Python", "Django", "Django REST Framework", "Angular", "PostgreSQL", "AWS"} <= skills
    assert result["education"][0]["institution"] == "Delhi Technological University"
    assert result["education"][0]["end_year"] == 2017
    assert result["projects"][0]["name"] == "ResumeRanker"
    assert result["certifications"][0]["name"].startswith("AWS Certified Developer")
    assert result["stated_experience_years"] == 7.0
    assert result["total_experience_years"] >= 9.0  # Jun 2017 -> today, merged


def test_overlapping_intervals_not_double_counted():
    entries = [
        {"start_date": "2020-01-01", "end_date": "2020-12-01", "is_current": False},
        {"start_date": "2020-06-01", "end_date": "2021-05-01", "is_current": False},
    ]
    assert parser.total_experience_years(entries) == 1.4  # Jan 2020 – May 2021 = 17 months


@pytest.mark.parametrize("maker,name,kind", [(make_docx, "cv.docx", "docx"), (make_pdf, "cv.pdf", "pdf")])
def test_extract_text_from_real_files(maker, name, kind):
    data = maker()
    upload = SimpleUploadedFile(name, data)
    got_kind, _ct, payload, sha = validate_resume_upload(upload)
    assert got_kind == kind and len(sha) == 64
    text = parser.extract_text(kind, payload)
    result = parser.parse_resume_text(text, use_llm=False)
    assert result["contact"]["email"] == "asha.verma@example.com"
    assert len(result["employment"]) == 3


@pytest.mark.parametrize(
    "name,data,code",
    [
        ("cv.exe", b"MZ...", "unsupported_file_type"),
        ("cv.pdf", b"not a pdf at all", "invalid_file"),
        ("cv.docx", b"%PDF-1.4 fake", "invalid_file"),
        ("cv.pdf", b"", "empty_file"),
    ],
)
def test_rejects_bad_uploads(name, data, code):
    with pytest.raises(DomainError) as exc:
        validate_resume_upload(SimpleUploadedFile(name, data))
    assert exc.value.code == code


def test_rejects_oversized(settings):
    settings.RESUME_MAX_UPLOAD_BYTES = 10
    with pytest.raises(DomainError) as exc:
        validate_resume_upload(SimpleUploadedFile("cv.pdf", make_pdf()))
    assert exc.value.code == "file_too_large"


def test_rejects_macro_docx():
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<w:document/>")
        zf.writestr("word/vbaProject.bin", "x")
    with pytest.raises(DomainError) as exc:
        validate_resume_upload(SimpleUploadedFile("cv.docx", buf.getvalue()))
    assert exc.value.code == "unsafe_file"


def test_llm_output_is_grounded(monkeypatch, settings):
    """Hallucinated employers/skills from the LLM must be dropped."""
    from apps.core import llm

    settings.LLM_PROVIDER = "anthropic"
    fake = {
        "headline": "Senior Python Developer",
        "summary": "",
        "skills": ["Python", "Haskell"],
        "employment": [
            {"company": "Nimbus Analytics Pvt Ltd", "title": "Senior Software Engineer", "location": "", "start_date": "2021-01-01",
             "end_date": None, "is_current": True, "achievements": ["x"]},
            {"company": "Orbit Software Solutions", "title": "Software Engineer", "location": "", "start_date": "2018-07-01",
             "end_date": "2020-12-01", "is_current": False, "achievements": []},
            {"company": "Pixel Labs", "title": "Associate Developer", "location": "", "start_date": "2017-06-01",
             "end_date": "2018-06-01", "is_current": False, "achievements": []},
            {"company": "Google", "title": "Staff Engineer", "location": "", "start_date": "2010-01-01",
             "end_date": "2012-01-01", "is_current": False, "achievements": []},
        ],
        "education": [],
    }
    monkeypatch.setattr(llm, "generate_json", lambda *a, **k: llm.LLMResult(fake, "anthropic", "test"))
    result = parser.parse_resume_text(sample_resume_text())
    companies = [e["company"] for e in result["employment"]]
    assert "Google" not in companies
    assert "Haskell" not in {s["name"] for s in result["skills"]}
    assert any("Ignored AI output" in w for w in result["warnings"])
    assert result["parser"] == "rules+anthropic"
