import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from apps.core.models import AuditLog
from apps.core.testing import auth_client, make_docx, make_pdf, make_user
from apps.profiles.models import CandidateSkill, EmploymentHistory, Resume, ResumeVersion

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def _media(tmp_path, settings):
    settings.PRIVATE_MEDIA_ROOT = tmp_path


def test_register_login_refresh_logout(client):
    from rest_framework.test import APIClient

    api = APIClient()
    r = api.post("/api/v1/auth/register/", {"email": "New@Example.com", "password": "Correct-Horse-9", "full_name": "N"}, format="json")
    assert r.status_code == 201, r.content
    assert r.data["user"]["email"] == "new@example.com"
    r = api.post("/api/v1/auth/login/", {"email": "new@example.com", "password": "Correct-Horse-9"}, format="json")
    assert r.status_code == 200
    refresh = r.data["refresh"]
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {r.data['access']}")
    assert api.get("/api/v1/auth/me/").data["email"] == "new@example.com"
    assert api.post("/api/v1/auth/logout/", {"refresh": refresh}, format="json").status_code == 205
    assert api.post("/api/v1/auth/refresh/", {"refresh": refresh}, format="json").status_code == 401


def test_weak_password_rejected():
    from rest_framework.test import APIClient

    r = APIClient().post("/api/v1/auth/register/", {"email": "a@example.com", "password": "password123"}, format="json")
    assert r.status_code == 400


def test_unauthenticated_access_denied():
    from rest_framework.test import APIClient

    assert APIClient().get("/api/v1/profile/").status_code == 401


def test_profile_defaults_and_preferences_update():
    user = make_user()
    api = auth_client(user)
    r = api.get("/api/v1/profile/")
    assert r.status_code == 200
    prefs = r.data["preferences"]
    assert "Python Backend Developer" in prefs["target_roles"]
    assert prefs["salary_min"] == 3_000_000 and prefs["work_models"] == ["hybrid", "remote"]
    r = api.patch("/api/v1/profile/preferences/", {"salary_min": 3_500_000, "salary_max": 4_500_000, "work_models": ["remote"]}, format="json")
    assert r.status_code == 200 and r.data["work_models"] == ["remote"]
    r = api.patch("/api/v1/profile/preferences/", {"salary_min": 9_000_000}, format="json")
    assert r.status_code == 400
    r = api.patch("/api/v1/profile/preferences/", {"work_models": ["moon"]}, format="json")
    assert r.status_code == 400
    r = api.patch("/api/v1/profile/", {"total_experience_years": "7.0", "headline": "Python + Angular lead"}, format="json")
    assert r.status_code == 200 and r.data["total_experience_years"] == "7.0"


@pytest.mark.parametrize("maker,name", [(make_docx, "asha.docx"), (make_pdf, "asha.pdf")])
def test_upload_parse_review_apply(maker, name):
    user = make_user()
    api = auth_client(user)
    payload = maker()
    upload = SimpleUploadedFile(name, payload)
    r = api.post("/api/v1/resumes/", {"file": upload, "title": "Master CV"}, format="multipart")
    assert r.status_code == 201, r.content
    resume_id = r.data["id"]
    # Celery runs eagerly in tests: parse has completed on commit.
    r = api.get(f"/api/v1/resumes/{resume_id}/")
    assert r.data["parse_status"] == "parsed", r.data
    draft = r.data["parsed_data"]
    assert len(draft["employment"]) == 3
    assert "raw_text" not in r.data
    # The user corrects a title before confirming.
    draft["employment"][2]["title"] = "Associate Software Developer"
    draft["total_experience_years"] = 7
    r = api.post(f"/api/v1/resumes/{resume_id}/apply/", {"data": draft}, format="json")
    assert r.status_code == 200, r.content
    assert r.data["total_experience_years"] == "7.0"
    assert r.data["completeness"]["checks"]["experience"] is True
    assert EmploymentHistory.objects.filter(profile__user=user, verified=True).count() == 3
    assert EmploymentHistory.objects.filter(title="Associate Software Developer").exists()
    assert CandidateSkill.objects.filter(profile__user=user, skill__name="Django", verified=True).exists()
    resume = Resume.objects.get(pk=resume_id)
    assert resume.is_master and resume.parse_status == "applied"
    master = ResumeVersion.objects.get(user=user, kind="master", status="approved")
    assert len(master.content["employment"]) == 3
    # Stored file is encrypted at rest, yet downloads decrypt for the owner.
    with open(resume.file.path, "rb") as fh:
        assert fh.read(6) == b"JPENC1"
    r = api.get(f"/api/v1/resumes/{resume_id}/download/")
    assert r.status_code == 200 and b"".join(r.streaming_content) == payload
    assert AuditLog.objects.filter(action="resume.applied", user=user).exists()


def test_duplicate_upload_rejected():
    user = make_user()
    api = auth_client(user)
    data = make_docx()
    assert api.post("/api/v1/resumes/", {"file": SimpleUploadedFile("a.docx", data)}, format="multipart").status_code == 201
    r = api.post("/api/v1/resumes/", {"file": SimpleUploadedFile("b.docx", data)}, format="multipart")
    assert r.status_code == 409 and r.data["error"]["code"] == "duplicate_resume"


def test_data_isolation_between_users():
    alice, bob = make_user("alice@example.com"), make_user("bob@example.com")
    a, b = auth_client(alice), auth_client(bob)
    r = a.post("/api/v1/resumes/", {"file": SimpleUploadedFile("a.docx", make_docx())}, format="multipart")
    resume_id = r.data["id"]
    r = a.post("/api/v1/profile/employment/", {"company": "Acme", "title": "Dev"}, format="json")
    emp_id = r.data["id"]
    assert r.data["verified"] is True
    assert b.get(f"/api/v1/resumes/{resume_id}/").status_code == 404
    assert b.get(f"/api/v1/resumes/{resume_id}/download/").status_code == 404
    assert b.delete(f"/api/v1/resumes/{resume_id}/").status_code == 404
    assert b.get(f"/api/v1/profile/employment/{emp_id}/").status_code == 404
    assert b.patch(f"/api/v1/profile/employment/{emp_id}/", {"title": "x"}, format="json").status_code == 404
    assert b.get("/api/v1/resumes/").data["count"] == 0


def test_skill_crud_canonicalizes():
    user = make_user()
    api = auth_client(user)
    r = api.post("/api/v1/profile/skills/", {"skill_name": "drf", "years": "5"}, format="json")
    assert r.status_code == 201 and r.data["name"] == "Django REST Framework"
    r = api.post("/api/v1/profile/skills/", {"skill_name": "Django REST Framework"}, format="json")
    assert r.status_code == 400


def test_employment_date_validation():
    api = auth_client(make_user())
    r = api.post("/api/v1/profile/employment/", {"company": "A", "title": "B", "start_date": "2020-01-01", "end_date": "2019-01-01"}, format="json")
    assert r.status_code == 400


def test_account_deletion_removes_files(settings):
    user = make_user()
    api = auth_client(user)
    api.post("/api/v1/resumes/", {"file": SimpleUploadedFile("a.docx", make_docx())}, format="multipart")
    user_dir = settings.PRIVATE_MEDIA_ROOT / "users" / str(user.pk)
    assert user_dir.exists()
    assert api.delete("/api/v1/auth/me/", {"password": "wrong"}, format="json").status_code == 403
    assert api.delete("/api/v1/auth/me/", {"password": "Sup3r-Secret-pw!"}, format="json").status_code == 204
    assert not user_dir.exists()
    assert not Resume.objects.exists()
