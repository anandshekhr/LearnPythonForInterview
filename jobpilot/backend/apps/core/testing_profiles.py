"""Build a verified candidate profile from the synthetic resume (for integration tests)."""
from apps.core.testing import make_user, sample_resume_text
from apps.profiles.models import Resume
from apps.profiles.services import parser
from apps.profiles.services.profile import apply_parsed_resume, get_or_create_profile


def verified_profile(email="asha@example.com", years=7):
    user = make_user(email)
    get_or_create_profile(user)
    resume = Resume.objects.create(user=user, original_filename="cv.docx", content_type="docx", size_bytes=1,
                                   sha256=email.ljust(64, "0")[:64], parse_status=Resume.ParseStatus.PARSED)
    data = parser.parse_resume_text(sample_resume_text(), use_llm=False)
    data["total_experience_years"] = years
    apply_parsed_resume(resume, data)
    return user
