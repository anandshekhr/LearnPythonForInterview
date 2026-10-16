"""Shared test helpers: synthetic (fictional) resume fixtures and API client helpers."""
import io

from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

SAMPLE_RESUME_LINES = [
    "Asha Verma",
    "Senior Python Developer",
    "asha.verma@example.com | +91 98765 43210 | Noida, India",
    "linkedin.com/in/asha-verma-example | github.com/ashaverma-example",
    "",
    "Professional Summary",
    "Backend engineer with 7+ years of experience building Django and Angular products, REST APIs and data pipelines on AWS.",
    "",
    "Technical Skills",
    "Languages: Python, TypeScript, SQL",
    "Frameworks: Django, DRF, Angular, Celery, FastAPI",
    "Data & Infra: PostgreSQL, Redis, Docker, AWS, Kubernetes",
    "",
    "Work Experience",
    "Senior Software Engineer | Nimbus Analytics Pvt Ltd | Jan 2021 - Present",
    "• Led a team of 5 engineers building a multi-tenant SaaS analytics platform on Django REST Framework and Angular.",
    "• Reduced report generation time by 60% by moving heavy jobs to Celery workers backed by Redis.",
    "• Designed PostgreSQL schemas and introduced pgvector-based semantic search.",
    "Software Engineer | Orbit Software Solutions | Jul 2018 - Dec 2020",
    "• Built REST APIs in Django for a logistics product used by 200+ enterprise customers.",
    "• Containerised services with Docker and deployed them on AWS ECS.",
    "Associate Developer",
    "Pixel Labs, Bengaluru    Jun 2017 – Jun 2018",
    "• Developed internal tools in Python and Flask.",
    "",
    "Projects",
    "ResumeRanker - semantic resume matching prototype",
    "• Built a RAG pipeline with LangChain and PostgreSQL to rank resumes against job descriptions.",
    "",
    "Education",
    "B.Tech in Computer Science, Delhi Technological University, 2013 - 2017, CGPA 8.4/10",
    "",
    "Certifications",
    "AWS Certified Developer – Associate - Amazon Web Services",
]


def sample_resume_text():
    return "\n".join(SAMPLE_RESUME_LINES)


def make_docx(lines=None) -> bytes:
    import docx

    document = docx.Document()
    for line in lines or SAMPLE_RESUME_LINES:
        document.add_paragraph(line)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def make_pdf(lines=None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 50
    for line in lines or SAMPLE_RESUME_LINES:
        text = line.replace("•", "-").replace("–", "-")
        c.setFont("Helvetica", 9)
        while pdfmetrics.stringWidth(text, "Helvetica", 9) > width - 80 and " " in text:
            # crude wrap
            cut = text.rfind(" ", 0, int(len(text) * (width - 80) / pdfmetrics.stringWidth(text, "Helvetica", 9)))
            c.drawString(40, y, text[:cut])
            y -= 12
            text = text[cut + 1:]
        c.drawString(40, y, text)
        y -= 12
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()
    return buf.getvalue()


def make_user(email="user@example.com", password="Sup3r-Secret-pw!", **kwargs):
    User = get_user_model()
    return User.objects.create_user(username=email, email=email, password=password, **kwargs)


def auth_client(user) -> APIClient:
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {RefreshToken.for_user(user).access_token}")
    return client
