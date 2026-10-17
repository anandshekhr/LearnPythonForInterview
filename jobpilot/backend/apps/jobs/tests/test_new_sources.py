"""Workday career sites and company career pages (schema.org JobPosting), against fixtures."""
import json
from datetime import timedelta

import httpx
import pytest
from django.utils import timezone

from apps.jobs import crawl
from apps.jobs.adapters.base import AdapterConfigError
from apps.jobs.adapters.career_page import CareerPageAdapter
from apps.jobs.adapters.workday import WorkdayAdapter
from apps.jobs.http import SourceHttpClient, SourceHTTPError
from apps.jobs.models import Company, JobPosting, JobSource
from apps.jobs.services.sync import sync_source

pytestmark = pytest.mark.django_db

# ----------------------------------------------------------------- Workday

WD_HOST = "https://acme.wd5.myworkdayjobs.com"
WD_API = f"{WD_HOST}/wday/cxs/acme/AcmeCareers"


def wd_posting(n, title):
    return {"title": title, "externalPath": f"/job/Noida/{title.replace(' ', '-')}_JR{n}",
            "locationsText": "Noida, India", "postedOn": "Posted 3 Days Ago", "bulletFields": [f"JR{n}"]}


def wd_detail(n, title, remote="Hybrid"):
    return {"jobPostingInfo": {
        "id": f"id{n}", "title": title, "jobReqId": f"JR{n}", "location": "Noida, India", "additionalLocations": ["Bengaluru, India"],
        "timeType": "Full time", "remoteType": remote, "startDate": "2026-10-10",
        "externalUrl": f"{WD_HOST}/AcmeCareers/job/Noida/{title.replace(' ', '-')}_JR{n}",
        "jobDescription": "<p>Build APIs.</p><h3>Requirements</h3><ul><li>5+ years of experience with Python and Django</li>"
                          "<li>PostgreSQL</li></ul>"},
        "hiringOrganization": {"name": "Acme"}}


@pytest.fixture
def workday(respx_mock):
    postings = [wd_posting(1, "Senior Python Developer"), wd_posting(2, "Data Engineer"), wd_posting(3, "Sales Manager")]
    respx_mock.get(f"{WD_HOST}/robots.txt").respond(200, text="User-agent: *\nDisallow: /wday/cxs/*/internal\n")
    listing = respx_mock.post(f"{WD_API}/jobs").mock(side_effect=lambda req: httpx.Response(200, json={
        "total": 3, "jobPostings": postings[json.loads(req.content)["offset"]:][:20]}))
    details = {}
    for n, p in enumerate(postings, 1):
        details[n] = respx_mock.get(f"{WD_API}{p['externalPath']}").respond(200, json=wd_detail(n, p["title"]))
    company = Company.objects.create(name="Acme")
    source = JobSource.objects.create(company=company, source_type="workday", identifier="acme.wd5.myworkdayjobs.com/AcmeCareers",
                                      min_request_interval_seconds=0)
    return source, listing, details, postings


def test_workday_identifier_normalisation():
    assert WorkdayAdapter.validate_identifier("https://acme.wd5.myworkdayjobs.com/en-US/AcmeCareers/job/x") == \
        "acme.wd5.myworkdayjobs.com/AcmeCareers"
    assert WorkdayAdapter.validate_identifier("acme.wd5.myworkdayjobs.com/AcmeCareers") == "acme.wd5.myworkdayjobs.com/AcmeCareers"
    for bad in ["https://evil.example.com/Acme", "https://acme.wd5.myworkdayjobs.com/", "https://acme.wd5.myworkdayjobs.com.evil.io/x"]:
        with pytest.raises(AdapterConfigError):
            WorkdayAdapter.validate_identifier(bad)
    with pytest.raises(AdapterConfigError):
        WorkdayAdapter.validate_config({"max_jobs": 0})
    with pytest.raises(AdapterConfigError):
        WorkdayAdapter.validate_config({"unknown": 1})


def test_workday_sync_normalises_and_skips_recent_details(workday):
    source, listing, details, _ = workday
    result = sync_source(source)
    assert len(result.created) == 3 and result.warnings == []
    job = JobPosting.objects.get(external_id="JR1")
    assert job.title == "Senior Python Developer" and job.work_model == "hybrid" and job.employment_type == "full_time"
    assert job.locations == ["Noida", "Bengaluru"] and job.published_at.date().isoformat() == "2026-10-10"
    assert job.canonical_url.startswith(f"{WD_HOST}/AcmeCareers/job/") and "Django" in job.description_text
    assert json.loads(listing.calls[0].request.content) == {"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""}
    # Second sync: listing only, no detail downloads for recently fetched jobs.
    before = sum(d.call_count for d in details.values())
    result = sync_source(source)
    assert result.unchanged == 3 and sum(d.call_count for d in details.values()) == before
    # Details older than the refresh window are downloaded again.
    JobPosting.objects.filter(external_id="JR2").update(details_fetched_at=timezone.now() - timedelta(days=8))
    sync_source(source)
    assert details[2].call_count == 2


def test_workday_capped_listing_never_expires_unseen(workday, respx_mock):
    source, *_ = workday
    sync_source(source)
    source.config = {"max_jobs": 1, "search_text": "python"}
    source.save()
    result = sync_source(source)
    assert result.expired == [] and any("fetched the first 1" in w for w in result.warnings)
    assert JobPosting.objects.filter(source=source, status="active").count() == 3


def test_workday_removed_job_expires_only_when_confirmed(workday, respx_mock):
    source, listing, details, postings = workday
    sync_source(source)
    remaining = postings[:2]
    listing.mock(side_effect=lambda req: httpx.Response(200, json={"total": 2, "jobPostings": remaining}))
    details[3].respond(404)
    result = sync_source(source)
    assert JobPosting.objects.get(external_id="JR3").status == "expired" and len(result.expired) == 1
    # Missing from listing but its page still exists (e.g. hidden by search filter): stays active.
    listing.mock(side_effect=lambda req: httpx.Response(200, json={"total": 1, "jobPostings": postings[:1]}))
    sync_source(source)
    assert JobPosting.objects.get(external_id="JR2").status == "active"


def test_workday_403_stops_without_workaround(respx_mock):
    respx_mock.get(f"{WD_HOST}/robots.txt").respond(404)
    route = respx_mock.post(f"{WD_API}/jobs").respond(403)
    source = JobSource.objects.create(company=Company.objects.create(name="Acme"), source_type="workday",
                                      identifier="acme.wd5.myworkdayjobs.com/AcmeCareers")
    with pytest.raises(SourceHTTPError) as exc:
        sync_source(source)
    assert "refused automated access" in str(exc.value) and route.call_count == 1
    source.refresh_from_db()
    assert source.status == "degraded" and "403" in source.last_error


def test_workday_respects_robots(respx_mock):
    respx_mock.get(f"{WD_HOST}/robots.txt").respond(200, text="User-agent: *\nDisallow: /wday/\n")
    route = respx_mock.post(f"{WD_API}/jobs").respond(200, json={"total": 0, "jobPostings": []})
    source = JobSource.objects.create(company=Company.objects.create(name="Acme"), source_type="workday",
                                      identifier="acme.wd5.myworkdayjobs.com/AcmeCareers")
    with pytest.raises(SourceHTTPError, match="robots.txt"):
        sync_source(source)
    assert route.call_count == 0


# ------------------------------------------------------------ career pages

SITE = "https://careers.acme.example"
LISTING = """<html><body><h1>Open roles</h1>
<a href="/jobs/101-senior-python-developer">Senior Python Developer</a>
<a href="/jobs/102-data-engineer?ref=list#top">Data Engineer</a>
<a href="/jobs/103-office-assistant">Office Assistant (no structured data)</a>
<a href="/jobs/secret-internal">Internal</a>
<a href="/about">About us</a>
<a href="https://boards.greenhouse.io/acme/jobs/1">Some roles are on Greenhouse</a>
<a href="mailto:jobs@acme.example">Email us</a>
</body></html>"""


def job_page(title, extra=None, graph=False):
    posting = {
        "@context": "https://schema.org", "@type": "JobPosting", "title": title,
        "description": "<p>Requirements</p><ul><li>6+ years of experience with Python, Django and AWS</li></ul>",
        "datePosted": "2026-10-05", "employmentType": "FULL_TIME", "identifier": {"@type": "PropertyValue", "value": "R-1"},
        "hiringOrganization": {"@type": "Organization", "name": "Acme"},
        "jobLocation": {"@type": "Place", "address": {"@type": "PostalAddress", "addressLocality": "Gurugram",
                                                      "addressRegion": "Haryana", "addressCountry": "IN"}},
        "baseSalary": {"@type": "MonetaryAmount", "currency": "INR",
                       "value": {"@type": "QuantitativeValue", "minValue": 3000000, "maxValue": 4200000, "unitText": "YEAR"}},
        **(extra or {}),
    }
    data = {"@context": "https://schema.org", "@graph": [{"@type": "WebPage"}, posting]} if graph else posting
    return f'<html><head><script type="application/ld+json">{json.dumps(data)}</script></head><body>{title}</body></html>'


@pytest.fixture
def career(respx_mock):
    respx_mock.get(f"{SITE}/robots.txt").respond(200, text="User-agent: *\nDisallow: /jobs/secret\n")
    listing = respx_mock.get(f"{SITE}/careers").respond(200, html=LISTING)
    pages = {
        "101": respx_mock.get(f"{SITE}/jobs/101-senior-python-developer").respond(200, html=job_page("Senior Python Developer")),
        "102": respx_mock.get(f"{SITE}/jobs/102-data-engineer", params={"ref": "list"}).respond(
            200, html=job_page("Data Engineer", {"jobLocationType": "TELECOMMUTE", "applicantLocationRequirements":
                                                  {"@type": "Country", "name": "India"}}, graph=True)),
        "103": respx_mock.get(f"{SITE}/jobs/103-office-assistant").respond(200, html="<html><body>Apply by email</body></html>"),
    }
    secret = respx_mock.get(f"{SITE}/jobs/secret-internal").respond(200, html=job_page("Secret"))
    source = JobSource.objects.create(company=Company.objects.create(name="Acme"), source_type="career_page",
                                      identifier=f"{SITE}/careers", min_request_interval_seconds=0)
    return source, listing, pages, secret


def test_career_page_sync(career):
    source, listing, pages, secret = career
    result = sync_source(source)
    jobs = {j.title: j for j in JobPosting.objects.filter(source=source)}
    assert set(jobs) == {"Senior Python Developer", "Data Engineer"}
    assert secret.call_count == 0  # robots.txt disallowed
    py = jobs["Senior Python Developer"]
    assert (py.salary_min, py.salary_max, py.salary_currency) == (3_000_000, 4_200_000, "INR")
    assert py.locations == ["Gurugram"] and py.employment_type == "full_time" and py.published_at.day == 5
    de = jobs["Data Engineer"]
    assert de.work_model == "remote" and de.location_raw == "Gurugram, Haryana, IN"  # remote role with a home office
    assert any("greenhouse" in w for w in result.warnings)
    assert any("1 linked pages had no schema.org JobPosting" in w for w in result.warnings)
    # Requirements are extracted like any other job.
    from apps.jobs.services.requirements import analyze_job

    analyze_job(py)
    assert {"Python", "Django", "AWS"} <= set(py.requirements["required_skills"]) and py.requirements["min_years"] == 6


def test_career_page_second_sync_skips_recent_pages_and_confirms_removal(career, respx_mock):
    source, listing, pages, _ = career
    sync_source(source)
    calls = pages["101"].call_count
    sync_source(source)
    assert pages["101"].call_count == calls  # fresh: not downloaded again
    listing.respond(200, html=LISTING.replace('<a href="/jobs/101-senior-python-developer">Senior Python Developer</a>', ""))
    pages["101"].respond(404)
    result = sync_source(source)
    assert JobPosting.objects.get(title="Senior Python Developer").status == "expired" and len(result.expired) == 1


def test_career_page_unconfirmed_removal_keeps_job(career, respx_mock):
    source, listing, pages, _ = career
    sync_source(source)
    listing.respond(200, html="<html><body>Temporarily no links</body></html>")
    respx_mock.get(f"{SITE}/sitemap.xml").respond(404)
    result = sync_source(source)  # page still serves an active JobPosting -> not removed
    assert result.expired == [] and JobPosting.objects.filter(source=source, status="active").count() == 2


def test_career_page_expired_validthrough(respx_mock):
    respx_mock.get(f"{SITE}/robots.txt").respond(404)
    respx_mock.get(f"{SITE}/careers").respond(200, html='<a href="/jobs/1-old">Old</a>')
    respx_mock.get(f"{SITE}/jobs/1-old").respond(200, html=job_page("Old role", {"validThrough": "2020-01-01T00:00:00Z"}))
    source = JobSource.objects.create(company=Company.objects.create(name="Acme"), source_type="career_page",
                                      identifier=f"{SITE}/careers")
    assert sync_source(source).created == []


def test_career_page_sitemap_fallback_and_custom_pattern(respx_mock):
    respx_mock.get(f"{SITE}/robots.txt").respond(200, text=f"Sitemap: {SITE}/sitemap_index.xml\n")
    respx_mock.get(f"{SITE}/careers").respond(200, html="<div id='app'></div>")
    respx_mock.get(f"{SITE}/sitemap_index.xml").respond(
        200, text=f"<sitemapindex><sitemap><loc>{SITE}/sitemap-jobs.xml</loc></sitemap></sitemapindex>")
    respx_mock.get(f"{SITE}/sitemap-jobs.xml").respond(
        200, text=f"<urlset><url><loc>{SITE}/openings/backend-1</loc></url><url><loc>{SITE}/blog/post</loc></url></urlset>")
    respx_mock.get(f"{SITE}/openings/backend-1").respond(200, html=job_page("Backend Engineer"))
    source = JobSource.objects.create(company=Company.objects.create(name="Acme"), source_type="career_page",
                                      identifier=f"{SITE}/careers", config={"job_link_pattern": r"/openings/"})
    assert [j.title for j in JobPosting.objects.filter(pk__in=sync_source(source).created)] == ["Backend Engineer"]


def test_career_page_postings_embedded_on_listing(respx_mock):
    blob = json.dumps({"@type": "ItemList", "itemListElement": [
        {"@type": "ListItem", "item": json.loads(job_page("A")[job_page("A").index("{"):job_page("A").index("</script>")]) |
         {"url": f"{SITE}/jobs/a"}}]})
    respx_mock.get(f"{SITE}/robots.txt").respond(404)
    respx_mock.get(f"{SITE}/careers").respond(200, html=f'<script type="application/ld+json">{blob}</script>')
    source = JobSource.objects.create(company=Company.objects.create(name="Acme"), source_type="career_page",
                                      identifier=f"{SITE}/careers")
    assert JobPosting.objects.get(pk=sync_source(source).created[0]).canonical_url == f"{SITE}/jobs/a"


def test_career_page_identifier_rules():
    assert CareerPageAdapter.validate_identifier("https://Careers.Acme.example/careers#top") == "https://careers.acme.example/careers"
    for bad, message in [("ftp://acme.example/jobs", "full careers page URL"), ("https://boards.greenhouse.io/acme", "greenhouse"),
                         ("https://acme.wd5.myworkdayjobs.com/Careers", "workday")]:
        with pytest.raises(AdapterConfigError, match=message):
            CareerPageAdapter.validate_identifier(bad)
    with pytest.raises(AdapterConfigError):
        CareerPageAdapter.validate_config({"job_link_pattern": "(unclosed"})


def test_server_error_on_robots_means_disallow(respx_mock):
    respx_mock.get(f"{SITE}/robots.txt").respond(500)
    page = respx_mock.get(f"{SITE}/careers").respond(200, html=LISTING)
    source = JobSource.objects.create(company=Company.objects.create(name="Acme"), source_type="career_page",
                                      identifier=f"{SITE}/careers")
    with pytest.raises(SourceHTTPError, match="robots.txt"):
        sync_source(source)
    assert page.call_count == 0


# -------------------------------------------------------------------- SSRF


@pytest.fixture
def real_ssrf_guard(settings, monkeypatch):
    settings.SOURCE_ALLOW_PRIVATE_HOSTS = False
    table = {"careers.acme.example": ["93.184.216.34"], "internal.acme.example": ["10.0.0.7"],
             "rebind.acme.example": ["93.184.216.34", "127.0.0.1"], "169.254.169.254": ["169.254.169.254"]}
    monkeypatch.setattr(crawl, "_resolve", lambda host: table[host])


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/", "https://internal.acme.example/careers", "https://rebind.acme.example/",
    "https://careers.acme.example:8443/careers", "https://user:pw@careers.acme.example/", "file:///etc/passwd",
])
def test_ssrf_guard_blocks(real_ssrf_guard, url):
    with pytest.raises(crawl.UnsafeURL):
        crawl.check_public_url(url)


def test_ssrf_guard_allows_public_and_rejects_private_identifier(real_ssrf_guard):
    assert crawl.check_public_url("https://careers.acme.example/careers")
    with pytest.raises(AdapterConfigError, match="non-public"):
        CareerPageAdapter.validate_identifier("https://internal.acme.example/careers")


def test_ssrf_guard_applies_to_redirects(real_ssrf_guard, respx_mock):
    respx_mock.get(f"{SITE}/careers").respond(302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})
    metadata = respx_mock.get("http://169.254.169.254/latest/meta-data/").respond(200, text="secret")
    http = SourceHttpClient("t", min_interval=0, public_only=True)
    with pytest.raises(SourceHTTPError, match="non-public"):
        http.get_text(f"{SITE}/careers")
    assert metadata.call_count == 0


def test_api_accepts_new_source_types(respx_mock):
    from apps.core.testing import auth_client, make_user

    api = auth_client(make_user())
    company = api.post("/api/v1/companies/", {"name": "Acme"}, format="json").data
    respx_mock.get(f"{WD_HOST}/robots.txt").respond(404)
    respx_mock.post(f"{WD_API}/jobs").respond(200, json={"total": 0, "jobPostings": []})
    r = api.post("/api/v1/sources/", {"company": company["id"], "source_type": "workday",
                                      "identifier": "https://acme.wd5.myworkdayjobs.com/en-US/AcmeCareers",
                                      "config": {"search_text": "python", "max_jobs": 200}}, format="json")
    assert r.status_code == 201, r.data
    assert r.data["identifier"] == "acme.wd5.myworkdayjobs.com/AcmeCareers"
    assert r.data["capabilities"]["api_submission"] is False and r.data["capabilities"]["limitations"]
    r = api.post("/api/v1/sources/", {"company": company["id"], "source_type": "career_page",
                                      "identifier": "https://careers.acme.example/careers", "config": {"render": True}}, format="json")
    assert r.status_code == 400 and "config" in r.data
    types = {t["source_type"] for t in api.get("/api/v1/sources/types/").data}
    assert types == {"greenhouse", "lever", "workday", "career_page"}
