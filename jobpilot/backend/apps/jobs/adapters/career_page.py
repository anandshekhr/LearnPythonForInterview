"""Company-hosted career pages that publish schema.org JobPosting structured data.

Companies that want their jobs in search engines' job results embed a JSON-LD `JobPosting`
on each job page. This adapter reads that data, and only that data:
  1. fetch the careers listing URL (optionally rendered with a headless browser for JS sites);
  2. collect JobPostings embedded on the listing page itself;
  3. follow same-site job links (a configurable regex, or a built-in heuristic), falling back to
     the site's sitemap when the page has no usable links;
  4. read the JSON-LD JobPosting on each job page; pages without one are skipped.
Every URL is checked against robots.txt and the SSRF guard. Removed jobs are expired only after
their page returns 404/410, redirects elsewhere, or its `validThrough` date has passed.
"""
from __future__ import annotations

import re
from datetime import timedelta
from urllib.parse import urlsplit, urlunsplit

from django.conf import settings
from django.utils import timezone as dj_timezone

from ..crawl import (
    JOB_LINK_HINT,
    RobotsPolicy,
    UnsafeURL,
    check_public_url,
    detect_ats,
    is_expired,
    iter_jobpostings,
    jobposting_to_job,
    parse_html,
    sitemap_locations,
    url_id,
)
from ..http import SourceHTTPError
from ..normalization import NormalizedJob
from .base import AdapterConfigError, JobSourceAdapter, SourceCapabilities

MAX_SITEMAPS = 5


class CareerPageAdapter(JobSourceAdapter):
    source_type = "career_page"
    allowed_config = {"job_link_pattern": str, "max_jobs": int, "render_js": bool, "detail_refresh_days": int}
    public_only = True
    min_request_interval = 2.0
    capabilities = SourceCapabilities(
        salary_data=True,
        application_questions=False,
        api_submission=False,
        browser_assist=False,
        documentation_url="https://schema.org/JobPosting",
        limitations=(
            "Only job pages that publish schema.org JobPosting structured data (JSON-LD) can be read.",
            "Follows links on the careers page or entries in its sitemap; jobs behind search forms or logins are not reachable.",
            "Pages built entirely with JavaScript need the 'render JavaScript' option and the browser worker.",
            "robots.txt is always obeyed. If the site disallows crawling, add its jobs manually or ask for an ATS feed.",
        ),
    )

    @classmethod
    def validate_identifier(cls, identifier: str) -> str:
        value = (identifier or "").strip()
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise AdapterConfigError("Enter the full careers page URL, e.g. https://www.example.com/careers")
        if len(value) > 500:
            raise AdapterConfigError("URL is too long.")
        try:
            check_public_url(value)
        except UnsafeURL as exc:
            raise AdapterConfigError(str(exc))
        ats = detect_ats(value)
        if ats:
            raise AdapterConfigError(f"This is a {ats} job board – add it with the '{ats}' source type instead.")
        return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))

    @classmethod
    def validate_config(cls, config: dict) -> dict:
        config = super().validate_config(config)
        pattern = config.get("job_link_pattern")
        if pattern:
            if len(pattern) > 200:
                raise AdapterConfigError("job_link_pattern is limited to 200 characters.")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise AdapterConfigError(f"job_link_pattern is not a valid regular expression: {exc}")
        if "max_jobs" in config and not 1 <= config["max_jobs"] <= 500:
            raise AdapterConfigError("max_jobs must be between 1 and 500.")
        return config

    # ------------------------------------------------------------------ fetching

    def _robots(self) -> RobotsPolicy:
        if not getattr(self, "_robots_policy", None):
            self._robots_policy = RobotsPolicy(self.http)
        return self._robots_policy

    def _get(self, url: str) -> str:
        if not self._robots().allowed(url):
            raise SourceHTTPError(f"robots.txt disallows {urlsplit(url).path}", status_code=None)
        return self.http.get_text(url)

    def _listing_html(self) -> str:
        if not self.config.get("render_js"):
            return self._get(self.identifier)
        if not self._robots().allowed(self.identifier):
            raise SourceHTTPError("robots.txt disallows the careers page")
        return render_page(self.identifier)

    def _is_job_link(self, url: str) -> bool:
        listing = urlsplit(self.identifier)
        target = urlsplit(url)
        if target.hostname != listing.hostname or url.rstrip("/") == self.identifier.rstrip("/"):
            return False
        pattern = self.config.get("job_link_pattern")
        if pattern:
            return bool(re.search(pattern, url))
        return bool(JOB_LINK_HINT.search(target.path + ("?" + target.query if target.query else "")))

    def _sitemap_links(self) -> list[str]:
        queue = self._robots().sitemaps(self.identifier) or [
            f"{urlsplit(self.identifier).scheme}://{urlsplit(self.identifier).netloc}/sitemap.xml"]
        links, seen = [], 0
        while queue and seen < MAX_SITEMAPS:
            sitemap = queue.pop(0)
            seen += 1
            try:
                pages, nested = sitemap_locations(self._get(sitemap))
            except SourceHTTPError:
                continue
            links.extend(u for u in pages if self._is_job_link(u))
            queue.extend(nested)
        return list(dict.fromkeys(links))

    def fetch_listings(self):
        max_jobs = int(self.config.get("max_jobs") or 100)
        refresh = timedelta(days=int(self.config.get("detail_refresh_days") or 7))
        now = dj_timezone.now()
        html = self._listing_html()
        blobs, links = parse_html(html, self.identifier)

        emitted = 0
        for posting in iter_jobpostings(blobs):  # postings embedded on the listing page itself
            if posting.get("url") and not is_expired(posting):
                job = jobposting_to_job(posting, str(posting["url"]), external_id=url_id(str(posting["url"])))
                if job.title:
                    emitted += 1
                    yield job

        ats_links = {detect_ats(u) for u in links} - {None}
        if ats_links:
            self.warnings.append(f"This careers page links to {', '.join(sorted(ats_links))}: add that job board as a "
                                 "source for complete, reliable results.")
        candidates = [u for u in links if self._is_job_link(u)]
        if not candidates and not emitted:
            candidates = self._sitemap_links()
        if len(candidates) > max_jobs:
            self.listing_complete = False
            self.warnings.append(f"Found {len(candidates)} job links; read the first {max_jobs} (raise max_jobs to read more).")
            candidates = candidates[:max_jobs]

        without_data = 0
        for url in candidates:
            ext_id = url_id(url)
            stored = self.known_jobs.get(ext_id)
            if stored and now - stored < refresh:
                yield NormalizedJob(external_id=ext_id, title="(unchanged)", canonical_url=url, partial=True)
                continue
            try:
                page = self._get(url)
            except SourceHTTPError as exc:
                if exc.status_code in (404, 410):
                    continue
                self.warnings.append(f"Skipped {url}: {exc}")
                continue
            postings = [p for p in iter_jobpostings(parse_html(page, url)[0])]
            if not postings:
                without_data += 1
                continue
            if is_expired(postings[0]):
                continue
            job = jobposting_to_job(postings[0], url, external_id=ext_id)
            if job.title:
                yield job
        if without_data:
            self.warnings.append(f"{without_data} linked pages had no schema.org JobPosting data and were skipped.")
        if not candidates and not emitted:
            self.warnings.append("No job links or JobPosting data found. Set a job link pattern, enable JavaScript "
                                 "rendering, or check whether the site uses an ATS such as Workday or Greenhouse.")

    def fetch_job(self, external_id: str):
        return None

    def confirm_removed(self, job) -> bool:
        page_url = (job.raw_metadata or {}).get("page_url") or job.canonical_url
        if not self._robots().allowed(page_url):
            return False
        try:
            response = self.http.request("GET", page_url, headers={"Accept": "text/html"})
        except SourceHTTPError as exc:
            return exc.status_code in (404, 410)
        if str(response.url).rstrip("/") != page_url.rstrip("/"):
            return True  # closed jobs typically redirect to the careers home page
        postings = list(iter_jobpostings(parse_html(response.text, page_url)[0]))
        return bool(postings) and is_expired(postings[0])


def render_page(url: str) -> str:
    """Render a JavaScript careers page with headless Chromium (browser worker only).

    Every request the page makes is checked by the SSRF guard; disallowed requests are aborted.
    """
    if not settings.BROWSER_AUTOMATION_ENABLED:
        raise SourceHTTPError("JavaScript rendering needs BROWSER_AUTOMATION_ENABLED=1 and the browser worker.")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SourceHTTPError("JavaScript rendering needs Playwright (run the browser worker).") from exc
    from apps.applications.browser.filler import launch_browser

    def guard(route):
        try:
            check_public_url(route.request.url)
        except UnsafeURL:
            return route.abort()
        return route.continue_()

    with sync_playwright() as p:
        browser = launch_browser(p, headless=True)
        try:
            page = browser.new_page(user_agent=settings.SOURCE_USER_AGENT)
            page.route("**/*", guard)
            page.goto(url, wait_until="networkidle", timeout=45000)
            return page.content()
        finally:
            browser.close()
