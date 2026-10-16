"""Assisted form filling with Playwright for permitted, hosted ATS application forms.

Guarantees:
  * only hosts in BROWSER_ASSIST_ALLOWED_HOSTS are opened (official Greenhouse/Lever forms);
  * the helper NEVER clicks a submit button – the user reviews and submits;
  * CAPTCHA / human-verification widgets and login walls stop automation and are reported;
  * sensitive questions are filled only with answers the user explicitly reviewed;
  * nothing is typed into fields it cannot identify with confidence.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from django.conf import settings

logger = logging.getLogger(__name__)

CAPTCHA_SELECTORS = [
    "iframe[src*='recaptcha']", "iframe[src*='hcaptcha']", "iframe[src*='challenges.cloudflare.com']", ".g-recaptcha",
    ".h-captcha", ".cf-turnstile", "[data-sitekey]",
]
LOGIN_SELECTORS = ["input[type='password']"]


@dataclass
class FieldValue:
    key: str
    value: str
    selectors: list[str]
    kind: str = "text"  # text | file | select | textarea
    sensitive: bool = False


@dataclass
class AssistPlan:
    url: str
    site: str
    fields: list[FieldValue] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass
class AssistReport:
    url: str
    site: str
    filled: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    human_verification_required: bool = False
    login_required: bool = False
    blocked_reason: str = ""
    screenshot: str = ""

    def as_dict(self):
        return self.__dict__.copy()


class AssistNotPermitted(Exception):
    pass


def check_host_allowed(url: str, allowed_hosts=None):
    host = (urlsplit(url).hostname or "").lower()
    allowed = [h.lower() for h in (allowed_hosts if allowed_hosts is not None else settings.BROWSER_ASSIST_ALLOWED_HOSTS)]
    if not host or host not in allowed:
        raise AssistNotPermitted(f"Browser assistance is not permitted for '{host}'. Open the official page yourself.")
    if urlsplit(url).scheme not in {"https"} and host not in {"127.0.0.1", "localhost"}:
        raise AssistNotPermitted("Only HTTPS application pages are supported.")


# Selector fallbacks per site. Greenhouse hosts two form generations; Lever uses one form.
SITE_SELECTORS = {
    "greenhouse": {
        "first_name": ["#first_name", "input[name='job_application[first_name]']", "input[name='first_name']"],
        "last_name": ["#last_name", "input[name='job_application[last_name]']", "input[name='last_name']"],
        "email": ["#email", "input[name='job_application[email]']", "input[type='email']"],
        "phone": ["#phone", "input[name='job_application[phone]']", "input[type='tel']"],
        "resume": ["input#resume[type='file']", "input[type='file'][name*='resume']", "#resume_fieldset input[type='file']"],
        "cover_letter_text": ["textarea#cover_letter_text", "textarea[name*='cover_letter']"],
        "question": ["#{key}", "[name='{key}']", "[name='job_application[answers_attributes][{key}][text_value]']"],
    },
    "lever": {
        "full_name": ["input[name='name']"],
        "email": ["input[name='email']"],
        "phone": ["input[name='phone']"],
        "current_company": ["input[name='org']"],
        "linkedin": ["input[name='urls[LinkedIn]']"],
        "github": ["input[name='urls[GitHub]']"],
        "website": ["input[name='urls[Portfolio]']", "input[name='urls[Other]']"],
        "resume": ["input[type='file'][name='resume']", "input#resume-upload-input"],
        "cover_letter_text": ["textarea[name='comments']"],
        "question": ["[name='{key}']"],
    },
}


def build_plan(app, resume_path: str, url: str | None = None) -> AssistPlan:
    from ..services.answers import profile_values

    site = app.source_type if app.source_type in SITE_SELECTORS else ""
    if not site:
        raise AssistNotPermitted("No browser-assist adapter exists for this source.")
    target = url or (app.job.apply_url if app.job else app.job_url)
    sel = SITE_SELECTORS[site]
    values = profile_values(app.user.profile)
    plan = AssistPlan(url=target, site=site)

    def add(key, value, kind="text", sensitive=False, selectors=None):
        if value:
            plan.fields.append(FieldValue(key, str(value), selectors or sel.get(key, []), kind, sensitive))

    if site == "lever":
        add("full_name", f"{values['first_name']} {values['last_name']}".strip())
        add("current_company", values["current_company"])
        for k in ("linkedin", "github", "website"):
            add(k, values[k])
    else:
        add("first_name", values["first_name"])
        add("last_name", values["last_name"])
    add("email", values["email"])
    add("phone", values["phone"])
    add("resume", resume_path, kind="file")
    if app.cover_letter and app.cover_letter.text_content:
        add("cover_letter_text", app.cover_letter.text_content, kind="textarea")
    handled = {"first_name", "last_name", "email", "phone", "resume", "cover_letter", "cover_letter_text"}
    for answer in app.answers.all():
        if answer.question_key in handled or answer.field_type == "file":
            continue
        if answer.sensitive and not answer.reviewed_at:
            plan.skipped.append(f"{answer.question_label[:80]} (sensitive – answer it yourself)")
            continue
        if not answer.answer:
            plan.skipped.append(f"{answer.question_label[:80]} (no answer prepared)")
            continue
        selectors = [s.format(key=answer.question_key) for s in sel["question"]]
        kind = "select" if answer.field_type in ("select", "multi_select") else ("textarea" if answer.field_type == "textarea" else "text")
        add(answer.question_key, answer.answer, kind, answer.sensitive, selectors)
    return plan


def _detect(page, selectors) -> bool:
    return any(page.locator(s).count() > 0 for s in selectors)


def fill_page(page, plan: AssistPlan, report: AssistReport):
    for f in plan.fields:
        locator = None
        for selector in f.selectors:
            candidate = page.locator(selector)
            if candidate.count() == 1:
                locator = candidate
                break
        if locator is None:
            report.not_found.append(f.key)
            continue
        try:
            if f.kind == "file":
                locator.set_input_files(f.value)
            elif f.kind == "select":
                locator.select_option(value=f.value) if locator.evaluate("e => e.tagName") == "SELECT" else locator.fill(f.value)
            else:
                locator.fill(f.value)
            report.filled.append(f.key)
        except Exception as exc:  # field disabled / hidden etc.
            logger.info("assist_fill_failed", extra={"field": f.key, "error": type(exc).__name__})
            report.not_found.append(f.key)


def launch_browser(playwright, headless: bool):
    kwargs = {"headless": headless}
    if settings.BROWSER_EXECUTABLE_PATH:
        kwargs["executable_path"] = settings.BROWSER_EXECUTABLE_PATH
    return playwright.chromium.launch(**kwargs)


def run_assist(plan: AssistPlan, headless=True, keep_open=False, allowed_hosts=None, screenshot_dir=None,
               wait_for_user=None) -> AssistReport:
    from playwright.sync_api import sync_playwright

    check_host_allowed(plan.url, allowed_hosts)
    report = AssistReport(url=plan.url, site=plan.site, skipped=list(plan.skipped))
    with sync_playwright() as p:
        browser = launch_browser(p, headless)
        try:
            page = browser.new_page()
            page.goto(plan.url, wait_until="domcontentloaded", timeout=45000)
            final_host = urlsplit(page.url).hostname
            if final_host != urlsplit(plan.url).hostname:
                check_host_allowed(page.url, allowed_hosts)  # redirected somewhere else: re-check
            if _detect(page, LOGIN_SELECTORS):
                report.login_required = True
                report.blocked_reason = "The page asks you to sign in. Sign in yourself; JobPilot does not handle credentials."
            else:
                fill_page(page, plan, report)
                if _detect(page, CAPTCHA_SELECTORS):
                    report.human_verification_required = True
                    report.blocked_reason = "Human verification (CAPTCHA) is present: complete it yourself before submitting."
            if screenshot_dir:
                path = os.path.join(screenshot_dir, f"assist-{plan.site}.png")
                page.screenshot(path=path, full_page=True)
                report.screenshot = path
            if keep_open and wait_for_user:
                wait_for_user(page, report)
        finally:
            browser.close()
    return report


def materialize_resume(app) -> str:
    """Decrypt the approved resume to a private temp file for upload. Caller deletes it."""
    doc = app.resume_document
    if doc is None:
        raise AssistNotPermitted("Attach an approved resume before using browser assistance.")
    fd, path = tempfile.mkstemp(suffix=f".{doc.format}", prefix="jobpilot-resume-")
    with os.fdopen(fd, "wb") as out, doc.file.open("rb") as src:
        out.write(src.read())
    os.chmod(path, 0o600)
    return path
