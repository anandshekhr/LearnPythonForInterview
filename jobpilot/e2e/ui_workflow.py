"""End-to-end UI workflow against a running stack (see docs/TESTING.md).

Registers a fresh user, uploads a synthetic resume, verifies the parsed profile, adds a
Greenhouse source (served by scripts/mock_job_boards.py), waits for discovery and
matching, tailors and approves a resume, records an assisted submission, and checks
the dashboard. Screenshots are written to --shots.
"""
import argparse
import glob
import os
import re
import sys
import time
import uuid

from playwright.sync_api import expect, sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://127.0.0.1:4200")
parser.add_argument("--resume", required=True)
parser.add_argument("--shots", default="e2e-shots")
parser.add_argument("--mobile", action="store_true")
args = parser.parse_args()
os.makedirs(args.shots, exist_ok=True)
CHROME = os.environ.get("BROWSER_EXECUTABLE_PATH") or (glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome") or [None])[0]


def shot(page, name):
    page.wait_for_timeout(400)
    page.screenshot(path=f"{args.shots}/{name}.png", full_page=True)


def step(msg):
    print(f"✔ {msg}", flush=True)


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME) if CHROME else p.chromium.launch()
    viewport = {"width": 390, "height": 844} if args.mobile else {"width": 1440, "height": 900}
    page = browser.new_page(viewport=viewport)
    errors = []
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    email = f"e2e-{uuid.uuid4().hex[:8]}@example.com"

    page.goto(f"{args.base}/login")
    page.get_by_role("button", name="New here? Create an account").click()
    page.get_by_label("Full name").fill("Asha Verma")
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password").fill("Correct-Horse-Battery-9")
    page.get_by_role("button", name="Create account").click()
    page.wait_for_url("**/resumes")
    step("registered and redirected to resume manager")

    page.locator("input[type=file]").set_input_files(args.resume)
    review = page.get_by_role("link", name="Review & verify")
    expect(review).to_be_visible(timeout=30000)
    shot(page, "01-resumes")
    review.click()
    page.wait_for_url("**/review")
    expect(page.get_by_label("Title").first).to_have_value("Senior Software Engineer")
    shot(page, "02-review")
    page.get_by_role("button", name="Confirm & apply to profile").click()
    page.wait_for_url("**/profile")
    expect(page.get_by_text("Profile 100% complete")).to_be_visible(timeout=10000)
    shot(page, "03-profile")
    step("resume parsed, reviewed and applied to verified profile")

    page.goto(f"{args.base}/sources")
    page.get_by_role("button", name="Add company").click()
    page.get_by_label("Name").fill("Example SaaS")
    page.get_by_role("dialog").get_by_role("button", name="Save").click()
    expect(page.get_by_role("cell", name=re.compile("Example SaaS"))).to_be_visible()
    page.get_by_role("button", name="Add job source").click()
    page.get_by_label("Company").click()
    page.get_by_role("option", name="Example SaaS").click()
    page.get_by_label("Board token / site name").fill("examplesaas")
    page.get_by_role("dialog").get_by_role("button", name="Save").click()
    expect(page.get_by_text(re.compile(r"3 active jobs"))).to_be_visible(timeout=30000)
    shot(page, "04-sources")
    step("company and Greenhouse source added; 3 jobs synchronised by the Celery worker")

    page.goto(f"{args.base}/recommended")
    first = page.locator(".match-card").first
    expect(first).to_be_visible(timeout=30000)
    expect(first).to_contain_text("Senior Python Developer")
    shot(page, "05-recommended")
    step("recommended jobs ranked with explanations")

    first.get_by_role("link").first.click()
    page.wait_for_url(re.compile(r".*/jobs/\d+$"))
    expect(page.get_by_text("Score breakdown")).to_be_visible()
    shot(page, "06-job-detail")
    page.get_by_role("button", name="Tailor resume").click()
    page.wait_for_url(re.compile(r".*/resume-versions/\d+$"), timeout=60000)
    expect(page.get_by_text("Master (verified profile)")).to_be_visible()
    shot(page, "07-tailored-compare")
    page.get_by_role("tab", name=re.compile("Requirement map")).click()
    shot(page, "08-requirement-map")
    page.get_by_role("button", name="Approve for applications").click()
    page.get_by_role("dialog").get_by_role("button", name="Approve", exact=True).click()
    expect(page.get_by_text(re.compile("approved", re.I)).first).to_be_visible()
    step("tailored resume generated, compared side by side and approved")

    page.goto(f"{args.base}/applications")
    page.get_by_role("row").nth(1).click()
    expect(page.get_by_text("Ready to apply").first).to_be_visible(timeout=10000)
    official = page.get_by_role("link", name="Open official application page")
    assert official.get_attribute("href").startswith("https://boards.greenhouse.io/examplesaas/jobs/4012001")
    shot(page, "09-application-ready")
    page.get_by_role("button", name="I submitted it").click()
    page.get_by_label("Confirmation number / reference (optional)").fill("GH-E2E-1")
    page.get_by_role("dialog").get_by_role("button", name="Save").click()
    expect(page.get_by_text(re.compile("via manual · ref GH-E2E-1"))).to_be_visible(timeout=10000)
    shot(page, "10-application-submitted")
    step("application approved, official link verified, submission recorded with history")

    page.goto(f"{args.base}/dashboard")
    submitted = page.locator("a.metric", has_text="Applications submitted")
    expect(submitted.locator(".value")).to_have_text("1")
    expect(page.locator("a.metric", has_text="Jobs discovered").locator(".value")).to_have_text("3")
    shot(page, "11-dashboard")
    page.goto(f"{args.base}/analytics")
    expect(page.locator("canvas").first).to_be_visible()
    shot(page, "12-analytics")
    for path in ["jobs", "interviews", "settings", "notifications", "documents"]:
        page.goto(f"{args.base}/{path}")
        page.wait_for_load_state("networkidle")
        shot(page, f"13-{path}")
    step("dashboard and analytics show real data; all pages render")
    browser.close()
    real_errors = [e for e in errors if "fonts.g" not in e and "net::ERR" not in e]
    if real_errors:
        print("Console errors:", *real_errors, sep="\n  ")
        sys.exit(1)
    print("UI workflow passed")
