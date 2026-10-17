"""UI check: add a company career page source and see its jobs discovered and scored.

Needs the mock board server (scripts/mock_job_boards.py on :8765) and the backend started with
SOURCE_ALLOW_PRIVATE_HOSTS=1 (development only), plus a user with a verified profile.
"""
import argparse
import glob
import os
import re

from playwright.sync_api import expect, sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://127.0.0.1:4200")
parser.add_argument("--email", required=True)
parser.add_argument("--password", required=True)
parser.add_argument("--careers", default="http://127.0.0.1:8765/careers")
parser.add_argument("--shots", default="e2e-shots")
args = parser.parse_args()
os.makedirs(args.shots, exist_ok=True)
CHROME = os.environ.get("BROWSER_EXECUTABLE_PATH") or (glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome") or [None])[0]

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME) if CHROME else p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{args.base}/login")
    page.get_by_label("Email").fill(args.email)
    page.get_by_label("Password").fill(args.password)
    page.get_by_role("button", name="Sign in").click()
    page.wait_for_url("**/dashboard")
    page.goto(f"{args.base}/sources")
    page.get_by_role("button", name="Add company").click()
    page.get_by_role("dialog").get_by_label("Name").fill("Example Careers Co")
    page.get_by_role("dialog").get_by_role("button", name="Save").click()
    expect(page.get_by_role("cell", name="Example Careers Co")).to_be_visible()
    page.get_by_role("button", name="Add job source").click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_label("Company").click()
    page.get_by_role("option", name="Example Careers Co").click()
    dialog.get_by_label("Source type").click()
    page.get_by_role("option", name=re.compile("Company career page")).click()
    dialog.get_by_label("Board token, site name or URL").fill(args.careers)
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text(re.compile(r"2 active jobs"))).to_be_visible(timeout=30000)
    expect(page.get_by_text("apply on company site").first).to_be_visible()
    page.screenshot(path=f"{args.shots}/career-source.png", full_page=True)
    page.goto(f"{args.base}/recommended")
    expect(page.locator(".match-card", has_text="Senior Python Developer").first).to_be_visible(timeout=30000)
    expect(page.locator(".match-card", has_text="AI Engineer").first).to_be_visible()
    page.screenshot(path=f"{args.shots}/career-recommended.png", full_page=True)
    browser.close()
    print("Career page source workflow passed")
