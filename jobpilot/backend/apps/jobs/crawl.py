"""Safe fetching of public web pages for career-site sources.

* `check_public_url` blocks SSRF: only http(s) on ports 80/443 to hosts that resolve
  exclusively to public IP addresses (checked again on every redirect hop).
* `RobotsPolicy` honours robots.txt (RFC 9309): 4xx → no rules, 5xx/unreachable → disallow.
* `parse_html` / `iter_jobpostings` / `jobposting_to_job` read schema.org JobPosting JSON-LD,
  the structured data career sites publish for search engines.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from django.conf import settings

from apps.core.text import clean_text, html_to_text

from .normalization import NormalizedJob, Salary, normalize_employment_type

ROBOTS_AGENT = "JobPilot"


class UnsafeURL(ValueError):
    pass


def _resolve(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)]


def check_public_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURL("Only http(s) URLs are allowed.")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeURL("The URL has no host name.")
    if parts.username or parts.password:
        raise UnsafeURL("URLs with credentials are not allowed.")
    if settings.SOURCE_ALLOW_PRIVATE_HOSTS:
        return url
    if parts.port not in (None, 80, 443):
        raise UnsafeURL("Only the standard web ports (80/443) are allowed.")
    try:
        addresses = _resolve(host)
    except OSError as exc:
        raise UnsafeURL(f"Cannot resolve host '{host}'.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global or ip.is_multicast:
            raise UnsafeURL(f"'{host}' resolves to a non-public address; refusing to fetch it.")
    return url


class RobotsPolicy:
    """Per-sync robots.txt cache keyed by origin."""

    def __init__(self, http):
        self.http = http
        self._parsers: dict[str, RobotFileParser | None] = {}

    def _origin(self, url):
        p = urlsplit(url)
        return f"{p.scheme}://{p.netloc}"

    def _parser(self, url) -> RobotFileParser | None:
        origin = self._origin(url)
        if origin not in self._parsers:
            from .http import SourceHTTPError

            parser = RobotFileParser()
            try:
                response = self.http.request("GET", f"{origin}/robots.txt", headers={"Accept": "text/plain"})
                parser.parse(response.text.splitlines())
            except SourceHTTPError as exc:
                if exc.status_code and 400 <= exc.status_code < 500:
                    parser.parse([])  # no robots.txt: everything allowed
                else:
                    parser = None  # server error / unreachable: be conservative
            self._parsers[origin] = parser
        return self._parsers[origin]

    def allowed(self, url) -> bool:
        parser = self._parser(url)
        return parser is not None and parser.can_fetch(ROBOTS_AGENT, url)

    def sitemaps(self, url) -> list[str]:
        parser = self._parser(url)
        return list(parser.site_maps() or []) if parser else []


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.jsonld: list[str] = []
        self.links: list[str] = []
        self._in_jsonld = False
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "script" and (a.get("type") or "").lower().strip() == "application/ld+json":
            self._in_jsonld, self._buf = True, []
        elif tag == "a" and a.get("href"):
            self.links.append(a["href"])

    def handle_endtag(self, tag):
        if tag == "script" and self._in_jsonld:
            self.jsonld.append("".join(self._buf))
            self._in_jsonld = False

    def handle_data(self, data):
        if self._in_jsonld:
            self._buf.append(data)


def parse_html(html: str, base_url: str) -> tuple[list[str], list[str]]:
    """Return (json-ld blobs, absolute links without fragments)."""
    parser = _PageParser()
    parser.feed(html or "")
    links = []
    for href in parser.links:
        href = href.strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        links.append(urljoin(base_url, href).split("#", 1)[0])
    return parser.jsonld, list(dict.fromkeys(links))


def _walk(node):
    if isinstance(node, list):
        for item in node:
            yield from _walk(item)
    elif isinstance(node, dict):
        yield node
        for key in ("@graph", "itemListElement", "item", "mainEntity"):
            if key in node:
                yield from _walk(node[key])


def _is_jobposting(node: dict) -> bool:
    t = node.get("@type")
    types = t if isinstance(t, list) else [t]
    return "JobPosting" in types


def iter_jobpostings(blobs: list[str]):
    for blob in blobs:
        try:
            data = json.loads(blob.strip())
        except ValueError:
            continue
        for node in _walk(data):
            if _is_jobposting(node):
                yield node


def _text(value) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("value") or "")
    if isinstance(value, list):
        return ", ".join(_text(v) for v in value if v)
    return str(value or "")


def _date(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _locations(posting) -> list[str]:
    out = []
    places = posting.get("jobLocation") or []
    for place in places if isinstance(places, list) else [places]:
        if not isinstance(place, dict):
            continue
        address = place.get("address") or {}
        if isinstance(address, str):
            out.append(address)
            continue
        parts = [_text(address.get(k)) for k in ("addressLocality", "addressRegion", "addressCountry")]
        label = ", ".join(p for p in parts if p)
        if label:
            out.append(label)
    return out


def _salary(posting) -> Salary:
    base = posting.get("baseSalary") or posting.get("estimatedSalary")
    if isinstance(base, list):
        base = base[0] if base else None
    if not isinstance(base, dict):
        return Salary()
    value = base.get("value") if isinstance(base.get("value"), dict) else {"value": base.get("value")}
    lo, hi = value.get("minValue") or value.get("value"), value.get("maxValue") or value.get("value")
    try:
        lo, hi = int(float(lo)), int(float(hi))
    except (TypeError, ValueError):
        return Salary()
    unit = str(value.get("unitText") or base.get("unitText") or "YEAR").upper()
    period = {"YEAR": "year", "MONTH": "month", "HOUR": "hour"}.get(unit)
    if not period or lo <= 0 or hi < lo:
        return Salary()
    return Salary(lo, hi, str(base.get("currency") or "")[:3].upper(), period, "")


def is_expired(posting, now=None) -> bool:
    until = _date(posting.get("validThrough"))
    return bool(until and until < (now or datetime.now(timezone.utc)))


def url_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


def jobposting_to_job(posting: dict, page_url: str, external_id: str | None = None) -> NormalizedJob:
    url = str(posting.get("url") or page_url)
    locations = _locations(posting)
    remote = str(posting.get("jobLocationType") or "").upper() == "TELECOMMUTE"
    if remote:
        requirement = _text(posting.get("applicantLocationRequirements"))
        locations = locations or [f"Remote{' - ' + requirement if requirement else ''}"]
    employment = posting.get("employmentType")
    employment = employment[0] if isinstance(employment, list) and employment else employment
    org = posting.get("hiringOrganization")
    return NormalizedJob(
        external_id=external_id or url_id(page_url),
        title=clean_text(_text(posting.get("title") or posting.get("name")), 300),
        canonical_url=url,
        apply_url=url,
        description_text=html_to_text(str(posting.get("description") or "")),
        department=_text(posting.get("occupationalCategory") or posting.get("industry"))[:200],
        location_raw=" / ".join(locations),
        locations=locations,
        work_model="remote" if remote else "unknown",
        employment_type=normalize_employment_type(str(employment or "").replace("_", " "), _text(posting.get("title"))),
        salary=_salary(posting),
        published_at=_date(posting.get("datePosted")),
        raw_metadata={"page_url": page_url, "identifier": _text(posting.get("identifier"))[:120],
                      "hiring_organization": _text(org)[:200], "valid_through": str(posting.get("validThrough") or "")},
    ).finalize()


JOB_LINK_HINT = re.compile(
    r"/(jobs?|careers?|positions?|openings?|opportunit(y|ies)|vacanc(y|ies)|requisitions?|roles?|job-details?)(/|-|\?|$)", re.I)
KNOWN_ATS_HOSTS = {
    "boards.greenhouse.io": "greenhouse", "job-boards.greenhouse.io": "greenhouse", "jobs.lever.co": "lever",
    "jobs.eu.lever.co": "lever", "myworkdayjobs.com": "workday",
}


def detect_ats(url: str) -> str | None:
    host = (urlsplit(url).hostname or "").lower()
    for known, name in KNOWN_ATS_HOSTS.items():
        if host == known or host.endswith("." + known):
            return name
    return None


def sitemap_locations(xml: str) -> tuple[list[str], list[str]]:
    """Return (page urls, nested sitemap urls) from a sitemap or sitemap index."""
    locs = [m.strip() for m in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml or "", re.I)]
    if re.search(r"<sitemapindex", xml or "", re.I):
        return [], locs
    return locs, []
