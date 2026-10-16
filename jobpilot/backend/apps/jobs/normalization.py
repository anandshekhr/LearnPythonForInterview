"""Normalisation of raw job data into consistent, comparable values."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

from apps.core.text import clean_text, normalize_token

# ------------------------------------------------------------------- titles

_TITLE_ABBREV = {
    r"\bsr\.?\b": "senior", r"\bjr\.?\b": "junior", r"\bswe\b": "software engineer", r"\bsde\b": "software development engineer",
    r"\beng\.?\b": "engineer", r"\bdev\b": "developer", r"\bml\b": "machine learning", r"\bai/ml\b": "ai machine learning",
    r"\bmgr\.?\b": "manager", r"\bfull[\s\-]?stack\b": "full stack", r"\bback[\s\-]?end\b": "backend", r"\bfront[\s\-]?end\b": "frontend",
}


def normalize_title(title: str) -> str:
    t = normalize_token(title)
    t = re.sub(r"\(.*?\)", " ", t)
    for pattern, repl in _TITLE_ABBREV.items():
        t = re.sub(pattern, repl, t)
    t = re.sub(r"[^a-z0-9+#/ ]", " ", t)
    t = re.sub(r"\b(i{1,3}|iv|v|[1-5])\b$", "", t.strip())  # "Engineer II"
    return re.sub(r"\s+", " ", t).strip()


_SENIORITY_RULES = [
    ("intern", r"\b(intern|internship|trainee|apprentice)\b"),
    ("principal", r"\b(principal|distinguished|fellow)\b"),
    ("manager", r"\b(engineering manager|manager|director|head of|vp|vice president)\b"),
    ("lead", r"\b(lead|staff|architect|tech lead|team lead)\b"),
    ("senior", r"\b(senior|sr|iii|level 3|l5|sde 3|sde iii)\b"),
    ("junior", r"\b(junior|jr|entry level|graduate|associate|fresher)\b"),
    ("mid", r"\b(ii|level 2|sde 2|sde ii|mid)\b"),
]


def detect_seniority(title: str) -> str:
    t = normalize_token(title)
    for level, pattern in _SENIORITY_RULES:
        if re.search(pattern, t):
            return level
    if re.search(r"\b(engineer|developer|scientist|analyst)\b", t):
        return "mid"
    return "unknown"


# ---------------------------------------------------------------- locations

_CITY_ALIASES = {
    "gurgaon": "Gurugram", "gurugram": "Gurugram", "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "bombay": "Mumbai",
    "mumbai": "Mumbai", "new delhi": "New Delhi", "delhi": "Delhi", "noida": "Noida", "greater noida": "Greater Noida",
    "pune": "Pune", "hyderabad": "Hyderabad", "chennai": "Chennai", "kolkata": "Kolkata", "ghaziabad": "Ghaziabad",
    "faridabad": "Faridabad", "ahmedabad": "Ahmedabad", "jaipur": "Jaipur", "chandigarh": "Chandigarh", "kochi": "Kochi",
}
NCR = {"Delhi", "New Delhi", "Noida", "Greater Noida", "Gurugram", "Ghaziabad", "Faridabad"}
_INDIA_TOKENS = {"india", "in", "ind"}


def split_locations(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"\s*(?:;|\||/|\bor\b|\band\b|•)\s*", raw, flags=re.I)
    out = [p.strip(" ,-") for p in parts]
    return list(dict.fromkeys(x for x in out if x))[:20]


def canonical_location(loc: str) -> str:
    lowered = normalize_token(loc)
    for alias, canonical in sorted(_CITY_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return canonical
    return loc.strip()


def location_tags(locations: list[str]) -> set[str]:
    """Expanded, lower-case tags used for preference matching (city, region, country)."""
    tags: set[str] = set()
    for loc in locations:
        lowered = normalize_token(loc)
        tags.add(lowered)
        city = canonical_location(loc)
        tags.add(city.lower())
        if city in NCR or re.search(r"\b(ncr|delhi ncr)\b", lowered):
            tags.update({"delhi ncr", "ncr"})
        if city in _CITY_ALIASES.values() or any(t in re.split(r"[\s,]+", lowered) for t in _INDIA_TOKENS):
            tags.add("india")
        if "remote" in lowered:
            tags.add("remote")
            if "india" in lowered or "apac" in lowered or "asia" in lowered:
                tags.add("remote india")
    return tags


def detect_work_model(location_raw: str, text: str = "", hint: str = "") -> str:
    hint = (hint or "").lower()
    if hint in {"remote", "hybrid", "onsite", "on-site", "in-office", "in office"}:
        return "onsite" if hint in {"on-site", "in-office", "in office"} else hint
    loc = (location_raw or "").lower()
    if "hybrid" in loc:
        return "hybrid"
    if "remote" in loc and "not remote" not in loc:
        return "remote"
    body = (text or "").lower()[:6000]
    if re.search(r"\bhybrid\b", body):
        return "hybrid"
    if re.search(r"\b(fully remote|100% remote|remote[- ]first|work from (home|anywhere)|this (role|position) is remote)\b", body):
        return "remote"
    if re.search(r"\b(on[- ]?site|in[- ]office|work from office|wfo)\b", body):
        return "onsite"
    return "unknown"


_EMPLOYMENT_MAP = {
    "fulltime": "full_time", "full time": "full_time", "full-time": "full_time", "permanent": "full_time", "regular": "full_time",
    "parttime": "part_time", "part time": "part_time", "part-time": "part_time",
    "contract": "contract", "contractor": "contract", "freelance": "contract", "temporary": "temporary", "temp": "temporary",
    "intern": "internship", "internship": "internship",
}


def normalize_employment_type(raw: str, title: str = "") -> str:
    key = normalize_token(raw).replace("_", " ")
    for k, v in _EMPLOYMENT_MAP.items():
        if k == key or k in key:
            return v
    if re.search(r"\bintern(ship)?\b", (title or "").lower()):
        return "internship"
    if re.search(r"\bcontract(or)?\b", (title or "").lower()):
        return "contract"
    return ""


# ------------------------------------------------------------------- salary


@dataclass
class Salary:
    minimum: int | None = None
    maximum: int | None = None
    currency: str = ""
    period: str = ""
    text: str = ""


_CURRENCY_SYMBOLS = {"₹": "INR", "rs": "INR", "inr": "INR", "$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP"}
_SALARY_RE = re.compile(
    r"(?P<cur>₹|rs\.?|inr|\$|usd|€|eur|£|gbp)?\s*(?P<lo>\d[\d,]*(?:\.\d+)?)\s*(?P<lounit>k|lpa|lakhs?|l|cr|crores?|m)?\s*"
    r"(?:-|–|to)\s*(?P<cur2>₹|rs\.?|inr|\$|usd|€|eur|£|gbp)?\s*(?P<hi>\d[\d,]*(?:\.\d+)?)\s*(?P<hiunit>k|lpa|lakhs?|l|cr|crores?|m)?"
    r"(?:\s*(?:per|/|a)\s*(?P<period>year|annum|yr|month|mo|hour|hr))?",
    re.I,
)
_UNIT = {"k": 1_000, "lpa": 100_000, "lakh": 100_000, "lakhs": 100_000, "l": 100_000, "cr": 10_000_000, "crore": 10_000_000,
         "crores": 10_000_000, "m": 1_000_000}


def _amount(value: str, unit: str | None) -> int:
    number = float(value.replace(",", ""))
    return int(number * _UNIT.get((unit or "").lower(), 1))


def parse_salary_text(text: str) -> Salary:
    """Best-effort extraction of a salary range from free text. Returns empty Salary if unsure."""
    if not text:
        return Salary()
    for m in _SALARY_RE.finditer(text):
        cur = (m.group("cur") or m.group("cur2") or "").lower().rstrip(".")
        unit = m.group("lounit") or m.group("hiunit")
        if not cur and not unit:
            continue  # bare number ranges ("3-5 years") are not salaries
        lo, hi = _amount(m.group("lo"), m.group("lounit") or unit), _amount(m.group("hi"), m.group("hiunit") or unit)
        if lo <= 0 or hi < lo:
            continue
        currency = _CURRENCY_SYMBOLS.get(cur, "INR" if (unit or "").lower() in {"lpa", "lakh", "lakhs", "l", "cr", "crore", "crores"} else "")
        period = (m.group("period") or "").lower()
        period = {"annum": "year", "yr": "year", "mo": "month", "hr": "hour"}.get(period, period)
        if not period:
            period = "year" if (unit or "").lower() in {"lpa", "lakh", "lakhs", "l", "cr", "k", "m"} or hi > 200_000 else ""
        if not currency or not period:
            continue
        return Salary(lo, hi, currency, period, m.group(0).strip()[:200])
    return Salary()


def annualize(amount: int | None, period: str) -> int | None:
    if amount is None:
        return None
    return {"year": amount, "month": amount * 12, "hour": amount * 2080}.get(period or "year", amount)


# ---------------------------------------------------------- normalized job


@dataclass
class NormalizedJob:
    external_id: str
    title: str
    canonical_url: str
    apply_url: str = ""
    description_text: str = ""
    department: str = ""
    location_raw: str = ""
    locations: list[str] = field(default_factory=list)
    work_model: str = "unknown"
    employment_type: str = ""
    salary: Salary = field(default_factory=Salary)
    published_at: datetime | None = None
    source_updated_at: datetime | None = None
    raw_metadata: dict = field(default_factory=dict)

    def finalize(self) -> "NormalizedJob":
        self.title = clean_text(self.title, 300)
        self.description_text = clean_text(self.description_text, 60000)
        if not self.locations:
            self.locations = split_locations(self.location_raw)
        self.locations = [canonical_location(x) for x in self.locations]
        if self.work_model == "unknown":
            self.work_model = detect_work_model(self.location_raw, self.description_text)
        if not self.salary.minimum:
            self.salary = parse_salary_text(self.description_text)
        self.canonical_url = canonical_url(self.canonical_url)
        return self

    @property
    def content_hash(self) -> str:
        material = "\x1f".join([
            self.title, self.description_text, self.location_raw, self.employment_type, self.department,
            str(self.salary.minimum), str(self.salary.maximum), self.apply_url,
        ])
        return hashlib.sha256(material.encode()).hexdigest()


def canonical_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") or "/"
    # Tracking parameters are dropped; ATS ids that live in the query (gh_jid) are kept.
    keep = "&".join(q for q in parts.query.split("&") if q.split("=")[0] in {"gh_jid", "jobId", "id"})
    return urlunsplit((parts.scheme.lower() or "https", parts.netloc.lower(), path, keep, ""))


def job_fingerprint(company_id: int, title: str, locations: list[str]) -> str:
    primary = sorted(normalize_token(x) for x in locations)[:1]
    material = f"{company_id}|{normalize_title(title)}|{primary[0] if primary else ''}"
    return hashlib.sha256(material.encode()).hexdigest()
