"""Helpers for handling untrusted text (job descriptions, resumes, web pages)."""
import html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_TAG_RE = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|tr|section)[^>]*>", re.I)
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_WS_RE = re.compile(r"[ \t\f\v]+")
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​-‏‪-‮⁠-⁤]")


def html_to_text(value: str) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    # Greenhouse returns HTML-escaped HTML, so unescape twice when needed.
    if "&lt;" in text or "&gt;" in text:
        text = html.unescape(text)
    text = _SCRIPT_RE.sub(" ", text)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    return clean_text(text)


def clean_text(value: str, max_length: int | None = None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = _CONTROL_RE.sub("", text)
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    out, blank = [], False
    for line in lines:
        if not line:
            if not blank and out:
                out.append("")
            blank = True
            continue
        out.append(line)
        blank = False
    text = "\n".join(out).strip()
    if max_length is not None:
        text = text[:max_length]
    return text


def wrap_untrusted(label: str, content: str, max_length: int = 20000) -> str:
    """Delimit untrusted content for LLM prompts so it is treated as data, not instructions."""
    safe = clean_text(content, max_length=max_length).replace("</untrusted", "</ untrusted")
    return f'<untrusted_{label}>\n{safe}\n</untrusted_{label}>'


def normalize_token(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())
