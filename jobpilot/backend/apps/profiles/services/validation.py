"""Defensive validation of uploaded resume files (treated as untrusted input)."""
import hashlib
import io
import zipfile

from django.conf import settings

from apps.core.exceptions import DomainError

ALLOWED = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MAX_PDF_PAGES = 30
MAX_DOCX_UNCOMPRESSED = 40 * 1024 * 1024
MAX_DOCX_ENTRIES = 2000


def validate_resume_upload(uploaded):
    """Return (kind, content_type, data, sha256) or raise DomainError."""
    name = (uploaded.name or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ALLOWED:
        raise DomainError("Only PDF and DOCX resumes are supported.", code="unsupported_file_type")
    if uploaded.size > settings.RESUME_MAX_UPLOAD_BYTES:
        raise DomainError(
            f"File is too large (max {settings.RESUME_MAX_UPLOAD_BYTES // (1024 * 1024)} MB).", code="file_too_large"
        )
    data = uploaded.read()
    if not data:
        raise DomainError("The uploaded file is empty.", code="empty_file")
    if ext == "pdf":
        _check_pdf(data)
    else:
        _check_docx(data)
    return ext, ALLOWED[ext], data, hashlib.sha256(data).hexdigest()


def _check_pdf(data: bytes):
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise DomainError("File content is not a valid PDF.", code="invalid_file")
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DomainError("Password-protected PDFs are not supported.", code="encrypted_file")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DomainError(f"PDF has too many pages (max {MAX_PDF_PAGES}).", code="too_many_pages")
        # Embedded JavaScript / launch actions are a red flag for a resume.
        root = reader.trailer.get("/Root", {})
        names = root.get("/Names", {}) if hasattr(root, "get") else {}
        if hasattr(names, "get") and names.get("/JavaScript") is not None:
            raise DomainError("PDFs containing scripts are not accepted.", code="unsafe_file")
        if hasattr(root, "get") and root.get("/OpenAction") is not None and b"/JavaScript" in data:
            raise DomainError("PDFs containing scripts are not accepted.", code="unsafe_file")
    except DomainError:
        raise
    except (PdfReadError, Exception) as exc:
        raise DomainError("The PDF could not be read.", code="invalid_file") from exc


def _check_docx(data: bytes):
    if not data.startswith(b"PK\x03\x04"):
        raise DomainError("File content is not a valid DOCX.", code="invalid_file")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_DOCX_ENTRIES:
                raise DomainError("DOCX structure is too large.", code="unsafe_file")
            if sum(i.file_size for i in infos) > MAX_DOCX_UNCOMPRESSED:
                raise DomainError("DOCX expands to an unsafe size.", code="unsafe_file")
            names = {i.filename for i in infos}
            if "word/document.xml" not in names:
                raise DomainError("File is not a Word document.", code="invalid_file")
            if any(n.lower().endswith("vbaproject.bin") for n in names):
                raise DomainError("Macro-enabled documents are not accepted.", code="unsafe_file")
            content_types = zf.read("[Content_Types].xml") if "[Content_Types].xml" in names else b""
            if b"macroEnabled" in content_types:
                raise DomainError("Macro-enabled documents are not accepted.", code="unsafe_file")
    except DomainError:
        raise
    except zipfile.BadZipFile as exc:
        raise DomainError("The DOCX could not be read.", code="invalid_file") from exc
