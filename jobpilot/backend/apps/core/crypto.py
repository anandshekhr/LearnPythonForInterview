"""Symmetric encryption for stored integration credentials (e.g. employer-issued API keys)."""
import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


def _fernet():
    key = settings.DATA_ENCRYPTION_KEY
    if not key:
        if settings.DEBUG or settings.TESTING:
            key = base64.urlsafe_b64encode(hashlib.sha256(settings.SECRET_KEY.encode()).digest()).decode()
        else:
            raise ImproperlyConfigured("DATA_ENCRYPTION_KEY is required to encrypt files and credentials")
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(value: str) -> str:
    if not value:
        return ""
    return _fernet().encrypt(value.encode()).decode()


def decrypt(token: str) -> str:
    if not token:
        return ""
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise ImproperlyConfigured("Stored credential cannot be decrypted with the configured key") from exc


def encrypt_bytes(data: bytes) -> bytes:
    return _fernet().encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    try:
        return _fernet().decrypt(data)
    except InvalidToken as exc:
        raise ImproperlyConfigured("Stored file cannot be decrypted with the configured key") from exc
