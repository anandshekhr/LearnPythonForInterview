"""Private, optionally encrypted file storage for resumes and generated documents.

Files live under PRIVATE_MEDIA_ROOT, which is never exposed by a web server.
Downloads go through API views that enforce ownership.
"""
import os

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from django.utils.functional import LazyObject

from . import crypto


class EncryptedFileSystemStorage(FileSystemStorage):
    def __init__(self, **kwargs):
        kwargs.setdefault("base_url", None)
        kwargs.setdefault("file_permissions_mode", 0o600)
        kwargs.setdefault("directory_permissions_mode", 0o700)
        super().__init__(**kwargs)

    # Resolved on every access so PRIVATE_MEDIA_ROOT can change (tests, multi-tenant volumes).
    @property
    def base_location(self):
        return str(settings.PRIVATE_MEDIA_ROOT)

    @property
    def location(self):
        return os.path.abspath(self.base_location)

    def _save(self, name, content):
        content.seek(0)
        data = content.read()
        if settings.ENCRYPT_FILES_AT_REST:
            data = b"JPENC1" + crypto.encrypt_bytes(data)
        return super()._save(name, ContentFile(data))

    def _open(self, name, mode="rb"):
        with super()._open(name, "rb") as fh:
            data = fh.read()
        if data.startswith(b"JPENC1"):
            data = crypto.decrypt_bytes(data[6:])
        return ContentFile(data, name=name)

    def url(self, name):  # pragma: no cover - deliberately unsupported
        raise NotImplementedError("Private files have no public URL")


class _LazyPrivateStorage(LazyObject):
    def _setup(self):
        self._wrapped = EncryptedFileSystemStorage()


private_storage = _LazyPrivateStorage()


def get_private_storage():
    return private_storage
