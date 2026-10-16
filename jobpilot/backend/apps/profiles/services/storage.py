import logging
import shutil

from django.conf import settings

logger = logging.getLogger(__name__)


def delete_user_files(user):
    """Remove every private file owned by the user (resumes and generated documents)."""
    root = settings.PRIVATE_MEDIA_ROOT / "users" / str(user.pk)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
        logger.info("user_files_deleted", extra={"user_id": user.pk})
