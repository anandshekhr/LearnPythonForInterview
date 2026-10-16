from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    """Enable pgvector before any model with a VectorField is created.

    Requires a role allowed to CREATE EXTENSION (superuser in the official
    pgvector image). If a DBA pre-installs the extension, this is a no-op.
    """

    initial = True
    dependencies = []
    operations = [VectorExtension()]
