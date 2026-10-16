"""Text embeddings for semantic matching (stored in pgvector columns).

The default ``hashing`` provider is a deterministic, dependency-free feature-hashing
embedding of unigrams and bigrams. It captures lexical overlap only, which is why
semantic similarity is just one weighted signal in the match score. Configure
``EMBEDDING_PROVIDER=openai`` with an OpenAI-compatible endpoint for model embeddings.
"""
from __future__ import annotations

import hashlib
import math
import re

import httpx
from django.conf import settings

DIM = 384
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")
_STOP = {
    "the", "and", "for", "with", "you", "our", "are", "will", "your", "that", "this", "have", "from", "work",
    "team", "who", "what", "about", "into", "their", "they", "them", "can", "all", "has", "not", "but", "a",
    "an", "of", "to", "in", "on", "at", "as", "is", "be", "or", "we", "by", "it", "us",
}


def _tokens(text: str) -> list[str]:
    return [t.strip(".-") for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def hashing_embedding(text: str) -> list[float]:
    vec = [0.0] * DIM
    tokens = _tokens(text)
    features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    for feat in features:
        digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def embed(text: str) -> list[float]:
    if settings.EMBEDDING_PROVIDER == "openai":
        return _openai_embedding(text)
    return hashing_embedding(text)


def _openai_embedding(text: str) -> list[float]:
    base = (settings.EMBEDDING_BASE_URL or "https://api.openai.com/v1").rstrip("/")
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{base}/embeddings",
            json={"model": settings.EMBEDDING_MODEL, "input": text[:16000], "dimensions": DIM},
            headers={"Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"},
        )
        resp.raise_for_status()
        vector = resp.json()["data"][0]["embedding"]
    if len(vector) != DIM:
        raise ValueError(f"Embedding provider returned {len(vector)} dimensions, expected {DIM}")
    return vector


def cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a, b = list(a), list(b)
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return float(dot / (na * nb)) if na and nb else 0.0
