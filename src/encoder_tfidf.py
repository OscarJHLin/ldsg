"""
Lightweight text encoder using TF-IDF + cosine similarity

This is a fallback encoder that doesn't require sentence-transformers.
It uses sklearn's TfidfVectorizer for basic semantic similarity.
"""
import os
import re

# Clear proxy before loading heavy deps
for k in list(os.environ.keys()):
    if any(p in k.lower() for p in ["proxy", "http", "https", "socks"]):
        os.environ.pop(k, None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_vectorizer = None


def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(
            max_features=512,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
    return _vectorizer


def encode(texts):
    """
    Encode a string or list of strings to TF-IDF vectors.
    Returns 2D numpy array (N, max_features).
    """
    vec = get_vectorizer()
    if isinstance(texts, str):
        texts = [texts]
    return vec.fit_transform(texts).toarray()


def encode_flat(text):
    """Encode a single string, return 1D vector (max_features,)."""
    return encode(text)[0]


def cosine_sim(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    v1 = np.asarray(vec1).flatten()
    v2 = np.asarray(vec2).flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))
