"""Encoder module - wraps sentence-transformers for embedding generation."""
import os

from sentence_transformers import SentenceTransformer
import numpy as np

_encoder = None


def _clear_proxy():
    """Clear proxy env vars to avoid socks:// issues with huggingface"""
    for k in list(os.environ.keys()):
        if any(p in k.lower() for p in ["proxy", "http", "https", "socks"]):
            os.environ.pop(k, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def get_encoder():
    global _encoder
    if _encoder is None:
        _clear_proxy()
        _encoder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    return _encoder

def encode(texts):
    """Encode a string or list of strings to embeddings. Always returns 2D (N, dim)."""
    encoder = get_encoder()
    if isinstance(texts, str):
        texts = [texts]
    embeddings = encoder.encode(texts, normalize_embeddings=True)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings

def encode_flat(text):
    """Encode a single string, return 1D vector (dim,)."""
    return encode(text)[0]

# Lazy-load projector
_projector = None
def _get_projector():
    global _projector
    if _projector is None:
        proj_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'projection_384_128.npy')
        try:
            _projector = np.load(proj_path)
        except Exception:
            return None
    return _projector

def project(v):
    """Project a 384-d vector to 128-d using the learned projection matrix."""
    proj = _get_projector()
    if proj is None:
        return np.asarray(v).flatten()  # fallback: no projection
    v = np.asarray(v).flatten()
    return (v @ proj) / (np.linalg.norm(v) + 1e-8)

def encode_128(text):
    """Encode to 128-d projected vector (for spatial graph operations)."""
    v384 = encode_flat(text)  # 384-d
    return project(v384)
