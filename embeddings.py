from typing import List, Optional
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return 2D array of embeddings, shape (len(texts), dim).

    Uses OpenAI embeddings when available; otherwise falls back to TF-IDF + SVD.
    The fallback uses character n-grams to be robust to typos and variants.
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            vecs = [d.embedding for d in resp.data]
            arr = np.asarray(vecs, dtype=np.float32)
            # L2 normalize
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            return arr / norms
        except Exception:
            pass

    # Fallback: TF-IDF on character n-grams + SVD to simulate dense embeddings
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_features=20000,
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
    )
    X = vectorizer.fit_transform(texts)
    k = min(256, max(16, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    dense = svd.fit_transform(X)
    norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
    return dense / norms


def cosine_rank(query: str, docs: List[str], top_k: int = 10):
    """Return list of (index, score) sorted by similarity.
    Embeds [query] + docs, computes cosine similarity.
    """
    all_texts = [query] + docs
    E = embed_texts(all_texts)
    qv = E[0:1]
    dv = E[1:]
    sims = cosine_similarity(qv, dv).ravel()
    order = np.argsort(-sims)
    return [(int(i), float(sims[i])) for i in order[:top_k]]
