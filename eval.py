from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def score_factual_consistency(text: str, fact_tokens: List[str]) -> float:
    if not text.strip():
        return 0.0
    if not fact_tokens:
        return 0.5
    text_lower = text.lower()
    hits = sum(1 for t in fact_tokens if t.lower() in text_lower)
    return min(1.0, hits / max(3, len(fact_tokens)))


def score_creativity(text: str) -> float:
    tokens = [t for t in text.split() if t.isalpha() or t.isalnum()]
    if len(tokens) < 10:
        return 0.4
    unique = len(set(tokens))
    ttr = unique / len(tokens)
    # Squash into [0,1] with a soft cap
    return float(max(0.0, min(1.0, 0.6 + 0.8 * (ttr - 0.4))))


def score_completeness(text: str, required_keywords: List[str]) -> float:
    if not required_keywords:
        return 1.0
    lower = text.lower()
    hits = sum(1 for k in required_keywords if k.lower() in lower)
    return min(1.0, hits / len(required_keywords))


def embedding_similarity(a: str, b: str) -> float:
    vec = TfidfVectorizer(min_df=1, max_features=5000)
    mat = vec.fit_transform([a, b])
    sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
    return float(max(0.0, min(1.0, sim)))


def evaluate_generation(text: str, location: Dict[str, any]) -> Dict[str, float]:
    name = location.get("name", "")
    type_ = location.get("type", "")
    dimension = location.get("dimension", "")
    residents = location.get("residents", [])

    fact_tokens: List[str] = [name, type_, dimension] + [r.get("name", "") for r in residents[:8]]
    required_keywords: List[str] = [name, type_, dimension]

    consistency = score_factual_consistency(text, [t for t in fact_tokens if t])
    creativity = score_creativity(text)
    completeness = score_completeness(text, [k for k in required_keywords if k])

    fact_sheet = f"Location: {name}. Type: {type_}. Dimension: {dimension}. Residents: " + \
                 ", ".join(r.get("name", "?") for r in residents[:12])
    grounding_sim = embedding_similarity(text, fact_sheet)

    overall = float(np.mean([consistency, creativity, completeness, grounding_sim]))

    return {
        "consistency": float(consistency),
        "creativity": float(creativity),
        "completeness": float(completeness),
        "grounding": float(grounding_sim),
        "overall": overall,
    }
