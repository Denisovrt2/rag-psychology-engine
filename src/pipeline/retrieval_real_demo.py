"""
Public fragment based on a real hybrid retrieval stage.

Идея этого слоя:
не принимать решение о релевантности одной "магической" формулой,
а собрать несколько retrieval-сигналов в единую feature-table,
которую уже можно использовать downstream:
- reranker
- scorer
- synthesis layer
"""

from __future__ import annotations

import math
import re
from typing import Optional


_TOKEN_RE_SIMPLE = re.compile(r"[а-яa-z0-9]+(?:-[а-яa-z0-9]+)?", re.IGNORECASE)
_DEFAULT_RRF_K = 20


def simple_tokenize(text: Optional[str]) -> list[str]:
    if not text:
        return []
    t = text.lower().replace("ё", "е")
    t = re.sub(r"(?<=\w)-\s+(?=\w)", "", t)
    tokens = _TOKEN_RE_SIMPLE.findall(t)
    return [w for w in tokens if len(w) >= 3]


def compute_keyword_overlap(query_text: Optional[str], chunk_text: Optional[str]) -> int:
    q = set(simple_tokenize(query_text))
    if not q:
        return 0
    c = set(simple_tokenize(chunk_text))
    return len(q & c)


def reciprocal_rank(rank: Optional[int], k: int = _DEFAULT_RRF_K) -> float:
    """
    Reciprocal-rank score. Higher is better.
    Missing rank -> 0.
    """
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / (k + rank)


def normalized_rank_score(rank: Optional[int], k: int = _DEFAULT_RRF_K) -> float:
    rr = reciprocal_rank(rank, k=k)
    rr_max = reciprocal_rank(1, k=k)
    if rr_max <= 0.0:
        return 0.0
    return max(0.0, min(rr / rr_max, 1.0))


def rank_agreement_score(
    dense_rank: Optional[int],
    sparse_rank: Optional[int],
    *,
    k: int = _DEFAULT_RRF_K,
) -> float:
    """
    Насколько согласованы dense и sparse retrieval.
    Это важный сигнал качества: если два retriever'а независимо поднимают один chunk,
    он обычно сильнее, чем результат только одного слоя.
    """
    if dense_rank is None or sparse_rank is None:
        return 0.0
    if dense_rank <= 0 or sparse_rank <= 0:
        return 0.0

    gap = abs(dense_rank - sparse_rank)
    return 1.0 / (1.0 + (gap / max(float(k), 1.0)))


def normalize_sparse_score(sparse_score: Optional[float]) -> float:
    """
    Сжимаем BM25-подобный raw score в bounded [0, 1).
    """
    s = max(float(sparse_score or 0.0), 0.0)
    return 1.0 - (1.0 / (1.0 + math.log1p(s)))


def dense_similarity_from_distance(dense_distance: Optional[float]) -> float:
    """
    Конвертация distance -> similarity.
    Lower distance -> better match.
    """
    d = max(float(dense_distance or 1.0), 0.0)
    return math.exp(-d)


def harmonic_mean(a: float, b: float, eps: float = 1e-9) -> float:
    a = max(float(a), 0.0)
    b = max(float(b), 0.0)
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return (2.0 * a * b) / (a + b + eps)


def build_hybrid_feature_row(
    *,
    query_text: str,
    chunk_text: str,
    dense_rank: Optional[int],
    sparse_rank: Optional[int],
    dense_distance: Optional[float],
    sparse_score: Optional[float],
) -> dict:
    """
    Вместо одной "магической" hybrid score функция возвращает retrieval feature row.

    Это архитектурно сильнее, потому что downstream слой может:
    - rerank
    - учитывать agreement
    - объяснять, почему chunk оказался высоко
    """
    keyword_overlap = compute_keyword_overlap(query_text, chunk_text)
    dense_similarity = dense_similarity_from_distance(dense_distance)
    sparse_norm = normalize_sparse_score(sparse_score)
    agreement = rank_agreement_score(dense_rank, sparse_rank)

    blended_signal = harmonic_mean(dense_similarity, sparse_norm)

    return {
        "keyword_overlap": keyword_overlap,
        "dense_rank_score": normalized_rank_score(dense_rank),
        "sparse_rank_score": normalized_rank_score(sparse_rank),
        "dense_similarity": round(dense_similarity, 6),
        "sparse_score_normalized": round(sparse_norm, 6),
        "rank_agreement": round(agreement, 6),
        "blended_signal": round(blended_signal, 6),
        "product_meaning": (
            "Hybrid retrieval строится не как одна формула, а как feature layer "
            "для downstream reranking и synthesis."
        ),
    }


if __name__ == "__main__":
    demo_row = build_hybrid_feature_row(
        query_text="Ребенок тревожится перед школой и плохо засыпает",
        chunk_text=(
            "Школьная тревожность часто проявляется нарушением сна, "
            "эмоциональным напряжением и ожиданием негативной оценки."
        ),
        dense_rank=3,
        sparse_rank=2,
        dense_distance=0.21,
        sparse_score=8.7,
    )

    print(demo_row)