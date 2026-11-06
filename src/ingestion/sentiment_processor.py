from __future__ import annotations

from dataclasses import dataclass
import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.ingestion.news_collector import NewsArticle

logger = logging.getLogger(__name__)

_finbert_pipeline = None
_vader = SentimentIntensityAnalyzer()
_FINBERT_MIN_CHARS = 10


@dataclass
class SentimentResult:
    label: str
    positive: float
    negative: float
    neutral: float
    scorer: str = "finbert"


def _get_finbert_pipeline():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _finbert_pipeline = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,
                device=-1,
            )
        except Exception as e:
            logger.warning("FinBERT unavailable (%s), VADER only", e)
    return _finbert_pipeline


def _vader_score(text: str) -> SentimentResult:
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    total = scores["pos"] + scores["neg"] + scores["neu"] or 1.0
    return SentimentResult(
        label=label,
        positive=scores["pos"] / total,
        negative=scores["neg"] / total,
        neutral=scores["neu"] / total,
        scorer="vader",
    )


def score_headline(text: str) -> SentimentResult:
    if len(text) < _FINBERT_MIN_CHARS:
        return _vader_score(text)
    pipeline_fn = _get_finbert_pipeline()
    if pipeline_fn is None:
        return _vader_score(text)
    try:
        results = pipeline_fn(text[:512])[0]
        scores = {r["label"].lower(): r["score"] for r in results}
        label = max(scores, key=scores.get)
        return SentimentResult(
            label=label,
            positive=scores.get("positive", 0.0),
            negative=scores.get("negative", 0.0),
            neutral=scores.get("neutral", 0.0),
        )
    except Exception as e:
        logger.warning("FinBERT error: %s — VADER fallback", e)
        return _vader_score(text)


def score_articles(
    articles: list[NewsArticle],
) -> list[tuple[NewsArticle, SentimentResult]]:
    return [(a, score_headline(a.headline)) for a in articles]


def score_texts(texts: list[str], batch_size: int = 32) -> list[SentimentResult]:
    """One FinBERT call per batch; VADER per text on error, unavailability, or short text."""
    results: list[SentimentResult | None] = [None] * len(texts)
    long_idx = []
    for i, t in enumerate(texts):
        if len(t) < _FINBERT_MIN_CHARS:
            results[i] = _vader_score(t)
        else:
            long_idx.append(i)
    pipeline_fn = _get_finbert_pipeline() if long_idx else None
    for start in range(0, len(long_idx), batch_size):
        idx = long_idx[start:start + batch_size]
        batch = [texts[i][:512] for i in idx]
        outs = None
        if pipeline_fn is not None:
            try:
                outs = pipeline_fn(batch, batch_size=batch_size)
                if len(outs) != len(batch):
                    raise ValueError("FinBERT returned wrong number of results")
            except Exception as e:
                logger.warning("FinBERT batch error: %s — VADER fallback", e)
                outs = None
        for j, i in enumerate(idx):
            if outs is None:
                results[i] = _vader_score(texts[i])
                continue
            scores = {r["label"].lower(): r["score"] for r in outs[j]}
            results[i] = SentimentResult(
                label=max(scores, key=scores.get),
                positive=scores.get("positive", 0.0),
                negative=scores.get("negative", 0.0),
                neutral=scores.get("neutral", 0.0),
            )
    return results  # type: ignore[return-value]
