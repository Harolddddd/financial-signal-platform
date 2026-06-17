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
