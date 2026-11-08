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


CN_MODELS = (
    "yiyanghkust/finbert-tone-chinese",
    "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
)
_cn_pipeline = None
_cn_scorer = "unscored"


def _get_cn_pipeline():
    """First Chinese sentiment model that loads; None when none can be loaded."""
    global _cn_pipeline, _cn_scorer
    if _cn_pipeline is None:
        from transformers import pipeline as hf_pipeline

        for model in CN_MODELS:
            try:
                _cn_pipeline = hf_pipeline("text-classification", model=model, top_k=None, device=-1)
                _cn_scorer = model.split("/")[-1]
                break
            except Exception as e:
                logger.warning("Chinese sentiment model %s unavailable: %s", model, e)
    return _cn_pipeline


def _cn_scores(out: list[dict]) -> dict[str, float]:
    """Both CN models name their labels (Positive/Negative[/Neutral]); lowercase them."""
    return {r["label"].lower(): r["score"] for r in out}


def score_texts_cn(texts: list[str], batch_size: int = 16) -> list[SentimentResult]:
    """Chinese FinBERT scoring; texts it cannot score come back as 'unscored' (null scores)."""
    unscored = SentimentResult("unscored", None, None, None, "unscored")  # type: ignore[arg-type]
    if not texts:
        return []
    pipe = _get_cn_pipeline()
    if pipe is None:
        return [unscored] * len(texts)
    results: list[SentimentResult] = []
    for start in range(0, len(texts), batch_size):
        batch = [t[:512] for t in texts[start:start + batch_size]]
        try:
            outs = pipe(batch, batch_size=batch_size, truncation=True, max_length=512)
        except Exception as e:
            logger.warning("Chinese sentiment batch error: %s", e)
            results.extend([unscored] * len(batch))
            continue
        for out in outs:
            s = _cn_scores(out)
            results.append(SentimentResult(
                label=max(s, key=s.get),
                positive=s.get("positive", 0.0),
                negative=s.get("negative", 0.0),
                neutral=s.get("neutral", 0.0),
                scorer=_cn_scorer,
            ))
    return results
