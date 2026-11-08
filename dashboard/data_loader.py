# dashboard/data_loader.py
from __future__ import annotations
import json
import logging
from pathlib import Path

import polars as pl

from config.markets import get_market
from src.backtesting.grader import Grade, ModelGrade, grade_model, build_leaderboard
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.walk_forward import FoldBacktestResult, WalkForwardBacktestResult
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.strategies.base import LiveSignal, Signal
from src.strategies.registry import list_strategies, load_strategy

logger = logging.getLogger(__name__)

def get_cache_dir(market: str) -> Path:
    return get_market(market).data_root / "cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Cache read failed (%s): %s — falling back to live compute", path, exc)
    return None


def _metrics_from_dict(d: dict) -> BacktestMetrics:
    return BacktestMetrics(
        n_trades=d["n_trades"],
        win_rate=d["win_rate"],
        profit_factor=d["profit_factor"],
        total_return_pct=d["total_return_pct"],
        sharpe_ratio=d["sharpe_ratio"],
        max_drawdown_pct=d["max_drawdown_pct"],
        precision_buy=d["precision_buy"],
        recall_buy=d["recall_buy"],
        f1_buy=d["f1_buy"],
        accuracy=d["accuracy"],
    )


def _grade_from_dict(d: dict) -> ModelGrade:
    return ModelGrade(
        model_name=d["model_name"],
        grade=Grade(d["grade"]),
        composite_score=d["composite_score"],
        metrics=_metrics_from_dict(d["metrics"]),
    )


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Public API — each function tries the cache first, falls back to live compute
# ---------------------------------------------------------------------------

def get_data_summary(parquet_dir: Path, market: str = "us") -> dict:
    cached = _load_cache(get_cache_dir(market) / "data_summary.json")
    if cached:
        return cached

    df = load_training_data(parquet_dir)
    tickers = df["ticker"].unique().to_list() if "ticker" in df.columns else []
    time_min = str(df["time"].min()) if "time" in df.columns else "N/A"
    time_max = str(df["time"].max()) if "time" in df.columns else "N/A"
    return {
        "n_tickers": len(tickers),
        "n_rows": len(df),
        "tickers": sorted(tickers),
        "date_range_start": time_min,
        "date_range_end": time_max,
    }


def get_leaderboard(
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    market: str = "us",
) -> list[ModelGrade]:
    cached = _load_cache(get_cache_dir(market) / "leaderboard.json")
    if cached:
        return [_grade_from_dict(g) for g in cached["grades"]]

    names = list_strategies()
    if not names:
        return []

    df = load_training_data(parquet_dir)
    grades: list[ModelGrade] = []
    for name in names:
        try:
            strategy = load_strategy(name)
            result = walk_forward_backtest_strategy(
                df, strategy, ohlcv_cols, feature_cols,
                train_window_days=400, test_window_days=21, step_days=21,
            )
            avg_metrics = BacktestMetrics(
                n_trades=sum(f.n_trades for f in result.folds),
                win_rate=result.mean_win_rate,
                profit_factor=0.0,
                total_return_pct=0.0,
                sharpe_ratio=result.mean_sharpe,
                max_drawdown_pct=result.worst_drawdown,
                precision_buy=result.weighted_precision_buy,
                recall_buy=0.0,
                f1_buy=0.0,
                accuracy=0.0,
            )
            grades.append(grade_model(name, avg_metrics))
        except Exception as e:
            logger.warning("Leaderboard skipping %s: %s", name, e)

    return build_leaderboard(grades)


def get_backtest_result(
    strategy_name: str,
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    market: str = "us",
) -> tuple[WalkForwardBacktestResult, ModelGrade]:
    cached = _load_cache(get_cache_dir(market) / f"backtest_{_safe(strategy_name)}.json")
    if cached:
        folds = [
            FoldBacktestResult(
                fold=f["fold"],
                train_start=f["train_start"],
                train_end=f["train_end"],
                test_start=f["test_start"],
                test_end=f["test_end"],
                metrics=_metrics_from_dict(f["metrics"]),
                n_trades=f["n_trades"],
            )
            for f in cached["folds"]
        ]
        wf = WalkForwardBacktestResult(
            folds=folds,
            mean_sharpe=cached["mean_sharpe"],
            mean_win_rate=cached["mean_win_rate"],
            mean_precision_buy=cached["mean_precision_buy"],
            worst_drawdown=cached["worst_drawdown"],
        )
        return wf, _grade_from_dict(cached["grade"])

    strategy = load_strategy(strategy_name)
    df = load_training_data(parquet_dir)
    result = walk_forward_backtest_strategy(
        df, strategy, ohlcv_cols, feature_cols,
        train_window_days=400, test_window_days=21, step_days=21,
    )
    avg_metrics = BacktestMetrics(
        n_trades=sum(f.n_trades for f in result.folds),
        win_rate=result.mean_win_rate,
        profit_factor=0.0,
        total_return_pct=0.0,
        sharpe_ratio=result.mean_sharpe,
        max_drawdown_pct=result.worst_drawdown,
        precision_buy=result.weighted_precision_buy,
        recall_buy=0.0,
        f1_buy=0.0,
        accuracy=0.0,
    )
    grade = grade_model(strategy_name, avg_metrics)
    return result, grade


def get_live_signals(
    parquet_dir: Path,
    ohlcv_cols: list[str],
    feature_cols: list[str],
    confidence_threshold: float = 0.75,
    market: str = "us",
) -> list[LiveSignal]:
    cached = _load_cache(get_cache_dir(market) / "signals.json")
    if cached:
        # Cache stores all signals at threshold=0; filter here so the slider works.
        return [
            LiveSignal(
                ticker=s["ticker"],
                date=s["date"],
                signal=Signal(s["signal"]),
                confidence=s["confidence"],
                entry_price=s["entry_price"],
                position_size=s["position_size"],
            )
            for s in cached["signals"]
            if s["signal"] == "Buy" and s["confidence"] >= confidence_threshold
        ]

    names = list_strategies()
    if not names:
        return []

    strategy = load_strategy(names[0])
    df = load_training_data(parquet_dir)
    df_pd = df.to_pandas()

    train_cutoff = max(0, len(df_pd) - 21)
    strategy.fit(df_pd.iloc[:train_cutoff])
    pred = strategy.predict(df_pd)

    df_with_pred = df.with_columns([
        pl.Series("_conf", pred.confidence.tolist()),
        pl.Series("_sig",  pred.signal.tolist()),
    ])
    latest = df_with_pred.sort("time").group_by("ticker", maintain_order=True).last()

    live_signals: list[LiveSignal] = []
    for row in latest.iter_rows(named=True):
        conf = float(row["_conf"])
        sig  = str(row["_sig"])
        if sig == "Buy" and conf >= confidence_threshold:
            live_signals.append(LiveSignal(
                ticker=str(row["ticker"]),
                date=str(row["time"]),
                signal=Signal.BUY,
                confidence=conf,
                entry_price=float(row["close"]),
                position_size=conf,
            ))
    return live_signals


def get_combined_ratings(market: str = "us") -> tuple[list[dict], dict[str, list[dict]]]:
    """Aggregate every strategy's live signal for each ticker into one
    overall Buy Rating, weighted by that strategy's leaderboard composite
    score — stronger track-record strategies count for more. Cache-only
    (no live-compute fallback): both signals.json and leaderboard.json must
    already exist.

    Returns (summary_rows, detail_by_ticker):
      summary_rows: one row per ticker — ticker, overall_rating (0-100),
        n_buy, n_strategies, date, entry_price — sorted by overall_rating
        descending.
      detail_by_ticker: ticker -> per-strategy contribution rows (each with
        its own date/entry_price too), sorted by contribution descending,
        for the drill-down view.
    """
    leaderboard = _load_cache(get_cache_dir(market) / "leaderboard.json")
    signals = _load_cache(get_cache_dir(market) / "signals.json")
    if not leaderboard or not signals:
        return [], {}

    weights = {g["model_name"]: g["composite_score"] for g in leaderboard["grades"]}

    by_ticker: dict[str, list[dict]] = {}
    for s in signals["signals"]:
        by_ticker.setdefault(s["ticker"], []).append(s)

    summary_rows: list[dict] = []
    detail_by_ticker: dict[str, list[dict]] = {}
    for ticker, rows in by_ticker.items():
        total_weight = 0.0
        weighted_buy = 0.0
        n_buy = 0
        detail: list[dict] = []
        latest_date = max((r["date"] for r in rows), default="")
        entry_price = next((r["entry_price"] for r in rows if r["date"] == latest_date), 0.0)
        for r in rows:
            w = weights.get(r["strategy"], 0.0)
            if w <= 0:
                continue
            is_buy = r["signal"] == "Buy"
            contribution = w * r["confidence"] if is_buy else 0.0
            total_weight += w
            weighted_buy += contribution
            n_buy += int(is_buy)
            detail.append({
                "strategy": r["strategy"],
                "weight": w,
                "signal": r["signal"],
                "confidence": r["confidence"],
                "contribution": contribution,
                "date": r["date"],
                "entry_price": r["entry_price"],
                "trade_status": r.get("trade_status"),
                "trade_open_date": r.get("trade_open_date"),
                "trade_close_date": r.get("trade_close_date"),
                "overlay_reason": r.get("overlay_reason") or "",
            })
        if total_weight <= 0:
            continue
        detail.sort(key=lambda d: d["contribution"], reverse=True)
        detail_by_ticker[ticker] = detail
        summary_rows.append({
            "ticker": ticker,
            "overall_rating": 100.0 * weighted_buy / total_weight,
            "n_buy": n_buy,
            "n_strategies": len(detail),
            "date": latest_date,
            "entry_price": entry_price,
        })

    summary_rows.sort(key=lambda r: r["overall_rating"], reverse=True)
    return summary_rows, detail_by_ticker


# ---------------------------------------------------------------------------
# League (live / bench / retired)
# ---------------------------------------------------------------------------

STALE_DAYS = {"live": 8, "bench": 15}
_TREND_EPS = 0.05


def _trend(history: list[dict]) -> str:
    ok = [h["net_sharpe"] for h in history if not h.get("error") and h.get("net_sharpe") is not None]
    if len(ok) < 2:
        return ""
    d = ok[-1] - ok[-2]
    return "↑" if d > _TREND_EPS else ("↓" if d < -_TREND_EPS else "→")


def _is_stale(section: str, tested_on: str, today) -> bool:
    if section not in STALE_DAYS:
        return False
    if not tested_on:
        return True
    from datetime import date as _date
    return (today - _date.fromisoformat(tested_on)).days > STALE_DAYS[section]


def get_league_table(market: str, today=None) -> dict[str, list[dict]]:
    """Rows for the Live / Bench / Retired tables, each sorted by net Sharpe."""
    from datetime import date as _date
    from src.league.league import STATE_FILE, latest_ok

    today = today or _date.today()
    cache_dir = get_cache_dir(market)
    out: dict[str, list[dict]] = {"live": [], "bench": [], "retired": []}
    state = _load_cache(cache_dir / STATE_FILE)
    if state:
        for name, m in state["methods"].items():
            h = latest_ok(m) or {}
            tested = m["history"][-1]["date"] if m["history"] else ""
            sec = m["status"] if m["status"] in out else "bench"
            if sec == "live":  # live methods are also re-tested by the weekly full backtest
                bt = _load_cache(cache_dir / f"backtest_{_safe(name)}.json") or {}
                tested = max(tested, str(bt.get("generated_at", ""))[:10])
            out[sec].append({
                "method": name,
                "net_sharpe": h.get("net_sharpe"),
                "precision": h.get("precision"),
                "n_trades": h.get("n_trades"),
                "tested_on": tested,
                "trend": _trend(m["history"]),
                "duplicate_of": m.get("duplicate_of"),
                "stale": _is_stale(sec, tested, today),
                "retired_because": m.get("retired_because"),
                "error": m["history"][-1].get("error") if m["history"] else None,
            })
    else:
        for name in list_strategies(market):
            bt = _load_cache(cache_dir / f"backtest_{_safe(name)}.json") or {}
            tested = str(bt.get("generated_at", ""))[:10]
            out["bench"].append({
                "method": name,
                "net_sharpe": bt.get("mean_sharpe"),
                "precision": bt.get("weighted_precision_buy"),
                "n_trades": ((bt.get("grade") or {}).get("metrics") or {}).get("n_trades"),
                "tested_on": tested,
                "trend": "",
                "duplicate_of": None,
                "stale": _is_stale("bench", tested, today),
                "retired_because": None,
                "error": None,
            })
    for rows in out.values():
        rows.sort(key=lambda r: (r["net_sharpe"] is None, -(r["net_sharpe"] or 0.0)))
    return out


def get_ticker_names(market: str) -> dict[str, str]:
    """China only: ticker -> company name (synced from Server B weekly)."""
    if market != "china":
        return {}
    path = get_market(market).data_root / "index_compositions" / "names.parquet"
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    return dict(zip(df["ticker"].to_list(), df["name"].to_list()))


def format_generated(iso: str, market: str) -> str:
    """Cache timestamp in US Eastern; China pages add Beijing time."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        ts = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return str(iso)
    et = ts.astimezone(ZoneInfo("America/New_York")).strftime("%m-%d %H:%M %Z")
    if market != "china":
        return et
    return f"{et} ({ts.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M')} Beijing)"
