"""Live / bench / retired state of every method in one market.

Pure state transitions (`update_league`) plus a small JSON store at
`<cache_dir>/league.json`. Rules (config/league.yaml):
- live holds at most `live_max` methods;
- a bench method replaces the weakest live one (or fills a free slot) only after
  winning `promote_after_passes` consecutive bench passes;
- methods whose predictions are near-identical are grouped; only the best of a
  group may be live, the rest are marked duplicate_of=<best>;
- a duplicate that stays similar to the same best method for
  `retire_after_passes` passes spanning >= `retire_min_days` days is retired.
Nothing is ever deleted; retirement is a flag.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import polars as pl
import yaml

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "league.yaml"
STATE_FILE = "league.json"


def load_config(path: Path = _CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- state

def _blank_method(status: str) -> dict:
    return {
        "status": status, "history": [], "beats_weakest_streak": 0,
        "duplicate_of": None, "similar_since": None, "similar_passes": 0,
        "retired_on": None, "retired_because": None,
    }


def new_state(market: str, seed: list[str]) -> dict:
    return {"market": market, "passes": [], "methods": {n: _blank_method("live") for n in seed}}


def ensure_method(state: dict, name: str) -> dict:
    return state["methods"].setdefault(name, _blank_method("bench"))


def load_state(cache_dir: Path, market: str, seed: list[str]) -> dict:
    path = Path(cache_dir) / STATE_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return new_state(market, seed)


def save_state(cache_dir: Path, state: dict) -> None:
    path = Path(cache_dir) / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def live_set(cache_dir: Path, market: str) -> list[str] | None:
    path = Path(cache_dir) / STATE_FILE
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    return [n for n, m in state["methods"].items() if m["status"] == "live"]


def last_completed_pass(state: dict) -> date | None:
    done = [p["date"] for p in state["passes"] if p.get("completed")]
    return date.fromisoformat(max(done)) if done else None


def latest_ok(m: dict) -> dict | None:
    """Most recent history entry without an error."""
    for h in reversed(m["history"]):
        if not h.get("error"):
            return h
    return None


# --------------------------------------------------------------------- similarity

def similarity(a: pl.DataFrame, b: pl.DataFrame) -> tuple[float, float]:
    """(mean daily Spearman of buy_score across tickers, Jaccard of buy picks)."""
    j = a.select("ticker", "time", pl.col("buy_score").alias("_a")).join(
        b.select("ticker", "time", pl.col("buy_score").alias("_b")), on=["ticker", "time"], how="inner"
    )
    corr = 0.0
    if j.height:
        per_day = j.group_by("time").agg(pl.corr("_a", "_b", method="spearman").alias("c"))["c"]
        per_day = per_day.filter(per_day.is_finite())
        corr = float(per_day.mean()) if len(per_day) else 0.0
    buys_a = set(a.filter(pl.col("buy_score") > 0).select("ticker", "time").iter_rows())
    buys_b = set(b.filter(pl.col("buy_score") > 0).select("ticker", "time").iter_rows())
    union = buys_a | buys_b
    jac = len(buys_a & buys_b) / len(union) if union else 0.0
    return corr, jac


def _groups(names: list[str], preds: dict[str, pl.DataFrame], cfg: dict) -> list[list[str]]:
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, x in enumerate(names):
        for y in names[i + 1:]:
            corr, jac = similarity(preds[x], preds[y])
            if corr >= cfg["similar_rank_corr"] or jac >= cfg["similar_jaccard"]:
                parent[find(x)] = find(y)
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(find(n), []).append(n)
    return [g for g in out.values() if len(g) > 1]


# ------------------------------------------------------------------------- update

def _score(m: dict) -> float:
    h = latest_ok(m)
    return h["net_sharpe"] if h else float("-inf")


def update_league(
    state: dict,
    results: dict[str, dict],
    preds: dict[str, pl.DataFrame],
    today: date,
    cfg: dict,
    completed: bool = True,
) -> dict:
    """Apply one bench pass. `results[name]` is a score dict or {"error": str}."""
    methods = state["methods"]
    iso = today.isoformat()

    # 1. history
    tested_ok: list[str] = []
    for name, r in results.items():
        m = ensure_method(state, name)
        if m["status"] == "retired":
            continue
        err = r.get("error")
        m["history"].append({
            "date": iso,
            "net_sharpe": None if err else r["net_sharpe"],
            "n_trades": None if err else r["n_trades"],
            "precision": None if err else r["precision"],
            "error": err,
        })
        if not err and name in preds:
            tested_ok.append(name)

    # 2. similarity groups among methods tested OK this pass
    dup_of: dict[str, str] = {}
    for group in _groups(sorted(tested_ok), preds, cfg):
        winner = max(group, key=lambda n: _score(methods[n]))
        for n in group:
            if n != winner:
                dup_of[n] = winner
    for n in tested_ok:
        m = methods[n]
        w = dup_of.get(n)
        if w is None:
            m.update(duplicate_of=None, similar_since=None, similar_passes=0)
        elif m["duplicate_of"] == w:
            m["similar_passes"] += 1
        else:
            m.update(duplicate_of=w, similar_since=iso, similar_passes=1)

    # 3. retirement
    for n in tested_ok:
        m = methods[n]
        if m["duplicate_of"] and m["similar_passes"] >= cfg["retire_after_passes"]:
            if (today - date.fromisoformat(m["similar_since"])).days >= cfg["retire_min_days"]:
                m.update(status="retired", retired_on=iso,
                         retired_because=f"similar to {m['duplicate_of']} since {m['similar_since']}",
                         beats_weakest_streak=0)

    # 4. live duplicates step down; then promotion
    for n, m in methods.items():
        if m["status"] == "live" and m["duplicate_of"]:
            m["status"] = "bench"

    def eligible(n: str) -> bool:
        m = methods[n]
        h = m["history"][-1] if m["history"] else None
        return (
            m["status"] == "bench" and not m["duplicate_of"] and n in tested_ok
            and h is not None and h["date"] == iso and not h["error"]
            and h["n_trades"] >= cfg["min_trades"]
        )

    def live() -> list[str]:
        return [n for n, m in methods.items() if m["status"] == "live"]

    def weakest() -> str | None:
        lv = live()
        return min(lv, key=lambda n: _score(methods[n])) if lv else None

    w = weakest()
    open_slot = len(live()) < cfg["live_max"]
    for n, m in methods.items():
        if m["status"] != "bench":
            continue
        if eligible(n) and (open_slot or w is None or _score(m) > _score(methods[w])):
            m["beats_weakest_streak"] += 1
        else:
            m["beats_weakest_streak"] = 0

    ready = sorted(
        (n for n in methods if eligible(n) and methods[n]["beats_weakest_streak"] >= cfg["promote_after_passes"]),
        key=lambda n: _score(methods[n]), reverse=True,
    )
    for n in ready:
        if len(live()) < cfg["live_max"]:
            methods[n].update(status="live", beats_weakest_streak=0)
            continue
        w = weakest()
        if w is not None and _score(methods[n]) > _score(methods[w]):
            methods[w].update(status="bench", beats_weakest_streak=0)
            methods[n].update(status="live", beats_weakest_streak=0)

    state["passes"].append({"date": iso, "completed": completed})
    return state


def seed_live_from_latest(state: dict, cfg: dict, n: int) -> dict:
    """One-time: live = top `n` non-duplicate methods by latest score with enough trades."""
    cands = [
        name for name, m in state["methods"].items()
        if m["status"] != "retired" and not m["duplicate_of"]
        and (h := latest_ok(m)) is not None and h["n_trades"] >= cfg["min_trades"]
    ]
    top = set(sorted(cands, key=lambda x: _score(state["methods"][x]), reverse=True)[:n])
    for name, m in state["methods"].items():
        if m["status"] != "retired":
            m["status"] = "live" if name in top else "bench"
    return state
