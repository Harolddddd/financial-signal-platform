from datetime import date, timedelta

import polars as pl
import pytest

from src.league import league as lg

CFG = lg.load_config()
D0 = date(2026, 11, 15)


def _preds(scores_by_day):
    """scores_by_day: list of per-day lists of buy_score for tickers T0..Tn."""
    rows = []
    for di, scores in enumerate(scores_by_day):
        for ti, s in enumerate(scores):
            rows.append((f"T{ti}", D0 + timedelta(days=di), float(s)))
    return pl.DataFrame(rows, schema=["ticker", "time", "buy_score"], orient="row")


# Distinct, non-similar prediction patterns per method.
_rng = __import__("numpy").random.default_rng(0)


def _rand_preds():
    return _preds((_rng.random((30, 8)) * (_rng.random((30, 8)) > 0.6)).tolist())


def _res(sharpe, trades=50):
    return {"net_sharpe": sharpe, "n_trades": trades, "precision": 0.5, "mean_net_return": 0.001}


def _state(live, bench):
    st = lg.new_state("us", seed=live)
    for b in bench:
        lg.ensure_method(st, b)
    return st


def _pass(st, scores, day, preds=None, **kw):
    preds = preds or {n: _rand_preds() for n in scores}
    results = {n: (s if isinstance(s, dict) else _res(s)) for n, s in scores.items()}
    return lg.update_league(st, results, preds, day, CFG, **kw)


def _status(st):
    return {n: m["status"] for n, m in st["methods"].items()}


def test_config_values():
    assert (CFG["live_max"], CFG["min_trades"], CFG["promote_after_passes"]) == (6, 20, 2)
    assert (CFG["similar_rank_corr"], CFG["similar_jaccard"]) == (0.85, 0.70)
    assert (CFG["retire_after_passes"], CFG["retire_min_days"], CFG["bench_window_days"]) == (3, 28, 1095)


def test_first_pass_no_promotion():
    st = _pass(_state(["a"] * 0 + ["a", "b", "c", "d", "e", "f"], ["x"]),
               {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5, "f": 0.6, "x": 2.0}, D0)
    assert _status(st)["x"] == "bench"
    assert st["passes"] == [{"date": D0.isoformat(), "completed": True}]


def test_promotion_needs_two_consecutive_wins():
    st = _state(["a", "b", "c", "d", "e", "f"], ["x"])
    scores = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5, "f": 0.6, "x": 2.0}
    st = _pass(st, scores, D0)
    st = _pass(st, scores, D0 + timedelta(days=14))
    s = _status(st)
    assert s["x"] == "live" and s["a"] == "bench"
    assert sum(v == "live" for v in s.values()) == 6


def test_streak_resets_on_loss():
    st = _state(["a", "b", "c", "d", "e", "f"], ["x"])
    base = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5, "f": 0.6}
    st = _pass(st, {**base, "x": 2.0}, D0)
    st = _pass(st, {**base, "x": 0.0}, D0 + timedelta(days=14))
    st = _pass(st, {**base, "x": 2.0}, D0 + timedelta(days=28))
    assert _status(st)["x"] == "bench"


def test_fills_empty_live_slot_after_two_passes():
    st = _state(["a"], ["x"])
    st = _pass(st, {"a": 0.5, "x": 0.1}, D0)
    st = _pass(st, {"a": 0.5, "x": 0.1}, D0 + timedelta(days=14))
    assert _status(st) == {"a": "live", "x": "live"}


def test_min_trades_blocks_live():
    st = _state(["a"], ["x"])
    st = _pass(st, {"a": 0.1, "x": _res(3.0, trades=19)}, D0)
    st = _pass(st, {"a": 0.1, "x": _res(3.0, trades=19)}, D0 + timedelta(days=14))
    assert _status(st)["x"] == "bench"


def test_error_method_keeps_status_and_never_promotes():
    st = _state(["a"], ["x"])
    st = _pass(st, {"a": 0.1, "x": 3.0}, D0)
    st = _pass(st, {"a": 0.1, "x": {"error": "boom"}}, D0 + timedelta(days=14),
               preds={"a": _rand_preds()})
    assert _status(st)["x"] == "bench"
    assert st["methods"]["x"]["history"][-1]["error"] == "boom"


def test_duplicate_keeps_best_and_marks_other():
    p = _rand_preds()
    st = _state(["a", "b"], [])
    st = _pass(st, {"a": 0.9, "b": 0.3}, D0, preds={"a": p, "b": p})
    assert st["methods"]["b"]["duplicate_of"] == "a"
    assert st["methods"]["a"]["duplicate_of"] is None
    assert _status(st)["b"] == "bench"  # live duplicates are demoted


def test_retire_after_three_passes_and_28_days():
    p = _rand_preds()
    st = _state(["a"], ["b"])
    for k in range(2):
        st = _pass(st, {"a": 0.9, "b": 0.3}, D0 + timedelta(days=14 * k), preds={"a": p, "b": p})
    assert _status(st)["b"] == "bench"
    st = _pass(st, {"a": 0.9, "b": 0.3}, D0 + timedelta(days=28), preds={"a": p, "b": p})
    m = st["methods"]["b"]
    assert m["status"] == "retired" and m["retired_on"] == (D0 + timedelta(days=28)).isoformat()
    assert "similar to a" in m["retired_because"]


def test_similarity_reset_when_not_similar():
    p = _rand_preds()
    st = _state(["a"], ["b"])
    st = _pass(st, {"a": 0.9, "b": 0.3}, D0, preds={"a": p, "b": p})
    st = _pass(st, {"a": 0.9, "b": 0.3}, D0 + timedelta(days=14))
    assert st["methods"]["b"]["duplicate_of"] is None and st["methods"]["b"]["similar_passes"] == 0


def test_similarity_identical_preds_is_1_1():
    p = _rand_preds()
    corr, jac = lg.similarity(p, p)
    assert corr == pytest.approx(1.0) and jac == pytest.approx(1.0)


def test_similarity_disjoint_buys_jaccard_0():
    a = _preds([[1, 0], [1, 0]])
    b = _preds([[0, 1], [0, 1]])
    assert lg.similarity(a, b)[1] == 0.0


def test_state_round_trip_and_live_set(tmp_path):
    assert lg.live_set(tmp_path, "us") is None
    st = lg.load_state(tmp_path, "us", seed=["a", "b"])
    lg.save_state(tmp_path, st)
    assert lg.live_set(tmp_path, "us") == ["a", "b"]
    assert lg.last_completed_pass(st) is None


def test_seed_from_latest_picks_top_eligible():
    st = _state([], ["a", "b", "c"])
    st = _pass(st, {"a": 0.1, "b": 0.9, "c": _res(5.0, trades=3)}, D0)
    lg.seed_live_from_latest(st, CFG, n=1)
    assert _status(st) == {"a": "bench", "b": "live", "c": "bench"}


def test_live_stays_when_duplicate_winner_is_ineligible():
    p = _rand_preds()
    st = _pass(_state(["a"], ["x"]), {"a": 0.3, "x": _res(0.9, trades=5)}, D0, preds={"a": p, "x": p})
    assert _status(st)["a"] == "live"
    assert st["methods"]["x"]["duplicate_of"] == "a"


def test_live_duplicate_waits_for_bench_winner_promotion():
    p = _rand_preds()
    st = _pass(_state(["a"], ["x"]), {"a": 0.3, "x": 0.9}, D0, preds={"a": p, "x": p})
    assert _status(st) == {"a": "live", "x": "bench"}            # x not promoted yet: a keeps its seat
    st = _pass(st, {"a": 0.3, "x": 0.9}, D0 + timedelta(days=14), preds={"a": p, "x": p})
    assert _status(st) == {"a": "bench", "x": "live"}            # x promoted -> a steps down


def test_open_slot_win_only_for_top_ranked():
    cfg = {**CFG, "live_max": 2}
    st = _state(["a"], ["x", "y"])
    st = lg.update_league(st, {"a": _res(0.5), "x": _res(0.9), "y": _res(0.1)},
                          {n: _rand_preds() for n in "axy"}, D0, cfg)
    st = lg.update_league(st, {"a": _res(0.5), "x": _res(0.9), "y": _res(0.8)},
                          {n: _rand_preds() for n in "axy"}, D0 + timedelta(days=14), cfg)
    assert _status(st) == {"a": "live", "x": "live", "y": "bench"}


def test_untested_method_keeps_streak():
    st = _state(["a", "b", "c", "d", "e", "f"], ["x"])
    base = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5, "f": 0.6}
    st = _pass(st, {**base, "x": 2.0}, D0)
    st = _pass(st, base, D0 + timedelta(days=14))                 # deadline: x not reached
    assert st["methods"]["x"]["beats_weakest_streak"] == 1
    st = _pass(st, {**base, "x": 2.0}, D0 + timedelta(days=28))
    assert _status(st)["x"] == "live"


def test_bench_only_methods_never_promoted():
    assert "stacked_live" in CFG["bench_only_prefixes"]
    st = _state(["a"], ["stacked_live_cn"])
    for k in range(3):
        st = _pass(st, {"a": 0.1, "stacked_live_cn": 3.0}, D0 + timedelta(days=14 * k))
    assert _status(st)["stacked_live_cn"] == "bench"
