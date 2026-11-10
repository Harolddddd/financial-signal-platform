"""Hand part of a market's bench pass to the compute server (B) and take its results back.

Trust model: B only ever *reads* a request + inputs that A publishes (read-only rrsync
key on /home/fspsync/outgoing) and *writes* results into A's inbox (write-only rrsync
key on /home/fspsync/incoming). A never logs in to B. Nothing from B is executed;
A accepts a result only if its manifest hash matches, the method was assigned to B,
and the prediction file has the exact expected schema, and A re-scores the
predictions itself, so B's own numbers are never trusted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REQUEST_FILE = "request.json"
MANIFEST_FILE = "manifest.json"
PRED_SCHEMA = {"ticker": pl.Utf8, "time": pl.Date, "buy_score": pl.Float64}
MAX_FILE_BYTES = 200 * 2**20
_REQUEST_KEYS = {"id", "market", "today", "deadline", "methods", "created"}


def _safe(name: str) -> str:
    return name.replace("/", "_")


def result_files(name: str) -> tuple[str, str]:
    return f"bench_{_safe(name)}.json", f"preds_{_safe(name)}.parquet"


def split_with_times(
    names: list[str],
    runtimes: dict[str, float],
    local_head_start_s: float = 0.0,
    remote_speed: float = 1.0,
    exclude: tuple[str, ...] | list[str] = (),
    max_share: float = 0.8,
    default_s: float = 300.0,
    remote_head_start_s: float = 0.0,
) -> tuple[list[str], list[str], float, float]:
    """Longest-first greedy split into (local, remote) that balances finish times.

    A is busy for `local_head_start_s` first (the other market), B for
    `remote_head_start_s` (downloads + its share of the other market); B runs at
    `remote_speed` x A's pace. Methods in `exclude` always stay local, and B never
    gets more than `max_share` of this market's estimated work. Also returns the
    estimated finish times (seconds from the start) of A and B.
    """
    known = [runtimes[n] for n in names if runtimes.get(n, 0) > 0]
    guess = sorted(known)[len(known) // 2] if known else default_s
    cost = {n: runtimes.get(n) or guess for n in names}
    total = sum(cost.values()) or 1.0
    local = [n for n in names if any(n.startswith(p) for p in exclude)]
    remote: list[str] = []
    t_local = local_head_start_s + sum(cost[n] for n in local)
    t_remote, remote_work = remote_head_start_s, 0.0
    for n in sorted((n for n in names if n not in local), key=lambda n: -cost[n]):
        c = cost[n]
        if t_remote + c / remote_speed <= t_local + c and remote_work + c <= max_share * total:
            remote.append(n); t_remote += c / remote_speed; remote_work += c
        else:
            local.append(n); t_local += c
    order = {n: i for i, n in enumerate(names)}
    return sorted(local, key=order.get), sorted(remote, key=order.get), t_local, t_remote


def split_methods(names: list[str], runtimes: dict[str, float], **kw) -> tuple[list[str], list[str]]:
    """(local, remote) from split_with_times."""
    local, remote, _, _ = split_with_times(names, runtimes, **kw)
    return local, remote


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_manifest(out_dir: Path, request: dict, done: list[str], complete: bool,
                   skipped: list[str] | None = None, now: datetime | None = None) -> dict:
    """B side: list every finished method with the hash and size of both its files."""
    entries = {}
    for name in done:
        files = {}
        for fn in result_files(name):
            p = out_dir / fn
            if not p.exists():
                break
            files[fn] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        else:
            entries[name] = files
    manifest = {
        "request_id": request["id"], "market": request["market"], "complete": complete,
        "updated": (now or datetime.now(timezone.utc)).isoformat(),
        "methods": entries, "skipped": skipped or [],
    }
    _atomic_write(out_dir / MANIFEST_FILE, json.dumps(manifest, indent=2))
    return manifest


def validate_request(req: dict) -> dict:
    missing = _REQUEST_KEYS - set(req)
    if missing:
        raise ValueError(f"request missing {sorted(missing)}")
    if not isinstance(req["methods"], list) or not all(isinstance(m, str) for m in req["methods"]):
        raise ValueError("request methods must be a list of names")
    return req


def load_request(path: Path) -> dict | None:
    try:
        return validate_request(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def publish_request(outgoing: Path, request: dict, feature_dir: Path, code_tar: Path | None = None) -> Path:
    """A side: snapshot the features (hard links, no copy), the code, then the request.

    The request file is written last, so B never sees a request whose inputs are
    still being written.
    """
    validate_request(request)
    outgoing.mkdir(parents=True, exist_ok=True)
    (outgoing / REQUEST_FILE).unlink(missing_ok=True)
    snap = outgoing / "features"
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir()
    for f in sorted(feature_dir.glob("*.parquet")):
        try:
            os.link(f, snap / f.name)
        except OSError:  # another filesystem: fall back to a copy
            shutil.copy2(f, snap / f.name)
    if code_tar is not None:
        shutil.copy2(code_tar, outgoing / "code.tar.gz")
    _atomic_write(outgoing / REQUEST_FILE, json.dumps(request, indent=2))
    return outgoing / REQUEST_FILE


def retract_request(outgoing: Path) -> None:
    """A side: the pass is over; B must not start (or restart) on this request."""
    (outgoing / REQUEST_FILE).unlink(missing_ok=True)
    shutil.rmtree(outgoing / "features", ignore_errors=True)


def read_manifest(inbox: Path, request: dict) -> dict | None:
    try:
        m = json.loads((inbox / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict) or m.get("request_id") != request["id"] or not isinstance(m.get("methods"), dict):
        return None
    return m


def remote_alive(manifest: dict | None, now: datetime, stale_s: float) -> bool:
    if manifest is None:
        return False
    try:
        updated = datetime.fromisoformat(manifest["updated"])
    except (KeyError, TypeError, ValueError):
        return False
    return (now - updated).total_seconds() <= stale_s


def _finite(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def import_results(inbox: Path, request: dict, manifest: dict, skip: set[str]) -> tuple[dict, list[str]]:
    """Verified results from B: {name: {"preds": DataFrame, "meta": dict}}, plus rejections.

    `meta` carries only B's runtime and fold metrics (finite numbers or None);
    the caller re-scores `preds` itself.
    """
    assigned = set(request["methods"])
    out, rejected = {}, []
    for name, files in manifest["methods"].items():
        if name in skip:
            continue
        why = None
        if name not in assigned:
            why = "not assigned to B"
        elif not isinstance(files, dict) or set(files) != set(result_files(name)):
            why = "wrong file list"
        else:
            for fn, info in files.items():
                p = inbox / fn
                if not p.is_file() or p.is_symlink():
                    why = f"{fn} missing"; break
                size = p.stat().st_size
                if size > MAX_FILE_BYTES or size != info.get("bytes"):
                    why = f"{fn} size mismatch"; break
                if sha256_file(p) != info.get("sha256"):
                    why = f"{fn} hash mismatch"; break
        if why is None:
            jf, pf = result_files(name)
            try:
                raw = json.loads((inbox / jf).read_text(encoding="utf-8"))
                preds = pl.read_parquet(inbox / pf)
            except Exception as exc:  # noqa: BLE001 - any unreadable file is a rejection
                why = f"unreadable: {str(exc)[:80]}"
            else:
                if dict(preds.schema) != PRED_SCHEMA:
                    why = "prediction schema mismatch"
                elif not isinstance(raw, dict) or raw.get("strategy_name") != name:
                    why = "result names another method"
        if why is not None:
            rejected.append(f"{name}: {why}")
            continue
        meta = {k: _finite(raw.get(k)) for k in ("runtime_s", "mean_sharpe", "weighted_precision_buy", "worst_drawdown")}
        n_folds = raw.get("n_folds")
        meta["n_folds"] = n_folds if isinstance(n_folds, int) and not isinstance(n_folds, bool) else None
        out[name] = {"preds": preds, "meta": meta}
    return out, rejected
