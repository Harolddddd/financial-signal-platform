"""Static checks on the server shell scripts (they only run on Linux)."""
import re
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"


def _scripts():
    return sorted(DEPLOY.glob("*.sh"))


def test_no_literal_backslash_n_in_commands():
    for p in _scripts():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            assert not re.search(r"\s\\n\s", code), f"{p.name}:{i} literal backslash-n"


def test_no_bare_test_and_command_as_loop_last_line():
    # `[[ cond ]] && cmd` as the last command of a loop body returns 1 when false,
    # which kills a `set -euo pipefail` script through the pipeline.
    txt = (DEPLOY / "backup.sh").read_text(encoding="utf-8")
    assert not re.search(r"\]\]\s*&&\s*coscli", txt)


def test_bench_runs_china_even_if_us_fails():
    txt = (DEPLOY / "run_pipeline.sh").read_text(encoding="utf-8")
    block = txt.split("  bench)", 1)[1].split(";;", 1)[0]
    assert "--market us" in block and "|| " in block.split("--market us", 1)[1].split("\n", 1)[0]


def test_bench_deadline_never_more_than_15h_away():
    txt = (DEPLOY / "run_pipeline.sh").read_text(encoding="utf-8")
    assert "15 hours" in txt
