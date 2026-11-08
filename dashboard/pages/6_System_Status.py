import psutil
import streamlit as st

from dashboard import system_status as ss

st.set_page_config(page_title="System Status", layout="wide")
st.header("System Status")
st.caption("All times are US Eastern (New York / North Carolina).")

if not ss.available():
    st.info("System status is only available on the Linux server.")
    st.stop()

if st.button("Refresh"):
    st.rerun()

# --- Server A resources ---
mem = psutil.virtual_memory()
disk = psutil.disk_usage("/")
c1, c2, c3 = st.columns(3)
c1.metric("CPU", f"{psutil.cpu_percent(interval=0.5):.0f}%")
c2.metric("Memory", f"{mem.used / 2**30:.1f} / {mem.total / 2**30:.1f} GB")
c3.metric("Disk", f"{disk.used / 2**30:.0f} / {disk.total / 2**30:.0f} GB")

# --- Services ---
st.subheader("Server A jobs")
rows = []
for unit, label in ss.SERVICES.items():
    info = ss.unit_info(unit)
    rows.append({
        "job": label,
        "unit": unit,
        "state": f"{info.get('ActiveState', '?')} ({info.get('SubState', '?')})",
        "last result": info.get("Result", "-"),
        "last start": info.get("ExecMainStartTimestamp") or "-",
        "last end": info.get("ExecMainExitTimestamp") or "-",
        "memory now": ss.human_bytes(info.get("MemoryCurrent", "")),
        "memory peak": ss.human_bytes(info.get("MemoryPeak", "")),
    })
st.dataframe(rows, hide_index=True, use_container_width=True)

st.subheader("Schedule")
trows = []
for timer in ss.TIMERS:
    t = ss.timer_info(timer)
    trows.append({
        "timer": timer,
        "enabled": t.get("ActiveState", "?"),
        "next run": t.get("NextElapseUSecRealtime") or "-",
        "last run": t.get("LastTriggerUSec") or "-",
    })
st.dataframe(trows, hide_index=True, use_container_width=True)

# --- Compute server (B) backfill ---
st.subheader("Server B: news backfill")
bf = ss.backfill_status()
if bf is None:
    st.caption("No status received from Server B yet.")
else:
    st.progress(bf.get("pct", 0.0), text=(
        f"{bf.get('state', '?')} - {bf.get('done', '?')}/{bf.get('total', '?')} tickers, "
        f"{bf.get('new_total', '?')} new articles"
    ))
    st.caption(
        f"{bf.get('days', '?')}-day backfill, started {ss.to_et(bf.get('started', '?'))}, "
        f"last update {ss.to_et(bf.get('updated', '?'))} (every 5 min), B memory {bf.get('mem', '?')}"
    )

# --- Last run summary ---
st.subheader("Last run")
unit = st.selectbox(
    "Job", [u for u in ss.SERVICES if u != "fsp-dashboard"], format_func=lambda u: ss.SERVICES[u]
)
info = ss.unit_info(unit)
summary = ss.summarize_run(ss.last_run_lines(unit), running=info.get("ActiveState") == "activating")
if not summary["steps"]:
    started = info.get("ExecMainStartTimestamp") or ""
    st.caption(f"Last run: {started}, result {info.get('Result', '-')}" if started else "No run recorded yet.")
else:
    st.dataframe(summary["steps"], hide_index=True, use_container_width=True)
    if summary["errors"]:
        st.warning(f"{summary['errors']} error lines, {summary['warnings']} warnings in this run "
                   "(delisted tickers show up here too).")
    else:
        st.caption(f"No errors, {summary['warnings']} warnings.")
with st.expander("Raw log (last 30 lines)"):
    st.code(ss.recent_log(unit, 30) or "(no log lines)", language="text")
