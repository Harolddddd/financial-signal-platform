import streamlit as st

from dashboard import system_status as ss

st.set_page_config(page_title="System Status", layout="wide")
st.header("System Status")
st.caption("All times are US Eastern (New York / North Carolina).")

if not ss.available():
    st.info("System status is only available on the Linux server.")
    st.stop()

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data(ttl=30)
def _server_a() -> dict:
    return ss.collect("A", ss.SERVICES, ss.TIMERS)


SERVERS = [
    ("A", "Server A · production", "2 cores · 8 GB. Runs the daily/weekly pipelines, bench re-tests, "
                                   "paper accounts and this dashboard.", _server_a()),
    ("B", "Server B · compute (Guangzhou)", "4 cores · 3.7 GB. Collects China news + macro and runs its "
                                            "share of the bench re-test. Reports every 5 min.",
     ss.read_snapshot()),
]


def _gb(n) -> str:
    return f"{n / 2**30:.1f}" if isinstance(n, (int, float)) else "?"


def _overview(name: str, role: str, snap: dict | None, host: str) -> None:
    st.subheader(name)
    st.caption(role)
    if snap is None:
        st.error("No status received yet.")
        return
    if host == "B" and ss.is_stale(snap):
        st.error(f"Not reporting since {ss.to_et(snap.get('updated', '?'))}: B may be down or offline.")
    else:
        st.success(f"Online · updated {ss.to_et(snap.get('updated', '?'))}")
    r = snap.get("resources", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("CPU", f"{r.get('cpu_pct', 0):.0f}%", help=f"Across {r.get('cpus', '?')} cores")
    c2.metric("Memory", f"{_gb(r.get('mem_used'))} / {_gb(r.get('mem_total'))} GB")
    c3.metric("Disk", f"{_gb(r.get('disk_used')).split('.')[0]} / {_gb(r.get('disk_total')).split('.')[0]} GB")
    running = [(u, i) for u, i in snap.get("units", {}).items() if i.get("running")]
    st.markdown("**Running now**")
    if not running:
        st.caption("Nothing; idle until the next scheduled job.")
    for unit, info in running:
        p = info.get("progress")
        steps = (info.get("summary") or {}).get("steps") or []
        step = steps[-1]["step"] if steps else ""
        if p:
            st.progress(min(p["pct"], 1.0), text=f"{info['label']}: {p['done']}/{p['total']} done, now {p['item']}")
        else:
            st.markdown(f"- {info['label']}" + (f": {step}" if step else ""))


def _details(snap: dict | None, host: str) -> None:
    if snap is None:
        st.caption("No status received yet.")
        return
    units = snap.get("units", {})
    st.markdown("**Jobs**")
    st.dataframe([{
        "job": i["label"],
        "state": ss.plain_state(i),
        "last result": i.get("Result", "-"),
        "last start": ss.systemd_to_et(i.get("ExecMainStartTimestamp", "")),
        "last end": ss.systemd_to_et(i.get("ExecMainExitTimestamp", "")),
        "memory peak": ss.human_bytes(i.get("MemoryPeak", "")),
        "unit": u,
    } for u, i in units.items()], hide_index=True, width="stretch")

    st.markdown("**Schedule**")
    st.dataframe([{
        "timer": t,
        "enabled": i.get("ActiveState", "?"),
        "next run": ss.systemd_to_et(i.get("NextElapseUSecRealtime", "")),
        "last run": ss.systemd_to_et(i.get("LastTriggerUSec", "")),
    } for t, i in snap.get("timers", {}).items()], hide_index=True, width="stretch")

    st.markdown("**Last run**")
    choices = [u for u in units if u != "fsp-dashboard"]
    if not choices:
        return
    unit = st.selectbox("Job", choices, key=f"job_{host}", format_func=lambda u: units[u]["label"])
    info = units[unit]
    summary = info.get("summary") or {}
    if not summary.get("steps"):
        started = info.get("ExecMainStartTimestamp") or ""
        st.caption(f"Last run: {ss.systemd_to_et(started)}, result {info.get('Result', '-')}"
                   if started else "No run recorded yet.")
    else:
        st.dataframe(summary["steps"], hide_index=True, width="stretch")
        if summary.get("errors"):
            st.warning(f"{summary['errors']} error lines, {summary.get('warnings', 0)} warnings in this run "
                       "(delisted tickers show up here too).")
        else:
            st.caption(f"No errors, {summary.get('warnings', 0)} warnings.")
    with st.expander("Raw log (last 30 lines)"):
        st.code(info.get("log") or "(no log lines)", language="text")


cols = st.columns(2, gap="large")
for col, (host, name, role, snap) in zip(cols, SERVERS):
    with col, st.container(border=True):
        _overview(name, role, snap, host)

st.divider()
tabs = st.tabs([name for _, name, _, _ in SERVERS])
for tab, (host, _, _, snap) in zip(tabs, SERVERS):
    with tab:
        _details(snap, host)
