import os

import streamlit as st

from src import analysis
from src.agent import ask
from src.data import PAIRS, build
from src.db import connect, read_sql

st.set_page_config(page_title="Pairs-Trading Research", layout="wide")
NOTE = "Research tool only, not investment advice. Past backtests do not predict future results."


@st.cache_resource
def get_con():
    con = connect()
    if con.execute("SELECT COUNT(*) FROM backtests").fetchone()[0] == 0:
        with st.spinner("First run: downloading prices and building the database..."):
            build(con)
    return con


def api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:  # no secrets.toml locally -> fall back to .env (loaded by src.agent)
        return os.getenv("GEMINI_API_KEY")


def explore(con):
    pair_id = st.selectbox("Pair", [f"{a}-{b}" for a, b in PAIRS])
    a, b, beta, pval = con.execute(
        "SELECT a, b, hedge_ratio, coint_pvalue FROM pairs WHERE pair_id = ?", [pair_id]).fetchone()
    bt = con.execute("SELECT * FROM backtests WHERE pair_id = ?", [pair_id]).df().iloc[0]

    c = st.columns(5)
    c[0].metric("Fit p (2015-20)", f"{pval:.3f}")
    c[1].metric("ADF p (2021+)", f"{bt.test_adf_pvalue:.3f}")
    c[2].metric("OOS Sharpe", f"{bt.sharpe:.2f}")
    c[3].metric("Max drawdown", f"{bt.max_drawdown:.3f}")
    c[4].metric("Trades", int(bt.n_trades))
    still = bt.test_adf_pvalue < 0.05
    (st.success if still else st.warning)(
        f"Hedge ratio {beta:.3f} fixed from 2015-2020. Out of sample the spread is "
        + ("still stationary (ADF p < 0.05)." if still else "no longer stationary: the relationship broke down."))

    px = analysis.get_prices(con, [a, b])
    st.subheader("Prices (rebased to 100)")
    st.line_chart(px / px.iloc[0] * 100)

    sig = con.execute("SELECT date, zscore, position FROM signals WHERE pair_id = ? ORDER BY date",
                      [pair_id]).df().set_index("date")
    latest = sig.zscore.dropna().iloc[-1]
    st.subheader(f"Spread z-score (latest {latest:+.2f} SD)")
    bands = sig[["zscore"]].assign(**{"+2": 2.0, "-2": -2.0, "+3": 3.0, "-3": -3.0})
    st.line_chart(bands)

    st.subheader("Out-of-sample equity curve (cumulative log-spread P&L, after costs)")
    daily = analysis.backtest_pair(con, pair_id, save=False)["daily"]
    st.area_chart(daily.cumsum().rename("cum_pnl"))

    st.subheader("Trades")
    st.dataframe(con.execute("SELECT entry_date, exit_date, round(pnl, 4) AS pnl, exit_reason FROM trades "
                             "WHERE pair_id = ? ORDER BY entry_date", [pair_id]).df(),
                 hide_index=True, width="stretch")


def scanner(con):
    st.markdown("Ranked by in-sample cointegration p-value, then out-of-sample Sharpe "
                "(`sql/scanner.sql`, a JOIN of `pairs` and `backtests`).")
    st.dataframe(con.execute(read_sql("scanner.sql")).df(), hide_index=True, width="stretch")
    with st.expander("SQL"):
        st.code(read_sql("scanner.sql"), language="sql")


def agent(con):
    key = api_key()
    if not key:
        st.info("Set GEMINI_API_KEY in .env (local) or Streamlit secrets (deployed) to use the agent.")
        return
    examples = ["Which pair held up best out of sample, and is it still cointegrated?",
                "Did KO-PEP break down after 2020? Compare fit and test-period statistics.",
                "How does V-MA's backtest change with entry at 2.5 instead of 2?"]
    q = st.text_area("Question", st.session_state.get("q", examples[0]))
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex[:40] + "...", key=ex):
            st.session_state.q = ex
            st.rerun()
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Calling tools..."):
            try:
                out = ask(con, q, api_key=key)
            except Exception as e:
                if getattr(e, "code", None) == 429:
                    st.warning("Gemini free-tier quota reached for this model. Try later, or set "
                               "GEMINI_MODEL to another model (quotas are per model).")
                else:
                    st.error(f"Agent error: {type(e).__name__}: {str(e)[:300]}")
                return
        st.markdown(out["answer"])
        st.caption(f"{len(out['tool_calls'])} tool calls, ~{out['tokens']} tokens")
        for i, c in enumerate(out["tool_calls"], 1):
            with st.expander(f"{i}. {c['name']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})"):
                st.json(c["result"])


st.title("Pairs-Trading Research Agent")
st.caption(NOTE)
con = get_con()
page = st.sidebar.radio("Page", ["Explore a pair", "Pair scanner", "Ask the agent"])
{"Explore a pair": explore, "Pair scanner": scanner, "Ask the agent": agent}[page](con)
