import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from src.db import read_sql

FIT_START, FIT_END = "2015-01-01", "2020-12-31"
TEST_START = "2021-01-01"
COST_PER_LEG = 0.0005  # 5 bps


def get_prices(con, tickers, start=None, end=None):
    df = con.execute(
        "SELECT date, ticker, close FROM prices WHERE ticker IN (SELECT UNNEST(?)) "
        "AND date >= COALESCE(?::DATE, DATE '1900-01-01') AND date <= COALESCE(?::DATE, DATE '2100-01-01')",
        [list(tickers), start, end],
    ).df()
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return wide[list(tickers)].dropna()


def coint_test(con, a, b, start=FIT_START, end=FIT_END):
    px = np.log(get_prices(con, [a, b], start, end))
    _, pvalue, _ = coint(px[a], px[b])
    beta = sm.OLS(px[a], sm.add_constant(px[b])).fit().params[b]
    return float(beta), float(pvalue), len(px)


def fit_pair(con, a, b, fit_start=FIT_START, fit_end=FIT_END):
    beta, pvalue, _ = coint_test(con, a, b, fit_start, fit_end)
    pair_id = f"{a}-{b}"
    con.execute(
        "INSERT OR REPLACE INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
        [pair_id, a, b, float(beta), float(pvalue), fit_start, fit_end],
    )
    return {"pair_id": pair_id, "hedge_ratio": float(beta), "coint_pvalue": float(pvalue)}


def spread_series(con, pair_id):
    a, b, beta = con.execute("SELECT a, b, hedge_ratio FROM pairs WHERE pair_id = ?", [pair_id]).fetchone()
    px = np.log(get_prices(con, [a, b]))
    return (px[a] - beta * px[b]).rename("spread")


def zscore_pandas(spread, window=60):
    roll = spread.rolling(window, min_periods=window)
    return ((spread - roll.mean()) / roll.std(ddof=1)).rename("zscore")


def zscore_sql(con, pair_id, spread, window=60):
    con.register("spread_input", pd.DataFrame({"pair_id": pair_id, "date": spread.index, "spread": spread.values}))
    out = con.execute(read_sql("zscore.sql", window=window)).df()
    con.unregister("spread_input")
    return out.set_index("date")["zscore"]


def save_signals(con, pair_id, window=60):
    s = spread_series(con, pair_id)
    df = pd.DataFrame({"pair_id": pair_id, "date": s.index, "spread": s.values,
                       "zscore": zscore_pandas(s, window).values, "position": None})
    con.execute("DELETE FROM signals WHERE pair_id = ?", [pair_id])
    con.execute("INSERT INTO signals SELECT pair_id, date, spread, zscore, position::INTEGER FROM df")
    return len(df)


def generate_positions(z, entry=2.0, exit=0.0, stop=3.0):
    """pos[t] is decided from z[t] (today's close) and earns from t+1 on. +1 = long spread."""
    pos, reasons, cur, locked = [], [], 0, False
    for zt in z:
        reason = None
        if np.isnan(zt):
            pos.append(cur); reasons.append(None); continue
        if locked and abs(zt) < entry:  # after a stop, wait for the spread to calm down
            locked = False
        if cur != 0 and cur * zt <= -stop:
            cur, reason, locked = 0, "stop", True
        elif cur != 0 and cur * zt >= -exit:
            cur, reason = 0, "exit"
        elif cur == 0 and not locked and abs(zt) >= entry:
            cur = -1 if zt > 0 else 1
        pos.append(cur); reasons.append(reason)
    return pd.Series(pos, index=z.index, name="position"), pd.Series(reasons, index=z.index)


def run_backtest(spread, z, beta, entry=2.0, exit=0.0, stop=3.0, cost=COST_PER_LEG):
    pos, reasons = generate_positions(z, entry, exit, stop)
    if pos.iloc[-1] != 0:  # close anything still open on the last test day
        pos.iloc[-1], reasons.iloc[-1] = 0, "end"
    trade_cost = cost * (1 + abs(beta))  # one leg in A, beta-sized leg in B
    dpos = pos.diff().fillna(pos).abs()
    daily = pos.shift(1).fillna(0) * spread.diff().fillna(0) - trade_cost * dpos

    trades, open_ = [], None
    for t, (prev, cur) in enumerate(zip(pos.shift(1).fillna(0), pos)):
        if prev != cur and prev != 0:
            pnl = prev * (spread.iloc[t] - spread.iloc[open_]) - 2 * trade_cost
            trades.append((spread.index[open_], spread.index[t], pnl, reasons.iloc[t] or "flip"))
            open_ = None
        if prev != cur and cur != 0:
            open_ = t
    trades = pd.DataFrame(trades, columns=["entry_date", "exit_date", "pnl", "exit_reason"])

    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    cum = daily.cumsum()
    return {"positions": pos, "daily": daily, "trades": trades, "sharpe": float(sharpe),
            "max_drawdown": float((cum - cum.cummax()).min()), "n_trades": len(trades)}


def backtest_pair(con, pair_id, window=60, test_start=TEST_START, save=True, **rules):
    beta = con.execute("SELECT hedge_ratio FROM pairs WHERE pair_id = ?", [pair_id]).fetchone()[0]
    s = spread_series(con, pair_id)
    z = zscore_pandas(s, window)  # rolling, so test-period z only sees past data
    test = s.index >= pd.Timestamp(test_start)
    res = run_backtest(s[test], z[test], beta, **rules)
    adf_p = float(adfuller(s[test], result_object=False)[1])  # fixed-beta spread still stationary out of sample?
    summary = {k: res[k] for k in ("sharpe", "max_drawdown", "n_trades")} | {"test_adf_pvalue": adf_p}
    if not save:
        return summary | {"trades": res["trades"], "daily": res["daily"]}

    con.execute("DELETE FROM trades WHERE pair_id = ?", [pair_id])
    tr = res["trades"].assign(pair_id=pair_id)
    con.execute("INSERT INTO trades SELECT pair_id, entry_date, exit_date, pnl, exit_reason FROM tr")
    con.execute("INSERT OR REPLACE INTO backtests VALUES (?, ?, ?, ?, ?, ?, ?)",
                [pair_id, res["sharpe"], res["max_drawdown"], res["n_trades"],
                 s.index[test][0], s.index[test][-1], adf_p])
    pos = res["positions"].reset_index().rename(columns={"index": "date"})
    con.execute("UPDATE signals SET position = NULL WHERE pair_id = ?", [pair_id])
    con.execute("UPDATE signals SET position = pos.position FROM pos "
                "WHERE signals.pair_id = ? AND signals.date = pos.date", [pair_id])
    return summary
