import pandas as pd
import yfinance as yf

PAIRS = [("KO", "PEP"), ("XOM", "CVX"), ("V", "MA"), ("HD", "LOW"), ("GLD", "GDX")]
START = "2015-01-01"


def tickers_to_fetch(con, tickers, start=START):
    rows = con.execute("SELECT ticker, MIN(date), MAX(date) FROM prices GROUP BY ticker").fetchall()
    have = {t: (first, last) for t, first, last in rows}
    start_ok = (pd.Timestamp(start) + pd.Timedelta(days=7)).date()
    fresh_ok = (pd.Timestamp.today() - pd.Timedelta(days=5)).date()
    return [t for t in tickers
            if t not in have or have[t][0] > start_ok or have[t][1] < fresh_ok]


def load_prices(con, tickers, start=START):
    todo = tickers_to_fetch(con, tickers, start)
    if not todo:
        return 0
    raw = yf.download(todo, start=start, auto_adjust=True, progress=False)["Close"]
    long = raw.reset_index().melt(id_vars="Date", var_name="ticker", value_name="close").dropna()
    long = long.rename(columns={"Date": "date"})[["ticker", "date", "close"]]
    long["date"] = pd.to_datetime(long["date"]).dt.date
    con.execute("DELETE FROM prices WHERE ticker IN (SELECT UNNEST(?))", [todo])
    con.execute("INSERT INTO prices SELECT * FROM long")
    return len(long)


def build(con):
    from src.analysis import backtest_pair, fit_pair, save_signals
    load_prices(con, sorted({t for p in PAIRS for t in p}))
    fits = [fit_pair(con, a, b) for a, b in PAIRS]
    for f in fits:
        save_signals(con, f["pair_id"])
        f.update(backtest_pair(con, f["pair_id"]))
    return pd.DataFrame(fits)


if __name__ == "__main__":
    from src.db import connect
    print(build(connect()).round(4).to_string(index=False))
