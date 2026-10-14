import json
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src import analysis

load_dotenv()
DEFAULT_MODEL = "gemini-3.5-flash-lite"  # free tier; 3.8-flash free quota is only ~20 req/day
MAX_TOOL_CALLS, TOKEN_CAP, ROW_CAP = 8, 60_000, 50

SYSTEM = """You are a quantitative research assistant for a pairs-trading research tool.
Answer only from tool results and cite the numbers you got (p-values, Sharpe, z-scores, dates).
Use descriptive language ("the spread is 2.3 SD wide", "the pair broke down after 2022").
Never give investment advice or say buy/sell/should trade. If a pair's out-of-sample ADF p-value is
>= 0.05 or its fit p-value is >= 0.05, say plainly that the relationship is not (or no longer)
statistically supported. Fit period is 2015-2020; backtests are out of sample from 2021 with a
fixed hedge ratio and 5 bps cost per leg.
DuckDB tables: prices(ticker,date,close), pairs(pair_id,a,b,hedge_ratio,coint_pvalue,fit_start,fit_end),
signals(pair_id,date,spread,zscore,position), trades(pair_id,entry_date,exit_date,pnl,exit_reason),
backtests(pair_id,sharpe,max_drawdown,n_trades,test_start,test_end,test_adf_pvalue).
pair_id looks like "KO-PEP". pnl and drawdown are in log-spread units."""

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|"
                       r"install|load|export|import|call|set|checkpoint|read_\w+|write_\w+)\b", re.I)


def query_results(con, sql):
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql or not re.match(r"^(select|with)\b", sql, re.I) or FORBIDDEN.search(sql):
        return {"error": "Only a single read-only SELECT is allowed."}
    df = con.execute(f"SELECT * FROM ({sql}) LIMIT {ROW_CAP}").df()
    return {"columns": list(df.columns), "rows": json.loads(df.to_json(orient="values", date_format="iso"))}


def tool_get_prices(con, tickers, start=None, end=None):
    px = analysis.get_prices(con, tickers, start, end)
    if px.empty:
        return {"error": f"No prices for {tickers} in that range."}
    return {"n_days": len(px), "start": str(px.index[0]), "end": str(px.index[-1]),
            "first_close": px.iloc[0].round(2).to_dict(), "last_close": px.iloc[-1].round(2).to_dict(),
            "total_return_pct": ((px.iloc[-1] / px.iloc[0] - 1) * 100).round(1).to_dict()}


def tool_test_cointegration(con, a, b, start=analysis.FIT_START, end=None):
    end = end or (analysis.FIT_END if start == analysis.FIT_START else None)  # custom start -> up to today
    beta, p, n = analysis.coint_test(con, a, b, start, end)
    return {"pair": f"{a}-{b}", "start": start, "end": end, "n_days": n, "hedge_ratio": round(beta, 4),
            "engle_granger_pvalue": round(p, 4), "cointegrated_at_5pct": p < 0.05}


def tool_run_backtest(con, pair_id, entry=2.0, exit=0.0, stop=3.0):
    r = analysis.backtest_pair(con, pair_id, save=False, entry=entry, exit=exit, stop=stop)
    reasons = r["trades"].exit_reason.value_counts().to_dict()
    return {"pair_id": pair_id, "rules": {"entry": entry, "exit": exit, "stop": stop},
            "sharpe": round(r["sharpe"], 3), "max_drawdown": round(r["max_drawdown"], 4),
            "n_trades": r["n_trades"], "total_pnl": round(float(r["trades"].pnl.sum()), 4),
            "exit_reasons": reasons, "test_adf_pvalue": round(r["test_adf_pvalue"], 4)}


S, N, STR, ARR = types.Schema, types.Type.NUMBER, types.Type.STRING, types.Type.ARRAY
DECLS = [
    types.FunctionDeclaration(
        name="get_prices", description="Summary of adjusted closes for tickers in the DB (2015-today).",
        parameters=S(type=types.Type.OBJECT, required=["tickers"], properties={
            "tickers": S(type=ARR, items=S(type=STR)), "start": S(type=STR, description="YYYY-MM-DD"),
            "end": S(type=STR, description="YYYY-MM-DD")})),
    types.FunctionDeclaration(
        name="test_cointegration", description="Engle-Granger test + OLS hedge ratio on log prices. Defaults to the 2015-2020 fit window; if only start is given, runs from start to today.",
        parameters=S(type=types.Type.OBJECT, required=["a", "b"], properties={
            "a": S(type=STR), "b": S(type=STR), "start": S(type=STR), "end": S(type=STR)})),
    types.FunctionDeclaration(
        name="run_backtest", description="Out-of-sample (2021+) z-score backtest of a stored pair with custom thresholds. Not saved.",
        parameters=S(type=types.Type.OBJECT, required=["pair_id"], properties={
            "pair_id": S(type=STR), "entry": S(type=N), "exit": S(type=N), "stop": S(type=N)})),
    types.FunctionDeclaration(
        name="query_results", description=f"Run one read-only DuckDB SELECT over the tables (max {ROW_CAP} rows).",
        parameters=S(type=types.Type.OBJECT, required=["sql"], properties={"sql": S(type=STR)})),
]
TOOLS = {"get_prices": tool_get_prices, "test_cointegration": tool_test_cointegration,
         "run_backtest": tool_run_backtest, "query_results": query_results}


def run_tool(con, name, args):
    try:
        return TOOLS[name](con, **args)
    except Exception as e:  # tool errors go back to the model, not up to the user
        return {"error": f"{type(e).__name__}: {e}"}


def _generate(client, model, contents, config):
    try:
        return client.models.generate_content(model=model, contents=contents, config=config)
    except errors.APIError as e:
        if e.code not in (429, 503):  # rate limit or overloaded: wait and retry once
            raise
        time.sleep(15)
        return client.models.generate_content(model=model, contents=contents, config=config)


def ask(con, question, api_key=None, model=None, client=None):
    client = client or genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
    model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM, tools=[types.Tool(function_declarations=DECLS)], temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    calls, tokens = [], 0
    while True:
        resp = _generate(client, model, contents, config)
        tokens += getattr(resp.usage_metadata, "total_token_count", 0) or 0
        fcs = resp.function_calls or []
        if not fcs:
            return {"answer": resp.text or "(no answer)", "tool_calls": calls, "tokens": tokens}
        if len(calls) + len(fcs) > MAX_TOOL_CALLS or tokens > TOKEN_CAP:
            # budget spent: one last call with tools switched off, so the model must answer from what it has
            contents.append(types.Content(role="user", parts=[types.Part.from_text(
                text="Tool budget reached. Answer now using only the tool results above.")]))
            final = config.model_copy(update={"tool_config": types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE"))})
            resp = _generate(client, model, contents, final)
            return {"answer": resp.text or "(no answer: budget reached)", "tool_calls": calls,
                    "tokens": tokens + (getattr(resp.usage_metadata, "total_token_count", 0) or 0)}
        contents.append(resp.candidates[0].content)
        parts = []
        for fc in fcs:
            result = run_tool(con, fc.name, dict(fc.args or {}))
            calls.append({"name": fc.name, "args": dict(fc.args or {}), "result": result})
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))


if __name__ == "__main__":
    import sys
    from src.db import connect
    out = ask(connect(), " ".join(sys.argv[1:]) or "Which pair held up best out of sample?")
    for c in out["tool_calls"]:
        print("tool:", c["name"], c["args"])
    print(out["answer"])
