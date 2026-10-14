from types import SimpleNamespace

from src.agent import ask, query_results
from src.db import connect


def test_query_guard(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    assert "rows" in query_results(con, "SELECT 1 AS x;")
    assert "rows" in query_results(con, "WITH t AS (SELECT 1 AS x) SELECT * FROM t")
    for bad in ["DROP TABLE prices", "SELECT 1; DELETE FROM prices", "SELECT * FROM read_csv('x')",
                "UPDATE pairs SET a='x'"]:
        assert "error" in query_results(con, bad)
    assert len(query_results(con, "SELECT * FROM range(1000)")["rows"]) == 50


class FakeModels:
    def __init__(self, replies):
        self.replies = iter(replies)

    def generate_content(self, **_):
        return next(self.replies)


def _resp(fcs=None, text=None):
    content = SimpleNamespace(role="model", parts=[])
    return SimpleNamespace(function_calls=fcs, text=text, candidates=[SimpleNamespace(content=content)],
                           usage_metadata=SimpleNamespace(total_token_count=10))


def test_agent_loop_runs_tools_then_answers(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    fc = SimpleNamespace(name="query_results", args={"sql": "SELECT 42 AS n"})
    client = SimpleNamespace(models=FakeModels([_resp([fc]), _resp(text="n is 42")]))
    out = ask(con, "q", client=client, model="m")
    assert out["answer"] == "n is 42"
    assert out["tool_calls"][0]["result"]["rows"] == [[42]]
