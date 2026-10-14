from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "pairs.duckdb"
SQL_DIR = ROOT / "sql"


def connect(path=DB_PATH):
    con = duckdb.connect(str(path))
    init_schema(con)
    return con


def init_schema(con):
    con.execute((SQL_DIR / "schema.sql").read_text())


def read_sql(name, **params):
    return (SQL_DIR / name).read_text().format(**params)
