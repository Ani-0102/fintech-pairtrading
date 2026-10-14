import numpy as np
import pandas as pd

from src.analysis import zscore_pandas, zscore_sql
from src.db import connect


def test_pandas_and_sql_zscore_match(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=300)
    spread = pd.Series(rng.normal(size=300).cumsum(), index=dates, name="spread")
    con = connect(tmp_path / "t.duckdb")

    pd_z = zscore_pandas(spread, 60)
    sql_z = zscore_sql(con, "X-Y", spread, 60)
    sql_z.index = pd.to_datetime(sql_z.index)

    assert pd_z.isna().sum() == sql_z.isna().sum() == 59
    np.testing.assert_allclose(pd_z.dropna().values, sql_z.dropna().values, rtol=1e-9)
