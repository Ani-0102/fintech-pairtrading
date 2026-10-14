import numpy as np
import pandas as pd

from src.analysis import generate_positions, run_backtest


def _s(vals):
    return pd.Series(vals, index=pd.bdate_range("2021-01-01", periods=len(vals)), dtype=float)


def test_rules_entry_exit_stop():
    z = _s([0, 2.1, 1.0, -0.1, 0, -2.5, -3.2, -2.5, -1.0, 0.5])
    pos, reasons = generate_positions(z)
    assert pos.tolist() == [0, -1, -1, 0, 0, 1, 0, 0, 0, 0]
    assert reasons.iloc[3] == "exit" and reasons.iloc[6] == "stop"


def test_pnl_on_hand_made_series():
    # short spread at 1.0 (z=2.1), exit at 0.4 -> gross 0.6, minus 2 trades * 5bps * (1+0.5)
    spread = _s([0.5, 1.0, 0.8, 0.4, 0.4])
    z = _s([0.0, 2.1, 1.0, -0.1, 0.0])
    res = run_backtest(spread, z, beta=0.5)
    cost = 2 * 0.0005 * 1.5
    assert res["n_trades"] == 1
    assert np.isclose(res["trades"].pnl.iloc[0], 0.6 - cost)
    assert np.isclose(res["daily"].sum(), 0.6 - cost)


def test_no_look_ahead():
    rng = np.random.default_rng(1)
    spread = _s(rng.normal(size=200).cumsum())
    z = _s(rng.normal(scale=2, size=200))
    base = run_backtest(spread, z, beta=1.0)
    z2, spread2 = z.copy(), spread.copy()
    z2.iloc[150:] = rng.normal(scale=2, size=50)  # rewrite the future
    spread2.iloc[150:] += 5
    alt = run_backtest(spread2, z2, beta=1.0)
    pd.testing.assert_series_equal(base["positions"].iloc[:150], alt["positions"].iloc[:150])
    pd.testing.assert_series_equal(base["daily"].iloc[:150], alt["daily"].iloc[:150])
