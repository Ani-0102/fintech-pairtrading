CREATE TABLE IF NOT EXISTS prices (
    ticker VARCHAR, date DATE, close DOUBLE,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS pairs (
    pair_id VARCHAR PRIMARY KEY, a VARCHAR, b VARCHAR, hedge_ratio DOUBLE,
    coint_pvalue DOUBLE, fit_start DATE, fit_end DATE
);
CREATE TABLE IF NOT EXISTS signals (
    pair_id VARCHAR, date DATE, spread DOUBLE, zscore DOUBLE, position INTEGER,
    PRIMARY KEY (pair_id, date)
);
CREATE TABLE IF NOT EXISTS trades (
    pair_id VARCHAR, entry_date DATE, exit_date DATE, pnl DOUBLE, exit_reason VARCHAR
);
CREATE TABLE IF NOT EXISTS backtests (
    pair_id VARCHAR PRIMARY KEY, sharpe DOUBLE, max_drawdown DOUBLE, n_trades INTEGER,
    test_start DATE, test_end DATE
);
ALTER TABLE backtests ADD COLUMN IF NOT EXISTS test_adf_pvalue DOUBLE;
