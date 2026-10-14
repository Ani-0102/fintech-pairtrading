-- Rank pairs: in-sample cointegration strength, then out-of-sample Sharpe.
SELECT p.pair_id,
       round(p.coint_pvalue, 4)     AS fit_pvalue,
       round(b.test_adf_pvalue, 4)  AS test_adf_pvalue,
       b.test_adf_pvalue < 0.05     AS still_coint,
       round(b.sharpe, 2)           AS sharpe,
       round(b.max_drawdown, 4)     AS max_drawdown,
       b.n_trades
FROM pairs p
JOIN backtests b USING (pair_id)
ORDER BY p.coint_pvalue ASC, b.sharpe DESC;
