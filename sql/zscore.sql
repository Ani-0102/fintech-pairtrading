-- Rolling z-score of the spread. Expects a table/view `spread_input(pair_id, date, spread)`
-- and the window size substituted for {window}. NULL until a full window is available.
SELECT
    pair_id,
    date,
    spread,
    CASE WHEN COUNT(spread) OVER w = {window}
         THEN (spread - AVG(spread) OVER w) / STDDEV_SAMP(spread) OVER w
    END AS zscore
FROM spread_input
WINDOW w AS (PARTITION BY pair_id ORDER BY date ROWS BETWEEN {window} - 1 PRECEDING AND CURRENT ROW)
ORDER BY pair_id, date;
