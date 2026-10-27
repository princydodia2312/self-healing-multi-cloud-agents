"""
Trino query client for the S3 data lake.

Connects to a real Trino coordinator when TRINO_HOST is set (e.g. after
`kubectl port-forward svc/trino 8080:8080 -n analytics`); falls back to demo
mode with realistic sample data otherwise.
"""
from __future__ import annotations
import os
import pandas as pd

# ── Pre-built analytical query shown in the UI ──────────────────────────────
# The eu_sales pipeline writes snappy Parquet to s3://eu-sales-insights-data-princy2312/processed/
# partitioned by run_date; sql/setup_trino.sql registers it as an external Hive table.
# Filtering on the partition column lets Trino prune whole S3 prefixes instead of
# scanning every file.
SAMPLE_QUERY = """\
-- 📊 Daily revenue per currency over the S3 Parquet data lake
-- Filtering on the partition column (run_date) → Trino reads only those S3 prefixes

SELECT
    run_date,
    currency,
    COUNT(*)                          AS orders,
    SUM(quantity)                     AS units_sold,
    ROUND(SUM(unit_price * quantity), 2) AS revenue,
    ROUND(AVG(unit_price), 2)         AS avg_unit_price
FROM
    hive.sales_eu.pipe_eu_sales_to_s3          -- external table over s3://.../processed/
WHERE
    run_date >= DATE '2026-05-01'              -- partition pruning
GROUP BY
    run_date, currency
ORDER BY
    run_date DESC, revenue DESC
LIMIT 100"""

# Backward-compatible alias (older UI code imported this name).
CROSS_CLOUD_QUERY = SAMPLE_QUERY

# ── Demo data returned when Trino is not reachable ──────────────────────────
_DEMO_DF = pd.DataFrame({
    "run_date":       ["2026-05-06", "2026-05-06", "2026-05-05", "2026-05-05",
                       "2026-05-04", "2026-05-04", "2026-05-03", "2026-05-03"],
    "currency":       ["EUR", "GBP", "EUR", "GBP", "EUR", "GBP", "EUR", "GBP"],
    "orders":         [742, 118, 701, 131, 688, 104, 659, 97],
    "units_sold":     [3710, 552, 3466, 619, 3398, 489, 3205, 455],
    "revenue":        [186_412.50, 27_904.10, 172_380.75, 31_022.40,
                       168_955.20, 24_610.85, 159_870.00, 22_745.30],
    "avg_unit_price": [50.24, 50.55, 49.73, 50.12, 49.72, 50.33, 49.88, 49.99],
})


def run_query(sql: str) -> tuple[pd.DataFrame, bool]:
    """
    Execute a Trino query. Returns (dataframe, is_live).
    is_live=False means demo mode — no real Trino connection.
    """
    host = os.getenv("TRINO_HOST", "")
    port = int(os.getenv("TRINO_PORT", "8080"))

    if host:
        try:
            import trino  # type: ignore
            conn = trino.dbapi.connect(
                host=host,
                port=port,
                user=os.getenv("TRINO_USER", "streamlit"),
                catalog="hive",
                schema="default",
            )
            cur = conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(200)
            return pd.DataFrame(rows, columns=cols), True
        except Exception:
            pass

    return _DEMO_DF.copy(), False
