# MISSION OBJECTIVE: PIPE_SHOP_ORDERS_TO_S3

**Natural Language Input:** Daily e-commerce orders from MongoDB (nested JSON documents) to the AWS S3 data lake

---

## 🏗️ 1. ARCHITECT SCOPE (DATA LOGIC)

**DATA PIPELINE (PYTHON) — SEMI-STRUCTURED SOURCE:**
- Source: **MongoDB** collection from `DATA_SOURCE.table` in your context (`raw_orders`). Call `read_data_schema` with `db_type="mongodb"`; it returns the INFERRED document schema (field paths, observed types, presence %) and `flattened_columns`.
- Credentials: `cloud_get("aws", "mongo_uri", db_type="mongodb")` and `cloud_get("aws", "mongo_db_name", db_type="mongodb") or "shop_raw"` inside the `if _CLOUD == "aws":` branch — never `os.getenv("MONGO_URI")`.
- Extraction: `pymongo.MongoClient(uri, serverSelectionTimeoutMS=20000)`; read the collection in batches of 500 documents with `.find({}).batch_size(500)`; client creation AND the batch loop in the SAME `try` block. This REPLACES the SQL `create_engine` / `pd.read_sql_query` steps of the standard script structure — every other step (idempotency check, business rules, Parquet write, Trino partition sync, all five metrics) stays exactly as in the Python pipeline standard.
- Flattening (per batch): one output row per order ITEM via `pd.json_normalize(docs, record_path="items", meta=[...], errors="ignore", sep="_")` — documents without an `items` list must be skipped, not crash. Order and customer fields are repeated on each item row (`customer.city` → `customer_city`). Follow the MongoDB extraction standard (query the vector store: "MONGODB EXTRACTION STANDARD flatten json_normalize ObjectId").
- `_id` is a BSON ObjectId: drop it or convert with `.astype(str)` — pyarrow cannot write ObjectId to Parquet.
- Output format: parquet/snappy
- Destination URI: `s3://princy2312-shop-orders-insights-data/processed/`
- Idempotency: check `s3://princy2312-shop-orders-insights-data/processed/run_date=<today>/` before writing. If data exists, exit 0.
- Save script to: `scripts/pipe_shop_orders_to_s3.py`

**BUSINESS RULES:**
  - See `TRANSFORMATION_LOGIC` in your context — this is the authoritative and complete rules list, applied to the FLATTENED item rows. Numeric fields (`price`, `qty`) arrive as numbers OR strings OR missing: coerce with `pd.to_numeric(..., errors="coerce")` BEFORE comparing. `order_date` arrives as a string, a date or garbage: coerce with `pd.to_datetime(..., errors="coerce")` BEFORE comparing.

**CATALOG & OBSERVABILITY:**
- Trino DDL: `sql/setup_trino.sql` — external table at `s3://princy2312-shop-orders-insights-data/processed/` with `run_date` partitioning; columns = the FLATTENED item-row columns (from `flattened_columns`), plus `is_suspicious BOOLEAN`, then `run_date DATE` last.
- Trino target: `hive.shop_orders.pipe_shop_orders_to_s3`
- Grafana: `dashboards/monitoring_specs.json` with 60-minute Data Silence alert

---

## 🛠️ 2. INFRA SCOPE (DEPLOYMENT & AUTOMATION)

**TERRAFORM:** Deploy AWS storage and identity/IAM resources (bucket `princy2312-shop-orders-insights-data`).
**K8S:** Deploy namespaces, Trino, Grafana, Prometheus + Pipeline Job.
**CI/CD:** `/.github/workflows/pipe_shop_orders_to_s3_pipeline.yml`

---

## 🔒 3. GLOBAL CONSTRAINTS

- All DB credentials via `cloud_get()` — never `os.getenv()` for the MongoDB URI or database name. GitHub Secrets are for CI/CD auth only (ECR, cloud CLI).
- Every agent MUST query the Vector Store for domain standards before writing.
- Final signal: `echo "Deployment Complete"`.
