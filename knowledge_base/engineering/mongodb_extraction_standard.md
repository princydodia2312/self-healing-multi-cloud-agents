# STANDARD: MONGODB EXTRACTION (SEMI-STRUCTURED SOURCE)

Applies when `DATA_SOURCE.type` is `mongodb`. It REPLACES only the credentials + extraction part
(steps 2–3) of the Python pipeline standard's MANDATORY SCRIPT STRUCTURE. The idempotency check,
business rules, Parquet write (`storage_options=dict()`), Trino partition sync and all five metrics
stay exactly as in the Python pipeline standard.

## Why this is different from a SQL source
A MongoDB collection has NO fixed schema. One document may nest objects (`customer.city`), hold
arrays (`items: [...]`), omit fields, or store the same field as a number in one document and a
string in the next. Code that indexes a field directly (`doc["customer"]["city"]`) or casts with
`.astype(float)` crashes on the first irregular document. Always: flatten with `pd.json_normalize`,
treat every field as optional, and coerce types with `errors="coerce"`.

## CRITICAL RULES
- **Credentials:** `cloud_get("aws", "mongo_uri", db_type="mongodb")` and
  `cloud_get("aws", "mongo_db_name", db_type="mongodb") or "shop_raw"` — inside the
  `if _CLOUD == "aws":` branch. `os.getenv("MONGO_URI")` is FORBIDDEN (bypasses SSM; the
  validator rejects it). Never log the URI — it contains the password.
- **Driver:** `from pymongo import MongoClient` (pymongo is in requirements.txt via the source DB type).
  `MongoClient(uri, serverSelectionTimeoutMS=20000)`.
- **Batching (Big Data):** never `list(collection.find())` the whole collection into memory. Iterate
  `collection.find({}, batch_size=500)` and flush every 500 documents into one Parquet part file.
- **Same try block:** `MongoClient(...)` AND the batch loop must be in ONE `try` block; close the
  client in `finally`.
- **ObjectId:** `_id` is a BSON `ObjectId` → pyarrow raises `ArrowTypeError: Expected bytes, got a
  'ObjectId' object`. Drop it (`meta` does not include `_id`) or `.astype(str)` it.
- **Arrays → rows:** one output row per array element with
  `pd.json_normalize(docs, record_path="items", meta=[...], errors="ignore", sep="_")`.
  A document whose `items` is missing or not a list must be SKIPPED (filter before normalizing),
  otherwise json_normalize raises `KeyError`/`TypeError`.
- **Nested objects → columns:** `meta=["order_id", "order_date", ["customer", "name"], ["customer", "city"]]`
  with `sep="_"` yields `customer_name`, `customer_city`. `errors="ignore"` fills missing meta with NaN.
- **Missing columns:** after normalizing, guarantee every expected column exists before the rules:
  `for col in EXPECTED: if col not in chunk.columns: chunk[col] = None`.
- **Mixed types:** `pd.to_numeric(chunk["price"], errors="coerce")`,
  `pd.to_datetime(chunk["order_date"], errors="coerce")` — assign back FIRST, compare SECOND.
- **Stable Parquet schema:** cast object columns that hold text to `str` / `string` dtype so every part
  file has the same column types (Trino reads all parts with one table schema).

## Extraction skeleton (replaces steps 2–3)
```python
from pymongo import MongoClient

EXPECTED = ["order_id", "order_date", "customer_name", "customer_city", "product", "price", "qty"]

    # ── 2. CREDENTIALS via cloud_get() ───────────────────────────────────────
    if _CLOUD == "aws":
        mongo_uri = cloud_get("aws", "mongo_uri", db_type="mongodb")
        mongo_db = cloud_get("aws", "mongo_db_name", db_type="mongodb") or "shop_raw"

    # ── 3. EXTRACTION + FLATTEN + TRANSFORMATION + WRITE (one try block) ─────
    start_time = time.time()
    total_rows = 0
    rejected_by_reason = {}

    def _flatten(docs):
        docs = [d for d in docs if isinstance(d.get("items"), list) and d["items"]]
        if not docs:
            return pd.DataFrame(columns=EXPECTED)
        chunk = pd.json_normalize(
            docs, record_path="items",
            meta=["order_id", "order_date", ["customer", "name"], ["customer", "city"]],
            errors="ignore", sep="_",
        )
        for col in EXPECTED:
            if col not in chunk.columns:
                chunk[col] = None
        return chunk[EXPECTED]

    client = None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=20000)
        cursor = client[mongo_db]["<collection>"].find({}, batch_size=500)
        batch, i = [], 0
        for doc in cursor:
            batch.append(doc)
            if len(batch) < 500:
                continue
            i = _process(_flatten(batch), i)   # business rules + to_parquet part_{i}, returns i + 1
            batch = []
        if batch:
            i = _process(_flatten(batch), i)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise
    finally:
        if client is not None:
            client.close()
```
`_process(chunk, i)` holds the SAME per-chunk body as the SQL standard: coerce types, apply every
`quality_standards` rule with its own `_before = len(chunk)` and `rejected_by_reason[...]`, cast
`qty` to `Int64`, write `f"{partition_uri}part_{i}.parquet"` with `storage_options=dict()`, add to
`total_rows`, return `i + 1`. Declare `total_rows` / `rejected_by_reason` with `nonlocal` if
`_process` is nested, or inline the body twice — but never skip the rules for the final partial batch.
An empty chunk (all documents malformed) is skipped without writing a file.

## Common runtime errors and their fix
| Error in CI log | Cause | Fix |
|---|---|---|
| `ArrowTypeError: ... 'ObjectId' object` | `_id` reached `to_parquet` | drop `_id` / `.astype(str)` |
| `KeyError: 'items'` / `TypeError: ... not iterable` | document without an items list | filter docs before `json_normalize` |
| `KeyError: 'customer_city'` | field absent in a whole batch | add missing EXPECTED columns as None |
| `TypeError: '>' not supported between 'str' and 'int'` | price/qty stored as text | `pd.to_numeric(..., errors="coerce")` first |
| `ServerSelectionTimeoutError` | Atlas network access list / wrong URI | infra/ops: allow the cluster's IPs in Atlas, check SSM `mongo_uri` |
