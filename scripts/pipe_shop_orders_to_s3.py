import os
import time
import datetime
import logging
from urllib.parse import urlparse

import pandas as pd
from pymongo import MongoClient
from trino.dbapi import connect as trino_connect
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from utils.cloud_config import cloud_get

_CLOUD = os.getenv("CLOUD_PROVIDER", "aws")
if _CLOUD == "aws":
    import boto3
elif _CLOUD == "gcp":
    from google.cloud import storage
elif _CLOUD == "azure":
    from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)

EXPECTED = [
    "order_id",
    "order_date",
    "customer_name",
    "customer_email",
    "customer_city",
    "product",
    "price",
    "qty",
]


def run():
    logging.info("Pipeline starting: pipe_shop_orders_to_s3")

    # ── 1. IDEMPOTENCY CHECK ──────────────────────────────────────────────────
    run_date = datetime.date.today().isoformat()
    destination_uri = os.getenv("DESTINATION_URI")
    partition_uri = f"{destination_uri}run_date={run_date}/"
    parsed = urlparse(partition_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')

    if _CLOUD == "aws":
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if response.get('KeyCount', 0) > 0:
            logging.info(f"Partition run_date={run_date} already populated. Skipping.")
            return
    elif _CLOUD == "gcp":
        client = storage.Client()
        blobs = list(client.list_blobs(bucket, prefix=prefix, max_results=1))
        if blobs:
            logging.info("Destination already populated. Skipping.")
            return
    elif _CLOUD == "azure":
        container_name = bucket.split('@')[0]
        client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
        container = client.get_container_client(container_name)
        blobs = list(container.list_blobs(name_starts_with=prefix))
        if blobs:
            logging.info("Destination already populated. Skipping.")
            return

    # ── 2. CREDENTIALS via cloud_get() ───────────────────────────────────────
    if _CLOUD == "aws":
        mongo_uri = cloud_get("aws", "mongo_uri", db_type="mongodb")
        mongo_db = cloud_get("aws", "mongo_db_name", db_type="mongodb") or "shop_raw"
    elif _CLOUD == "gcp":
        mongo_uri = cloud_get("gcp", "mongo_uri", db_type="mongodb")
        mongo_db = cloud_get("gcp", "mongo_db_name", db_type="mongodb") or "shop_raw"
    elif _CLOUD == "azure":
        mongo_uri = cloud_get("azure", "mongo_uri", db_type="mongodb")
        mongo_db = cloud_get("azure", "mongo_db_name", db_type="mongodb") or "shop_raw"

    # ── 3. EXTRACTION + FLATTEN + TRANSFORMATION + WRITE ───────────────────────
    start_time = time.time()
    total_rows = 0
    rejected_by_reason = {}

    def _flatten(docs):
        docs = [d for d in docs if isinstance(d.get("items"), list) and d["items"]]
        if not docs:
            return pd.DataFrame(columns=EXPECTED)
        chunk = pd.json_normalize(
            docs,
            record_path="items",
            meta=["order_id", "order_date", ["customer", "name"], ["customer", "email"], ["customer", "city"]],
            errors="ignore",
            sep="_",
        )
        for col in EXPECTED:
            if col not in chunk.columns:
                chunk[col] = None
        return chunk[EXPECTED]

    def _process(chunk, i):
        nonlocal total_rows, rejected_by_reason
        if chunk.empty:
            return i

        # 3b. Business rules
        # Rule 1: completeness_enforcement (DROP_RECORD)
        _before = len(chunk)
        _cond_id = chunk['order_id'].notna() & (chunk['order_id'].astype(str).str.strip() != '')
        _cond_prod = chunk['product'].notna() & (chunk['product'].astype(str).str.strip() != '')
        chunk = chunk[_cond_id & _cond_prod]
        rejected_by_reason['completeness_enforcement'] = (
            rejected_by_reason.get('completeness_enforcement', 0) + (_before - len(chunk))
        )

        # Rule 2: monetary_integrity (DROP_RECORD)
        chunk['price'] = pd.to_numeric(chunk['price'], errors='coerce')
        _before = len(chunk)
        chunk = chunk[chunk['price'] > 0.0]
        rejected_by_reason['monetary_integrity'] = (
            rejected_by_reason.get('monetary_integrity', 0) + (_before - len(chunk))
        )

        # Rule 3: quantity_validity (EXCLUDE_AND_LOG)
        chunk['qty'] = pd.to_numeric(chunk['qty'], errors='coerce')
        _before = len(chunk)
        _mask_qty = chunk['qty'] > 0
        if (~_mask_qty).any():
            logging.warning(f"Excluded {(~_mask_qty).sum()} rows: invalid or non-positive quantity.")
        chunk = chunk[_mask_qty]
        rejected_by_reason['quantity_validity'] = (
            rejected_by_reason.get('quantity_validity', 0) + (_before - len(chunk))
        )

        # Rule 4: temporal_validity (EXCLUDE_AND_LOG)
        chunk['order_date'] = pd.to_datetime(chunk['order_date'], errors='coerce')
        _before = len(chunk)
        _now = pd.Timestamp.now()
        _mask_date = chunk['order_date'].notna() & (chunk['order_date'] <= _now)
        if (~_mask_date).any():
            logging.warning(f"Excluded {(~_mask_date).sum()} rows: invalid or future order date.")
        chunk = chunk[_mask_date]
        rejected_by_reason['temporal_validity'] = (
            rejected_by_reason.get('temporal_validity', 0) + (_before - len(chunk))
        )

        # Rule 5: location_default (DEFAULT_VALUE 'UNKNOWN')
        _city_valid = chunk['customer_city'].notna() & (chunk['customer_city'].astype(str).str.strip() != '')
        chunk['customer_city'] = chunk['customer_city'].where(_city_valid, other='UNKNOWN')

        # Rule 6: bulk_order_outlier (FLAG_AS_SUSPICIOUS)
        chunk['is_suspicious'] = chunk['qty'] >= 100

        if chunk.empty:
            return i

        # 3c. Type casting — float64 → Int64 for quantity/count columns
        int_cols = [c for c in chunk.select_dtypes(include='float64').columns
                    if any(kw in c.lower() for kw in ['quantity', 'qty', 'count', 'units'])]
        for col in int_cols:
            chunk[col] = chunk[col].astype('Int64')

        # 3d. Write
        chunk.to_parquet(
            f"{partition_uri}part_{i}.parquet",
            engine="pyarrow",
            compression="snappy",
            index=False,
            storage_options=dict()
        )
        logging.info(f"Chunk {i}: {len(chunk)} rows processed")
        total_rows += len(chunk)
        return i + 1

    client = None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=20000)
        cursor = client[mongo_db]["raw_orders"].find({}, batch_size=500)
        batch, i = [], 0
        for doc in cursor:
            batch.append(doc)
            if len(batch) < 500:
                continue
            i = _process(_flatten(batch), i)
            batch = []
        if batch:
            i = _process(_flatten(batch), i)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise
    finally:
        if client is not None:
            client.close()

    rejected_rows = sum(rejected_by_reason.values())
    duration_seconds = time.time() - start_time
    logging.info(f"Pipeline completed. Rows: {total_rows}, rejected: {rejected_rows}, duration: {duration_seconds:.1f}s")

    # ── 4. TRINO PARTITION REGISTRATION ──────────────────────────────────────
    trino_host = os.getenv("TRINO_HOST", "trino.analytics.svc.cluster.local")
    conn = trino_connect(host=trino_host, port=8080, user="pipeline")
    cursor = conn.cursor()

    for _attempt in range(5):
        try:
            cursor.execute("CALL hive.system.sync_partition_metadata('shop_orders', 'pipe_shop_orders_to_s3', 'ADD')")
            cursor.fetchall()
            break
        except Exception as _e:
            if "not found" in str(_e).lower() and _attempt < 4:
                logging.warning(f"Trino table not visible yet (attempt {_attempt + 1}/5) — retrying in 3s.")
                time.sleep(3)
                continue
            raise
    logging.info(f"Trino partition run_date={run_date} registered.")

    # ── 5. METRICS EMISSION ───────────────────────────────────────────────────
    pushgateway_url = os.getenv("PUSHGATEWAY_URL", "http://pushgateway.monitoring.svc.cluster.local:9091")
    project_id = os.getenv("PROJECT_ID", "unknown")
    cloud_provider = os.getenv("CLOUD_PROVIDER", "unknown")

    registry = CollectorRegistry()
    Gauge('pipeline_rows_processed_total', 'Total rows written to storage after business rules',
          ['project_id', 'cloud_provider'], registry=registry) \
        .labels(project_id=project_id, cloud_provider=cloud_provider).set(total_rows)
    Gauge('pipeline_last_success_timestamp', 'Unix timestamp of last successful run',
          ['project_id', 'cloud_provider'], registry=registry) \
        .labels(project_id=project_id, cloud_provider=cloud_provider).set(time.time())
    Gauge('pipeline_rows_rejected_total', 'Rows removed by DROP_RECORD / EXCLUDE_AND_LOG rules',
          ['project_id', 'cloud_provider'], registry=registry) \
        .labels(project_id=project_id, cloud_provider=cloud_provider).set(rejected_rows)
    Gauge('pipeline_duration_seconds', 'Wall-clock duration of the extract-transform-write phase',
          ['project_id', 'cloud_provider'], registry=registry) \
        .labels(project_id=project_id, cloud_provider=cloud_provider).set(duration_seconds)

    rejected_by_reason_gauge = Gauge(
        'pipeline_rows_rejected_by_reason', 'Rows rejected per business rule, labelled by rule name',
        ['project_id', 'cloud_provider', 'reason'], registry=registry)
    for _reason, _count in rejected_by_reason.items():
        rejected_by_reason_gauge.labels(
            project_id=project_id, cloud_provider=cloud_provider, reason=_reason).set(_count)

    push_to_gateway(pushgateway_url, job=project_id, registry=registry)
    logging.info(
        f"Metrics pushed: rows={total_rows}, rejected={rejected_rows}, "
        f"by_reason={rejected_by_reason}, duration={duration_seconds:.1f}s, cloud={cloud_provider}"
    )


if __name__ == "__main__":
    run()
