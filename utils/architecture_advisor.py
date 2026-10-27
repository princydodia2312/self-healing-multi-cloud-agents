"""
Architecture Advisor — takes a plain-English business description and returns a costed
assessment of how the workload maps onto this project's AWS big-data stack
(PostgreSQL → Python ETL on EKS → Parquet on S3 partitioned by run_date → Trino → Grafana).
"""
from __future__ import annotations
import json

from utils.cost_estimator import estimate_monthly_cost
from utils.llm_defaults import NL_MODEL, make_nl_client, nl_api_key

_SYSTEM = """\
You are an expert AWS data architect. A user will describe a data pipeline need in plain
English. The target stack is FIXED: a relational source (PostgreSQL/MySQL on RDS) → a Python
batch ETL job on EKS → Parquet (snappy) on S3 partitioned by run_date → Trino (Hive/Glue
catalog) for SQL → Prometheus + Grafana for pipeline metrics.

Return a JSON object with this exact schema:

{
  "summary": "<one sentence — what this pipeline does>",
  "specs": {
    "data_volume_gb_day": <number, estimate if not stated, default 50>,
    "frequency": "<hourly|daily|weekly|streaming>",
    "region": "<closest AWS region slug, e.g. eu-central-1>",
    "compliance": ["GDPR"|"HIPAA"|"PCI-DSS"|...],
    "source_type": "<PostgreSQL|MySQL|S3|API|Kafka|...>",
    "destination_type": "<data lake|data warehouse|real-time|...>",
    "needs_ml": <true|false>,
    "heavy_transforms": <true|false>
  },
  "pros": ["<2-3 concrete reasons this stack fits THIS workload>"],
  "cons": ["<1-2 concrete limitations of this stack for THIS workload>"],
  "label": "<8 words max>",
  "fit_reason": "<2 sentences: how the workload maps onto the stack and what to watch>"
}

Rules:
- pros/cons must be specific to the described workload, not generic copy
- Mention partitioning, file format or query patterns where relevant
- Return ONLY the JSON object, no markdown, no commentary
"""


def analyze(description: str) -> dict:
    api_key = nl_api_key()
    if api_key:
        try:
            raw    = _call_gpt(description, api_key)
            parsed = json.loads(raw)
            return _build_report(parsed)
        except Exception:
            pass
    return _heuristic_report(description)


def _call_gpt(description: str, api_key: str) -> str:
    client = make_nl_client()
    resp = client.chat.completions.create(
        model=NL_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": description},
        ],
        temperature=0.2,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _option(vol_gb: int, pros: list, cons: list, label: str) -> dict:
    est = estimate_monthly_cost("aws", storage_gb=vol_gb)
    return {"cloud": "aws", "total": est["total"], "items": est["items"],
            "pros": pros, "cons": cons, "label": label}


def _build_report(parsed: dict) -> dict:
    specs  = parsed.get("specs", {})
    vol_gb = int(specs.get("data_volume_gb_day", 50))
    return {
        "summary":    parsed.get("summary", ""),
        "specs":      specs,
        "option":     _option(vol_gb, parsed.get("pros", []), parsed.get("cons", []),
                              parsed.get("label", "")),
        "fit_reason": parsed.get("fit_reason", ""),
    }


def _heuristic_report(description: str) -> dict:
    desc_lower = description.lower()
    vol_gb     = 50
    compliance = ["GDPR"] if any(w in desc_lower for w in ["eu", "europe", "gdpr", "pii"]) else []
    needs_ml   = any(w in desc_lower for w in ["ml", "model", "predict", "train", "machine learning"])
    heavy_etl  = any(w in desc_lower for w in ["transform", "aggregate", "join", "spark", "warehouse"])
    streaming  = any(w in desc_lower for w in ["real-time", "realtime", "streaming", "kafka"])

    cons = ["Batch-oriented: not built for sub-minute latency"] if streaming else \
           ["Single-node pandas ETL — very large daily volumes need chunking or Spark"]
    if needs_ml:
        cons.append("No managed ML tooling in the stack — train models downstream of S3")
    reason = (
        "Source rows are extracted by a Python job on EKS, cleaned by the business rules, and "
        "written as snappy Parquet to S3 partitioned by run_date, so Trino scans only the days "
        "a query needs. "
        + ("Heavy transforms should be pushed into Trino SQL over the Parquet layer."
           if heavy_etl else "Grafana tracks rows processed/rejected per run for data quality.")
    )
    return {
        "summary": description[:100],
        "specs":   {"data_volume_gb_day": vol_gb, "frequency": "daily",
                    "region": "eu-central-1", "compliance": compliance,
                    "source_type": "PostgreSQL", "destination_type": "data lake",
                    "needs_ml": needs_ml, "heavy_transforms": heavy_etl},
        "option":  _option(
            vol_gb,
            ["Columnar Parquet + run_date partitions keep Trino scans cheap",
             "S3 decouples storage from compute",
             "Business rules quarantine bad rows before they reach the lake"],
            cons,
            "S3 data lake + Trino SQL on EKS",
        ),
        "fit_reason": reason,
    }
