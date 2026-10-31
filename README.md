# Self-Healing Big Data Pipeline on AWS

**Big Data mini project.** A team of AI agents builds a batch Big Data pipeline on AWS, deploys it, and repairs it when the deployment fails.

| | |
|---|---|
| **Team** | _Member 1_ · _Member 2_ · _Member 3_ |
| **Subject** | Big Data Analytics — mini project |
| **Stack** | Python · pandas · PyArrow/Parquet · PostgreSQL (RDS) · Amazon S3 · Trino · Prometheus + Grafana · Kubernetes (EKS) · Terraform · GitHub Actions · LangGraph · Gemini (free) or OpenAI |

---

## 1. Problem statement

Building a data pipeline takes many separate pieces of work: ETL code, a storage layout, a SQL catalog, infrastructure, CI/CD, and monitoring. Pipelines also break, usually because of dirty source data or a configuration mistake, and someone has to read the logs and fix them.

This project automates both parts:

1. **Build.** Starting from a YAML config (or a plain-English description), agents generate a working batch pipeline:
   - extract sales orders from **PostgreSQL**
   - apply **data-quality rules**
   - write **Parquet partitioned by `run_date`** to an **S3 data lake**
   - register the data as a **Trino** table for SQL analytics
   - publish run metrics to **Grafana**
2. **Self-heal.** When the deployment fails, a *Medic* agent reads the real CI logs, finds the faulty file using quoted evidence, sends the problem to the agent that owns that file, patches it, and redeploys.

## 2. Big Data pipeline (what gets built)

```mermaid
flowchart LR
    SRC[("RDS PostgreSQL<br/>raw_eu_sales<br/>(dirty data)")] -->|chunked extract| ETL
    subgraph EKS["Kubernetes job on EKS"]
        ETL["Python ETL<br/>pandas · business rules"]
    end
    ETL -->|rejected rows counted per rule| PG["Prometheus<br/>Pushgateway"]
    ETL -->|snappy Parquet| S3[("S3 data lake<br/>processed/run_date=YYYY-MM-DD/")]
    S3 --> TR["Trino<br/>hive.sales_eu.pipe_eu_sales_to_s3<br/>(Glue metastore)"]
    PG --> GR["Grafana dashboard<br/>rows · rejections · duration"]
    TR --> BI["SQL analytics<br/>(Streamlit 'Trino SQL' tab)"]
```

| Big Data concept | How it is implemented |
|---|---|
| **Ingestion** | Chunked SQLAlchemy reads from RDS PostgreSQL, and batched `pymongo` reads from MongoDB (500 documents at a time). The whole source is never loaded into memory at once. |
| **Variety (structured + semi-structured)** | Two sources: a fixed-schema PostgreSQL table (`eu_sales`) and schema-less nested JSON documents in MongoDB (`shop_orders`, see §3b). |
| **Data quality / cleansing** | Six rules in [`configs/business_rules/sales_logic.yaml`](configs/business_rules/sales_logic.yaml), compiled into real pandas conditions (see §3) |
| **Columnar storage** | Apache **Parquet** with **snappy** compression (PyArrow + s3fs) |
| **Partitioning** | Hive-style `s3://eu-sales-insights-data-princy2312/processed/run_date=<date>/` |
| **Idempotency** | If today's `run_date` partition already exists, the job exits without rewriting it |
| **Schema-on-read / SQL engine** | Trino external table over S3 (`sql/setup_trino.sql`) with the Glue metastore. The partitions are synced, so a filter on `run_date` only scans the matching S3 prefixes (partition pruning). |
| **Observability** | The job pushes `rows_processed`, `rows_rejected` (per rule) and `duration_seconds` to Prometheus Pushgateway. A Grafana dashboard plus a 60-minute *data silence* alert are generated with the pipeline. |
| **Infrastructure as code** | Terraform for the S3 bucket and IAM (IRSA). Kubernetes manifests for Trino, Grafana, Prometheus and the pipeline Job. |

The generated artifacts land in the repo under `scripts/`, `sql/`, `dashboards/`, `k8s/`, `terraform/`, `Dockerfile` and `.github/workflows/pipe_*.yml`. Snapshots of the deterministic artifacts are pinned as test goldens in [`tests/goldens/aws/`](tests/goldens/aws/).

## 3. Data-quality rules

The source table is deliberately polluted by [`scripts/seed_chaos.py`](scripts/seed_chaos.py) with negative prices, NULL quantities, future dates and unknown currencies. Each row is then checked against these rules:

| Rule | Target | Logic | On failure |
|---|---|---|---|
| `monetary_integrity` | price / amount / cost columns | value > 0 | `DROP_RECORD` |
| `temporal_validity` | order date | ≤ now | `EXCLUDE_AND_LOG` |
| `completeness_enforcement` | `order_id` | not NULL / empty | `DROP_RECORD` |
| `currency_standardization` | `currency` | ∈ {EUR, GBP} | `DEFAULT_VALUE` → `EUR` |
| `volume_sanity_check` | `quantity` | < 1000 | `FLAG_AS_SUSPICIOUS` |
| `quantity_validity` | `quantity` | > 0 | `FLAG_AS_SUSPICIOUS` |

Rejection counts per rule are exported as metrics, so data quality shows up in Grafana.

## 3b. Second pipeline: MongoDB → S3 (`shop_orders`, semi-structured data)

A MongoDB collection has no fixed schema. Each order document nests a customer object and holds a list of items, and the seeder makes them inconsistent on purpose:

```json
{ "_id": ObjectId("6721a9f..."), "order_id": "ORD-482193", "order_date": "2026-10-20T11:02:00",
  "customer": { "name": "Asha", "email": "...", "city": "Pune" },
  "items": [ { "product": "Laptop", "price": 55000, "qty": 1 },
             { "product": "Mouse",  "price": "500", "qty": 2 } ] }
```

The pipeline reads the collection in batches, **flattens** every document into one row per item (`pd.json_normalize` with `record_path="items"`), cleans it, and writes the same partitioned Parquet layout as `eu_sales`:

| order_id | order_date | customer_name | customer_city | product | price | qty | is_suspicious |
|---|---|---|---|---|---|---|---|
| ORD-482193 | 2026-10-20 | Asha | Pune | Laptop | 55000.0 | 1 | false |
| ORD-482193 | 2026-10-20 | Asha | Pune | Mouse | 500.0 | 2 | false |

Before generating code, the Architect calls `read_data_schema(db_type="mongodb")`. This samples 200 documents and **infers** the schema: every field path, the types seen, how often it is present, and warnings such as "MIXED TYPES" or "BSON ObjectId" ([`utils/mongo_schema.py`](utils/mongo_schema.py)). The extraction rules are in [`knowledge_base/engineering/mongodb_extraction_standard.md`](knowledge_base/engineering/mongodb_extraction_standard.md).

Rules ([`configs/business_rules/shop_orders_logic.yaml`](configs/business_rules/shop_orders_logic.yaml)), applied to the flattened rows:

| Rule | Target | Logic | On failure | Messy data the seeder injects |
|---|---|---|---|---|
| `completeness_enforcement` | `order_id`, `product` | present, not empty | `DROP_RECORD` | missing order IDs / product names |
| `monetary_integrity` | `price` | numeric (text converted first) and > 0 | `DROP_RECORD` | `-10`, `0`, `"abc"`, `"500"` |
| `quantity_validity` | `qty` | > 0 | `EXCLUDE_AND_LOG` | `-1`, `0`, `"2"`, missing |
| `temporal_validity` | `order_date` | valid date, not in the future | `EXCLUDE_AND_LOG` | `"2030-01-01"`, `"not-a-date"`, missing |
| `location_default` | `customer_city` | present | `DEFAULT_VALUE` → `UNKNOWN` | customers without a city |
| `bulk_order_outlier` | `qty` | < 100 | `FLAG_AS_SUSPICIOUS` | quantities of 500 |

The messy documents also produce new failures for the self-healing loop: an `ObjectId` reaching Parquet (`ArrowTypeError`), a field missing from a whole batch (`KeyError`), or text compared with numbers (`TypeError`). The eval corpus includes the first two.

MongoDB runs on the free **MongoDB Atlas M0** tier, so it costs nothing. The connection string is stored in SSM as an encrypted SecureString and read with `cloud_get`, exactly like the PostgreSQL credentials.

## 4. The agents and the self-healing loop

```mermaid
flowchart TD
    U["Pipeline YAML<br/>or plain-English description"] --> S
    subgraph G["LangGraph state machine"]
        S["🧭 Supervisor<br/>deterministic router"]
        A["📐 Architect<br/>ETL code · Trino DDL · dashboard"]
        I["🏗️ Infra<br/>Terraform · K8s · CI/CD · git push"]
        M["🩺 Medic<br/>evidence-grounded diagnosis"]
        S --> A --> S
        S --> I --> S
        S --> M
        M -- "request_fix(file, evidence_quote)" --> S
    end
    KB[("Pinecone<br/>engineering standards (RAG)")] -. retrieval .-> A & I & M
    I -- "git push" --> GH["GitHub Actions"]
    GH -- "build · terraform · deploy" --> AWS["AWS: EKS · S3 · ECR"]
    GH -- "CI logs" --> M
```

- **Supervisor**: a deterministic router. The routing rules are Python code, not prompt text.
- **Architect**: generates the parts that need judgment: the ETL script and the Trino DDL. It first inspects the real source schema, then writes code that follows the standards it retrieves from the knowledge base ([`knowledge_base/`](knowledge_base/)).
- **Infra**: generates the pipeline Terraform. The Dockerfile, Kubernetes manifests and deploy workflow are rendered deterministically from config by [`agents/codegen.py`](agents/codegen.py). It then pushes everything and triggers CI.
- **Medic**: polls the CI run and turns the failure into a structured summary. It may only call `request_fix` with a **verbatim quote** from real log output. Speculation is rejected, which acts as an anti-hallucination gate. The quoted file is then patched surgically.

```mermaid
sequenceDiagram
    participant CI as GitHub Actions
    participant M as 🩺 Medic
    participant S as 🧭 Supervisor
    participant O as Owning agent (Architect / Infra)
    CI-->>M: failing logs
    M->>M: structured summary + evidence gate
    alt real, quoted evidence
        M->>S: request_fix(file, evidence, owner)
        S->>O: healing context
        O->>O: patch file + auto-validate
        O->>CI: re-push → redeploy
        CI-->>M: re-poll → mission_status = "verified" ✅
    else no convergence (3 identical errors / 8 rounds)
        M-->>S: mission_status = "escalated" ❌ (exit 1)
    end
```

Safety limits:
- **Bounded:** the healing loop stops after 3 identical errors or 8 rounds.
- **Fail-closed:** a run counts as successful only when its status is `verified`.
- **Pre-push security gate:** before every push, [`policy/security_analyzer.py`](policy/security_analyzer.py) blocks generated bundles that contain:
  - containers running as root
  - inline secrets
  - public buckets or databases
  - wildcard IAM
  - unpinned GitHub Actions

## 5. Repository layout

| Path | Contents |
|---|---|
| `agents/`, `graph.py`, `main.py` | The multi-agent system (LangGraph nodes, tools, prompts, deterministic codegen) |
| `configs/` | The `eu_sales` (PostgreSQL) and `shop_orders` (MongoDB) pipelines: objective, source DB, business rules, S3 target |
| `knowledge_base/` | Engineering standards used for RAG (Python, MongoDB extraction, SQL, Grafana, CI/CD, Terraform-S3, Docker, K8s) |
| `bootstrap/aws/` | One-time baseline Terraform: EKS, RDS PostgreSQL, ECR, S3 state, SSM parameters |
| `scripts/` | `seed_chaos.py` (dirty rows for PostgreSQL, messy documents for MongoDB), `ingest_to_pinecone.py` (RAG sync), `export_bootstrap_outputs.py` |
| `utils/` | Config/credential resolution, MongoDB schema inference, rules loader, Trino client, observability, NL parser, cost estimate |
| `streamlit_app.py` | Web UI: describe a pipeline in plain English, upload a dataset, run the sample pipeline, query Trino, view metrics |
| `evals/` | Offline self-heal eval: a corpus of 12 real failure classes scored against the real Medic routing + evidence gate |
| `policy/` | Security gate for generated infrastructure + unsafe test fixtures |
| `tests/` | Hermetic unit tests (no cloud, no credentials) |

## 6. How to run

**Prerequisites:**
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Terraform ≥ 1.6 and the AWS CLI
- an LLM API key: a **free Google Gemini key** (default, see below) or an OpenAI key
- a Pinecone index (dimension 1536, cosine; Pinecone's starter plan is free)
- an AWS account
- a GitHub token with `repo` and `workflow` scopes

```bash
cp .env.example .env      # fill in keys
make install              # uv sync
make bootstrap-aws        # one-time: EKS + RDS + ECR + S3 state + SSM (exports outputs)
make demo                 # ingest standards → seed 100 dirty rows → run the eu_sales pipeline
```

Or run each step yourself:

```bash
make ingest               # sync knowledge_base/ to Pinecone
make chaos rows=200       # seed dirty rows into raw_eu_sales
make run p=eu_sales       # agents build, deploy and (if needed) heal the pipeline
make streamlit            # web UI on http://localhost:8501
```

**MongoDB pipeline (`shop_orders`).** Create a free Atlas M0 cluster (AWS, Frankfurt), add `MONGO_URI` and `MONGO_DB_NAME=shop_raw` to `.env`, then:

```bash
make mongo-ssm                 # store the connection string in SSM (encrypted) for the EKS job
make chaos target=shop_orders  # insert 500 messy order documents into shop_raw.raw_orders
make run p=shop_orders         # agents build, deploy and heal the MongoDB → S3 pipeline
```

Run one pipeline at a time. Both pipelines share one cluster and one set of generated files (`scripts/`, `sql/`, `k8s/`, ...), so the most recent run's files are the ones in the repo.

The same steps are available from GitHub Actions in `run_agent.yml` (manual dispatch with bootstrap, knowledge-base sync, chaos and pipeline inputs).

**Free LLM option (Google Gemini).** The project runs on the Gemini API free tier as well as on OpenAI. One setting in `.env` switches the agents, the Streamlit UI and the RAG embeddings together:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=...          # free key from https://aistudio.google.com/apikey
# defaults: agents gemini-3.8-flash, falling back to gemini-3.6-flash → gemini-2.5-flash → gemini-3.5-flash-lite
#           NL/UI gemini-3.5-flash-lite · embeddings gemini-embedding-2 (1536-dim)
```

To use OpenAI instead, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`. After switching provider, run `make ingest` again, because embeddings from different models can't be mixed in one index. The free tier allows only about **20 requests per day per model** (reset at midnight Pacific time; your limits are shown in AI Studio), while one agent run needs roughly 40–80 model calls. That's why the agents fall back through several models, each with its own daily quota. Google may use free-tier prompts to improve its products.

> **Cost warning.** The AWS baseline (EKS + RDS) bills by the hour, about $230/month while running (`make cost`). Use `make aws-pause` between sessions and `make aws-resume` before a demo. Tear everything down with `make destroy-aws` or the `destroy.yml` workflow after the evaluation.

**Querying the data lake:**

```bash
kubectl port-forward svc/trino 8080:8080 -n analytics   # then TRINO_HOST=localhost in .env
```

```sql
SELECT run_date, currency, COUNT(*) AS orders, ROUND(SUM(unit_price * quantity), 2) AS revenue
FROM hive.sales_eu.pipe_eu_sales_to_s3
WHERE run_date >= DATE '2026-05-01'          -- partition pruning
GROUP BY run_date, currency
ORDER BY run_date DESC;
```

## 7. Self-healing demo (for the viva)

1. Run `make chaos` to load dirty data, then `make run p=eu_sales`.
2. Watch the log, or the **Agent graph** in Streamlit. The Architect generates the ETL, and `validate_generated_code` checks it.
3. If the deploy fails, the Medic quotes the failing line from the CI log, for example:

   ```
   ValueError: could not convert string to float: 'not_a_number'
   ```

   It routes the fix to the Architect, which patches the file (for example `.astype(float)` → `pd.to_numeric(..., errors="coerce")`). Then the run redeploys.
4. The run ends with `✅ MISSION VERIFIED`. The S3 bucket now contains `run_date=…/` Parquet files, Trino can query them, and the Grafana dashboard shows rows processed vs rejected.
5. To show the Medic's routing without any cloud, run `make heal LOG=<some_failing_log>.txt`.

## 8. Testing and evaluation

```bash
make test          # hermetic unit tests — no cloud, no credentials
make lint          # ruff
make eval-replay   # offline self-heal eval → evals/report/REPORT.md
make gate-proof    # security gate refuses unsafe fixtures, passes the AWS goldens
```

The test suite covers:
- supervisor routing
- the generated-code validators
- golden-file tests for every deterministic artifact (Dockerfile, K8s manifests, Grafana dashboard, deploy workflow)
- credential resolution
- MongoDB schema inference, messy-document generation and the `shop_orders` config
- the state lifecycle

The eval harness replays 12 failure cases (script bugs including two MongoDB ones, Trino DDL errors, S3/IAM and SSM failures, and hallucination traps) against the real Medic routing and evidence gate. Both currently score 100%.

## 9. Limitations and future work

- **Batch only:** no streaming ingestion (Kafka/Kinesis would be the next step).
- **Single-node processing:** the ETL uses pandas, so very large volumes would need Spark (EMR) or pushing transforms into Trino.
- **Demo-grade networking:** the RDS endpoint is public but restricted to allow-listed IP ranges (CIDR), and Grafana is on a password-protected public load balancer. Trino is internal-only and has no authentication.
- **The LLM is not unit-tested:** generated-code quality is checked by the validators, the eval harness and real runs, not by asserting on model output.

## Credits and license

This project is a scoped-down, AWS-only adaptation of [`self-healing-multi-cloud-agents`](https://github.com/theofanis-tsakanikas/self-healing-multi-cloud-agents). MIT License, see [LICENSE](LICENSE). Architecture details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
