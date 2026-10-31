# Architecture — As Built

This document describes how the system works in its AWS-only, mini-project form. For setup and the demo, see the [README](../README.md).

---

## 1. The agent graph

The system is a LangGraph state machine with four nodes that share one `AgentState`:

```
SUPERVISOR  ── deterministic router (LLM emits ONE word; invariants live in Python)
    ├── ARCHITECT   ETL script · Trino DDL · dashboard spec · (fix-mode: surgical patches)
    ├── INFRA       Terraform · K8s manifests · Dockerfile · deploy workflow · git push
    └── MEDIC       CI polling · evidence-grounded diagnosis · request_fix handoffs
```

- **Architect** and **Infra** always return to the Supervisor.
- **Medic** runs its tools through a ToolNode: `fetch_github_action_logs`, `query_vector_store`, `request_fix` and `store_architectural_insight`. A successful `request_fix` goes straight back to the Supervisor for a deterministic handoff, with no extra LLM turn.
- The entry point and every routing decision are conditional edges on `state["next_step"]` (see `graph.py`).

### Phase-gated tool control

Agents do not choose their tools freely. Each turn is a gated phase with exactly one permitted tool, `tool_choice="required"`, and pre-computed arguments:

- **Architect:** Discovery → Schema → Implementation, with one artifact per implementation step.
- **Medic:** two tools. `request_fix` needs a verbatim `evidence_quote` from real logs, and the tool **rejects** any quote that contains no recognised error marker.
- **Supervisor:** no tools at all.

---

## 2. Execution model (AWS)

```
RDS PostgreSQL ──pandas (chunked)──▶ business rules ──▶ snappy Parquet on S3
                                                         (Hive-partitioned by run_date)
                                                                │
                                       Trino on EKS ── external table (Glue metastore) + partition sync
                                                                │
                                       Prometheus Pushgateway ──▶ Grafana dashboard + data-silence alert
```

| Layer | Component |
|---|---|
| Compute | EKS (pipeline runs as a Kubernetes `Job`) |
| Storage | S3: `s3://eu-sales-insights-data-princy2312/processed/run_date=<date>/` |
| Registry | ECR |
| Source DB | RDS PostgreSQL (`raw_eu_sales`, pipeline `eu_sales`) or MongoDB Atlas (`raw_orders` documents, pipeline `shop_orders` — flattened with `pd.json_normalize`, schema inferred by `read_data_schema`) |
| Catalog / SQL | Trino `hive` catalog backed by AWS Glue |
| Identity | IRSA: the pipeline's ServiceAccount assumes an IAM role (no static keys in the pod) |
| Image tag | `:${{ github.sha }}`, written into `job.yaml` by the deploy workflow |

The pipeline is **idempotent**: if today's `run_date` partition already exists, the job exits 0 without rewriting it. After writing, it calls `sync_partition_metadata`, so the new partition can be queried immediately.

---

## 3. Credential resolution

`cloud_get()` (in `utils/cloud_config.py`) is the **only** sanctioned way for a generated script to read DB credentials. Using `os.getenv()` for DB variables is a policy violation, and the validator catches it.

It resolves in three tiers: **SSM Parameter Store** (via IRSA) → `.bootstrap_outputs.json` → environment variables (`POSTGRES_DB_*`).

Infrastructure values follow the same rule. For example, the ECR URL has a single source of truth: bootstrap writes it to SSM, and `cloud_get_infra()` reads it back.

---

## 4. The self-healing loop

1. The CI deploy fails because of chaos-seeded data, a defect in the generated code, or an infrastructure error.
2. The **Medic** polls the run with exponential backoff (30 s → 300 s, at most 5 attempts). After that it escalates instead of looping until the recursion limit.
3. Python, not the LLM, parses the message history into a structured **VALIDATION SUMMARY**: FAILED files with their verbatim error text, plus the CLEAN files.
4. `request_fix(file, evidence_quote)` is rejected unless the quote contains a real error marker. Only FAILED files may be named. The node also checks **provenance**: the quote must appear in real tool or log output from this run.
5. The fix goes to the owning agent as a **one-shot `healing_context`** (Architect for code, Infra for infrastructure). Whoever consumes it clears it.
6. In fix mode, the Architect uses `patch_project_file` for a surgical, auto-validated change, never a full rewrite.
7. The agent pushes, the Medic re-polls, and the loop repeats until the run is green. The loop is **bounded**: it stops after 3 identical errors or 8 heal rounds. **Fail-closed**: only `mission_status = "verified"` counts as success.

Before every push, `push_to_github` runs the **security gate** (`policy/security_analyzer.py`). The push is refused if the bundle has any HIGH finding: root containers, inline secrets, public buckets/DBs, wildcard IAM, open ingress, or unpinned actions.

---

## 5. The LLM-vs-deterministic boundary

- **The LLM owns decisions that depend on variable input:** the real source schema, natural-language business rules turned into pandas logic, the Trino DDL, the pipeline Terraform, and log diagnosis.
- **Code owns everything that can be mechanically derived.** `agents/codegen.py` renders these deterministically from config, and they are golden-tested in `tests/goldens/aws/`:
  - `requirements.txt` and the Dockerfile
  - the Grafana dashboard JSON
  - the six K8s manifests
  - the deploy workflow
- **Knowledge base (`knowledge_base/*.md`):** versioned engineering standards that agents retrieve through Pinecone RAG. When generated output is wrong, the **standard** is fixed, never the output.
- **Validator (`validate_generated_code`):** a policy safety net. It checks credential access, literal regions, `${{ ... }}` inside K8s manifests, and dashboard template variables.

---

## 6. CI/CD

- **`run_agent.yml`** (manual dispatch) runs the agent itself. Its steps are an optional AWS bootstrap, knowledge-base sync to Pinecone, chaos injection, and the pipeline run.
- **The generated `pipe_eu_sales_to_s3_pipeline.yml`** is the deploy workflow that the Infra agent writes. It builds and pushes the image to ECR, applies the pipeline Terraform, deploys to EKS, and verifies. It only triggers on pushes that touch **artifact paths**.
- **`tests.yml`** runs the hermetic unit tests, lint, the coverage floor, the offline eval check and the security gate.
- **`cleanup_k8s.yml`** and **`destroy.yml`** handle workload teardown and the full AWS destroy (with a typed confirmation). For pausing, use `make aws-pause` / `make aws-resume`.

---

## 7. Repository zones

| Zone | Paths | Change policy |
|---|---|---|
| Agent system | `agents/`, `graph.py`, `main.py`, `utils/` | Code review + unit tests |
| Standards (RAG corpus) | `knowledge_base/` | Edit, then **re-sync to Pinecone** (`make ingest`) |
| Pipeline definition | `configs/` | Declarative inputs for `eu_sales` |
| One-time baseline | `bootstrap/aws/` | Hand-written Terraform, applied once |
| **Generated outputs** | `scripts/pipe_*.py`, `k8s/`, `sql/`, `dashboards/`, `terraform/`, `Dockerfile`, `requirements.txt`, `.github/workflows/pipe_*.yml` | **Never edited by hand.** Fix them through standards/prompts; the next run regenerates them. |
| Demo surface | `streamlit_app.py`, `utils/nlp_parser.py`, `utils/cost_estimator.py`, `utils/architecture_advisor.py` | NL authoring + cost preview, kept separate from the agent generation path |
