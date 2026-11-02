import os

# 1. Project Root Discovery (Absolute path to project base)
# This goes two levels up from agents/constants.py to reach the root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Prompt Files Names
ARCHITECT_PROMPT_FILE = "architect.md"
INFRA_PROMPT_FILE = "infra.md"
MEDIC_PROMPT_FILE = "medic.md"
SUPERVISOR_PROMPT_FILE = "supervisor.md"

# 3. Directory Paths
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
INFRA_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "infra")
PROMPTS_DIR = os.path.join(BASE_DIR, "agents", "prompts")

# 4. Global Settings
TEMPERATURE = 0

# Canonical knowledge-base queries. These used to be issued by the LLM (it was told the exact
# strings in agents/prompts/architect.md); now Python issues them directly, so discovery costs
# zero LLM calls. Keep the strings identical to the prompt so retrieval stays the same.
KB_QUERIES = {
    "arch_standard_python": (
        "STANDARD PYTHON DATA PIPELINES CRITICAL RULES: cloud_get mandatory credentials, "
        "storage_options to_parquet, create_engine extraction loop same try block, "
        "FLAG_AS_SUSPICIOUS is_suspicious quality_standards pandas, type casting float64 Int64 "
        "quantity columns, destination_uri os.getenv DESTINATION_URI, idempotency partition "
        "run_date, push_to_gateway prometheus_client, requirements.txt repo root"
    ),
    "arch_standard_trino": (
        "STANDARD TRINO DDL GENERATION setup_trino.sql: CREATE SCHEMA DROP TABLE CREATE TABLE "
        "3-part catalog.schema.table, external_location PARQUET partitioned_by ARRAY run_date, "
        "data types VARCHAR DECIMAL INTEGER TIMESTAMP BOOLEAN, hive external table s3 gs abfss protocol"
    ),
    "arch_standard_mongodb": (
        "MONGODB EXTRACTION STANDARD semi-structured source: pymongo batch flatten "
        "json_normalize record_path ObjectId mixed types"
    ),
}


def kb_result_usable(result) -> bool:
    """True when a query_vector_store result actually carries a standard (not an error / empty)."""
    text = str(result or "").strip().lower()
    return bool(text) and not any(bad in text for bad in (
        "no relevant guidelines found", "pinecone not initialized", "error querying"))

DEFAULT_REQUIRED_ARTIFACTS = [
    "scripts/{pipeline_name}.py",
    "sql/setup_trino.sql",
    "dashboards/monitoring_specs.json",
    "requirements.txt",
]

# Fallback when infrastructure config omits required_k8s_manifests.
DEFAULT_REQUIRED_K8S_MANIFESTS = [
    "k8s/00_namespaces.yaml",
    "k8s/trino_deployment.yaml",
    "k8s/grafana_deployment.yaml",
    "k8s/prometheus_deployment.yaml",
    "k8s/configmaps.yaml",
    "k8s/job.yaml",
]

# Required Terraform files for Databricks deployments.
# No K8s manifests or Dockerfile — Databricks manages its own compute.
DEFAULT_REQUIRED_DATABRICKS_TF_FILES = [
    "terraform/providers.tf",
    "terraform/main.tf",
    "terraform/variables.tf",
    "terraform/outputs.tf",
    "terraform/terraform.tfvars",
]

# LLM timeout / retries — read directly by llm_factory.py from env vars.
# Defined here only as documentation; do not import these — use os.getenv() in llm_factory.
# LLM_TIMEOUT_SEC  (default: 120s)
# LLM_MAX_RETRIES  (default: 1)

# Pinned image versions for K8s manifests.
# Single source of truth — keep in sync with knowledge_base/infrastructure/k8s_deployment_rules.md.
# Used by: validate_generated_code (enforcement), the infra.py orchestration phase-instruction,
# and generate_k8s_manifest docstring (guidance).
K8S_PINNED_IMAGES: dict[str, str] = {
    "trinodb/trino":    "trinodb/trino:403",
    "grafana/grafana":  "grafana/grafana:10.4.2",
    "prom/prometheus":  "prom/prometheus:v2.51.0",
    "prom/pushgateway": "prom/pushgateway:v1.8.0",
}
