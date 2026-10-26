"""
Reads `terraform output -json` from bootstrap/aws and writes the values into
.bootstrap_outputs.json at the project root (the local-dev tier of utils/cloud_config).

Run this ONCE after `terraform apply` in bootstrap/aws (make bootstrap-aws does it):

    python scripts/export_bootstrap_outputs.py aws
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_DIR = PROJECT_ROOT / "bootstrap"
OUTPUT_FILE = PROJECT_ROOT / ".bootstrap_outputs.json"


# ---------------------------------------------------------------------------
# Core: run terraform output -json
# ---------------------------------------------------------------------------

def _terraform_output(cloud: str) -> dict:
    tf_dir = BOOTSTRAP_DIR / cloud
    if not tf_dir.exists():
        print(f"  ⚠️  bootstrap/{cloud}/ not found — skipping.")
        return {}

    result = subprocess.run(
        ["terraform", "output", "-json"],
        cwd=str(tf_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ❌ terraform output failed for {cloud}:\n{result.stderr[:500]}")
        return {}

    raw = json.loads(result.stdout)
    # terraform output -json wraps each value: {"key": {"value": ..., "type": ...}}
    return {k: v["value"] for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Normalizers: map Terraform output keys → canonical keys used by nlp_parser
# ---------------------------------------------------------------------------

def _ecr_base_url(full_url: str) -> str:
    """'123456789.dkr.ecr.eu-central-1.amazonaws.com/repo' → base URL without repo."""
    return full_url.rsplit("/", 1)[0]


def _normalize_aws(raw: dict) -> dict:
    out = {}
    if v := raw.get("state_bucket_name"):
        out["state_bucket"] = v
    if v := raw.get("lock_table_name"):
        out["lock_table"] = v
    if v := raw.get("ecr_repository_url"):
        out["ecr_repository_url"] = v
        out["ecr_base_url"] = _ecr_base_url(v)
    if v := raw.get("eks_cluster_name"):
        out["eks_cluster_name"] = v
    if v := raw.get("eks_oidc_issuer"):
        out["eks_oidc_issuer"] = v
    if v := raw.get("irsa_role_arn"):
        out["irsa_role_arn"] = v
    # NL/Streamlit-authored pipelines: account id + the shared IRSA role they assume.
    for k in ("aws_account_id", "oidc_provider_arn",
              "pipeline_irsa_role_name", "pipeline_irsa_role_arn"):
        if v := raw.get(k):
            out[k] = v
    for k in ("rds_host", "rds_port", "rds_db_name", "rds_username"):
        if v := raw.get(k):
            out[k] = v
    return out


_NORMALIZERS = {
    "aws": _normalize_aws,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export(clouds: list):
    # Load existing file so we only overwrite the clouds being updated
    existing = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)

    for cloud in clouds:
        print(f"\n🔍 Reading terraform outputs for {cloud.upper()}...")
        raw = _terraform_output(cloud)
        if not raw:
            continue
        normalized = _NORMALIZERS[cloud](raw)
        existing[cloud] = normalized
        print(f"  ✅ {len(normalized)} values captured.")
        for k, v in normalized.items():
            display = str(v)[:60] + "..." if len(str(v)) > 60 else v
            print(f"     {k}: {display}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n✅ Saved to {OUTPUT_FILE}\n")


if __name__ == "__main__":
    # AWS is the only supported cloud; the optional arg is kept for `make bootstrap-aws`.
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "aws"
    if target not in _NORMALIZERS:
        sys.exit(f"Unsupported cloud {target!r} — only 'aws' is supported.")
    export([target])
