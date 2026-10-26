"""
Static monthly cost estimator for the AWS infrastructure this project provisions.

Deterministic, not a live pricing API and not an LLM guess: it enumerates the FIXED footprint
the AWS bootstrap creates (bootstrap/aws) and multiplies by published list prices kept here as
documented constants, so the estimate is explainable line-by-line.

Footprint:  EKS (Auto Mode) + RDS PostgreSQL db.t4g.micro + S3 + ECR + data transfer.

Honesty: list prices, eu-central-1, verified 2026-08. EKS Auto Mode compute is usage-billed, so
this is a typical steady-run estimate, not a fixed 24/7 bill. Excludes data-scan. Real bills vary —
use the AWS Pricing Calculator before making infrastructure decisions. Pausing the baseline
between sessions (`make aws-pause`) removes most of the compute + RDS cost.

    python -m utils.cost_estimator
"""
from __future__ import annotations

PRICES_LAST_UPDATED = "2026-08"
PRICES_DISCLAIMER = (
    "Estimate based on public AWS list prices (verified August 2026), eu-central-1, on-demand. "
    "EKS Auto Mode compute is usage-billed — the figure is a typical steady run, not a guaranteed "
    "24/7 bill. Excludes data-scan. Check the AWS Pricing Calculator before infrastructure decisions."
)

_PRICES: dict[str, float] = {
    "eks_control_plane":    73.0,   # $0.10/hr × 730
    "eks_automode_compute": 140.0,  # general-purpose + system node pools, EC2 + Auto Mode fee
    "rds_t4g_micro":        14.0,   # db.t4g.micro ($0.016/hr) + 20 GB gp3
    "s3_gb":                 0.023,
    "ecr_gb":                0.10,
    "data_transfer_gb":      0.09,
}

_DEFAULT_STORAGE_GB  = 50
_DEFAULT_TRANSFER_GB = 10
_DEFAULT_IMAGE_GB    = 5


def estimate_monthly_cost(cloud: str = "aws", storage_gb: int = _DEFAULT_STORAGE_GB) -> dict:
    """Return the itemised monthly AWS cost estimate ({} for any other cloud)."""
    if cloud != "aws":
        return {}
    p = _PRICES
    items = {
        "EKS Control Plane":                  p["eks_control_plane"],
        "EKS Compute (Auto Mode node pools)": p["eks_automode_compute"],
        "RDS PostgreSQL (db.t4g.micro)":      p["rds_t4g_micro"],
        f"S3 Storage ({storage_gb} GB)":      round(storage_gb * p["s3_gb"], 2),
        f"ECR ({_DEFAULT_IMAGE_GB} GB)":      round(_DEFAULT_IMAGE_GB * p["ecr_gb"], 2),
        f"Data Transfer ({_DEFAULT_TRANSFER_GB} GB out)": round(_DEFAULT_TRANSFER_GB * p["data_transfer_gb"], 2),
    }
    return {
        "cloud": "aws",
        "items": items,
        "total": round(sum(items.values()), 2),
        "last_updated": PRICES_LAST_UPDATED,
        "disclaimer": PRICES_DISCLAIMER,
    }


if __name__ == "__main__":
    _e = estimate_monthly_cost()
    print(f"AWS monthly cost estimate (list prices, {_DEFAULT_STORAGE_GB} GB, {PRICES_LAST_UPDATED}):\n")
    for _k, _v in _e["items"].items():
        print(f"  {_k:38} ${_v:>8.2f}")
    print(f"  {'TOTAL':38} ${_e['total']:>8.2f}")
    print(f"\n{_e['disclaimer']}")
