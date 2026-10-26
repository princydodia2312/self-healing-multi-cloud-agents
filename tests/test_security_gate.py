"""Adversarial proof that the generated-infrastructure security gate holds.

The gate must PASS the clean AWS goldens (what codegen renders for eu_sales) and REFUSE the
deliberately-unsafe fixtures — one violation per HIGH rule.
"""
from pathlib import Path

import pytest

from policy.security_analyzer import analyze, derive_findings, extract_context

_ROOT = Path(__file__).resolve().parent.parent
_CLEAN = _ROOT / "tests" / "goldens" / "aws"
_UNSAFE = _ROOT / "policy" / "fixtures" / "unsafe"

_EXPECTED_UNSAFE_RULES = {
    "DOCKERFILE_ROOT_USER",
    "DOCKERFILE_COPIES_ENV",
    "K8S_INLINE_SECRET",
    "IMAGE_PUBLIC_LATEST",
    "WORKFLOW_INLINE_SECRET",
    "TF_PUBLIC_DB",
    "TF_IAM_WILDCARD_RESOURCE",
    "TF_OPEN_INGRESS",
    "TF_PUBLIC_BUCKET_ACL",
    "K8S_HOST_NETWORK",
    "K8S_PRIVILEGED",
    "WORKFLOW_UNPINNED_ACTION",
}


def test_clean_goldens_pass_the_gate():
    result = analyze(_CLEAN)
    highs = [f for f in result["findings"] if f["severity"] == "HIGH"]
    assert result["passed"] is True, f"clean AWS goldens should pass; got HIGH findings: {highs}"
    assert result["high_count"] == 0


def test_unsafe_fixture_is_refused():
    result = analyze(_UNSAFE)
    assert result["passed"] is False
    fired = {f["rule"] for f in result["findings"] if f["severity"] == "HIGH"}
    assert fired == _EXPECTED_UNSAFE_RULES, f"unsafe fixture should trip every HIGH rule; got {fired}"


@pytest.mark.parametrize("rule", sorted(_EXPECTED_UNSAFE_RULES))
def test_each_high_rule_fires_on_the_unsafe_fixture(rule):
    fired = {f.rule for f in derive_findings(extract_context(_UNSAFE)) if f.severity == "HIGH"}
    assert rule in fired


def test_resolve_from_tf_sentinel_image_is_not_flagged(tmp_path):
    # The pre-sed job.yaml image sentinel is a valid placeholder, not a public :latest image.
    from policy.security_analyzer import RESOLVE_FROM_TF

    k8s = tmp_path / "k8s"
    k8s.mkdir()
    (k8s / "job.yaml").write_text(
        "apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: j\nspec:\n  template:\n    spec:\n"
        f"      containers:\n        - name: pipeline\n          image: {RESOLVE_FROM_TF}\n"
    )
    fired = {f["rule"] for f in analyze(tmp_path)["findings"] if f["severity"] == "HIGH"}
    assert "IMAGE_PUBLIC_LATEST" not in fired
