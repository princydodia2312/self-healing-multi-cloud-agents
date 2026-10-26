"""Golden-file + invariant tests for the deterministic artifact generators (AWS).

Goldens in tests/goldens/aws/ are snapshots of the artifacts agents/codegen.py
renders for the eu_sales pipeline (configs/pipelines/eu_sales_pipeline.yaml):
Dockerfile, requirements, Grafana dashboard, the K8s manifests (Trino, Grafana,
Prometheus, pipeline Job, ConfigMaps) and the generated GitHub Actions deploy
workflow. Any unintended change to the generators fails a test here.
"""
import json
from pathlib import Path

import pytest
import yaml

from agents import codegen

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).parent / "goldens" / "aws"

AWS_CONF = yaml.safe_load(
    (ROOT / "configs" / "pipelines" / "eu_sales_pipeline.yaml").read_text(encoding="utf-8"))
REGISTRY = "000000000000.dkr.ecr.eu-central-1.amazonaws.com/eu-sales-pipeline-repo"
# The Trino DDL embedded in the configmaps golden.
SQL = "CREATE SCHEMA IF NOT EXISTS hive.sales_eu;\n"


def _read(rel: str) -> str:
    return (GOLDEN / rel).read_text(encoding="utf-8")


def _docs(text: str):
    return [d for d in yaml.safe_load_all(text) if d is not None]


class TestDockerfile:
    def test_matches_golden(self):
        rendered = codegen.render_dockerfile("scripts/pipe_eu_sales_to_s3.py")
        assert rendered == _read("Dockerfile")

    def test_pins_python_312(self):
        assert "python:3.12-slim" in codegen.render_dockerfile("scripts/x.py")


class TestRequirements:
    def test_matches_golden(self):
        assert codegen.render_requirements("aws", "postgres") == _read("requirements.txt")

    def test_aws_postgres_drivers(self):
        lines = codegen.render_requirements("aws", "postgres").splitlines()
        assert lines[:5] == ["pandas", "sqlalchemy", "pyarrow", "trino", "prometheus-client"]
        assert "boto3" in lines and "s3fs" in lines and "psycopg2-binary" in lines


class TestMonitoringSpecs:
    def test_matches_golden(self):
        rendered = json.loads(codegen.render_monitoring_specs(AWS_CONF, "aws"))
        assert rendered == json.loads(_read("dashboards/monitoring_specs.json"))

    def test_uid_stable_and_slug_derived(self):
        spec = json.loads(codegen.render_monitoring_specs(AWS_CONF, "aws"))
        assert spec["uid"] == "eu-sales-data-observability"
        assert spec["tags"] == ["data-pipeline", "eu-sales", "aws"]


class TestK8sGoldens:
    @pytest.mark.parametrize("golden,render", [
        ("k8s/00_namespaces.yaml", codegen.render_namespaces),
        ("k8s/trino_deployment.yaml", codegen.render_trino_deployment),
        ("k8s/grafana_deployment.yaml", codegen.render_grafana_deployment),
        ("k8s/prometheus_deployment.yaml", codegen.render_prometheus_deployment),
    ])
    def test_shared_services_match_golden(self, golden, render):
        assert _docs(render(AWS_CONF, "aws")) == _docs(_read(golden))

    def test_grafana_has_admin_secret_and_nlb(self):
        deploy, svc = _docs(codegen.render_grafana_deployment(AWS_CONF, "aws"))[:2]
        env = deploy["spec"]["template"]["spec"]["containers"][0]["env"]
        admin = [e for e in env if e["name"] == "GF_SECURITY_ADMIN_PASSWORD"]
        assert admin and admin[0]["valueFrom"]["secretKeyRef"]["name"] == "grafana-admin"
        assert svc["metadata"]["annotations"][
            "service.beta.kubernetes.io/aws-load-balancer-scheme"] == "internet-facing"

    def test_job_matches_golden(self):
        assert _docs(codegen.render_job(AWS_CONF, "aws", REGISTRY)) == _docs(_read("k8s/job.yaml"))

    def test_job_writes_parquet_to_s3(self):
        doc = _docs(codegen.render_job(AWS_CONF, "aws", REGISTRY))[0]
        env = {e["name"]: e.get("value") for e in doc["spec"]["template"]["spec"]["containers"][0]["env"]}
        assert env["DESTINATION_URI"] == "s3://eu-sales-insights-data/processed/"
        assert env["CLOUD_PROVIDER"] == "aws"

    def test_configmaps_match_golden(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "sql").mkdir()
        (tmp_path / "dashboards").mkdir()
        (tmp_path / "sql" / "setup_trino.sql").write_text(SQL, encoding="utf-8")
        (tmp_path / "dashboards" / "monitoring_specs.json").write_text(
            codegen.render_monitoring_specs(AWS_CONF, "aws"), encoding="utf-8")
        rendered = _docs(codegen.render_configmaps(AWS_CONF, "aws"))
        assert rendered == _docs(_read("k8s/configmaps.yaml"))

    def test_hive_uses_glue_metastore(self):
        hive = codegen._hive_properties(AWS_CONF, "aws")
        assert "hive.metastore=glue" in hive and "hive.metastore.glue.region=eu-central-1" in hive


class TestWorkflow:
    def test_matches_golden(self):
        rendered = codegen.render_workflow(AWS_CONF, "aws", REGISTRY, is_databricks=False)
        assert yaml.safe_load(rendered) == yaml.safe_load(_read("workflows/pipe_eu_sales_to_s3_pipeline.yml"))

    def test_invariants(self):
        wf = codegen.render_workflow(AWS_CONF, "aws", REGISTRY, is_databricks=False)
        parsed = yaml.safe_load(wf)
        # Least privilege + narrowed trigger (never scripts/**)
        assert parsed["permissions"] == {"contents": "read"}
        assert "scripts/pipe_*.py" in parsed[True]["push"]["paths"]  # YAML parses `on` as True
        assert "scripts/**" not in parsed[True]["push"]["paths"]
        steps = [s.get("name") for s in parsed["jobs"]["deploy"]["steps"]]
        assert steps.index("Create Grafana Admin Secret") < steps.index("Deploy Shared Services to Kubernetes")
        assert "Deploy Pipeline Job to Kubernetes" in steps and "Check Deployment Status" in steps
        assert "Set Image Tag in Job Manifest" in steps
        assert 'echo "Deployment Complete"' in parsed["jobs"]["deploy"]["steps"][-1]["run"]
        # Region comes from the GitHub Variable, never a literal.
        assert "aws-region: ${{ vars.AWS_DEFAULT_REGION }}" in wf
        assert "aws-region: eu-central-1" not in wf

    def test_job_image_equals_workflow_build_target(self):
        job = _docs(codegen.render_job(AWS_CONF, "aws", REGISTRY))[0]
        assert job["spec"]["template"]["spec"]["containers"][0]["image"] == f"{REGISTRY}:latest"
        wf = codegen.render_workflow(AWS_CONF, "aws", REGISTRY, is_databricks=False)
        assert f"docker build -t {REGISTRY}:" in wf
        assert f"docker push {REGISTRY}:" in wf
        assert f"image: {REGISTRY}" in wf


class TestEnsureOrchestrators:
    """Behavioral: writes + validates via the REAL validate_generated_code safety net."""

    def test_architect_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        written, errors = codegen.ensure_architect_artifacts(
            AWS_CONF, {"db_type": "postgres"}, {"provider": "kubernetes"}, [])
        assert errors == []
        assert sorted(written) == ["dashboards/monitoring_specs.json", "requirements.txt"]
        # Idempotent: nothing re-written when already tracked.
        again, _ = codegen.ensure_architect_artifacts(
            AWS_CONF, {"db_type": "postgres"}, {"provider": "kubernetes"}, written)
        assert again == []

    def test_infra_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CLOUD_PROVIDER", "aws")
        monkeypatch.setattr(codegen, "REPO_ROOT", tmp_path)
        (tmp_path / "sql").mkdir()
        (tmp_path / "dashboards").mkdir()
        (tmp_path / "sql" / "setup_trino.sql").write_text("SELECT 1;", encoding="utf-8")
        (tmp_path / "dashboards" / "monitoring_specs.json").write_text(
            codegen.render_monitoring_specs(AWS_CONF, "aws"), encoding="utf-8")
        written, errors = codegen.ensure_infra_artifacts(
            AWS_CONF, {"provider": "kubernetes"}, REGISTRY, [])
        assert errors == []
        assert "Dockerfile" in written
        assert "k8s/job.yaml" in written and "k8s/configmaps.yaml" in written
        assert ".github/workflows/pipe_eu_sales_to_s3_pipeline.yml" in written
        assert len([f for f in written if f.startswith("k8s/")]) == 6

    def test_removes_stale_pipeline_workflows_keeps_repo_workflows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CLOUD_PROVIDER", "aws")
        monkeypatch.setattr(codegen, "REPO_ROOT", tmp_path)
        (tmp_path / "sql").mkdir()
        (tmp_path / "dashboards").mkdir()
        (tmp_path / "sql" / "setup_trino.sql").write_text("SELECT 1;", encoding="utf-8")
        (tmp_path / "dashboards" / "monitoring_specs.json").write_text(
            codegen.render_monitoring_specs(AWS_CONF, "aws"), encoding="utf-8")
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "pipe_old_pipeline.yml").write_text("stale")
        (wf_dir / "tests.yml").write_text("repo infra — must survive", encoding="utf-8")
        _, errors = codegen.ensure_infra_artifacts(AWS_CONF, {"provider": "kubernetes"}, REGISTRY, [])
        assert errors == []
        assert not (wf_dir / "pipe_old_pipeline.yml").exists()
        assert (wf_dir / "pipe_eu_sales_to_s3_pipeline.yml").exists()
        assert (wf_dir / "tests.yml").exists()
