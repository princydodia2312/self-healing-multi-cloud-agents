"""Gemini-call reduction: steps with fully-known arguments run in Python, not through the LLM.

Hermetic: the LLM, Pinecone, the source DBs, Terraform and git are mocked. Each test asserts both
the behaviour (state fields, files) and the number of LLM invocations.
"""
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from agents import architect

PIPE = {
    "pipeline_id": "pipe_eu_sales_to_s3",
    "cloud_provider": "aws",
    "project_structure": {"python_script_path": "scripts/pipe_eu_sales_to_s3.py"},
    "required_artifacts": ["scripts/{pipeline_name}.py", "sql/setup_trino.sql",
                           "dashboards/monitoring_specs.json", "requirements.txt"],
    "aws_setup": {"bucket_name": "b", "region": "eu-central-1"},
}
SCHEMA_OK = {"status": "success", "database": "postgres", "table": "raw_eu_sales",
             "schema": ["order_id (TEXT)", "unit_price (DOUBLE PRECISION)"], "sample_data": []}
SPEC = "🛡️ [OFFICIAL SPEC]\nSource: x.md (Relevance: 0.80)\nContent: rules"


def _state(**kw):
    s = {
        "task": "build it", "messages": [], "project_id": "P", "raw_configs": {
            "pipeline": PIPE,
            "database": {"db_type": "postgres", "default_table": "raw_eu_sales"},
            "rules": {"quality_standards": []},
            "infrastructure": {"service_name": "s3", "data_format": "parquet"},
        },
        "written_files": [], "collected_specs": {}, "schema_discovered": False,
        "last_agent": "supervisor", "medic_fix_requested": False, "github_done": False,
        "healing_context": "",
    }
    s.update(kw)
    return s


def _llm_returning(*responses):
    llm = MagicMock()
    llm.bind_tools.return_value.invoke.side_effect = list(responses)
    return llm


def _write_call(name, i):
    return {"name": "write_project_file", "id": f"c{i}", "args": {"filename": name, "content": "x"}}


class TestArchitectDeterministicDiscoveryAndSchema:
    def test_discovery_and_schema_need_no_llm_and_implementation_is_one_turn(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        one_turn = AIMessage(content="", tool_calls=[
            _write_call("scripts/pipe_eu_sales_to_s3.py", 1), _write_call("sql/setup_trino.sql", 2)])
        llm = _llm_returning(one_turn)
        written = []

        def fake_write(args):
            written.append(args["filename"])
            return f"File saved successfully to {args['filename']}"

        with patch.object(architect, "get_llm", return_value=llm), \
             patch.object(architect, "query_vector_store", MagicMock(invoke=MagicMock(return_value=SPEC))) as kb, \
             patch.object(architect, "read_data_schema", MagicMock(invoke=MagicMock(return_value=SCHEMA_OK))), \
             patch.object(architect, "write_project_file", MagicMock(invoke=MagicMock(side_effect=fake_write))), \
             patch.object(architect, "validate_generated_code",
                          MagicMock(invoke=MagicMock(return_value="CLEAN: ok"))), \
             patch("agents.codegen.ensure_architect_artifacts",
                   return_value=(["requirements.txt", "dashboards/monitoring_specs.json"], [])):
            out = architect.architect_node(_state())

        assert kb.invoke.call_count == 2                       # python + trino standard, issued by Python
        assert out["schema_discovered"] is True
        assert set(out["collected_specs"]) == {"arch_standard_python", "arch_standard_trino"}
        assert llm.bind_tools.return_value.invoke.call_count == 1   # ONLY the implementation turn
        assert written == ["scripts/pipe_eu_sales_to_s3.py", "sql/setup_trino.sql"]
        assert out["architect_status"] == "completed"
        assert any("SCHEMA DISCOVERY RESULT" in str(m.content) for m in out["messages"])

    def test_mongo_source_also_fetches_mongodb_standard(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = _state()
        st["raw_configs"]["database"] = {"db_type": "mongodb", "default_table": "raw_orders"}
        schema = MagicMock(invoke=MagicMock(return_value={**SCHEMA_OK, "database": "mongodb"}))
        with patch.object(architect, "get_llm", return_value=_llm_returning(AIMessage(content="done"))), \
             patch.object(architect, "query_vector_store", MagicMock(invoke=MagicMock(return_value=SPEC))), \
             patch.object(architect, "read_data_schema", schema), \
             patch("agents.codegen.ensure_architect_artifacts", return_value=([], [])):
            out = architect.architect_node(st)
        assert "arch_standard_mongodb" in out["collected_specs"]
        schema.invoke.assert_called_once_with({"table_name": "raw_orders", "db_type": "mongodb"})

    def test_empty_kb_result_falls_back_to_llm_discovery(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        llm = _llm_returning(AIMessage(content="no tool"))
        with patch.object(architect, "get_llm", return_value=llm), \
             patch.object(architect, "query_vector_store", MagicMock(invoke=MagicMock(
                 return_value="No relevant guidelines found. Proceed with standard engineering practices."))), \
             patch.object(architect, "read_data_schema", MagicMock()) as schema:
            out = architect.architect_node(_state())
        assert out["collected_specs"] == {}
        assert llm.bind_tools.return_value.invoke.call_count == 1    # LLM discovery fallback used
        schema.invoke.assert_not_called()                             # never read schema without standards

    def test_failed_schema_read_routes_to_medic_without_llm(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        llm = _llm_returning()
        with patch.object(architect, "get_llm", return_value=llm), \
             patch.object(architect, "query_vector_store", MagicMock(invoke=MagicMock(return_value=SPEC))), \
             patch.object(architect, "read_data_schema", MagicMock(invoke=MagicMock(
                 return_value="Database Error on postgres: connection timed out [OperationalError]"))):
            out = architect.architect_node(_state())
        assert out["agent_error"] is True and out["schema_discovered"] is False
        assert out["architect_status"] == "pending"
        assert "SCHEMA DISCOVERY FAILED" in str(out["messages"][-1].content)
        llm.bind_tools.return_value.invoke.assert_not_called()

    def test_fix_mode_still_uses_llm_patch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        llm = _llm_returning(AIMessage(content="no tool"))
        st = _state(collected_specs={"arch_standard_python": SPEC, "arch_standard_trino": SPEC},
                    schema_discovered=True, last_agent="medic", healing_context="fix scripts/pipe_eu_sales_to_s3.py")
        with patch.object(architect, "get_llm", return_value=llm), \
             patch.object(architect, "query_vector_store", MagicMock()) as kb:
            architect.architect_node(st)
        kb.invoke.assert_not_called()
        names = [t.name for t in llm.bind_tools.call_args.args[0]]
        assert names == ["patch_project_file"]



# ── Infra: deterministic iac discovery, terraform and push ──────────────────────────────────

from agents import infra  # noqa: E402

TF_FILES = ["terraform/providers.tf", "terraform/main.tf", "terraform/variables.tf",
            "terraform/outputs.tf", "terraform/terraform.tfvars"]
K8S_FILES = ["k8s/00_namespaces.yaml", "k8s/trino_deployment.yaml", "k8s/grafana_deployment.yaml",
             "k8s/prometheus_deployment.yaml", "k8s/configmaps.yaml", "k8s/job.yaml"]
ALL_ORCH = K8S_FILES + ["Dockerfile", ".github/workflows/pipe_eu_sales_to_s3_pipeline.yml"]
SHA = "a" * 40


def _infra_state(**kw):
    s = _state(collected_specs={"infra_standard_iac": SPEC}, last_agent="supervisor",
               infra_provisioned=False, ecr_repository_url="123.dkr.ecr.eu-central-1.amazonaws.com/repo")
    s.update(kw)
    return s


class TestInfraDeterministic:
    def test_iac_discovery_is_python_only(self):
        llm = _llm_returning(AIMessage(content="no tool"))
        kb = MagicMock(invoke=MagicMock(return_value=SPEC))
        with patch.object(infra, "get_llm", return_value=llm), patch.object(infra, "query_vector_store", kb):
            out = infra.infra_node(_infra_state(collected_specs={}))
        assert "infra_standard_iac" in out["collected_specs"]
        q = kb.invoke.call_args.args[0]["query"]
        assert q == infra._IAC_QUERIES["aws"]
        # The LLM is then asked ONLY to write the .tf files (no execute_terraform, no discovery).
        names = [t.name for t in llm.bind_tools.call_args.args[0]]
        assert names == ["write_terraform_config"]

    def test_terraform_runs_in_python_after_llm_writes_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        writes = AIMessage(content="", tool_calls=[
            {"name": "write_terraform_config", "id": f"t{i}", "args": {"filename": f.split("/")[1], "content": "x"}}
            for i, f in enumerate(TF_FILES)])
        llm = _llm_returning(writes)
        tf = MagicMock(invoke=MagicMock(side_effect=lambda a: f"SUCCESS: Terraform {a['command']}\n"
                                        + ("Apply complete! Resources: 8 added" if a["command"] == "apply" else "")))
        # The REAL write_terraform_config writes into tmp_path (backend pinning included).
        with patch.object(infra, "get_llm", return_value=llm), patch.object(infra, "execute_terraform", tf), \
             patch("agents.tools.validate_generated_code", MagicMock(invoke=MagicMock(return_value="CLEAN"))):
            out = infra.infra_node(_infra_state())
        assert [c.args[0]["command"] for c in tf.invoke.call_args_list] == ["init", "plan", "apply"]
        assert out["infra_provisioned"] is True and out["agent_error"] is False
        assert llm.bind_tools.return_value.invoke.call_count == 1          # only the .tf-writing turn

    def test_terraform_failure_is_surfaced_for_medic(self):
        llm = _llm_returning()
        tf = MagicMock(invoke=MagicMock(side_effect=lambda a: (
            f"SUCCESS: Terraform {a['command']}\n" if a["command"] != "apply"
            else "FAILED: Terraform apply\nERROR: AccessDenied: not authorized to perform s3:CreateBucket")))
        with patch.object(infra, "get_llm", return_value=llm), patch.object(infra, "execute_terraform", tf):
            out = infra.infra_node(_infra_state(written_files=list(TF_FILES)))
        llm.bind_tools.return_value.invoke.assert_not_called()            # files existed → no LLM turn
        assert out["agent_error"] is True and out["infra_provisioned"] is False
        from agents.medic import _terraform_command_failure
        assert "AccessDenied" in _terraform_command_failure(out["messages"])

    def test_push_runs_in_python_without_llm(self):
        llm = _llm_returning()
        push = MagicMock(invoke=MagicMock(return_value=f"STATUS: SUCCESS | SHA: {SHA} | Message: pushed"))
        with patch.object(infra, "get_llm", return_value=llm), patch.object(infra, "push_to_github", push), \
             patch("agents.codegen.ensure_infra_artifacts", return_value=([], [])):
            out = infra.infra_node(_infra_state(infra_provisioned=True, written_files=TF_FILES + ALL_ORCH))
        llm.bind_tools.return_value.invoke.assert_not_called()
        assert push.invoke.call_args.args[0]["commit_message"] == "deploy: generated pipeline artifacts"
        assert out["github_done"] is True and out["last_push_sha"] == SHA
        assert out["infra_status"] == "completed"

    def test_security_gate_block_routes_to_medic(self):
        push = MagicMock(invoke=MagicMock(return_value="Error: SECURITY GATE FAILED — refusing to push 1 HIGH"))
        with patch.object(infra, "get_llm", return_value=_llm_returning()), \
             patch.object(infra, "push_to_github", push), \
             patch("agents.codegen.ensure_infra_artifacts", return_value=([], [])):
            out = infra.infra_node(_infra_state(infra_provisioned=True, written_files=TF_FILES + ALL_ORCH))
        assert out["github_done"] is False and out["agent_error"] is True
        assert out["infra_status"] == "pending"

    def test_heal_after_architect_fix_pushes_without_llm(self):
        llm = _llm_returning()
        push = MagicMock(invoke=MagicMock(return_value=f"STATUS: SUCCESS | SHA: {SHA} | Message: ok"))
        st = _infra_state(infra_provisioned=True, written_files=TF_FILES + ALL_ORCH, github_done=True,
                          medic_fix_requested=True, last_agent="architect", healing_context="")
        with patch.object(infra, "get_llm", return_value=llm), patch.object(infra, "push_to_github", push):
            out = infra.infra_node(st)
        llm.bind_tools.return_value.invoke.assert_not_called()
        assert push.invoke.called and out["github_done"] is True


# ── Medic fast path: local validation failures need no LLM diagnosis ────────────────────────

from agents import medic  # noqa: E402
from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402

_RUFF = ("VALIDATION FAILED (ruff):\nF821 Undefined name `pd`\n --> scripts/pipe_eu_sales.py:12:5")


def _medic_state(msgs, **kw):
    s = _state(messages=msgs, infra_status="pending", last_agent="architect", agent_error=True)
    s.update(kw)
    return s


class TestMedicFastPath:
    def _run(self, st):
        llm = _llm_returning(AIMessage(content="should not be called"))
        with patch.object(medic, "get_llm", return_value=llm), \
             patch.object(medic, "query_vector_store", MagicMock(invoke=MagicMock(return_value="none"))):
            out = medic.medic_node(st)
        return out, llm.bind_tools.return_value.invoke

    def test_architect_validation_failure_routes_without_llm(self):
        tm = ToolMessage(tool_call_id="w1", content=(
            "File scripts/pipe_eu_sales.py successfully written\n\n"
            f"AUTO-VALIDATION FAILED — fix these errors in 'scripts/pipe_eu_sales.py':\n{_RUFF}"))
        out, invoke = self._run(_medic_state([tm]))
        invoke.assert_not_called()
        assert out["medic_fix_target"] == "architect"
        assert "scripts/pipe_eu_sales.py" in out["healing_context"]
        assert "F821 Undefined name `pd`" in out["healing_context"]
        assert out["fix_attempt"] == 1

    def test_infra_validation_failure_routes_to_infra(self):
        tm = ToolMessage(tool_call_id="t1", content=(
            "File terraform/main.tf successfully written\n\nAUTO-VALIDATION FAILED — fix these errors in "
            "'terraform/main.tf' and call the same generation tool again with corrected content:\n"
            "VALIDATION FAILED (terraform validate):\nError: Unsupported argument \"acl\""))
        out, invoke = self._run(_medic_state([tm], last_agent="infra"))
        invoke.assert_not_called()
        assert out["medic_fix_target"] == "infra"
        assert 'Unsupported argument "acl"' in out["healing_context"]
        assert "and call the same" not in out["healing_context"]

    def test_same_error_three_times_escalates(self):
        tm = ToolMessage(tool_call_id="w1", content=(
            f"AUTO-VALIDATION FAILED — fix these errors in 'scripts/pipe_eu_sales.py':\n{_RUFF}"))
        st = _medic_state([tm])
        out, _ = self._run(st)
        st.update(fix_attempt=out["fix_attempt"], last_fix_signature=out["last_fix_signature"],
                  total_fix_attempts=out["total_fix_attempts"])
        out, _ = self._run(st)
        st.update(fix_attempt=out["fix_attempt"], last_fix_signature=out["last_fix_signature"],
                  total_fix_attempts=out["total_fix_attempts"])
        out, _ = self._run(st)
        assert out.get("fix_loop_escalated") is True

    def test_terraform_apply_failure_keeps_llm_path(self):
        msgs = [HumanMessage(content="FAILED: Terraform apply\nERROR: AccessDenied s3:CreateBucket")]
        _, invoke = self._run(_medic_state(msgs, last_agent="infra"))
        assert invoke.called                     # diagnosis of a cloud error stays with the LLM

    def test_old_failure_outside_window_is_ignored(self):
        old = ToolMessage(tool_call_id="w0", content=(
            f"AUTO-VALIDATION FAILED — fix these errors in 'scripts/pipe_eu_sales.py':\n{_RUFF}"))
        later = [HumanMessage(content=f"step {i}") for i in range(10)]
        _, invoke = self._run(_medic_state([old] + later))
        assert invoke.called
