"""Provider selection: OpenAI (paid) vs the free Gemini API — chat model, NL client, embeddings.

Hermetic: the SDK classes are patched, so nothing touches the network.
"""
import importlib
from unittest.mock import MagicMock, patch

import pytest

from agents import llm_factory


_KEYS = ("LLM_PROVIDER", "LLM_MODEL", "NL_MODEL", "EMBED_MODEL", "EMBED_DIMENSIONS",
         "LLM_MAX_RETRIES", "LLM_FALLBACK_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
         "LLM_MODEL_ARCHITECT", "LLM_MODEL_INFRA", "LLM_MODEL_MEDIC", "LLM_MODEL_SUPERVISOR")


@pytest.fixture
def env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch
    # Leave utils.llm_defaults in its default (OpenAI) state for the rest of the suite.
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    _reload_defaults()


def _reload_defaults():
    import utils.llm_defaults as d
    # Skip .env so a developer's local file can't change the outcome.
    with patch("dotenv.load_dotenv"):
        return importlib.reload(d)


class TestProviderInference:
    def test_gemini_prefix_maps_to_gemini(self):
        assert llm_factory._infer_provider("gemini-3.8-flash") == "gemini"

    def test_openai_and_anthropic_prefixes(self):
        assert llm_factory._infer_provider("gpt-4o") == "openai"
        assert llm_factory._infer_provider("claude-sonnet-4-20250514") == "anthropic"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            llm_factory._infer_provider("llama-3")


class TestGetLlmGemini:
    def test_provider_only_uses_gemini_default_model_with_fallback(self, env):
        env.setenv("LLM_PROVIDER", "gemini")
        env.setenv("GEMINI_API_KEY", "g-key")
        fake = MagicMock()
        with patch.object(llm_factory, "_gemini_chat_cls", return_value=fake):
            llm = llm_factory.get_llm(temperature=0)
        primary, *fallbacks = (c.kwargs for c in fake.call_args_list)
        assert primary["model"] == "gemini-3.8-flash"
        assert [f["model"] for f in fallbacks] == ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                                                   "gemini-2.5-flash", "gemini-3.5-flash-lite"]
        assert primary["google_api_key"] == "g-key"
        assert primary["max_retries"] == 1           # switch sooner from an overloaded primary...
        assert "temperature" not in primary          # Gemini 3+: keep Google's default sampling
        fake.return_value.with_fallbacks.assert_called_once()   # ...and switch to the fallback
        assert llm is fake.return_value.with_fallbacks.return_value

    def test_fallback_can_be_disabled(self, env):
        env.setenv("LLM_PROVIDER", "gemini")
        env.setenv("LLM_FALLBACK_MODEL", "none")
        fake = MagicMock()
        with patch.object(llm_factory, "_gemini_chat_cls", return_value=fake):
            llm = llm_factory.get_llm(temperature=0)
        assert fake.call_count == 1
        assert fake.call_args.kwargs["max_retries"] == 6   # patient retries without a fallback
        assert llm is fake.return_value

    def test_older_gemini_model_keeps_temperature(self, env):
        env.setenv("LLM_MODEL", "gemini-2.5-flash")
        env.setenv("GOOGLE_API_KEY", "g-key")
        fake = MagicMock()
        with patch.object(llm_factory, "_gemini_chat_cls", return_value=fake):
            llm_factory.get_llm(temperature=0.1)
        assert fake.call_args_list[0].kwargs["temperature"] == 0.1
        assert fake.call_args_list[0].kwargs["google_api_key"] == "g-key"

    def test_real_gemini_class_supports_bind_tools_with_fallback(self, env):
        env.setenv("LLM_PROVIDER", "gemini")
        env.setenv("GEMINI_API_KEY", "g-key")
        from langchain_core.tools import tool

        @tool
        def ping(x: str) -> str:
            """Echo."""
            return x

        llm = llm_factory.get_llm()
        assert llm.bind_tools([ping], tool_choice="required") is not None   # no network call

    def test_openai_path_unchanged(self, env):
        fake = MagicMock()
        with patch("langchain_openai.ChatOpenAI", fake):
            llm_factory.get_llm(temperature=0)
        assert fake.call_args.kwargs["model"] == "gpt-4o"
        assert fake.call_args.kwargs["max_retries"] == 1


class TestGeminiUserTurnGuard:
    def test_system_only_request_gets_a_user_turn(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        out = llm_factory._with_user_turn([SystemMessage(content="rules")])
        assert isinstance(out[-1], HumanMessage) and len(out) == 2

    def test_existing_conversation_untouched(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        msgs = [SystemMessage(content="rules"), HumanMessage(content="hi")]
        assert llm_factory._with_user_turn(msgs) == msgs

    def test_leading_tool_call_gets_a_user_turn_before_it(self):
        # The exact shape that crashed the Architect's schema phase on Gemini:
        # [system, ai(tool_call), tool, user(task)].
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        call = AIMessage(content="", tool_calls=[{"name": "q", "args": {}, "id": "c1"}])
        msgs = [SystemMessage(content="rules"), call,
                ToolMessage(content="spec", tool_call_id="c1"), HumanMessage(content="task")]
        out = llm_factory._with_user_turn(msgs)
        assert [type(m).__name__ for m in out] == [
            "SystemMessage", "HumanMessage", "AIMessage", "ToolMessage", "HumanMessage"]
        assert out[2] is call                           # model turn (and its signature) untouched

    def test_tool_call_after_model_text_gets_a_user_turn(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        msgs = [SystemMessage(content="s"), HumanMessage(content="u"), AIMessage(content="thinking"),
                AIMessage(content="", tool_calls=[{"name": "q", "args": {}, "id": "c2"}])]
        out = llm_factory._with_user_turn(msgs)
        assert isinstance(out[3], HumanMessage) and out[4] is msgs[3]

    def test_valid_tool_chain_untouched(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        msgs = [SystemMessage(content="s"), HumanMessage(content="u"),
                AIMessage(content="", tool_calls=[{"name": "q", "args": {}, "id": "c3"}]),
                ToolMessage(content="r", tool_call_id="c3"),
                AIMessage(content="", tool_calls=[{"name": "q", "args": {}, "id": "c4"}])]
        assert llm_factory._with_user_turn(msgs) == msgs


class TestDefaults:
    def test_gemini_defaults_and_nl_client(self, env):
        env.setenv("LLM_PROVIDER", "gemini")
        env.setenv("GEMINI_API_KEY", "g-key")
        d = _reload_defaults()
        assert d.NL_MODEL == "gemini-3.5-flash-lite"
        assert d.EMBED_MODEL == "gemini-embedding-2"
        assert d.EMBED_DIMENSIONS == 1536           # same Pinecone index as OpenAI
        fake = MagicMock()
        with patch("openai.OpenAI", fake):
            d.make_nl_client()
        assert fake.call_args.kwargs == {"api_key": "g-key", "base_url": d.GEMINI_BASE_URL}

    def test_gemini_embeddings_request_1536_dims(self, env):
        env.setenv("LLM_PROVIDER", "gemini")
        env.setenv("GEMINI_API_KEY", "g-key")
        d = _reload_defaults()
        fake = MagicMock()
        with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", fake):
            d.make_embeddings()
        assert fake.call_args.kwargs == {"model": "gemini-embedding-2", "output_dimensionality": 1536,
                                         "google_api_key": "g-key"}

    def test_openai_defaults(self, env):
        d = _reload_defaults()
        assert d.llm_provider() == "openai"
        assert d.NL_MODEL == "gpt-4o-mini"
        assert d.EMBED_MODEL == "text-embedding-3-small"

    def test_anthropic_embeddings_stay_on_openai(self, env):
        env.setenv("LLM_MODEL", "claude-sonnet-4-20250514")
        d = _reload_defaults()
        assert d.llm_provider() == "anthropic"
        assert d.EMBED_MODEL == "text-embedding-3-small"


class TestPerAgentModels:
    def _models(self, env, agent):
        fake = MagicMock()
        with patch.object(llm_factory, "_gemini_chat_cls", return_value=fake):
            llm_factory.get_llm(agent=agent)
        return [c.kwargs["model"] for c in fake.call_args_list]

    def test_each_agent_starts_on_its_own_model_with_rotated_chain(self, env):
        env.setenv("LLM_PROVIDER", "gemini")
        env.setenv("LLM_MODEL", "gemini-3.8-flash")       # must NOT override per-agent primaries
        assert self._models(env, "architect") == [
            "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
            "gemini-2.5-flash", "gemini-3.5-flash-lite"]
        assert self._models(env, "infra") == [
            "gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.8-flash",
            "gemini-3.7-flash", "gemini-3.5-flash-lite"]               # lite always last
        assert self._models(env, "medic")[0] == "gemini-3.7-flash"
        assert self._models(env, "supervisor")[0] == "gemini-3.5-flash-lite"

    def test_404_model_is_skipped_for_the_run(self):
        llm_factory._EXHAUSTED_MODELS.clear()
        err = Exception("404 NOT_FOUND. {'error': {'code': 404, 'status': 'NOT_FOUND'}}")
        with pytest.raises(Exception):
            llm_factory._guarded_call("gemini-2.5-flash", lambda: (_ for _ in ()).throw(err))
        assert "gemini-2.5-flash" in llm_factory._EXHAUSTED_MODELS
        llm_factory._EXHAUSTED_MODELS.clear()

    def test_agent_override_env(self, env):
        env.setenv("LLM_PROVIDER", "gemini")
        env.setenv("LLM_MODEL_INFRA", "gemini-2.5-flash")
        assert self._models(env, "infra")[0] == "gemini-2.5-flash"

    def test_openai_ignores_agent(self, env):
        fake = MagicMock()
        with patch("langchain_openai.ChatOpenAI", fake):
            llm_factory.get_llm(agent="architect")
        assert fake.call_args.kwargs["model"] == "gpt-4o"


class TestDailyQuotaSkip:
    _DAILY = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
              "'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'retryDelay': '85579s'")
    _MINUTE = ("429 RESOURCE_EXHAUSTED. 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', "
               "'retryDelay': '23s'")

    def setup_method(self):
        llm_factory._EXHAUSTED_MODELS.clear()

    def teardown_method(self):
        llm_factory._EXHAUSTED_MODELS.clear()

    def test_daily_quota_marks_model_and_skips_it_next_time(self):
        calls = []

        def boom():
            calls.append(1)
            raise Exception(self._DAILY)

        with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
            llm_factory._guarded_call("gemini-3.8-flash", boom)
        assert "gemini-3.8-flash" in llm_factory._EXHAUSTED_MODELS
        with pytest.raises(llm_factory.ModelQuotaExhausted):
            llm_factory._guarded_call("gemini-3.8-flash", boom)
        assert len(calls) == 1                     # second time: no network call at all

    def test_per_minute_429_is_not_marked(self):
        with pytest.raises(Exception):
            llm_factory._guarded_call("gemini-3.6-flash", lambda: (_ for _ in ()).throw(Exception(self._MINUTE)))
        assert "gemini-3.6-flash" not in llm_factory._EXHAUSTED_MODELS

    def test_fallback_chain_moves_past_exhausted_model(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_core.runnables import RunnableLambda

        llm_factory._EXHAUSTED_MODELS.add("gemini-3.8-flash")
        primary = RunnableLambda(lambda x: llm_factory._guarded_call("gemini-3.8-flash", lambda: "never"))
        chain = primary.with_fallbacks([FakeListChatModel(responses=["from model 2"])])
        assert chain.invoke("hi").content == "from model 2"


class TestNoModelPrefill:
    def test_history_ending_on_model_text_gets_a_user_turn(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        msgs = [SystemMessage(content="s"), HumanMessage(content="u"), AIMessage(content="done")]
        out = llm_factory._with_user_turn(msgs)
        assert isinstance(out[-1], HumanMessage) and out[:3] == msgs


class TestDbUnreachableGuard:
    def test_connection_failure_escalates_without_llm(self):
        from agents import medic
        from langchain_core.messages import HumanMessage
        st = {"messages": [HumanMessage(content=(
                  "SCHEMA DISCOVERY FAILED (read_data_schema table_name='raw_eu_sales', db_type='postgres'): "
                  "Database Error on postgres: (psycopg2.OperationalError) connection to server at \"x\" "
                  "failed: Connection timed out\n\tIs the server running on that host and accepting TCP/IP "
                  "connections? [OperationalError]"))],
              "infra_status": "pending", "project_id": "P"}
        llm = MagicMock()
        with patch.object(medic, "get_llm", return_value=llm):
            out = medic.medic_node(st)
        llm.bind_tools.return_value.invoke.assert_not_called()
        assert out["mission_status"] == "escalated" and out["fix_loop_escalated"] is True
        assert "TF_VAR_rds_allowed_cidrs" in out["messages"][0].content

    def test_sql_error_is_not_treated_as_network(self):
        from agents.medic import _source_db_unreachable
        assert not _source_db_unreachable(
            "Database Error on postgres: relation \"raw_x\" does not exist [ProgrammingError]")
