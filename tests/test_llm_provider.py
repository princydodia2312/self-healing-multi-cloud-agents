"""Provider selection: OpenAI (paid) vs the free Gemini API — chat model, NL client, embeddings.

Hermetic: the SDK classes are patched, so nothing touches the network.
"""
import importlib
from unittest.mock import MagicMock, patch

import pytest

from agents import llm_factory


_KEYS = ("LLM_PROVIDER", "LLM_MODEL", "NL_MODEL", "EMBED_MODEL", "EMBED_DIMENSIONS",
         "LLM_MAX_RETRIES", "LLM_FALLBACK_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY")


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
        primary, fallback = (c.kwargs for c in fake.call_args_list)
        assert primary["model"] == "gemini-3.8-flash" and fallback["model"] == "gemini-3.6-flash"
        assert primary["google_api_key"] == "g-key"
        assert primary["max_retries"] == 2           # give up sooner on an overloaded primary...
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
