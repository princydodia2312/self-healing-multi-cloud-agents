"""Provider selection: OpenAI (paid) vs the free Gemini API — chat model, NL client, embeddings.

Hermetic: the SDK classes are patched, so nothing touches the network.
"""
import importlib
from unittest.mock import MagicMock, patch

import pytest

from agents import llm_factory


_KEYS = ("LLM_PROVIDER", "LLM_MODEL", "NL_MODEL", "EMBED_MODEL", "EMBED_DIMENSIONS",
         "LLM_MAX_RETRIES", "GEMINI_API_KEY", "GOOGLE_API_KEY")


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
    def test_provider_only_uses_gemini_default_model_and_key(self, env):
        env.setenv("LLM_PROVIDER", "gemini")
        env.setenv("GEMINI_API_KEY", "g-key")
        fake = MagicMock()
        with patch("langchain_google_genai.ChatGoogleGenerativeAI", fake):
            llm_factory.get_llm(temperature=0)
        kwargs = fake.call_args.kwargs
        assert kwargs["model"] == "gemini-3.8-flash"
        assert kwargs["google_api_key"] == "g-key"
        assert kwargs["max_retries"] == 6           # patient retries for free-tier rate limits
        assert "temperature" not in kwargs          # Gemini 3+: keep Google's default sampling

    def test_older_gemini_model_keeps_temperature(self, env):
        env.setenv("LLM_MODEL", "gemini-2.5-flash")
        env.setenv("GOOGLE_API_KEY", "g-key")
        fake = MagicMock()
        with patch("langchain_google_genai.ChatGoogleGenerativeAI", fake):
            llm_factory.get_llm(temperature=0.1)
        assert fake.call_args.kwargs["temperature"] == 0.1
        assert fake.call_args.kwargs["google_api_key"] == "g-key"

    def test_openai_path_unchanged(self, env):
        fake = MagicMock()
        with patch("langchain_openai.ChatOpenAI", fake):
            llm_factory.get_llm(temperature=0)
        assert fake.call_args.kwargs["model"] == "gpt-4o"
        assert fake.call_args.kwargs["max_retries"] == 1


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
