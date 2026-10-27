"""Shared LLM / embedding provider settings.

One switch picks the provider for EVERYTHING (agents, NL/demo surface, RAG embeddings):

    LLM_PROVIDER=openai   → OpenAI (paid)            key: OPENAI_API_KEY
    LLM_PROVIDER=gemini   → Google Gemini API (free tier available)  key: GEMINI_API_KEY
    LLM_PROVIDER=anthropic→ Anthropic (paid, agents only)

If LLM_PROVIDER is unset it is inferred from LLM_MODEL (gemini-* → gemini, claude-* → anthropic,
anything else → openai).

* The AGENT pipeline builds its chat model in `agents/llm_factory.get_llm` (LLM_MODEL).
* The NL / demo surface (`nlp_parser`, `rules_loader`, `architecture_advisor`) uses a small model
  for constrained JSON extraction on its OWN knob (NL_MODEL) via `make_nl_client()` — an OpenAI-SDK
  client that, for Gemini, points at Google's OpenAI-compatible endpoint.
* RAG embeddings (`agents/tools.py`, `scripts/ingest_to_pinecone.py`) come from `make_embeddings()`.
  Both providers produce EMBED_DIMENSIONS-sized vectors (default 1536), so the same Pinecone index
  works — but vectors from different models are NOT comparable: re-run `make ingest` after switching.

Temperatures are intentionally NOT centralised here: they differ by purpose and stay at each call site.
"""
import os

from dotenv import load_dotenv

# Load .env HERE (not only in the entry points): the module-level defaults below are computed at
# import time, and several importers import this module before they call load_dotenv themselves.
load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_DEFAULT_AGENT_MODEL = {"openai": "gpt-4o", "gemini": "gemini-3.8-flash",
                        "anthropic": "claude-sonnet-4-20250514"}
_DEFAULT_NL_MODEL = {"openai": "gpt-4o-mini", "gemini": "gemini-3.5-flash-lite"}
_DEFAULT_EMBED_MODEL = {"openai": "text-embedding-3-small", "gemini": "gemini-embedding-2"}


def llm_provider() -> str:
    """The configured provider: LLM_PROVIDER if set, else inferred from LLM_MODEL."""
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        return explicit
    model = os.getenv("LLM_MODEL") or ""
    if model.startswith("gemini-"):
        return "gemini"
    if model.startswith("claude-"):
        return "anthropic"
    return "openai"


def using_gemini() -> bool:
    return llm_provider() == "gemini"


def gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def default_agent_model(provider: str) -> str:
    return _DEFAULT_AGENT_MODEL.get(provider, "gpt-4o")


def nl_api_key() -> str | None:
    """API key for the NL / embedding provider (Gemini key or OpenAI key)."""
    return gemini_api_key() if using_gemini() else os.getenv("OPENAI_API_KEY")


def make_nl_client():
    """OpenAI-SDK client for the NL surface — OpenAI itself, or Gemini's OpenAI-compatible API.

    A missing key does NOT raise here (modules build this client at import time); the request
    fails with an auth error instead, and main.py reports the missing key up front."""
    from openai import OpenAI

    key = nl_api_key() or "MISSING_API_KEY"
    if using_gemini():
        return OpenAI(api_key=key, base_url=os.getenv("GEMINI_BASE_URL") or GEMINI_BASE_URL)
    return OpenAI(api_key=key)


def _embed_provider() -> str:
    # Anthropic has no embedding API — RAG embeddings then stay on OpenAI.
    return "gemini" if using_gemini() else "openai"


NL_MODEL = os.getenv("NL_MODEL") or _DEFAULT_NL_MODEL.get(llm_provider(), "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL") or _DEFAULT_EMBED_MODEL[_embed_provider()]
# Must match the Pinecone index dimension (the index was created for 1536).
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS") or "1536")


def make_embeddings():
    """LangChain embeddings object (embed_query / embed_documents) for the configured provider."""
    if _embed_provider() == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            output_dimensionality=EMBED_DIMENSIONS,
            google_api_key=gemini_api_key(),
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBED_MODEL)
