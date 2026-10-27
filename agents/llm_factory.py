import os

from utils.llm_defaults import default_agent_model, gemini_api_key

# Model-name prefix → provider
_PREFIX_TO_PROVIDER = {
    "gpt-":    "openai",
    "o1":      "openai",
    "o3":      "openai",
    "claude-": "anthropic",
    # Google AI Studio (Gemini API key, free tier). Vertex AI needs LLM_PROVIDER=vertexai explicitly.
    "gemini-": "gemini",
}


def _infer_provider(model: str) -> str:
    for prefix, provider in _PREFIX_TO_PROVIDER.items():
        if model.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot infer LLM provider from model name '{model}'. "
        f"Set LLM_PROVIDER explicitly or use a recognised model prefix: "
        f"{list(_PREFIX_TO_PROVIDER.keys())}"
    )


def get_llm(temperature=0):
    # `or <default>` (not getenv's default arg) so a set-but-EMPTY env var — e.g. an unset GitHub
    # Actions `vars.LLM_MODEL` that expands to "" — falls back instead of yielding "" → _infer_provider("")
    # raising, or int("") crashing.
    explicit_provider = (os.getenv("LLM_PROVIDER") or "").lower()
    model       = os.getenv("LLM_MODEL") or default_agent_model(explicit_provider or "openai")
    timeout     = float(os.getenv("LLM_TIMEOUT_SEC") or "120")

    # LLM_PROVIDER can override auto-detection (edge cases / private endpoints)
    provider = explicit_provider or _infer_provider(model)
    # The Gemini free tier has low per-minute limits → retry 429s more patiently by default.
    max_retries = int(os.getenv("LLM_MAX_RETRIES") or ("6" if provider == "gemini" else "1"))

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    if provider == "gemini":
        # Native LangChain integration (not the OpenAI-compatible endpoint): it round-trips Gemini's
        # thought signatures across multi-turn tool calls, which the agents' tool loops rely on.
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {}
        # Google recommends leaving sampling params at their defaults for Gemini 3+ models (low
        # temperatures can degrade reasoning / cause loops). Older models keep the agent's value.
        if not model.startswith(("gemini-3", "gemini-4")):
            kwargs["temperature"] = temperature
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=gemini_api_key(),
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    if provider == "vertexai":
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            # timeout handled by Google SDK internally
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. "
        "Valid options: 'openai', 'gemini', 'anthropic', 'vertexai'."
    )
