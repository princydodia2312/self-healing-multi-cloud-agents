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

# Used when the primary Gemini model is overloaded (503), times out (504) or hits its per-model
# free-tier quota (429). A different model has separate capacity AND a separate quota.
_DEFAULT_GEMINI_FALLBACK = "gemini-3.6-flash"


def _infer_provider(model: str) -> str:
    for prefix, provider in _PREFIX_TO_PROVIDER.items():
        if model.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot infer LLM provider from model name '{model}'. "
        f"Set LLM_PROVIDER explicitly or use a recognised model prefix: "
        f"{list(_PREFIX_TO_PROVIDER.keys())}"
    )


def _with_user_turn(messages):
    """Gemini rejects a request that carries only a system instruction ("contents are required");
    OpenAI accepts it. Nodes that run with an empty history (e.g. the Medic's first turn) would
    otherwise crash, so append a minimal user turn in that case."""
    from langchain_core.messages import HumanMessage, SystemMessage

    msgs = list(messages)
    if not any(not isinstance(m, SystemMessage) for m in msgs):
        msgs.append(HumanMessage(content="Begin. Follow the instructions above."))
    return msgs


_GEMINI_CHAT_CLS = None


def _gemini_chat_cls():
    """ChatGoogleGenerativeAI with the `_with_user_turn` guard (built lazily: optional dependency)."""
    global _GEMINI_CHAT_CLS
    if _GEMINI_CHAT_CLS is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        class SafeGeminiChat(ChatGoogleGenerativeAI):
            def _generate(self, messages, *args, **kwargs):
                return super()._generate(_with_user_turn(messages), *args, **kwargs)

            async def _agenerate(self, messages, *args, **kwargs):
                return await super()._agenerate(_with_user_turn(messages), *args, **kwargs)

            def _stream(self, messages, *args, **kwargs):
                yield from super()._stream(_with_user_turn(messages), *args, **kwargs)

        _GEMINI_CHAT_CLS = SafeGeminiChat
    return _GEMINI_CHAT_CLS


def _gemini_model(model: str, temperature: float, timeout: float, max_retries: int):
    # Native LangChain integration (not the OpenAI-compatible endpoint): it round-trips Gemini's
    # thought signatures across multi-turn tool calls, which the agents' tool loops rely on.
    kwargs = {}
    # Google recommends leaving sampling params at their defaults for Gemini 3+ models (low
    # temperatures can degrade reasoning / cause loops). Older models keep the agent's value.
    if not model.startswith(("gemini-3", "gemini-4")):
        kwargs["temperature"] = temperature
    return _gemini_chat_cls()(
        model=model,
        google_api_key=gemini_api_key(),
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
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

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=int(os.getenv("LLM_MAX_RETRIES") or "1"),
        )

    if provider == "gemini":
        fallback = (os.getenv("LLM_FALLBACK_MODEL") or _DEFAULT_GEMINI_FALLBACK).strip()
        use_fallback = fallback.lower() not in ("", "none", "off") and fallback != model
        # With a fallback model available, give up on an overloaded primary sooner and switch.
        retries = int(os.getenv("LLM_MAX_RETRIES") or ("2" if use_fallback else "6"))
        primary = _gemini_model(model, temperature, timeout, retries)
        if not use_fallback:
            return primary
        # RunnableWithFallbacks forwards bind_tools(...) to the primary AND the fallback.
        return primary.with_fallbacks([_gemini_model(fallback, temperature, timeout, retries)])

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=int(os.getenv("LLM_MAX_RETRIES") or "1"),
        )

    if provider == "vertexai":
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(
            model=model,
            temperature=temperature,
            max_retries=int(os.getenv("LLM_MAX_RETRIES") or "1"),
            # timeout handled by Google SDK internally
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. "
        "Valid options: 'openai', 'gemini', 'anthropic', 'vertexai'."
    )
