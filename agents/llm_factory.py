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
# free-tier quota (429). Each model has separate capacity AND a separate quota — the free tier
# allows ~20 requests/day PER MODEL, so a chain of models multiplies the free daily budget.
_DEFAULT_GEMINI_FALLBACK = "gemini-3.6-flash,gemini-2.5-flash,gemini-3.5-flash-lite"


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
    """Make a message list satisfy Gemini's turn-order rules (OpenAI is more lenient):

    * a request must contain at least one non-system turn ("contents are required");
    * a model turn may not open the conversation, and a model FUNCTION-CALL turn must come
      right after a user turn or a tool-response turn ("function call turn comes immediately
      after a user turn or after a function response turn").

    The agents re-send a trimmed window of earlier history, which can start with — or contain —
    an earlier tool-call turn without the user turn before it. A minimal user turn is inserted
    where needed; the model turns themselves (and their thought signatures) are left untouched.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    out, prev = [], None   # prev = last non-system message
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append(m)
            continue
        if isinstance(m, AIMessage) and (
            prev is None or (m.tool_calls and not isinstance(prev, (HumanMessage, ToolMessage)))
        ):
            out.append(HumanMessage(content="Continue."))
        out.append(m)
        prev = m
    if prev is None:
        out.append(HumanMessage(content="Begin. Follow the instructions above."))
    return out


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
        raw = (os.getenv("LLM_FALLBACK_MODEL") or _DEFAULT_GEMINI_FALLBACK).strip()
        fallbacks = [] if raw.lower() in ("", "none", "off") else [
            m.strip() for m in raw.split(",") if m.strip() and m.strip() != model]
        # With fallback models available, give up on an overloaded/exhausted model sooner and switch.
        retries = int(os.getenv("LLM_MAX_RETRIES") or ("2" if fallbacks else "6"))
        primary = _gemini_model(model, temperature, timeout, retries)
        if not fallbacks:
            return primary
        # RunnableWithFallbacks forwards bind_tools(...) to the primary AND every fallback.
        return primary.with_fallbacks([_gemini_model(m, temperature, timeout, retries) for m in fallbacks])

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
