import logging
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

# Free-tier Gemini models used by the agents. The free tier allows ~20 requests/day PER MODEL, so
# each agent starts on its OWN model (separate daily quota) and falls back through the others —
# the stronger "flash" models first, the weaker "flash-lite" LAST (it writes one file per turn and
# strays outside its task, which costs more calls than it saves).
_GEMINI_MODEL_POOL = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                      "gemini-2.5-flash", "gemini-3.5-flash-lite")
_DEFAULT_AGENT_MODELS = {
    "architect": "gemini-3.8-flash",       # writes the ETL script + DDL (hardest task)
    "infra": "gemini-3.6-flash",           # writes the Terraform
    "medic": "gemini-3.7-flash",           # diagnoses CI failures
    "supervisor": "gemini-3.5-flash-lite",  # rare LLM fallback only
}
# Kept for backward compatibility (LLM_FALLBACK_MODEL default before per-agent models).
_DEFAULT_GEMINI_FALLBACK = ",".join(_GEMINI_MODEL_POOL[1:])


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
    elif isinstance(prev, AIMessage) and not prev.tool_calls:
        # Gemini 3.x rejects "model prefilling": the final turn must be a user message or a
        # function response. A history ending on a model turn (e.g. an agent re-entered after
        # another agent's plain-text reply) gets a minimal user turn appended.
        out.append(HumanMessage(content="Continue with the task."))
    return out


_GEMINI_CHAT_CLS = None

# ── Daily free-quota tracking ────────────────────────────────────────────────────────────────
# Gemini's free tier caps requests PER DAY PER MODEL (quotaId ...PerDay...). Once a model hits it
# the remaining ~24h of the run can never succeed on that model, so retrying it only burns time.
# Models that report a DAILY quota error are remembered for this process and skipped instantly,
# letting the fallback chain move straight to the next model. Per-minute 429s are NOT marked.
_EXHAUSTED_MODELS: set[str] = set()


class ModelQuotaExhausted(RuntimeError):
    """Raised instead of calling a model whose daily free quota is already used up."""


def _is_daily_quota_error(err: Exception) -> bool:
    import re as _re

    text = str(err)
    # A model the key cannot use (404 NOT_FOUND) will not start working mid-run either —
    # skip it like an exhausted one instead of paying a failed round trip on every call.
    if "404" in text and "NOT_FOUND" in text.upper().replace(" ", "_"):
        return True
    if "RESOURCE_EXHAUSTED" not in text and "429" not in text:
        return False
    if "PerDay" in text or "per day" in text.lower():
        return True
    m = _re.search(r"retryDelay'?\"?:\s*'?\"?(\d+)s", text)
    return bool(m and int(m.group(1)) > 3600)


def _guarded_call(model: str, fn):
    if model in _EXHAUSTED_MODELS:
        raise ModelQuotaExhausted(
            f"{model}: daily free quota exhausted (skipped). If every model is exhausted, wait for the "
            f"reset (midnight Pacific) and use './run_project.sh <pipeline> --resume', or use another key.")
    try:
        return fn()
    except Exception as e:
        if _is_daily_quota_error(e):
            _EXHAUSTED_MODELS.add(model)
            logging.getLogger("LLM").warning(
                f"⛔ {model} daily free quota exhausted — skipping it for this run.")
        raise


def _gemini_chat_cls():
    """ChatGoogleGenerativeAI with the `_with_user_turn` guard and daily-quota skipping
    (built lazily: optional dependency)."""
    global _GEMINI_CHAT_CLS
    if _GEMINI_CHAT_CLS is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        class SafeGeminiChat(ChatGoogleGenerativeAI):
            def _generate(self, messages, *args, **kwargs):
                return _guarded_call(self.model, lambda: super(SafeGeminiChat, self)._generate(
                    _with_user_turn(messages), *args, **kwargs))

            async def _agenerate(self, messages, *args, **kwargs):
                if self.model in _EXHAUSTED_MODELS:
                    _guarded_call(self.model, lambda: None)   # raises ModelQuotaExhausted
                try:
                    return await super()._agenerate(_with_user_turn(messages), *args, **kwargs)
                except Exception as e:
                    if _is_daily_quota_error(e):
                        _EXHAUSTED_MODELS.add(self.model)
                    raise

            def _stream(self, messages, *args, **kwargs):
                if self.model in _EXHAUSTED_MODELS:
                    _guarded_call(self.model, lambda: None)   # raises ModelQuotaExhausted
                try:
                    yield from super()._stream(_with_user_turn(messages), *args, **kwargs)
                except Exception as e:
                    if _is_daily_quota_error(e):
                        _EXHAUSTED_MODELS.add(self.model)
                    raise

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


def get_llm(temperature=0, agent: str | None = None):
    """Chat model for an agent ("supervisor" | "architect" | "infra" | "medic").

    Gemini: each agent gets its OWN primary model (separate free-tier daily quota) — override with
    LLM_MODEL_<AGENT>; LLM_MODEL applies only to calls without an agent. The fallback chain is the
    rest of the model pool (or LLM_FALLBACK_MODEL). Other providers ignore `agent` and use LLM_MODEL.
    """
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
        primary = model
        if agent:
            # Per-agent primary model → each agent drains its OWN daily quota first.
            primary = (os.getenv(f"LLM_MODEL_{agent.upper()}") or "").strip() or \
                _DEFAULT_AGENT_MODELS.get(agent, model)
        raw = (os.getenv("LLM_FALLBACK_MODEL") or "").strip()
        if raw.lower() in ("none", "off"):
            fallbacks = []
        elif raw:
            fallbacks = [m.strip() for m in raw.split(",") if m.strip() and m.strip() != primary]
        else:
            # Rotate the pool so the chain starts after this agent's primary; flash-lite goes last.
            pool = list(_GEMINI_MODEL_POOL)
            if primary in pool:
                i = pool.index(primary)
                pool = pool[i + 1:] + pool[:i]
            fallbacks = [m for m in pool if m != primary and "lite" not in m] + \
                        [m for m in pool if m != primary and "lite" in m]
        # With fallback models available, switch to the next model after ONE retry: a busy (503)
        # model is better replaced than hammered, and every retry is one more request.
        retries = int(os.getenv("LLM_MAX_RETRIES") or ("1" if fallbacks else "6"))
        primary_llm = _gemini_model(primary, temperature, timeout, retries)
        if not fallbacks:
            return primary_llm
        # RunnableWithFallbacks forwards bind_tools(...) to the primary AND every fallback.
        return primary_llm.with_fallbacks([_gemini_model(m, temperature, timeout, retries) for m in fallbacks])

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
