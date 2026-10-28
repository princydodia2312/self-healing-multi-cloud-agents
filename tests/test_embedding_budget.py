"""Regression net for the Pinecone embedding token budget.

The knowledge base is ingested one vector PER FILE with no chunking, so every standard must fit the
embed model's 8191-token per-input limit. If one grows past it, the ingest fails and (before the
loud-fail guard) silently left a STALE version live in Pinecone. This test refuses a PR that pushes
any standard over the limit BEFORE it can poison the vector store — the visible half of the guard in
`scripts/ingest_to_pinecone.py`.
"""
from pathlib import Path

import pytest

from utils import embedding_budget
from utils.embedding_budget import (
    EMBED_TOKEN_SAFE_CEILING,
    EMBED_TOKEN_WARN,
    count_tokens,
    count_tokens_detail,
)

_KB = Path(__file__).resolve().parent.parent / "knowledge_base"
_STANDARDS = sorted(_KB.rglob("*.md"))


def test_knowledge_base_has_standards():
    # Guard against the glob silently matching nothing (which would make the budget test vacuous).
    assert _STANDARDS, f"No standards found under {_KB}"


def test_check_file_reports_within_limit():
    tokens, within = embedding_budget.check_file(_STANDARDS[0])
    assert isinstance(tokens, int) and tokens > 0
    assert within is True


def test_count_tokens_falls_back_when_tiktoken_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("simulated: no tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert count_tokens("a" * 40) == 10  # conservative chars/4, rounded up


@pytest.mark.parametrize("path", _STANDARDS, ids=lambda p: p.name)
def test_standard_within_embedding_token_limit(path):
    tokens, accurate = count_tokens_detail(path.read_text(encoding="utf-8"))
    if not accurate:
        pytest.skip("tiktoken unavailable — a chars/4 estimate over-counts; the ingest lets the API verify")
    assert tokens <= EMBED_TOKEN_SAFE_CEILING, (
        f"{path.relative_to(_KB.parent)} is {tokens} tokens > {EMBED_TOKEN_SAFE_CEILING} — it would be "
        f"SILENTLY DROPPED by ingest and served stale from Pinecone. Split it into two standards "
        f"(and fetch both at retrieval time)."
    )
    if tokens > EMBED_TOKEN_WARN:
        # Not a failure, but surfaces the near-limit standards so the next editor knows to split
        # BEFORE adding content rather than after an overflow.
        print(
            f"NEAR LIMIT: {path.name} at {tokens}/{EMBED_TOKEN_SAFE_CEILING} tokens "
            f"({EMBED_TOKEN_SAFE_CEILING - tokens} to spare) — split before adding more."
        )


# ── Rate-limit retry ─────────────────────────────────────────────────────────────────────────

_GEMINI_429 = ("Error embedding content (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
               "Quota exceeded for metric: embed_content_free_tier_tokens. Please retry in 36.220261269s.")


def test_retry_delay_uses_server_hint():
    assert embedding_budget.retry_delay_seconds(Exception(_GEMINI_429)) == pytest.approx(38.22, abs=0.01)


def test_retry_delay_defaults_and_ignores_other_errors():
    assert embedding_budget.retry_delay_seconds(Exception("HTTP 429 Too Many Requests")) == 60.0
    assert embedding_budget.retry_delay_seconds(ValueError("invalid api key")) is None


def test_embed_with_retry_recovers_after_rate_limit():
    calls, slept = [], []

    def flaky(text):
        calls.append(text)
        if len(calls) < 3:
            raise Exception(_GEMINI_429)
        return [0.1, 0.2]

    out = embedding_budget.embed_with_retry(flaky, "doc", sleep=slept.append)
    assert out == [0.1, 0.2]
    assert len(calls) == 3 and len(slept) == 2


def test_embed_with_retry_raises_non_rate_limit_and_gives_up():
    def broken(_):
        raise ValueError("invalid api key")

    with pytest.raises(ValueError):
        embedding_budget.embed_with_retry(broken, "doc", sleep=lambda _s: None)

    def always_429(_):
        raise Exception(_GEMINI_429)

    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        embedding_budget.embed_with_retry(always_429, "doc", attempts=3, sleep=lambda _s: None)
