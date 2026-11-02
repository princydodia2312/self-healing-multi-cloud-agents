"""--resume: a crashed run continues from its SQLite checkpoint instead of starting over."""
import operator
import sqlite3
from types import SimpleNamespace
from typing import Annotated, TypedDict
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

pytest.importorskip("langgraph.checkpoint.sqlite")
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

import main  # noqa: E402


class _S(TypedDict):
    messages: Annotated[list, add_messages]
    steps: Annotated[list, operator.add]
    project_id: str


def _graph(cp, crash_on_b):
    def a(s):
        # Gemini-style tool-call turn: thought signature bytes + tool_calls survive the checkpoint.
        return {"steps": ["a"], "messages": [
            AIMessage(content="", tool_calls=[{"name": "t", "args": {"x": 1}, "id": "c1"}],
                      additional_kwargs={"__gemini_function_call_thought_signatures__": {"c1": b"\x01sig"}}),
            ToolMessage(tool_call_id="c1", content="ok")]}

    def b(s):
        if crash_on_b["on"]:
            raise RuntimeError("429 all models exhausted")
        return {"steps": ["b"]}

    g = StateGraph(_S)
    g.add_node("a", a)
    g.add_node("b", b)
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile(checkpointer=cp)


def test_crash_then_resume_continues_without_redoing_steps(tmp_path):
    cp = SqliteSaver(sqlite3.connect(tmp_path / "c.sqlite", check_same_thread=False))
    crash = {"on": True}
    app = _graph(cp, crash)
    cfg = {"configurable": {"thread_id": "eu_sales"}}
    with pytest.raises(RuntimeError):
        list(app.stream({"messages": [HumanMessage(content="go")], "steps": [], "project_id": "EU_SALES-1"}, cfg))
    snap = app.get_state(cfg)
    assert snap.next == ("b",)
    ai = [m for m in snap.values["messages"] if isinstance(m, AIMessage)][0]
    assert ai.tool_calls[0]["args"] == {"x": 1}
    assert ai.additional_kwargs["__gemini_function_call_thought_signatures__"]["c1"] == b"\x01sig"

    crash["on"] = False
    list(app.stream(None, cfg))
    assert app.get_state(cfg).values["steps"] == ["a", "b"]       # "a" not re-run


def test_resume_input_restores_project_id(monkeypatch):
    snap = SimpleNamespace(values={"project_id": "EU_SALES-20261102-0714"}, next=("medic",))
    monkeypatch.setenv("PROJECT_ID", "EU_SALES-NEW")
    with patch.object(main, "app", SimpleNamespace(get_state=lambda cfg: snap)):
        assert main._resume_input({}, {"fresh": True}) is None
    assert main.os.environ["PROJECT_ID"] == "EU_SALES-20261102-0714"


def test_resume_input_without_pending_run_starts_fresh():
    done = SimpleNamespace(values={"project_id": "X"}, next=())
    with patch.object(main, "app", SimpleNamespace(get_state=lambda cfg: done)):
        assert main._resume_input({}, {"fresh": True}) == {"fresh": True}
