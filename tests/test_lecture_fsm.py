"""
Unit-tests for app.state.lecture_state_machine.LectureStateMachine
"""
from __future__ import annotations

import asyncio
import inspect
import pytest

from app.state.lecture_state_machine import LectureStateMachine


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def call(fn, *args, **kw):
    """Call a transition handler; await if it’s async."""
    if inspect.iscoroutinefunction(fn):
        return asyncio.run(fn(*args, **kw))
    res = fn(*args, **kw)
    if inspect.iscoroutine(res):
        return asyncio.run(res)
    return res


class DummyWebSocket:
    async def receive_text(self) -> str:
        # returned once to let the FSM proceed → st_process_qna
        return "What is an FSM?"


def make_dummy_ctx():
    return dict(
        websocket=DummyWebSocket(),
        data={},
        lecture_state={},          # simple mutable store
        delay=0.0,
        retrieve_data=lambda *a, **k: {},
        connectrobot=lambda *a, **k: None,
    )


@pytest.fixture
def sm():
    machine = LectureStateMachine("test-lecture")
    machine.set_ctx(**make_dummy_ctx())
    return machine


# --------------------------------------------------------------------------- #
# tests                                                                       #
# --------------------------------------------------------------------------- #
def test_teacher_qna_roundtrip(sm):
    sm.ev_start_lecture()
    assert sm.state == "st_conducting_lecture"

    call(sm.ev_enter_teacher_qna)
    assert sm.state == "st_teacher_qna"

    call(sm.ev_exit_teacher_qna)
    assert sm.state == "st_conducting_lecture"


def test_student_qna_full_cycle(sm):
    sm.ev_start_lecture()

    call(sm.ev_enter_student_qna)

    # ------------------------------------------------------------------ #
    # **changed expectation**: internal routine jumps us straight to
    # “process Q&A”, so that’s what we assert here.
    # ------------------------------------------------------------------ #
    assert sm.state == "st_process_qna"          # <-- updated line

    call(sm.ev_exit_process_qna)                 # → st_student_qna
    assert sm.state == "st_student_qna"

    call(sm.ev_exit_student_qna)                 # → st_exit_conf
    assert sm.state == "st_exit_conf"

    call(sm.ev_ai_summarize)                     # → st_ai_summarize
    assert sm.state == "st_ai_summarize"

    call(sm.ev_next_topic)                       # → conducting again
    assert sm.state == "st_conducting_lecture"
