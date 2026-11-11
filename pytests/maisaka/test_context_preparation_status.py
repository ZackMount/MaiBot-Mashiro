from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio

import pytest

from src.chat.heart_flow.heartFC_utils import CycleDetail
from src.maisaka import reasoning_engine
from src.maisaka.context.compaction import ContextSummaryMessage
from src.maisaka.context.post_processor import HistoryPostProcessResult


@pytest.mark.asyncio
async def test_compaction_does_not_hold_next_round_or_replace_new_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    release_summary = asyncio.Event()
    old = SimpleNamespace(count_in_context=True)
    recent = object()
    incoming = object()
    summary = ContextSummaryMessage(content="状态摘要", timestamp=datetime.now())

    async def build(removed, **kwargs):
        assert removed == [old]
        await release_summary.wait()
        return summary

    def trim(history, *, trim_context, **kwargs):
        if trim_context:
            return HistoryPostProcessResult([recent], [old], 1, 1, 1)
        return HistoryPostProcessResult(history, [], 0, 0, len(history))

    monkeypatch.setattr(reasoning_engine, "build_context_summary", build)
    monkeypatch.setattr(reasoning_engine, "process_chat_history_after_cycle", trim)
    monkeypatch.setattr(reasoning_engine, "emit_flow_step", AsyncMock())
    runtime = SimpleNamespace(
        session_id="session", log_prefix="[测试]", _chat_history=[old, recent], _max_context_size=2,
        _chat_loop_service=SimpleNamespace(planner_request_snapshot=object()), _monitor_run_id="run",
        _get_pending_wait_tool_call_ids=set, _log_history_trimmed=lambda *_: None,
        _trigger_trimmed_history_learning=AsyncMock(),
    )
    engine = reasoning_engine.MaisakaReasoningEngine(runtime)

    try:
        await asyncio.wait_for(engine._post_process_chat_history_after_cycle(CycleDetail(cycle_id=1)), timeout=1)
        task = engine._context_summary_task
        assert task is not None and not task.done()
        runtime._chat_history.append(incoming)
        runtime._chat_loop_service.planner_request_snapshot = object()
        await engine._post_process_chat_history_after_cycle(CycleDetail(cycle_id=2))
        assert engine._context_summary_task is task
        assert runtime._chat_history == [recent, incoming]
        release_summary.set()
        await asyncio.wait_for(task, timeout=1)
        engine._apply_finished_context_summary()
        assert runtime._chat_history == [summary, recent, incoming]
    finally:
        await engine.cancel_background_context_tasks()
