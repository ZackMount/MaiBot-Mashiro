from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.common.data_models.message_component_data_model import AtComponent, TextComponent
from src.maisaka.memory import heuristic_injector
from src.services.memory_service import MemoryHit, MemorySearchResult


@pytest.mark.asyncio
async def test_question_recall_uses_current_chat_before_cached_impression(monkeypatch: pytest.MonkeyPatch) -> None:
    session = SimpleNamespace(session_id="group-a", group_id="123", user_id=None)
    message = SimpleNamespace(
        session_id=session.session_id,
        message_id="request",
        platform="qq",
        raw_message=SimpleNamespace(components=[AtComponent("bot", "真白"), TextComponent(" imgg 现在哪")]),
    )
    monkeypatch.setattr(heuristic_injector, "is_bot_self", lambda _, user_id: user_id == "bot")
    monkeypatch.setattr(
        heuristic_injector.chat_manager, "get_existing_session_by_session_id", lambda _: session
    )
    config = heuristic_injector.global_config.a_memorix.integration
    monkeypatch.setattr(config, "heuristic_memory_recall_enabled", True)
    monkeypatch.setattr(config, "heuristic_memory_cross_chat_enabled", True)
    search = AsyncMock(
        side_effect=[
            MemorySearchResult(hits=[
                MemoryHit("另一群的 imgg 近况", hit_type="episode", metadata={"source": "chat_summary:group-b"}),
            ]),
            MemorySearchResult(hits=[
                MemoryHit("imgg已回美国", hit_type="paragraph", hash_value="same-chat"),
            ]),
        ]
    )
    monkeypatch.setattr(heuristic_injector.memory_service, "search", search)
    injector = heuristic_injector.HeuristicMemoryInjector()
    monkeypatch.setattr(
        injector, "_resolve_hit_sources_by_hash", AsyncMock(return_value={"same-chat": "chat_summary:group-a"})
    )
    injector._states[session.session_id] = heuristic_injector.HeuristicMemoryRecallState(
        cached_reference="无关印象", cache_expires_at=time() + 600
    )

    reference = await injector.build_injection_message(session_id=session.session_id, question_message=message)

    assert "imgg已回美国" in reference
    assert "另一群" not in reference
    assert "无关印象" not in reference
    assert {call.kwargs["mode"] for call in search.await_args_list} == {"episode", "search"}
    for call in search.await_args_list:
        assert call.args == ("imgg 现在哪",)
        assert call.kwargs["chat_id"] == session.session_id
        assert call.kwargs["cross_group"] is False

    assert await injector.build_injection_message(session_id=session.session_id, question_message=message) == reference
    assert search.await_count == 2

    message.raw_message.components = [TextComponent("哈哈，好有意思")]
    monkeypatch.setattr(heuristic_injector, "count_messages", lambda **_: config.heuristic_memory_recall_window_size)
    assert await injector.build_injection_message(session_id=session.session_id, question_message=message) == "无关印象"
    assert search.await_count == 2
