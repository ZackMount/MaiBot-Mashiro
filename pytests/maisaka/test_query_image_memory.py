from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.maisaka.builtin_tool import query_image_memory


@pytest.mark.asyncio
async def test_image_lookup_requires_a_path_only_for_multiple_images(monkeypatch: pytest.MonkeyPatch) -> None:
    first = SimpleNamespace(type="image", binary_hash="first")
    second = SimpleNamespace(type="image", binary_hash="second")
    components = [SimpleNamespace(type="text"), first]
    message = SimpleNamespace(message_id="123", raw_message=SimpleNamespace(components=components))
    queries = []

    def find_messages(**kwargs):
        queries.append(kwargs)
        return [message]

    search = AsyncMock(return_value={"success": True, "hits": []})
    monkeypatch.setattr(query_image_memory, "find_messages", find_messages)
    monkeypatch.setattr(query_image_memory.memory_service, "image_memory", search)
    context = SimpleNamespace(
        runtime=SimpleNamespace(session_id="this-chat"),
        build_failure_result=lambda _, content, **kwargs: (False, content),
        build_success_result=lambda _, content, **kwargs: (True, content),
    )
    call = SimpleNamespace(tool_name="query_image_memory", arguments={"message_id": "123"})

    ok, _ = await query_image_memory.handle_tool(context, call)
    assert ok
    assert queries == [{"session_id": "this-chat", "message_id": "123", "limit": 1}]
    assert search.await_args.kwargs["current_external_ref"] == "chat:this-chat:123:1"

    components.append(second)
    ok, detail = await query_image_memory.handle_tool(context, call)
    assert not ok and "1、2" in detail
    assert search.await_count == 1

    call.arguments["component_path"] = "2"
    ok, _ = await query_image_memory.handle_tool(context, call)
    assert ok
    assert search.await_args.kwargs["content_hash"] == "second"
