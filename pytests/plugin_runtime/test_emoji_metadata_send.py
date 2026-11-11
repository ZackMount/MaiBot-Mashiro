from types import SimpleNamespace
from unittest.mock import AsyncMock

import base64

import pytest

from src.common.utils import image_path
from src.emoji_system.emoji_manager import emoji_manager
from src.plugin_runtime.capabilities.core import RuntimeCoreCapabilityMixin
from src.plugin_runtime.capabilities.data import RuntimeDataCapabilityMixin
from src.services import send_service


@pytest.mark.asyncio
async def test_emoji_metadata_does_not_read_images(monkeypatch):
    record = SimpleNamespace(file_hash="one", description="开心")
    monkeypatch.setattr(emoji_manager, "emojis", [record])
    result = await RuntimeDataCapabilityMixin()._cap_emoji_get_all(
        "selector", "emoji.get_all", {"include_binary_data": False}
    )
    assert result == {"success": True, "emojis": [{"emoji_hash": "one", "description": "开心"}]}


@pytest.mark.asyncio
async def test_send_emoji_reads_only_the_selected_hash_and_preserves_base64_callers(monkeypatch, tmp_path):
    image = tmp_path / "selected.gif"
    original = b"GIF89a-original-bytes"
    image.write_bytes(original)
    monkeypatch.setattr(image_path, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        emoji_manager, "emojis", [SimpleNamespace(file_hash="chosen", full_path=image)]
    )
    send = AsyncMock(return_value=SimpleNamespace(platform_message_id="receipt"))
    monkeypatch.setattr(send_service, "emoji_to_stream_with_message", send)
    host = RuntimeCoreCapabilityMixin()
    args = {"stream_id": "chat", "sync_to_maisaka_history": True}

    result = await host._cap_send_emoji("selector", "send.emoji", {**args, "emoji_hash": "chosen"})
    assert result["success"]
    assert base64.b64decode(send.call_args.kwargs["emoji_base64"]) == original
    assert send.call_args.kwargs["sync_to_maisaka_history"]

    result = await host._cap_send_emoji("selector", "send.emoji", {**args, "emoji_hash": "removed"})
    assert not result["success"]
    assert send.await_count == 1

    result = await host._cap_send_emoji("existing-plugin", "send.emoji", {**args, "emoji_base64": "old-value"})
    assert result["success"]
    assert send.call_args.kwargs["emoji_base64"] == "old-value"
