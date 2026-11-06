from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import asyncio
import pytest

from PIL import Image

from src.agent.images import resolve_images
from src.agent.messaging import source_images, tell
from src.common.data_models.message_component_data_model import EmojiComponent, ImageComponent, MessageSequence, ReplyComponent, TextComponent
from src.maisaka.context.messages import ContentContextMessage


def gif_bytes():
    output = BytesIO()
    frames = [Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")]
    frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return output.getvalue()


def test_sticker_reference_keeps_original_animation(tmp_path):
    async def scenario():
        data = gif_bytes()
        sticker = EmojiComponent(binary_hash="", binary_data=data)
        message = ContentContextMessage(raw_message=MessageSequence([sticker]), visible_text="sticker",
                                        timestamp=datetime.now(), message_id="123")
        runtime = SimpleNamespace(_chat_history=[message])
        with patch("src.chat.heart_flow.heartflow_manager.heartflow_manager.get_or_create_heartflow_chat", AsyncMock(return_value=runtime)):
            available = await source_images("chat-a")
            assert "123" in available
            service = SimpleNamespace(agent_id="agent", workspace=tmp_path)
            images, attachments = await resolve_images(service, "chat-a", ["123"])
        assert images[0].binary_data == data
        saved = tmp_path / "relay_images" / (images[0].binary_hash + ".gif")
        assert attachments[0]["path"] == str(saved)
        assert saved.read_bytes() == data
        with Image.open(saved) as decoded:
            assert decoded.n_frames == 2
    asyncio.run(scenario())


def test_quoted_sticker_lookup_is_restricted_to_source_chat(tmp_path):
    async def scenario():
        data = gif_bytes()
        command = SimpleNamespace(raw_message=MessageSequence([ReplyComponent("original"), TextComponent("draw")]))
        quoted = SimpleNamespace(raw_message=MessageSequence([EmojiComponent(binary_hash="", binary_data=data)]))
        def lookup(message_id, chat_id):
            assert chat_id == "chat-a"
            return {"command": command, "original": quoted}.get(message_id)
        service = SimpleNamespace(agent_id="agent", workspace=tmp_path)
        with patch("src.agent.messaging.source_images", AsyncMock(return_value={})), \
             patch("src.services.message_service.get_message_by_id", side_effect=lookup):
            images, _ = await resolve_images(service, "chat-a", ["command"])
        assert images[0].binary_data == data
    asyncio.run(scenario())


def test_database_reference_never_leaks_to_another_chat(tmp_path):
    async def scenario():
        sticker = SimpleNamespace(raw_message=MessageSequence([EmojiComponent(binary_hash="", binary_data=gif_bytes())]))
        def lookup(message_id, chat_id):
            return sticker if message_id == "123" and chat_id == "chat-a" else None
        service = SimpleNamespace(agent_id="agent", workspace=tmp_path)
        with patch("src.agent.messaging.source_images", AsyncMock(return_value={})), \
             patch("src.services.message_service.get_message_by_id", side_effect=lookup):
            await resolve_images(service, "chat-a", ["123"])
            with pytest.raises(ValueError, match="当前聊天上下文中没有图片"):
                await resolve_images(service, "chat-b", ["123"])
    asyncio.run(scenario())


def test_agent_gif_reply_preserves_all_frames_for_chat(tmp_path):
    async def scenario():
        data = gif_bytes()
        path = tmp_path / "result.gif"
        path.write_bytes(data)
        runtime = SimpleNamespace(enqueue_internal_task=Mock(return_value={"queued": True}))
        service = SimpleNamespace(agent_id="agent", workspace=tmp_path, name="test agent", relay_tool="agent_talk",
                                  relay_kind="agent_relay", image_path=lambda value: Path(value),
                                  require_agent=lambda: object())
        with patch("src.agent.messaging.resolve_chat"), \
             patch("src.agent.messaging.chat_manager.get_session_name", return_value="test chat"), \
             patch("src.chat.heart_flow.heartflow_manager.heartflow_manager.get_or_create_heartflow_chat", AsyncMock(return_value=runtime)):
            result = await tell(service, "agent", "done", target="chat-a", images=[str(path)])
        assert result["delivered"]
        returned = runtime.enqueue_internal_task.call_args.kwargs["images"][0]
        assert isinstance(returned, ImageComponent)
        assert returned.binary_data == data
        segment = await returned.to_seg()
        import base64
        assert base64.b64decode(segment.data) == data
    asyncio.run(scenario())
