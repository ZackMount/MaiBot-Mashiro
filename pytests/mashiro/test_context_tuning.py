"""回归验证引用归属、真实聊天上下文与当前轮次的边界。"""

from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio
import tomllib

from PIL import Image
import pytest

from src.chat.message_receive.message import SessionMessage
from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import (
    ImageComponent,
    MessageSequence,
    ReplyComponent,
    TextComponent,
)
from src.common.i18n import get_locale, set_locale
from src.common.prompt_i18n import PROMPTS_ROOT, extract_prompt_placeholders, load_prompt
from src.config.config import global_config
from src.llm_models.payload_content.context_item import (
    AssistantMessageItem,
    ContextImagePart,
    ContextItemBuilder,
    ContextItemMeta,
    ContextToolCall,
    FunctionCallItem,
    RoleType,
    SystemMessageItem,
    UserMessageItem,
    get_item_text,
)
from src.maisaka.context.history import build_prefixed_message_sequence, build_session_message_visible_text
from src.maisaka.context.messages import (
    InternalContextMessage,
    ModelOutputContextMessage,
    SessionBackedMessage,
    ToolResultMessage,
)
from src.maisaka.context.planner_messages import (
    build_planner_prefix,
    build_planner_turn_anchor,
    build_planner_user_prefix_from_session_message,
)
from src.maisaka.reasoning_engine import CycleRuntimeState, MaisakaReasoningEngine
from src.maisaka.runtime import MaisakaHeartFlowChatting
from src.services.send_service import _build_processed_plain_text

PROFILE_PATH = Path(__file__).resolve().parents[2] / "profiles/mashiro.toml"


@pytest.fixture(autouse=True)
def synthetic_settings(monkeypatch):
    """使用合成配置，不读取部署账号或模型密钥。"""
    profile = tomllib.loads(PROFILE_PATH.read_text())
    previous_locale = get_locale()
    set_locale("zh-CN")
    monkeypatch.setattr(global_config.bot, "nickname", "真白")
    monkeypatch.setattr(global_config.bot, "qq_account", "10000")
    monkeypatch.setattr(global_config.bot, "alias_names", [])
    monkeypatch.setattr(global_config.debug, "replyer_retro_prompt", False)
    monkeypatch.setattr(global_config.visual, "max_image_num", 10)
    monkeypatch.setattr(global_config.personality, "personality", "温和随和的群友。")
    monkeypatch.setattr(global_config.personality, "multiple_reply_style", [])
    monkeypatch.setattr(global_config.personality, "reply_style", profile["personality"]["reply_style"])
    monkeypatch.setattr(
        global_config.chat.reply_style,
        "group_chat_prompt",
        profile["chat"]["reply_style"]["group_chat_prompt"],
    )
    yield
    set_locale(previous_locale)


@pytest.fixture
def generator(monkeypatch):
    obj = object.__new__(BaseMaisakaReplyGenerator)
    obj.chat_stream = SimpleNamespace(session_id="offline-only", is_group_session=True)
    obj._load_prompt = load_prompt
    monkeypatch.setattr(
        obj, "_build_group_chat_attention_block", lambda session_id: global_config.chat.reply_style.group_chat_prompt
    )
    monkeypatch.setattr(obj, "_build_keyword_reaction_prompt", lambda **kwargs: "")
    return obj


def make_message(
    message_id: str,
    text: str,
    timestamp: datetime = datetime(2026, 11, 1, 12),
    *,
    user_id: str = "other-user",
    name: str = "群友甲",
    session_id: str = "offline-only",
) -> SessionMessage:
    message = SessionMessage(message_id=message_id, timestamp=timestamp, platform="qq")
    message.session_id = session_id
    message.message_info = MessageInfo(user_info=UserInfo(user_id, name))
    message.raw_message = MessageSequence([TextComponent(text)])
    message.processed_plain_text = text
    return message


def chat_entry(message: SessionMessage, source_kind: str = "user") -> SessionBackedMessage:
    prefix = build_planner_user_prefix_from_session_message(message, is_self_message=source_kind == "guided_reply")
    return SessionBackedMessage.from_session_message(
        message,
        raw_message=build_prefixed_message_sequence(message.raw_message, prefix),
        visible_text=build_session_message_visible_text(message),
        source_kind=source_kind,
    )


def test_outbound_quote_is_not_concatenated_as_own_speech():
    message = make_message("response", "尽量不挂，我尽量", user_id="10000", name="真白")
    message.raw_message.components.insert(
        0, ReplyComponent("original", "大家不要挂科哈", target_message_sender_nickname="群友甲")
    )
    text = _build_processed_plain_text(message)
    assert "发送者=群友甲" in text
    assert text.index("大家不要挂科哈") < text.index("[/引用消息]") < text.index("尽量不挂，我尽量")


def test_full_dates_distinguish_previous_day_from_new_messages():
    old = build_planner_prefix(timestamp=datetime(2026, 10, 31, 21, 42, 36), user_name="甲", message_id="old")
    new = build_planner_prefix(timestamp=datetime(2026, 11, 1, 15, 18, 22), user_name="乙", message_id="new")
    assert 'time="2026-10-31 21:42:36"' in old
    assert 'time="2026-11-01 15:18:22"' in new


def test_turn_anchor_ignores_internal_notices_and_other_sessions():
    current = make_message("current", "新话题", datetime(2026, 11, 1, 15, 18, 22), name="乙")
    old = chat_entry(make_message("old", "昨天的话题", datetime(2026, 10, 31, 21, 42, 36)))
    foreign = chat_entry(make_message("foreign", "别的群", datetime(2026, 11, 1, 15), session_id="another-chat"))
    notification = make_message("notice", "戳一戳", datetime(2026, 11, 1, 15, 1))
    notification.is_notify = True
    internal = InternalContextMessage(
        raw_message=MessageSequence([TextComponent("内部参考")]),
        visible_text="内部参考",
        timestamp=datetime(2026, 11, 1, 15, 10),
    )
    text = build_planner_turn_anchor(current, 0, [old, foreign, chat_entry(notification), internal])
    assert "msg_id=current" in text
    assert "2026-11-01 15:18:22" in text
    assert "发送者=乙" in text
    assert "previous_message_id=old" in text
    assert "gap_to_trigger=17h 35m 46s" in text
    assert "重提或引用旧事" in text


def test_reply_context_omits_planner_speech_and_preserves_own_quote_metadata(generator):
    original = make_message("own", "尽量不挂", datetime(2026, 11, 1, 22, 44), user_id="10000", name="真白")
    original.raw_message.components.insert(0, ReplyComponent("original"))
    own = chat_entry(original, "guided_reply")
    own.visible_text = "[真白]大家不要挂科哈 尽量不挂"
    plan = ModelOutputContextMessage(
        ContextItemBuilder().set_role(RoleType.Assistant).add_text_content("内部分析：引用甲，然后说我也是").build()
    )
    items = generator._build_request_messages(
        chat_history=[own, plan],
        reply_message=None,
        reply_reason="完整分析：先调用 show_emoji_list 再发言",
        stream_id="offline-only",
        reply_tool_args={"reply_reference": "甲正在调侃名字缩写。"},
    )
    text = "\n".join(get_item_text(item) for item in items)
    assert items[0].role == RoleType.System
    assert "内部分析：" not in text and "完整分析：" not in text
    assert "show_emoji_list" not in text
    assert "大家不要挂科哈" not in text
    assert 'is_self_message="true"' in text and 'quote="original"' in text
    assert "2026-11-01 22:44:00" in text
    assert "【回复信息参考】\n甲正在调侃名字缩写。" in text


def test_reply_target_keeps_quote_and_own_body_distinct(generator):
    target = make_message("target", "你叫啥名字", datetime(2026, 11, 1, 19, 22))
    target.raw_message.components.insert(
        0, ReplyComponent("original", "我怎么就成z姓了", target_message_sender_nickname="真白")
    )
    text = generator._build_target_message_block(target)
    assert "2026-11-01 19:22:00" in text
    assert text.index("发送者=真白") < text.index("我怎么就成z姓了") < text.index("[/引用消息]")
    assert "- 本条发言内容：你叫啥名字" in text


def test_normal_replyer_receives_latest_ten_original_images(generator):
    with BytesIO() as buffer:
        Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    history = []
    for index in range(12):
        message = make_message(str(index), "", datetime(2026, 11, 1, 12, index))
        message.raw_message.components.append(
            ImageComponent(binary_hash="", binary_data=image_bytes, content="image description")
        )
        history.append(chat_entry(message))
    items = generator._build_request_messages(
        chat_history=history,
        reply_message=None,
        reply_reason="",
        stream_id="offline-only",
        enable_visual_message=True,
    )
    images = [
        part
        for item in items
        if isinstance(item, (SystemMessageItem, UserMessageItem, AssistantMessageItem))
        for part in item.parts
        if isinstance(part, ContextImagePart)
    ]
    assert len(images) == 10
    assert all(not image.image_base64.startswith("deferred-image:") for image in images)


def test_shared_prompt_does_not_instruct_writer_to_call_tools():
    shared = global_config.chat.reply_style.group_chat_prompt
    assert "show_emoji_list" not in shared and "attach_emoji" not in shared


@pytest.mark.asyncio
@pytest.mark.parametrize("question_second", [3, 6])
async def test_queued_question_precedes_bot_reply_and_keeps_tool_pair(generator, question_second):
    """补充消息无论在规划前还是工具执行中到达，都不能变成对机器人回复的追问。"""
    video = make_message("video", "分享视频", datetime(2026, 11, 1, 12))
    question = make_message("question", "？", datetime(2026, 11, 1, 12, 0, question_second))
    own = chat_entry(
        make_message("own", "像考古", datetime(2026, 11, 1, 12, 0, 7), user_id="10000", name="真白"),
        "guided_reply",
    )
    call = ModelOutputContextMessage(FunctionCallItem(
        meta=ContextItemMeta.create(logical_turn_id="turn", timestamp=datetime(2026, 11, 1, 12, 0, 5)),
        tool_call=ContextToolCall("call", "reply", b"{}"),
    ))
    result = ToolResultMessage(
        content="回复已发送", timestamp=datetime(2026, 11, 1, 12, 0, 8),
        tool_call_id="call", logical_turn_id="turn", tool_name="reply",
    )
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = SimpleNamespace(
        _chat_history=[chat_entry(video), call, own, result],
        _is_focus_mode_active_for_current_chat=lambda: False,
    )

    await engine._ingest_messages([question])

    history = engine._runtime._chat_history
    assert [message.message_id for message in history if isinstance(message, SessionBackedMessage)] == [
        "video", "question", "own",
    ]
    assert history[history.index(call) + 1] is result
    for entry in (call, result):
        assert sum(message is entry for message in history) == 1
    items = generator._build_request_messages(
        chat_history=history, reply_message=question, reply_reason="", stream_id="offline-only",
    )
    text = "\n".join(get_item_text(item) for item in items)
    assert text.index('msg_id="question"') < text.index('msg_id="own"')


def test_equal_timestamp_messages_keep_arrival_order():
    first = chat_entry(make_message("first", "第一条"))
    second = chat_entry(make_message("second", "第二条"))
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = SimpleNamespace(_chat_history=[first])

    engine._insert_chat_history_message(second)

    assert engine._runtime._chat_history == [first, second]


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_second,expected_target", [(3, "pending"), (-1, "first")])
async def test_planner_includes_messages_received_during_context_preparation(monkeypatch, pending_second, expected_target):
    first = make_message("first", "分享视频", datetime(2026, 11, 1, 12))
    pending = make_message("pending", "？", first.timestamp + timedelta(seconds=pending_second))
    runtime = object.__new__(MaisakaHeartFlowChatting)
    runtime.session_id = "offline-only"
    runtime.log_prefix = "[测试]"
    runtime.message_cache = [first]
    runtime._last_processed_index = 1
    runtime._chat_history = [chat_entry(first)]
    runtime._fixed_message_debouncer = SimpleNamespace(consume=lambda: None)
    runtime._wait_for_message_quiet_period = AsyncMock()
    runtime._mark_message_turn_unscheduled = lambda: None
    runtime._is_focus_mode_active_for_current_chat = lambda: False
    runtime._update_stage_status = lambda *args, **kwargs: None
    runtime._cycle_counter = 1
    runtime._consecutive_wait_count = 0
    engine = MaisakaReasoningEngine(runtime)
    engine._emit_flow_step = AsyncMock()
    engine._refresh_jargon_reference_message = lambda: None
    engine._run_interruptible_planner = AsyncMock(return_value=SimpleNamespace())
    started = asyncio.Event()
    release = asyncio.Event()
    memory_search = AsyncMock(side_effect=AssertionError("Planner 发出前不应查长期记忆"))
    image_search = AsyncMock(side_effect=AssertionError("Planner 发出前不应查图片记忆"))
    from src.services.memory_service import memory_service
    monkeypatch.setattr(memory_service, "search", memory_search)
    monkeypatch.setattr(memory_service, "image_memory", image_search)

    async def prepare_tools():
        started.set()
        await release.wait()
        return [], "可发现工具"

    monkeypatch.setattr(engine, "_build_action_tool_definitions", prepare_tools)
    state = CycleRuntimeState()
    task = asyncio.create_task(engine._run_planner_request(
        trigger_message=first, source_messages=[first], round_index=0, round_text="第 1 轮", state=state,
    ))
    await asyncio.wait_for(started.wait(), timeout=1)
    runtime.message_cache.append(pending)
    release.set()
    trigger, sources = await task

    assert trigger.message_id == expected_target
    assert state.reply_target_message.message_id == expected_target
    assert sources == [first, pending]
    assert not runtime._has_pending_messages()
    assert [message.message_id for message in runtime._chat_history] == (
        ["first", "pending"] if pending_second > 0 else ["pending", "first"]
    )
    engine._run_interruptible_planner.assert_awaited_once()
    args = engine._run_interruptible_planner.call_args.kwargs
    assert f"msg_id={expected_target}" in args["tail_user_messages"][0]
    assert args["injected_user_messages"] == ["可发现工具"]
    memory_search.assert_not_awaited()
    image_search.assert_not_awaited()


@pytest.mark.parametrize("name", ["maisaka_chat", "maisaka_replyer", "maisaka_turn_anchor"])
def test_translated_templates_keep_matching_placeholders(name):
    placeholders = [
        extract_prompt_placeholders((PROMPTS_ROOT / locale / (name + ".prompt")).read_text())
        for locale in ["zh-CN", "en-US", "ja-JP"]
    ]
    assert placeholders[0] == placeholders[1] == placeholders[2]
