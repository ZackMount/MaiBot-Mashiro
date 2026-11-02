"""回归验证引用归属、真实聊天上下文与当前轮次的边界。"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

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
    RoleType,
    SystemMessageItem,
    UserMessageItem,
    get_item_text,
)
from src.maisaka.context.history import build_prefixed_message_sequence, build_session_message_visible_text
from src.maisaka.context.messages import InternalContextMessage, ModelOutputContextMessage, SessionBackedMessage
from src.maisaka.context.planner_messages import (
    build_planner_prefix,
    build_planner_turn_anchor,
    build_planner_user_prefix_from_session_message,
)
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


@pytest.mark.parametrize("name", ["maisaka_chat", "maisaka_replyer", "maisaka_turn_anchor"])
def test_translated_templates_keep_matching_placeholders(name):
    placeholders = [
        extract_prompt_placeholders((PROMPTS_ROOT / locale / (name + ".prompt")).read_text())
        for locale in ["zh-CN", "en-US", "ja-JP"]
    ]
    assert placeholders[0] == placeholders[1] == placeholders[2]
