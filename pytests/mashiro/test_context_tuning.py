"""Regression tests for observed quote attribution and reply-context failures."""
from datetime import datetime
from io import BytesIO
from pathlib import Path
import tomllib
from types import SimpleNamespace

import pytest
from PIL import Image

from src.chat.message_receive.message import MsgIDMapping, SessionMessage
from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator
from src.common.data_models.message_component_data_model import (
    ImageComponent, MessageSequence, ReplyComponent, TextComponent,
)
from src.common.prompt_i18n import load_prompt, extract_prompt_placeholders, PROMPTS_ROOT
from src.common.i18n import get_locale, set_locale
from src.config.config import global_config
from src.llm_models.payload_content.context_item import (
    ContextItemBuilder, RoleType, get_item_text,
)
from src.maisaka.context.messages import ModelOutputContextMessage, SessionBackedMessage
from src.maisaka.context.planner_messages import build_planner_prefix, build_session_backed_text_message, build_planner_turn_anchor
from src.services.send_service import _build_processed_plain_text


PROFILE_PATH = Path(__file__).resolve().parents[2] / "profiles/mashiro.toml"


@pytest.fixture(autouse=True)
def synthetic_settings(monkeypatch):
    """Every test uses synthetic settings, never a deployed bot's config or API keys."""
    profile = tomllib.loads(PROFILE_PATH.read_text())
    previous_locale = get_locale()
    set_locale("zh-CN")
    monkeypatch.setattr(global_config.bot, "nickname", "真白")
    monkeypatch.setattr(global_config.bot, "qq_account", "10000")
    monkeypatch.setattr(global_config.bot, "alias_names", [])
    monkeypatch.setattr(global_config.experimental, "replyer_retro_prompt", False)
    monkeypatch.setattr(global_config.experimental, "emotion_trait", "neutral")
    monkeypatch.setattr(global_config.visual, "max_image_num", 10)
    monkeypatch.setattr(global_config.personality, "personality", "温和随和的群友。")
    monkeypatch.setattr(global_config.personality, "multiple_reply_style", [])
    monkeypatch.setattr(global_config.personality, "reply_style", profile["personality"]["reply_style"])
    monkeypatch.setattr(global_config.chat.reply_style, "group_chat_prompt", profile["chat"]["reply_style"]["group_chat_prompt"])
    yield
    set_locale(previous_locale)


@pytest.fixture
def generator(monkeypatch):
    obj = object.__new__(BaseMaisakaReplyGenerator)
    obj.chat_stream = SimpleNamespace(session_id='offline-only', is_group_session=True)
    obj._load_prompt = load_prompt
    monkeypatch.setattr(obj, '_build_group_chat_attention_block', lambda session_id: global_config.chat.reply_style.group_chat_prompt)
    monkeypatch.setattr(obj, '_build_keyword_reaction_prompt', lambda **kwargs: '')
    return obj


def test_outbound_quote_is_not_concatenated_as_own_speech():
    quote = ReplyComponent('original', '大家不要挂科哈', target_message_sender_nickname='群友甲')
    msg = SimpleNamespace(raw_message=MessageSequence([quote, TextComponent('尽量不挂，我尽量')]))
    text = _build_processed_plain_text(msg)
    assert '发送者=群友甲' in text
    assert text.index('大家不要挂科哈') < text.index('[/引用消息]') < text.index('尽量不挂，我尽量')
    assert text != '大家不要挂科哈 尽量不挂，我尽量'


@pytest.mark.asyncio
async def test_cached_inbound_quote_preserves_its_author():
    quote = ReplyComponent('original', '不是你说的', target_message_sender_id='user-1')
    result = await SessionMessage.process_reply_component(SimpleNamespace(), quote, MsgIDMapping())
    assert '发送者=user-1' in result
    assert '[/引用消息]' in result


@pytest.mark.asyncio
async def test_mapping_resolved_quote_keeps_attribution():
    quote = ReplyComponent('original')
    mapping = MsgIDMapping()
    mapping.mapping['original'] = ('他人的话', SimpleNamespace(user_cardname='甲', user_nickname='甲', user_id='user-1'))
    result = await SessionMessage.process_reply_component(SimpleNamespace(), quote, mapping)
    assert '发送者=甲' in result
    assert '他人的话' in result


def test_full_dates_distinguish_previous_day_from_new_messages():
    old = build_planner_prefix(timestamp=datetime(2026,10,31,21,42,36),user_name='甲',message_id='old')
    new = build_planner_prefix(timestamp=datetime(2026,11,1,15,18,22),user_name='乙',message_id='new')
    assert '2026-10-31 21:42:36' in old
    assert '2026-11-01 15:18:22' in new


def test_turn_anchor_uses_trigger_not_an_old_unanswered_message():
    current=SimpleNamespace(message_id='current',timestamp=datetime(2026,11,1,15,18,22),
        message_info=SimpleNamespace(user_info=SimpleNamespace(user_id='u',user_nickname='乙',user_cardname='')))
    old=build_session_backed_text_message(speaker_name='甲',text='昨天的话题',
        timestamp=datetime(2026,10,31,21,42,36),source_kind='user',message_id='old')
    text=build_planner_turn_anchor(current,0,[old])
    assert 'msg_id=current' in text
    assert '2026-11-01 15:18:22' in text
    assert '发送者=乙' in text
    assert '内部步骤 1' in text
    assert 'previous_message_id=old' in text
    assert 'gap_to_trigger=17h 35m 46s' in text
    assert '重提或引用旧事' in text


def test_reply_context_omits_planner_speech_and_preserves_own_quote_metadata(generator):
    own = build_session_backed_text_message(speaker_name='真白',text='尽量不挂',
        timestamp=datetime(2026,11,1,22,44),source_kind='guided_reply',message_id='own',
        quote_ids=['original'],is_self_message=True)
    own.visible_text = '[真白]大家不要挂科哈 尽量不挂'
    plan = ModelOutputContextMessage(ContextItemBuilder().set_role(RoleType.Assistant)
        .add_text_content('内部分析：引用甲，然后说我也是').build())
    items = generator._build_request_messages(chat_history=[own,plan],reply_message=None,
        reply_reason='完整分析：先调用 show_emoji_list 再发言',stream_id='offline-only',
        reply_tool_args={'reply_reference':'甲正在调侃名字缩写。'})
    text = '\n'.join(get_item_text(i) for i in items)
    assert items[0].role == RoleType.System
    assert '内部分析：' not in text
    assert '完整分析：' not in text
    assert 'show_emoji_list' not in text
    assert '大家不要挂科哈' not in text
    assert 'is_self_message="true"' in text and 'quote="original"' in text
    assert '2026-11-01 22:44:00' in text
    assert '【回复信息参考】\n甲正在调侃名字缩写。' in text


def test_reply_target_keeps_quote_and_own_body_distinct(generator):
    target = SimpleNamespace(platform='qq',message_id='target',timestamp=datetime(2026,11,1,19,22),
        message_info=SimpleNamespace(user_info=SimpleNamespace(user_id='someone',user_nickname='乙',user_cardname='')),
        raw_message=MessageSequence([ReplyComponent('original','我怎么就成z姓了',target_message_sender_nickname='真白'),
                                     TextComponent('你叫啥名字')]))
    text=generator._build_target_message_block(target)
    assert '2026-11-01 19:22:00' in text
    assert text.index('发送者=真白') < text.index('我怎么就成z姓了') < text.index('[/引用消息]')
    assert '- 本条发言内容：你叫啥名字' in text


def test_normal_replyer_receives_latest_ten_original_images(generator):
    buf=BytesIO();Image.new('RGB',(2,2),'white').save(buf,format='PNG')
    history=[]
    for n in range(12):
        history.append(SessionBackedMessage(raw_message=MessageSequence([
            TextComponent(build_planner_prefix(timestamp=datetime(2026,11,1,12,n),user_name='甲',message_id=str(n))),
            ImageComponent(binary_hash='',binary_data=buf.getvalue(),content='image description')]),
            visible_text='[甲]image description',timestamp=datetime(2026,11,1,12,n)))
    items=generator._build_request_messages(chat_history=history,reply_message=None,reply_reason='',
        stream_id='offline-only',enable_visual_message=True)
    image_parts=[p for i in items if hasattr(i,'parts') for p in i.parts if p.__class__.__name__=='ContextImagePart']
    assert len(image_parts)==10


def test_shared_prompt_does_not_instruct_writer_to_call_tools():
    shared=global_config.chat.reply_style.group_chat_prompt
    assert 'show_emoji_list' not in shared and 'attach_emoji' not in shared
    assert global_config.experimental.replyer_retro_prompt is False
    assert global_config.experimental.emotion_trait=='neutral'
    profile = tomllib.loads(PROFILE_PATH.read_text())
    assert profile['expression']['learning_list'][0]['use'] is False
    assert profile['expression']['learning_list'][0]['learn'] is True


@pytest.mark.parametrize('name',['maisaka_chat','maisaka_replyer','maisaka_turn_anchor'])
def test_translated_templates_keep_matching_placeholders(name):
    sets=[extract_prompt_placeholders((PROMPTS_ROOT/l/(name+'.prompt')).read_text()) for l in ['zh-CN','en-US','ja-JP']]
    assert sets[0]==sets[1]==sets[2]
