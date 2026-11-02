from datetime import datetime
from types import SimpleNamespace

import pytest

from src.chat.message_receive.message import SessionMessage
from src.chat.utils import utils
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.config.config import global_config
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
from src.maisaka.builtin_tool.reply import _resolve_reply_quote
from src.maisaka.quote.counter import ReplyQuoteCounter
from src.maisaka.quote.rules import quote_probability


def quote_context(account_id):
    now = datetime.now()
    target = SessionMessage("target", now, "qq")
    target.session_id = "chat"
    target.message_info = MessageInfo(user_info=UserInfo("someone", "群友"))
    own = SessionMessage("own", now, "qq")
    own.session_id = "chat"
    own.message_info = MessageInfo(user_info=UserInfo("bot-a", "机器人"))
    counter = ReplyQuoteCounter("chat")
    counter.record(target)
    counter.record(own)
    runtime = SimpleNamespace(
        chat_stream=SimpleNamespace(platform="qq", account_id=account_id, is_group_session=True),
        get_reply_quote_context=lambda message: counter.snapshot(message, now=now),
    )
    return BuiltinToolRuntimeContext(None, runtime), target


@pytest.mark.asyncio
@pytest.mark.parametrize("account_id,known", [(None, {"bot-a"}), ("bot-a", {"bot-a", "bot-b"})])
async def test_reply_quote_resolves_known_sender(monkeypatch, account_id, known):
    monkeypatch.setattr(utils, "get_bot_accounts", lambda platform: known)
    monkeypatch.setattr(global_config.chat.reply_style, "group_quote_frequency", "occasionally")
    context, target = quote_context(account_id)

    _, probability = await _resolve_reply_quote(context, "收到", target)

    assert probability == pytest.approx(quote_probability(
        text_length=2, previous_sender_self=True, previous_seconds=0,
        frequency="occasionally", target_distance=1,
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("known", [set(), {"bot-a", "bot-b"}])
async def test_reply_quote_rejects_unknown_or_ambiguous_sender(monkeypatch, known):
    monkeypatch.setattr(utils, "get_bot_accounts", lambda platform: known)
    monkeypatch.setattr(global_config.chat.reply_style, "group_quote_frequency", "occasionally")
    context, target = quote_context(None)

    with pytest.raises(ValueError, match="需要明确的 bot 发送账号"):
        await _resolve_reply_quote(context, "收到", target)
