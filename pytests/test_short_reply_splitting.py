from types import SimpleNamespace

import pytest

from src.chat.utils import utils


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["rule", "llm"])
async def test_short_reply_with_spaces_is_one_message(monkeypatch, mode):
    monkeypatch.setattr(utils, "global_config", SimpleNamespace(
        response_post_process=SimpleNamespace(enable_response_post_process=True),
        response_splitter=SimpleNamespace(mode=mode, enable=True, enable_kaomoji_protection=False,
            max_length=512, max_sentence_num=8, max_split_num=3, enable_overflow_return_all=True),
        chinese_typo=SimpleNamespace(enable=False),
    ))
    async def unexpected_model_call(_text):
        raise AssertionError("short reply should not be sent to splitter model")
    monkeypatch.setattr(utils, "split_text_with_llm", unexpected_model_call)
    text = "说点大家不知道的，我就打了个 kb，它自己冒出来了"
    result = await utils.process_llm_response_segments_async(text)
    assert [item.text for item in result] == [text]


def test_rule_splitter_never_treats_spaces_as_boundaries(monkeypatch):
    monkeypatch.setattr(utils.random, "random", lambda: 1.0)
    result = utils._split_into_sentence_segments("你好 kb，hello world 再见。")
    assert [part for part, _ in result] == ["你好 kb", "hello world 再见"]
