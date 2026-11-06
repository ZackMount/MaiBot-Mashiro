"""验证 Gemini 接收同一轮多个工具结果的消息结构。"""

from src.llm_models.model_client.gemini_client import _convert_messages
from src.llm_models.payload_content.context_item import (
    ContextItemMeta,
    ContextToolCall,
    FunctionCallItem,
    FunctionCallOutputItem,
)


def _call(call_id: str, name: str, turn: str) -> FunctionCallItem:
    return FunctionCallItem(
        meta=ContextItemMeta.create(logical_turn_id=turn),
        tool_call=ContextToolCall.create(call_id=call_id, func_name=name, args={}),
    )


def _output(call_id: str, name: str, turn: str) -> FunctionCallOutputItem:
    return FunctionCallOutputItem(
        meta=ContextItemMeta.create(logical_turn_id=turn),
        call_id=call_id,
        tool_name=name,
        output='{"ok":true}',
    )


def test_parallel_function_responses_are_one_content():
    items = [
        _call("one", "agent_talk", "turn-1"),
        _call("two", "reply", "turn-1"),
        _output("one", "agent_talk", "turn-1"),
        _output("two", "reply", "turn-1"),
    ]

    contents, _ = _convert_messages(items)

    assert len(contents) == 2
    assert [part.function_call.name for part in contents[0].parts] == ["agent_talk", "reply"]
    assert [part.function_response.name for part in contents[1].parts] == ["agent_talk", "reply"]
    assert [part.function_response.id for part in contents[1].parts] == ["one", "two"]


def test_sequential_calls_keep_separate_responses():
    items = [
        _call("one", "agent_status", "turn-1"),
        _output("one", "agent_status", "turn-1"),
        _call("two", "agent_talk", "turn-2"),
        _output("two", "agent_talk", "turn-2"),
    ]

    contents, _ = _convert_messages(items)

    assert len(contents) == 4
    assert contents[1].parts[0].function_response.name == "agent_status"
    assert contents[3].parts[0].function_response.name == "agent_talk"
