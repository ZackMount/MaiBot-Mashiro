from src.llm_models.model_client.openai_responses_client import _convert_tool_options
from src.llm_models.payload_content.tool_option import normalize_tool_options


def test_optional_reply_attachments_are_not_made_mandatory():
    tools = normalize_tool_options([
        {
            "type": "function",
            "function": {
                "name": "reply",
                "description": "Send a reply",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "msg_id": {"type": "string"},
                        "attach_emoji": {"type": "integer"},
                    },
                    "required": ["msg_id"],
                },
            },
        }
    ])
    converted = _convert_tool_options(tools)
    assert converted[0]["strict"] is False
    assert converted[0]["parameters"]["required"] == ["msg_id"]
    assert "attach_emoji" in converted[0]["parameters"]["properties"]
