"""query_image_memory 内置工具。"""

from __future__ import annotations

from typing import Optional

import asyncio

from src.A_memorix.core.image.component_paths import build_chat_external_ref, iter_message_image_components
from src.common.message_repository import find_messages
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.services.memory_service import memory_service

from .context import BuiltinToolRuntimeContext


def get_tool_spec(*, enabled: bool = True) -> ToolSpec:
    return ToolSpec(
        name="query_image_memory",
        description="查询当前聊天中某张图片之前的出现记录和相关讨论。",
        parameters_schema={
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "description": "包含目标图片的消息 ID。"},
                "component_path": {"type": "string", "description": "消息内有多张图片时填写组件路径。只有一张图片时可以省略。"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
            },
            "required": ["message_id"],
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
        enabled=enabled,
    )


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    del context
    runtime = tool_ctx.runtime
    session_id = str(runtime.session_id or "").strip()
    requested_message_id = str(invocation.arguments.get("message_id") or "").strip()
    requested_path = str(invocation.arguments.get("component_path") or "").strip()
    if not requested_message_id:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "查询图片记忆工具需要提供有效的 `message_id` 参数。",
        )
    try:
        limit = max(1, min(20, int(invocation.arguments.get("limit") or 8)))
    except (TypeError, ValueError):
        limit = 8

    messages = await asyncio.to_thread(find_messages, session_id=session_id, message_id=requested_message_id, limit=1)
    if not messages:
        return tool_ctx.build_failure_result(invocation.tool_name, "当前聊天没有这条消息。")
    message = messages[-1]
    images = dict(iter_message_image_components(message.raw_message.components))
    if not requested_path and len(images) > 1:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"这条消息有多张图片，请指定 component_path。可选路径为 {'、'.join(images)}。",
        )
    component_path = requested_path or next(iter(images), "")
    component = images.get(component_path)
    if component is None:
        return tool_ctx.build_failure_result(invocation.tool_name, "没有找到指定图片。")

    content_hash = str(component.binary_hash or "").strip()
    external_ref = build_chat_external_ref(
        chat_id=session_id,
        message_id=str(message.message_id),
        component_path=component_path,
    )
    result = await memory_service.image_memory(
        action="search",
        content_hash=content_hash,
        current_external_ref=external_ref,
        chat_id=session_id,
        session_id=session_id,
        candidate_limit=limit,
    )
    if not result.get("success", True) and result.get("error"):
        return tool_ctx.build_failure_result(invocation.tool_name, str(result["error"]), structured_content=result)

    lines: list[str] = []
    for index, hit in enumerate(result.get("hits") or [], start=1):
        kind = "同一图片" if hit.get("match_kind") == "exact_hash" else "相似图片"
        lines.append(f"{index}. {kind}，相似度 {float(hit.get('similarity') or 0.0):.3f}")
        for observation in hit.get("observations") or []:
            text = str(observation.get("text") or "").strip()
            if text:
                lines.append(f"   图片认知：{text}")
        for memory in hit.get("related_memories") or []:
            text = str(memory.get("content") or "").strip()
            if text:
                lines.append(f"   关联记忆：{text}")
    content = "\n".join(lines) if lines else "没有找到同图或达到阈值的相似图片记录。"
    return tool_ctx.build_success_result(invocation.tool_name, content, structured_content=result)
