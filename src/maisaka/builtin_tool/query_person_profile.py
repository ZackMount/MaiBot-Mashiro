"""query_person_profile 内置工具。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.services.memory_service import memory_service

from .context import BuiltinToolRuntimeContext

DEFAULT_PROFILE_LIMIT = 8


def get_tool_spec(*, enabled: bool = True) -> ToolSpec:
    """获取 query_person_profile 工具声明。"""

    return ToolSpec(
        name="query_person_profile",
        description="查询当前聊天中与某人相关的记忆、身份和互动；用户明确提到其他群时可跨群查询。",
        parameters_schema={
            "type": "object",
            "properties": {
                "person_id": {
                    "type": "string",
                    "description": "内部人物ID；明确给出时填。",
                },
                "person_name": {
                    "type": "string",
                    "description": "名称/昵称/关键词；通常填这个。",
                },
                "limit": {
                    "type": "integer",
                    "description": "证据上限。",
                    "default": DEFAULT_PROFILE_LIMIT,
                },
                "scope": {"type": "string", "enum": ["current", "groups"], "default": "current",
                          "description": "默认只查当前聊天；只有用户明确要求查其他群或主动提到其他群经历时才用 groups。"},
            },
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
        enabled=enabled,
    )


def _normalize_limit(raw_limit: Any) -> int:
    try:
        limit = int(raw_limit or DEFAULT_PROFILE_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_PROFILE_LIMIT
    return max(1, min(limit, 20))


def _extract_profile_text(payload: Dict[str, Any]) -> str:
    return str(payload.get("profile_text") or payload.get("summary") or "").strip()


def _extract_traits(profile_text: str) -> list[str]:
    traits: list[str] = []
    for line in profile_text.splitlines():
        clean_line = line.strip().strip("-").strip()
        if clean_line:
            traits.append(clean_line)
        if len(traits) >= 8:
            break
    return traits


def _build_structured_content(
    *,
    payload: Dict[str, Any],
    requested_person_id: str,
    requested_person_name: str,
    limit: int,
) -> Dict[str, Any]:
    profile_text = _extract_profile_text(payload)
    return {
        "success": bool(payload.get("success")),
        "summary": profile_text,
        "traits": _extract_traits(profile_text),
        "person_id": str(payload.get("person_id") or requested_person_id or "").strip(),
        "person_name": str(payload.get("person_name") or requested_person_name or "").strip(),
        "profile_source": str(payload.get("profile_source") or "").strip(),
        "has_manual_override": bool(payload.get("has_manual_override", False)),
        "from_cache": bool(payload.get("from_cache", False)),
        "limit": limit,
    }


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """执行 query_person_profile 内置工具。"""

    person_id = str(invocation.arguments.get("person_id") or "").strip()
    person_name = str(invocation.arguments.get("person_name") or "").strip()
    if not person_id and not person_name:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "query_person_profile 需要提供 person_id 或 person_name 中的一个。",
        )

    from .query_memory import handle_tool as query_memory

    arguments = {
        "query": person_name or person_id,
        "person_name": person_name,
        "person_id": person_id,
        "limit": _normalize_limit(invocation.arguments.get("limit")),
        "scope": invocation.arguments.get("scope") or "current",
    }
    return await query_memory(tool_ctx, ToolInvocation(
        tool_name=invocation.tool_name, arguments=arguments, call_id=invocation.call_id,
        session_id=invocation.session_id, stream_id=invocation.stream_id,
        reasoning=invocation.reasoning, metadata=invocation.metadata,
    ), context)
