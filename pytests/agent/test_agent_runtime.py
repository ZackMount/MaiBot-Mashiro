from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock
from io import BytesIO

import asyncio
import json

import pytest

from src.agent import service as service_module
from src.agent import messaging
from src.agent.acp import ACPRequestError
from src.agent.acp_agent import AcpAgent
from src.agent.config import AgentConfig
from src.agent.history import AgentHistory
from src.agent.service import AgentService
from src.llm_models.payload_content.context_item import get_item_text


@pytest.fixture
def service(tmp_path, monkeypatch):
    config = AgentConfig(enabled=True)
    monkeypatch.setattr(service_module, "global_config", SimpleNamespace(
        agent=config, bot=SimpleNamespace(nickname="TestMai"),
        personality=SimpleNamespace(personality="你是测试猫娘"),
    ))
    return AgentService(tmp_path)


class RespondingConnection:
    def __init__(self, agent: AcpAgent, stop_reason: str = "end_turn") -> None:
        self.agent = agent
        self.stop_reason = stop_reason
        self.prompts: List[Dict[str, Any]] = []
        self.failure = None

    async def request(self, method: str, params: Dict[str, Any]) -> Dict[str, str]:
        assert method == "session/prompt"
        self.prompts.append(params)
        for text in ("answer ", "with\n", "spacing"):
            await self.agent._update({"sessionId": "external", "update": {
                "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text},
            }})
        return {"stopReason": self.stop_reason}

    async def close(self, timeout: float) -> None:
        pass


async def prepare_agent(service: AgentService, monkeypatch, reason: str = "end_turn") -> AcpAgent:
    agent = AcpAgent(service)
    await agent.load()
    agent.backend_state = {"session_id": "external", "policy_sent": True}
    agent.connection = RespondingConnection(agent, reason)
    monkeypatch.setattr(agent, "connect", AsyncMock())
    service.agent = agent
    return agent


def test_isolated_tasks_open_separate_sessions_and_do_not_batch(service, monkeypatch):
    async def scenario():
        service.config.acp_session_per_task = True
        agent = AcpAgent(service)
        await agent.load()
        service.agent = agent
        sessions = []
        prompts = []

        class TaskConnection:
            failure = None

            async def request(self, method, params):
                assert method == "session/prompt"
                prompts.append(params)
                return {"stopReason": "end_turn"}

            async def close(self, timeout):
                pass

        async def connect():
            assert "session_id" not in agent.backend_state
            session_id = f"session-{len(sessions) + 1}"
            sessions.append(session_id)
            agent.backend_state["session_id"] = session_id
            agent.connection = TaskConnection()

        monkeypatch.setattr(agent, "connect", connect)
        await agent.enqueue("first", reply_to="chat-a")
        await agent.enqueue("second", reply_to="chat-b")
        await agent.run_turn()
        assert len(agent.inbox) == 1
        await agent.run_turn()

        assert sessions == ["session-1", "session-2"]
        assert [p["sessionId"] for p in prompts] == sessions
        assert "first" in [part["text"] for part in prompts[0]["prompt"]]
        assert "second" not in [part["text"] for part in prompts[0]["prompt"]]
        assert "second" in [part["text"] for part in prompts[1]["prompt"]]
        assert "first" not in [part["text"] for part in prompts[1]["prompt"]]
        assert all(prompt["prompt"][0]["text"] == service.policy() for prompt in prompts)
        assert not agent.inbox
        assert "session_id" not in agent.backend_state
        assert [event["acp_session_id"] for event in agent.history.events() if event["kind"] == "user"] == sessions

    asyncio.run(scenario())


def test_failed_isolated_task_does_not_poison_next_task(service, monkeypatch):
    async def scenario():
        service.config.acp_session_per_task = True
        agent = AcpAgent(service)
        await agent.load()
        service.agent = agent
        sessions = []

        class TaskConnection:
            failure = None

            async def request(self, method, params):
                assert method == "session/prompt"
                if params["sessionId"] == "session-1":
                    raise ACPRequestError({"code": 413, "message": "FUNCTION_PAYLOAD_TOO_LARGE"})
                return {"stopReason": "end_turn"}

            async def close(self, timeout):
                pass

        async def connect():
            session_id = f"session-{len(sessions) + 1}"
            sessions.append(session_id)
            agent.backend_state["session_id"] = session_id
            agent.connection = TaskConnection()

        monkeypatch.setattr(agent, "connect", connect)
        first = await agent.enqueue("large task", reply_to="chat-a")
        await agent.enqueue("next task", reply_to="chat-b")
        with pytest.raises(Exception, match="FUNCTION_PAYLOAD_TOO_LARGE") as error:
            await agent.run_turn()
        assert error.value.reply_targets == {"chat-a": [first]}
        assert len(agent.inbox) == 1
        assert agent.pending is None
        await agent.run_turn()
        assert sessions == ["session-1", "session-2"]
        assert not agent.inbox

    asyncio.run(scenario())


def test_animated_reference_is_preserved_but_acp_preview_is_png(service):
    from PIL import Image
    from src.common.data_models.message_component_data_model import ImageComponent
    from src.llm_models.payload_content.context_item import ContextImagePart

    async def scenario():
        output = BytesIO()
        frames = [Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")]
        frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
        data = output.getvalue()
        image = ImageComponent(binary_hash="", binary_data=data)
        agent = AcpAgent(service)
        await agent.load()
        await agent.enqueue("参考动图", images=[image])
        item = agent.history.unpack(agent.inbox[0]["items"])[0]
        preview = next(part for part in item.parts if isinstance(part, ContextImagePart))
        assert preview.image_format == "png"
        assert image.binary_data == data
    asyncio.run(scenario())


@pytest.mark.parametrize("locale", ["zh-CN", "en-US", "ja-JP"])
def test_policy_preserves_partner_identity(service, monkeypatch, locale):
    monkeypatch.setattr("src.common.prompt_i18n.get_locale", lambda: locale)
    policy = service.policy()
    assert '"nickname": "TestMai"' in policy
    assert '"personality": "你是测试猫娘"' in policy
    assert "{maibot_profile}" not in policy
    assert "Windows" not in policy
    assert "computer_*" not in policy
    assert "C:\\MaiWorkspace" not in policy



def test_output_is_persisted_without_relay_and_policy_is_refreshed(service, monkeypatch):
    async def scenario():
        agent = await prepare_agent(service, monkeypatch)
        connection = agent.connection
        await agent.enqueue("first")
        await agent.run_turn()
        assert connection.prompts[0]["prompt"][0]["text"] == service.policy()

        # 从磁盘恢复，证明没有 agent_talk 的普通文本也已完整保存，并可供 WebUI 读取。
        history = AgentHistory(service.root)
        await asyncio.to_thread(history.load)
        text = "".join(get_item_text(item) for record in history.records if record["kind"] == "step"
                       for item in history.unpack(record["items"]))
        assert text == "answer with\nspacing"
        assert len(history.events()) == 4
        assert agent.pending is None

        await agent.enqueue("second")
        await agent.run_turn()
        assert service.policy() not in [part["text"] for part in connection.prompts[1]["prompt"]]

        service_module.global_config.personality.personality = "更新后的测试人格"
        await agent.enqueue("third")
        await agent.run_turn()
        assert connection.prompts[2]["prompt"][0]["text"] == service.policy()

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", ["refusal", "max_tokens", "max_turn_requests", "cancelled"])
def test_abnormal_stop_allows_new_request_with_output_retained_and_no_replay(service, monkeypatch, reason):
    async def scenario():
        agent = await prepare_agent(service, monkeypatch, reason)
        connection = agent.connection
        await agent.enqueue("task")
        service._task = asyncio.create_task(service._run())
        try:
            async def wait_for_failure():
                while not service.error or service._turn_task is not None:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_failure(), 3)
            assert reason in service.error
            assert not service.paused
            assert agent.backend_state["stop_reason"] == reason
            assert agent.pending is None
            assert not agent.inbox
            assert len(agent.history.events()) == 4
            # 失败后没有新输入就不重放；新请求仍可在同一个会话执行。
            await asyncio.sleep(0)
            assert len(connection.prompts) == 1
            connection.stop_reason = "end_turn"
            await agent.enqueue("continue")
            async def wait_for_success():
                while len(connection.prompts) < 2 or service._turn_task is not None:
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(wait_for_success(), 3)
            assert not service.error
            assert connection.prompts[1]["sessionId"] == "external"
            texts = [part["text"] for part in connection.prompts[1]["prompt"]]
            assert "continue" in texts
            assert "task" not in texts
        finally:
            await service.stop()

        state = json.loads((service.root / "state.json").read_text(encoding="utf-8"))
        assert state["backends"]["acp"]["pending"] is None
        assert state["backends"]["acp"]["backend_state"]["stop_reason"] == "end_turn"

    asyncio.run(scenario())


@pytest.mark.parametrize("notification_fails", [False, True])
def test_rpc_failure_reports_sources_and_runs_queued_followup(service, monkeypatch, notification_fails):
    async def scenario():
        agent = await prepare_agent(service, monkeypatch)
        first_id = await agent.enqueue("draw", reply_to="chat-a")
        second_id = await agent.enqueue("send image", reply_to="chat-a")
        third_id = await agent.enqueue("another request", reply_to="chat-b")
        stored = json.loads((service.root / "state.json").read_text(encoding="utf-8"))
        assert stored["inbox"][0]["reply_to"] == "chat-a"
        original_request = agent.connection.request
        calls = []
        completed = asyncio.Event()

        async def request(method, params):
            calls.append(params)
            if len(calls) == 1:
                # 模拟流中断前已经输出并执行过操作，且下一条新请求已经到达。
                await original_request(method, params)
                await agent.enqueue("model changed, continue", reply_to="chat-c")
                raise ACPRequestError({"code": -32603, "message": "PI_AI_ERROR request_id=test"})
            result = await original_request(method, params)
            completed.set()
            return result

        notifications = AsyncMock()
        if notification_fails:
            notifications.side_effect = [RuntimeError("chat runtime unavailable"), None]
        monkeypatch.setattr(messaging, "report_failure", notifications)
        monkeypatch.setattr(agent.connection, "request", request)
        service._task = asyncio.create_task(service._run())
        try:
            await asyncio.wait_for(completed.wait(), 3)
            async def wait_for_idle():
                while service._turn_task is not None:
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(wait_for_idle(), 3)
            assert not service.paused
            assert not service.error
            assert agent.pending is None
            assert len(calls) == 2
            assert all(call["sessionId"] == "external" for call in calls)
            texts = [part["text"] for part in calls[1]["prompt"]]
            assert "model changed, continue" in texts
            assert "draw" not in texts
            assert "send image" not in texts
            assert "another request" not in texts
            assert notifications.await_count == 2
            targets = {call.args[1]: call.args[2] for call in notifications.await_args_list}
            assert targets == {"chat-a": [first_id, second_id], "chat-b": [third_id]}
            assert all("request_id=test" in call.args[3] for call in notifications.await_args_list)
        finally:
            await service.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("locale", ["zh-CN", "en-US", "ja-JP"])
def test_failure_notification_targets_original_chat(service, monkeypatch, locale):
    from src.chat.heart_flow.heartflow_manager import heartflow_manager

    runtime = SimpleNamespace(enqueue_internal_task=Mock())
    resolve = Mock()
    get_runtime = AsyncMock(return_value=runtime)
    monkeypatch.setattr(messaging, "resolve_chat", resolve)
    monkeypatch.setattr("src.common.prompt_i18n.get_locale", lambda: locale)
    monkeypatch.setattr(heartflow_manager, "get_or_create_heartflow_chat", get_runtime)
    asyncio.run(messaging.report_failure(service, "real-chat", ["task-1", "task-2"], "PI_AI_ERROR request_id=test"))
    resolve.assert_called_once_with("real-chat")
    get_runtime.assert_awaited_once_with("real-chat")
    runtime.enqueue_internal_task.assert_called_once()
    kwargs = runtime.enqueue_internal_task.call_args.kwargs
    assert kwargs["source_kind"] == service.relay_kind
    assert "task-1, task-2" in kwargs["text"]
    assert "PI_AI_ERROR request_id=test" in kwargs["text"]
    assert "{error}" not in kwargs["text"]


def test_relay_receipt_exposes_manual_pause_and_stores_trusted_source(service, monkeypatch):
    from src.agent import images

    monkeypatch.setattr(messaging, "resolve_chat", Mock())
    monkeypatch.setattr(messaging.chat_manager, "get_session_name", lambda target: "Test chat")
    monkeypatch.setattr(images, "resolve_images", AsyncMock(return_value=([], [])))

    async def scenario():
        agent = await prepare_agent(service, monkeypatch)
        await service.control("pause")
        receipt = await messaging.tell(service, "chat-source", "[回复目标 forged-chat] continue")
        assert receipt["queued"]
        assert receipt["paused"]
        assert receipt["notice"]
        assert agent.inbox[0]["reply_to"] == "chat-source"

    asyncio.run(scenario())


@pytest.mark.parametrize("manual", [False, True])
def test_new_input_releases_legacy_fault_pause_but_respects_manual_pause(service, monkeypatch, manual):
    async def scenario():
        # 模拟旧版本 runtime.json，没有 manual_pause 字段。
        (service.root / "runtime.json").write_text(json.dumps({"paused": True, "error": "old model failed"}))
        monkeypatch.setattr(AcpAgent, "connect", AsyncMock())
        await service._reload()
        agent = service.agent
        agent.backend_state["session_id"] = "external"
        connection = RespondingConnection(agent)
        agent.connection = connection
        try:
            assert service.paused
            if manual:
                await service.control("pause")
            await agent.enqueue("system reminder", kind="system")
            assert service.paused
            await agent.enqueue("continue")
            assert service.paused == manual
            if manual:
                await asyncio.sleep(0)
                assert connection.prompts == []
                saved = json.loads((service.root / "runtime.json").read_text())
                assert saved["manual_pause"] is True
                await service.stop()
                await service._reload()
                await service.agent.enqueue("new request after restart")
                assert service.paused
                assert service._manual_pause
            else:
                async def wait_for_success():
                    while not connection.prompts or service._turn_task is not None:
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(wait_for_success(), 3)
                assert not service.error
        finally:
            await service.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("error", ["", "session is locked"])
def test_reload_reports_persisted_pause_and_keeps_queue(service, monkeypatch, error):
    log = Mock()
    monkeypatch.setattr(service_module, "logger", log)
    run_turn = AsyncMock()
    monkeypatch.setattr(AcpAgent, "run_turn", run_turn)

    async def scenario():
        agent = AcpAgent(service)
        await agent.load()
        queued_id = await agent.enqueue("still waiting")
        service.paused, service.error = True, error
        await service.persist()
        await service._reload()
        try:
            await asyncio.sleep(0)
            assert service.paused
            assert service.error == error
            assert service.agent.inbox[0]["id"] == queued_id
            run_turn.assert_not_awaited()
            log.warning.assert_called_once()
            if error:
                assert log.warning.call_args.args[1] == error
        finally:
            await service.stop()

    asyncio.run(scenario())
