"""Vertex AI 客户端，复用 Gemini 消息与工具协议。"""

from typing import Any

import os

import httpx
from google import genai
from google.oauth2 import service_account

from src.config.model_configs import APIProvider
from src.llm_models.model_client.base_client import client_registry

from .gemini_client import GeminiClient, _build_http_options


class _HttpxAuthResponse:
    """google-auth 所需的同步响应接口。"""

    def __init__(self, response: httpx.Response) -> None:
        self.status = response.status_code
        self.headers = response.headers
        self.data = response.content


class _VertexCredentials(service_account.Credentials):
    """仅将此服务账号的令牌刷新请求发送到指定代理。"""

    def refresh(self, request: Any) -> None:
        proxy = os.environ.get("MAIBOT_VERTEX_PROXY", "").strip()
        if not proxy:
            super().refresh(request)
            return

        with httpx.Client(proxy=proxy, timeout=30, trust_env=False) as client:
            def send(url: str, method: str = "GET", body: bytes | None = None,
                     headers: dict[str, str] | None = None, **kwargs: Any) -> _HttpxAuthResponse:
                response = client.request(method, url, content=body, headers=headers, timeout=kwargs.get("timeout", 30))
                return _HttpxAuthResponse(response)

            super().refresh(send)


@client_registry.register_client_class("vertex")
class VertexClient(GeminiClient):
    """通过服务账号访问 Vertex AI；代理仅作用于该提供商。"""

    @staticmethod
    def _create_client(api_provider: APIProvider) -> genai.Client:
        credential_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        credentials = _VertexCredentials.from_service_account_file(
            credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        options = _build_http_options(api_provider)
        proxy = os.environ.get("MAIBOT_VERTEX_PROXY", "").strip()
        if proxy:
            options.client_args["proxy"] = proxy
            # google-genai 有 aiohttp 时优先使用它，指定 httpx transport 才能使用 SOCKS。
            options.async_client_args["transport"] = httpx.AsyncHTTPTransport(
                proxy=proxy, verify=options.async_client_args["verify"],
            )
        return genai.Client(
            vertexai=True,
            project=credentials.project_id,
            location="global",
            credentials=credentials,
            http_options=options,
        )
