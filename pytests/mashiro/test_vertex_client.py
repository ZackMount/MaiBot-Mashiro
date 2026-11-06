"""验证 Vertex 身份认证和模型请求使用独立代理。"""

from types import SimpleNamespace

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.config.model_configs import APIProvider
from src.llm_models.model_client import vertex_client


def test_service_account_refresh_uses_vertex_proxy(monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    credentials = vertex_client._VertexCredentials.from_service_account_info(
        {
            "type": "service_account",
            "client_email": "test@example.invalid",
            "token_uri": "https://oauth2.googleapis.com/token",
            "private_key": private_key,
        },
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    calls = []
    original_client = httpx.Client

    def mock_client(*, proxy, timeout, trust_env):
        calls.append((proxy, timeout, trust_env))

        def respond(request):
            assert str(request.url) == "https://oauth2.googleapis.com/token"
            assert request.method == "POST"
            return httpx.Response(200, json={"access_token": "synthetic", "expires_in": 3600, "token_type": "Bearer"})

        return original_client(transport=httpx.MockTransport(respond))

    monkeypatch.setenv("MAIBOT_VERTEX_PROXY", "socks5h://127.0.0.1:11888")
    monkeypatch.setattr(vertex_client.httpx, "Client", mock_client)
    credentials.refresh(None)
    assert credentials.token == "synthetic"
    assert calls == [("socks5h://127.0.0.1:11888", 30, False)]


def test_vertex_uses_project_credentials_and_socks_for_async_requests(monkeypatch):
    fake_credentials = SimpleNamespace(project_id="test-project")
    observed = {}
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/test/credentials.json")
    monkeypatch.setenv("MAIBOT_VERTEX_PROXY", "socks5h://127.0.0.1:11888")
    monkeypatch.setattr(vertex_client._VertexCredentials, "from_service_account_file", lambda *a, **kw: fake_credentials)
    monkeypatch.setattr(vertex_client.genai, "Client", lambda **kwargs: observed.update(kwargs))
    provider = APIProvider(
        name="Vertex", client_type="vertex", base_url="https://aiplatform.googleapis.com/v1", auth_type="none",
    )

    vertex_client.VertexClient(provider)

    assert observed["vertexai"] is True
    assert observed["project"] == "test-project"
    assert observed["location"] == "global"
    assert observed["credentials"] is fake_credentials
    assert observed["http_options"].client_args["proxy"] == "socks5h://127.0.0.1:11888"
    assert isinstance(observed["http_options"].async_client_args["transport"], httpx.AsyncHTTPTransport)
