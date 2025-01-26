import pytest
import pytest_httpx

from llm_cli.clients import _base, _models
from llm_cli.clients._vendors import mistral


class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_parses_and_returns_response_when_configured_correctly(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ):
        client = mistral.MistralClient(
            model=_models.MISTRAL, api_key="fake-key", system_prompt="fake"
        )

        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/chat/completions#stream",
            method="POST",
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            stream=self._stream_response_ok(),
            is_reusable=True,
        )

        result = client.stream_response(user_prompt="Bonjour?")

        response = "".join([text async for text in result])
        assert response == "Salut!"

    @pytest.mark.asyncio
    async def test_raises_when_fails_to_authenticate(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ):
        client = mistral.MistralClient(
            model=_models.MISTRAL, api_key="fake-key", system_prompt="fake"
        )

        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/chat/completions#stream",
            method="POST",
            status_code=401,
            is_reusable=True,
        )

        with pytest.raises(_base.VendorAPIError) as exc:
            async for _ in client.stream_response(user_prompt="Error?"):
                pass

        assert exc.value.vendor == _models.Vendor.MISTRAL
        assert exc.value.status_code == 401

    @staticmethod
    def _stream_response_ok() -> pytest_httpx.IteratorStream:
        return pytest_httpx.IteratorStream(
            [
                b'id: 1\nevent: data\ndata: {"id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY", "type": "data", "model": "mistral-latest", "choices": [{"index": 0, "finish_reason": "sufficient", "delta": {"content": "Sa"}}]}\n\n',
                b'id: 1\nevent: data\ndata: {"id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY", "type": "data", "model": "mistral-latest", "choices": [{"index": 0, "finish_reason": "sufficient", "delta": {"content": "lut"}}]}\n\n',
                b'id: 1\nevent: data\ndata: {"id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY", "type": "data", "model": "mistral-latest", "choices": [{"index": 0, "finish_reason": "sufficient", "delta": {"content": "!"}}]}\n\n',
            ]
        )
