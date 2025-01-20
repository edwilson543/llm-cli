import pytest
import pytest_httpx

from llm_cli.clients._vendors import anthropic


class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_parses_and_returns_response_when_configured_correctly(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ):
        client = anthropic.AnthropicClient(api_key="fake-key", system_prompt="fake")

        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            status_code=200,
            stream=self._stream_response_ok(),
            is_reusable=True,
        )

        result = client.stream_response(user_prompt="Tell me about your constitution.")

        response = "".join([text async for text in result])
        assert response == "I have no idea!"

    @pytest.mark.asyncio
    async def test_raises_when_fails_to_authenticate(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ):
        client = anthropic.AnthropicClient(api_key="fake-key", system_prompt="fake")

        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            status_code=401,
            stream=self._stream_response_401(),
            is_reusable=True,
        )

        with pytest.raises(anthropic.AnthropicAPIError) as exc:
            async for _ in client.stream_response(
                user_prompt="Tell me about your constitution."
            ):
                pass

        assert exc.value.status_code == 401

    @staticmethod
    def _stream_response_ok() -> pytest_httpx.IteratorStream:
        """
        The Anthropic message stream API response schema, per:  https://docs.anthropic.com/en/api/messages-streaming#basic-streaming-request.
        """
        return pytest_httpx.IteratorStream(
            [
                b'id: 1\nevent: message_start\ndata: {"type": "message_start", "message": {"id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY", "type": "message", "role": "assistant", "content": [], "model": "claude-3-5-sonnet-20241022", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 1}}}\n\n',
                b'id: 2\nevent: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n\n',
                b'id: 3\nevent: ping\ndata {"type": "ping"}\n\n',
                b'id: 4\nevent: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "I have "}}\n\n',
                b'id: 5\nevent: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "no idea!"}}\n\n',
                b'id: 6\nevent: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n',
                b'id: 7\nevent: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence":null}, "usage": {"output_tokens": 15}}\n\n',
                b'id: 8\nevent: message_stop\ndata: {"type": "message_stop"}\n\n',
            ]
        )

    @staticmethod
    def _stream_response_401() -> pytest_httpx.IteratorStream:
        """
        The Anthropic messages API response schema, per: https://docs.anthropic.com/en/api/messages-streaming#error-events.
        """
        return pytest_httpx.IteratorStream(
            [
                b'id: 1\nevent: error\ndata: {"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}\n\n',
            ]
        )
