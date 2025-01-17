from typing import Any

import pytest
import pytest_httpx

from llm_cli.domain.llm_client import _anthropic


class TestGetResponseAsync:
    @pytest.mark.asyncio
    async def test_parses_and_returns_response_when_configured_correctly(
        self, httpx_mock: pytest_httpx.HTTPXMock
    ):
        client = _anthropic.AnthropicClient(api_key="fake-key")

        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            status_code=200,
            stream=self._stream_response_json(),
            is_reusable=True,
        )

        result = client.stream_response(user_prompt="What's nine minus eight?")

        response = "".join([text async for text in result])
        assert response == "I have no idea!"

    # def test_raises_when_fails_to_authenticate(
    #     self, httpx_mock: pytest_httpx.HTTPXMock
    # ):
    #     client = _anthropic.AnthropicClient(api_key="fake-key")
    #
    #     response_json = self._response_json_4xx()
    #
    #     httpx_mock.add_response(
    #         url="https://api.anthropic.com/v1/messages",
    #         method="POST",
    #         status_code=401,
    #         json=response_json,
    #     )
    #
    #     with pytest.raises(_anthropic.AnthropicAPIError) as exc:
    #         client.get_response(user_prompt="What's eight minus nine?")
    #
    #     assert exc.value.status_code == 401

    @staticmethod
    def _stream_response_json():
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
    def _response_json_4xx() -> dict[str, Any]:
        """
        The Anthropic messages API response schema, per: https://docs.anthropic.com/en/api/messages.
        """
        return {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "Invalid request"},
        }
