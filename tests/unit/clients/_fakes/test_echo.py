import pytest

from llm_cli.clients._fakes import echo


class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_returns_user_prompt(self):
        client = echo.EchoClient()
        user_prompt = "Can you hear the echo?"

        response = [
            chunk async for chunk in client.stream_response(user_prompt=user_prompt)
        ]

        assert "".join(chunk for chunk in response) == user_prompt
