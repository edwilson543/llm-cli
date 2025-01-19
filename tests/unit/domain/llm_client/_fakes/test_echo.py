import pytest

from llm_cli.domain.llm_client._fakes import echo


class TestGetResponseAsync:
    @pytest.mark.asyncio
    async def test_returns_user_prompt(self):
        client = echo.EchoClient()
        user_prompt = "Can you hear the echo?"

        response = [
            chunk async for chunk in client.stream_response(user_prompt=user_prompt)
        ]

        assert "".join(chunk for chunk in response) == user_prompt
