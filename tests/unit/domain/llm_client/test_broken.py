import pytest

from llm_cli.domain.llm_client import _base, _broken


class TestGetResponse:
    def test_raises_whenever_called(self):
        client = _broken.BrokenClient()

        with pytest.raises(_base.LLMClientError):
            client.get_response(user_prompt="Do you work?")


class TestGetResponseAsync:
    @pytest.mark.asyncio
    async def test_raises_whenever_called(self):
        client = _broken.BrokenClient()

        with pytest.raises(_base.LLMClientError):
            await client.get_response_async(user_prompt="Do you work?")
