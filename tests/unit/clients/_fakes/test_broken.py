import pytest

from llm_cli.clients import _base
from llm_cli.clients._fakes import broken


class TestGetResponseAsync:
    @pytest.mark.asyncio
    async def test_raises_whenever_called(self):
        client = broken.BrokenClient()

        with pytest.raises(_base.LLMClientError):
            async for _ in client.stream_response(user_prompt="Do you work?"):
                pass
