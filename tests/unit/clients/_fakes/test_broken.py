import pytest

from llm_cli.clients import _base, _models
from llm_cli.clients._fakes import broken


class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_raises_whenever_called(self):
        client = broken.BrokenClient()

        with pytest.raises(_base.VendorAPIError) as exc:
            async for _ in client.stream_response(user_prompt="Do you work?"):
                pass

        assert exc.value.vendor == _models.Vendor.FAKE_AI
