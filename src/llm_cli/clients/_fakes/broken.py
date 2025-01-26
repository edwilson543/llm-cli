from typing import AsyncGenerator

from llm_cli.clients import _base, _models


class BrokenClient(_base.LLMClient):
    """
    Raise an exception whenever any method is called.
    """

    vendor = _models.Vendor.FAKE_AI

    def __init__(self, system_prompt: str = "Be broken."):
        super().__init__(system_prompt=system_prompt)

    async def _stream_response(self) -> AsyncGenerator[str, None]:
        for _ in range(0, 1):
            yield ""
        raise _base.VendorAPIError(status_code=503, vendor=self.vendor)
