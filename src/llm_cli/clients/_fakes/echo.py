import re
from typing import AsyncGenerator

from llm_cli.clients import _base, _models


class EchoClient(_base.LLMClient):
    """
    Return the prompt as the response message - an echo.
    """

    vendor = _models.Vendor.FAKE_AI

    def __init__(self, system_prompt: str = "Provide an echo."):
        super().__init__(system_prompt=system_prompt)

    async def _stream_response(self) -> AsyncGenerator[str, None]:
        latest_user_message = self._messages[-1]["content"]

        for word in re.split(r"(\s+)", latest_user_message):
            yield word
