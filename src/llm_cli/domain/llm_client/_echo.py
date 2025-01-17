import re
from typing import AsyncGenerator

from llm_cli.domain.llm_client import _base


class EchoClient(_base.LLMClient):
    """
    Return the prompt as the response message - an echo.
    """

    async def _stream_response(
        self, *, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        latest_user_message = self._messages[-1]["content"]

        for word in re.split(r"(\s+)", latest_user_message):
            yield word
