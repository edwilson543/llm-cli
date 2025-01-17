import re
from typing import AsyncGenerator

from llm_cli.domain.llm_client import _base


class EchoClient(_base.LLMClient):
    """
    Return the prompt as the response message - an echo.
    """

    async def stream_response(
        self, *, user_prompt: str, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        for word in re.split(r"(\s+)", user_prompt):
            yield word
