from typing import AsyncGenerator

from llm_cli.domain.llm_client import _base


class EchoClient(_base.LLMClient):
    """
    Return the prompt as the response message - an echo.
    """

    async def stream_response(
        self, *, user_prompt: str, persona: str | None = None
    ) -> AsyncGenerator[str, None]:
        for letter in user_prompt:
            yield letter
