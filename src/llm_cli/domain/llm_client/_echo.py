from typing import AsyncGenerator

from llm_cli.domain.llm_client import _base


class EchoClient(_base.LLMClient):
    """
    Return the prompt as the response message - an echo.
    """

    def get_response(self, *, user_prompt: str, persona: str | None = None) -> str:
        return user_prompt

    async def get_response_async(
        self, *, user_prompt: str, persona: str | None = None
    ) -> AsyncGenerator[str, None]:
        for letter in user_prompt:
            yield letter
