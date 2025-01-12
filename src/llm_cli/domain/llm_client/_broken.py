from typing import AsyncGenerator

from llm_cli.domain.llm_client import _base


class BrokenClient(_base.LLMClient):
    """
    Raise an exception whenever any method is called.
    """

    def get_response(self, *, user_prompt: str, persona: str | None = None) -> str:
        raise _base.LLMClientError()

    async def get_response_async(
        self, *, user_prompt: str, persona: str | None = None
    ) -> AsyncGenerator[str, None]:
        raise _base.LLMClientError()
