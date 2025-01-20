from typing import AsyncGenerator

from llm_cli.clients import _base


class BrokenClientError(_base.LLMClientError):
    def __str__(self) -> str:
        return "Fake AI is permanently broken."


class BrokenClient(_base.LLMClient):
    """
    Raise an exception whenever any method is called.
    """

    def __init__(self, system_prompt: str = "Be broken."):
        super().__init__(system_prompt=system_prompt)

    async def _stream_response(self) -> AsyncGenerator[str, None]:
        for _ in range(0, 1):
            yield ""
        raise BrokenClientError()
