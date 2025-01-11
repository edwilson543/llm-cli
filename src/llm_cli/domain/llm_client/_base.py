import abc
import dataclasses

from collections.abc import AsyncGenerator


@dataclasses.dataclass
class LLMClientError(Exception):
    pass


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def get_response(self, *, user_prompt: str, character: str | None = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_response_async(
        self, *, user_prompt: str, character: str | None = None
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError
