import abc
import dataclasses
import typing
from collections.abc import AsyncGenerator

from llm_cli import env


@dataclasses.dataclass(frozen=True)
class LLMClientError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class APIKeyNotSet(LLMClientError):
    env_var: str

    def __str__(self) -> str:
        return f"The '{self.env_var}' environment variable is not set."


class Message(typing.TypedDict):
    role: typing.Literal["user"] | typing.Literal["assistant"]
    content: str


class LLMClient(abc.ABC):
    _api_key_env_var: str | None = None

    def __init__(self) -> None:
        super().__init__()
        self._messages: list[Message] = []

    async def stream_response(
        self, *, user_prompt: str, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """
        Wrapper for the client-specific streaming logic.

        The user message and assistant messages are appended to the response,
        so that the conversation history is retained on the client instance.
        """
        self._append_user_message(user_prompt)
        chunks: list[str] = []

        async for text in self._stream_response(system_prompt=system_prompt):
            chunks.append(text)
            yield text

        assistant_message = "".join(chunk for chunk in chunks)
        self._append_assistant_message(assistant_message)

    # Private interface.

    @abc.abstractmethod
    def _stream_response(self, *, system_prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream the next response in the conversation stored in `self._messages`.
        """
        raise NotImplementedError

    def _append_user_message(self, message: str) -> None:
        self._messages.append({"role": "user", "content": message})

    def _append_assistant_message(self, message: str) -> None:
        self._messages.append({"role": "assistant", "content": message})

    def _get_api_key(self, api_key: str | None = None) -> str:
        if api_key:
            return api_key
        if not api_key and self._api_key_env_var:
            try:
                return env.as_str(self._api_key_env_var)
            except env.EnvironmentVariableNotSet as exc:
                raise APIKeyNotSet(env_var=self._api_key_env_var) from exc

        raise APIKeyNotSet(env_var="UNKNOWN")
