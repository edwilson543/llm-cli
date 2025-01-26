import abc
import dataclasses
import typing
from collections.abc import AsyncGenerator

from llm_cli import env

from . import _models


@dataclasses.dataclass(frozen=True)
class LLMClientError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class VendorAPIError(LLMClientError):
    vendor: _models.Vendor
    status_code: int = -1

    def __str__(self) -> str:
        return f"The {self.vendor.value} API responded with status code: {self.status_code}."


@dataclasses.dataclass(frozen=True)
class APIKeyNotSet(LLMClientError):
    env_var: str

    def __str__(self) -> str:
        return f"The '{self.env_var}' environment variable is not set."


Role = typing.Literal["system"] | typing.Literal["user"] | typing.Literal["assistant"]


class Message(typing.TypedDict):
    role: Role
    content: str


class LLMClient(abc.ABC):
    vendor: _models.Vendor

    def __init__(self, *, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._messages: list[Message] = []

    async def stream_response(self, *, user_prompt: str) -> AsyncGenerator[str, None]:
        """
        Wrapper for the client-specific streaming logic.

        The user message and assistant messages are appended to the response,
        so that the conversation history is retained on the client instance.
        """
        self._append_user_message(user_prompt)
        chunks: list[str] = []

        async for text in self._stream_response():
            chunks.append(text)
            yield text

        assistant_message = "".join(chunk for chunk in chunks)
        self._append_assistant_message(assistant_message)

    # Private interface.

    @abc.abstractmethod
    def _stream_response(self) -> AsyncGenerator[str, None]:
        """
        Stream the next response in the conversation stored in `self._messages`.
        """
        raise NotImplementedError

    def _append_user_message(self, message: str) -> None:
        self._messages.append({"role": "user", "content": message})

    def _append_assistant_message(self, message: str) -> None:
        self._messages.append({"role": "assistant", "content": message})

    @property
    def _api_key_env_var(self) -> str:
        return f"{self.vendor.value}_API_KEY"

    def _get_api_key(self, api_key: str | None = None) -> str:
        if api_key:
            return api_key
        elif not self._api_key_env_var:
            raise APIKeyNotSet(env_var="UNKNOWN")

        try:
            return env.as_str(self._api_key_env_var)
        except env.EnvironmentVariableNotSet as exc:
            raise APIKeyNotSet(env_var=self._api_key_env_var) from exc
