import abc
import dataclasses
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


class LLMClient(abc.ABC):
    _api_key_env_var: str | None = None

    @abc.abstractmethod
    async def stream_response(
        self, *, user_prompt: str, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError

    def _get_api_key(self, api_key: str | None = None) -> str:
        if api_key:
            return api_key
        if not api_key and self._api_key_env_var:
            try:
                return env.as_str(self._api_key_env_var)
            except env.EnvironmentVariableNotSet as exc:
                raise APIKeyNotSet(env_var=self._api_key_env_var) from exc

        raise APIKeyNotSet(env_var="UNKNOWN")
