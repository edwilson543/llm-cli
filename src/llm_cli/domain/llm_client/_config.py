from __future__ import annotations

import dataclasses
import enum

from . import _anthropic, _base, _broken, _echo


class Model(enum.Enum):
    CLAUDE_3_5_SONNET = "CLAUDE_3_5_SONNET"

    # Local implementations.
    ECHO = "ECHO"
    BROKEN = "BROKEN"

    @classmethod
    def available_models(cls) -> list[Model]:
        return [model for model in cls if model not in cls.unavailable_models()]

    @classmethod
    def unavailable_models(cls) -> list[Model]:
        return [cls.BROKEN]


@dataclasses.dataclass
class ModelNotConfigured(_base.LLMClientError):
    model: Model

    def __str__(self) -> str:
        return (
            f"No LLMClient implementation is installed for model '{self.model.value}'."
        )


def get_llm_client(*, model: Model) -> _base.LLMClient:
    """
    Return an LLMClient instance that integrates with the specified model.
    """
    if model == Model.CLAUDE_3_5_SONNET:
        return _anthropic.AnthropicClient()
    if model == Model.ECHO:
        return _echo.EchoClient()
    if model == Model.BROKEN:
        return _broken.BrokenClient()
    raise ModelNotConfigured(model=model)
