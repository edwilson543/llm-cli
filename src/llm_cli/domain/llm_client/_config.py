import dataclasses
import enum

from . import _base, _claude, _echo


class Model(enum.Enum):
    CLAUDE_3_5_SONNET = "CLAUDE_3_5_SONNET"
    ECHO = "ECHO"


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
        return _claude.ClaudeClient()
    if model == Model.ECHO:
        return _echo.EchoClient()
    raise ModelNotConfigured(model=model)
