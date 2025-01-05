import dataclasses
import enum

from . import _base, _claude


class Model(enum.Enum):
    CLAUDE_3_5_SONNET = "CLAUDE_3_5_SONNET"


@dataclasses.dataclass
class ModelUnavailable(Exception):
    model: Model


def get_llm_client(*, model: Model) -> _base.LLMClient:
    if model == Model.CLAUDE_3_5_SONNET:
        return _claude.ClaudeClient()
    raise ModelUnavailable(model=model)
