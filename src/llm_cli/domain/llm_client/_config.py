import dataclasses

from . import _anthropic, _base, _broken, _echo, _models, _xai


@dataclasses.dataclass(frozen=True)
class ModelNotConfigured(_base.LLMClientError):
    model: _models.Model

    def __str__(self) -> str:
        return (
            f"No LLMClient implementation is installed for model '{self.model.value}'."
        )


def get_llm_client(*, model: _models.Model) -> _base.LLMClient:
    """
    Return an LLMClient instance that integrates with the specified model.
    """
    if model == _models.Model.CLAUDE_SONNET:
        return _anthropic.AnthropicClient()
    if model == _models.Model.GROK_2:
        return _xai.XAIClient()
    if model == _models.Model.ECHO:
        return _echo.EchoClient()
    if model == _models.Model.BROKEN:
        return _broken.BrokenClient()
    raise ModelNotConfigured(model=model)
