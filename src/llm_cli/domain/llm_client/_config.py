import dataclasses

from llm_cli import env

from . import _anthropic, _base, _broken, _echo, _models, _xai


@dataclasses.dataclass(frozen=True)
class ModelNotConfigured(_base.LLMClientError):
    model: _models.Model

    def __str__(self) -> str:
        return f"No LLMClient implementation is installed for model '{self.model.official_name}'."


def get_llm_client(*, model: _models.Model, system_prompt: str) -> _base.LLMClient:
    """
    Return an LLMClient instance that integrates with the specified model.
    """
    if model.vendor == _models.Vendor.ANTHROPIC:
        return _anthropic.AnthropicClient(model=model, system_prompt=system_prompt)
    elif model.vendor == _models.Vendor.XAI:
        return _xai.XAIClient(model=model, system_prompt=system_prompt)
    elif model.friendly_name == "echo":
        return _echo.EchoClient(system_prompt=system_prompt)
    elif model.friendly_name == "broken":
        return _broken.BrokenClient(system_prompt=system_prompt)
    raise ModelNotConfigured(model=model)


def get_available_models() -> list[_models.Model]:
    return [
        _models.CLAUDE_HAIKU,
        _models.CLAUDE_SONNET,
        _models.CLAUDE_OPUS,
        _models.GROK_2,
        _models.ECHO,
        _models.BROKEN,
    ]


def get_default_model() -> _models.Model:
    """
    Get the model to use by default.
    """
    try:
        default_model = env.as_str("DEFAULT_MODEL")
    except env.EnvironmentVariableNotSet:
        return _models.CLAUDE_SONNET

    for model in get_available_models():
        if model.friendly_name == default_model:
            return model

    return _models.CLAUDE_SONNET
