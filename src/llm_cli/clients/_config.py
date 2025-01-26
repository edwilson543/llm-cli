import dataclasses

from llm_cli import env

from . import _base, _models
from ._fakes import broken, echo
from ._vendors import anthropic, meta, mistral, openai, xai


@dataclasses.dataclass(frozen=True)
class ModelNotConfigured(Exception):
    model: _models.Model

    def __str__(self) -> str:
        return f"No LLMClient implementation is installed for model '{self.model.official_name}'."


def get_llm_client(*, model: _models.Model, system_prompt: str) -> _base.LLMClient:
    """
    Return an LLMClient instance that integrates with the specified model.
    """
    if model.vendor == _models.Vendor.ANTHROPIC:
        return anthropic.AnthropicClient(model=model, system_prompt=system_prompt)
    elif model.vendor == _models.Vendor.META:
        return meta.MetaClient(model=model, system_prompt=system_prompt)
    elif model.vendor == _models.Vendor.MISTRAL:
        return mistral.MistralClient(model=model, system_prompt=system_prompt)
    elif model.vendor == _models.Vendor.OPENAI:
        return openai.OpenAIClient(model=model, system_prompt=system_prompt)
    elif model.vendor == _models.Vendor.XAI:
        return xai.XAIClient(model=model, system_prompt=system_prompt)
    elif model.friendly_name == "echo":
        return echo.EchoClient(system_prompt=system_prompt)
    elif model.friendly_name == "broken":
        return broken.BrokenClient(system_prompt=system_prompt)
    raise ModelNotConfigured(model=model)


def get_available_models() -> list[_models.Model]:
    return [
        # Anthropic.
        _models.CLAUDE_HAIKU,
        _models.CLAUDE_SONNET,
        _models.CLAUDE_OPUS,
        # Meta.
        _models.LLAMA_3,
        # Mistral.
        _models.CODESTRAL,
        _models.MISTRAL,
        _models.MINISTRAL,
        # OpenAI.
        _models.GPT_4,
        _models.GPT_4_MINI,
        # xAI.
        _models.GROK_2,
        # Fakes.
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
