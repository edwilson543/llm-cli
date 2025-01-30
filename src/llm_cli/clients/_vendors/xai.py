from llm_cli.clients import _base, _models

from . import anthropic


class XAIClient(anthropic.AnthropicClient):
    """
    The xAI API is compatible with the Anthropic SDK, hence the Anthropic client is subclassed.
    https://docs.x.ai/docs/overview#migrating-from-another-llm-provider
    """

    vendor = _models.Vendor.XAI

    def __init__(
        self,
        parameters: _base.ModelParameters,
        model: _models.Model,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            parameters=parameters,
            api_key=api_key,
            base_url="https://api.x.ai",
            model=model,
        )
