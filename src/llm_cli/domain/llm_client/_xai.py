from . import _anthropic


class XAIClient(_anthropic.AnthropicClient):
    """
    The xAI API is compatible with the Anthropic SDK, hence the Anthropic client is subclaseed.
    https://docs.x.ai/docs/overview#migrating-from-another-llm-provider
    """
    _api_key_env_var = "XAI_API_KEY"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            api_key=api_key, base_url="https://api.x.ai", model="grok-2-1212"
        )
