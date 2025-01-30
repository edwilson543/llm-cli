from llm_cli.clients import _base, _models

from . import openai


class MetaClient(openai.OpenAIClient):
    vendor = _models.Vendor.META

    def __init__(
        self,
        *,
        parameters: _base.ModelParameters,
        model: _models.Model,
        api_key: str | None = None,
    ) -> None:
        api_key = self._get_api_key(api_key=api_key)
        super().__init__(
            parameters=parameters,
            model=model,
            api_key=api_key,
            base_url="https://api.llama-api.com",
        )
