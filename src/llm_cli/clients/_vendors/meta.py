
from llm_cli.clients import _models

from . import openai


class MetaClient(openai.OpenAIClient):
    vendor = _models.Vendor.META

    def __init__(
        self,
        *,
        system_prompt: str,
        model: _models.Model,
        api_key: str | None = None,
    ) -> None:
        api_key = self._get_api_key(api_key=api_key)
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            api_key=api_key,
            base_url="https://api.llama-api.com",
        )
