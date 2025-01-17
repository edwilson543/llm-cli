import dataclasses
from collections.abc import AsyncGenerator

import anthropic

from llm_cli.domain.llm_client import _base, _models


@dataclasses.dataclass(frozen=True)
class AnthropicAPIError(_base.LLMClientError):
    status_code: int

    def __str__(self) -> str:
        return f"The API responded with status code: {self.status_code}."


class AnthropicClient(_base.LLMClient):
    _api_key_env_var = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: _models.Model | None = None,
    ) -> None:
        super().__init__()

        api_key = self._get_api_key(api_key=api_key)

        self._client = anthropic.Client(api_key=api_key, base_url=base_url)
        self._async_client = anthropic.AsyncClient(api_key=api_key, base_url=base_url)
        self._model = (
            model.official_name if model else _models.CLAUDE_SONNET.official_name
        )
        self._max_tokens = 1024

    async def _stream_response(
        self, *, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        try:
            async with self._async_client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=self._messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.APIStatusError as exc:
            raise AnthropicAPIError(status_code=exc.status_code)
