import dataclasses
from collections.abc import AsyncGenerator

import anthropic

from llm_cli.domain.llm_client import _base, _models


@dataclasses.dataclass(frozen=True)
class AnthropicAPIError(_base.LLMClientError):
    status_code: int

    def __str__(self) -> str:
        return f"The API responded with status code: {self.status_code}."


@dataclasses.dataclass(frozen=True)
class AnthropicResponseTypeError(_base.LLMClientError):
    type: str


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

        self._system_prompt = "Please be as succinct as possible in your answer. "

    async def stream_response(
        self, *, user_prompt: str, persona: str | None = None
    ) -> AsyncGenerator[str, None]:
        if persona:
            self._add_persona_to_system_prompt(persona=persona)

        try:
            async with self._async_client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.APIStatusError as exc:
            raise AnthropicAPIError(status_code=exc.status_code)

    def _add_persona_to_system_prompt(self, persona: str) -> None:
        self._system_prompt += f"Please assume the persona of {persona}."
