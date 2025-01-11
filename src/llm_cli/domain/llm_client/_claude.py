import dataclasses

from collections.abc import AsyncGenerator

import anthropic

from anthropic import types as anthropic_types

from llm_cli import env
from llm_cli.domain.llm_client import _base


@dataclasses.dataclass
class AnthropicAPIKeyNotSet(_base.LLMClientError):
    def __str__(self) -> str:
        return "The 'ANTHROPIC_API_KEY' environment variable is not set."


@dataclasses.dataclass
class AnthropicAPIError(_base.LLMClientError):
    status_code: int

    def __str__(self) -> str:
        return f"Unable to get a response. The Anthropic API responded with status code: {self.status_code}."


@dataclasses.dataclass
class AnthropicResponseTypeError(_base.LLMClientError):
    type: str


class ClaudeClient(_base.LLMClient):
    def __init__(self) -> None:
        super().__init__()

        try:
            api_key = env.as_str("ANTHROPIC_API_KEY")
        except env.EnvironmentVariableNotSet as exc:
            raise AnthropicAPIKeyNotSet from exc

        self._client = anthropic.Client(api_key=api_key)
        self._async_client = anthropic.AsyncClient(api_key=api_key)
        self._model = "claude-3-5-sonnet-20241022"
        self._max_tokens = 1024

        self._system_prompt = "Please be as succinct as possible in your answer. "

    def get_response(self, *, user_prompt: str, character: str | None = None) -> str:
        if character:
            self._add_character_to_system_prompt(character=character)

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIStatusError as exc:
            raise AnthropicAPIError(status_code=exc.status_code)

        response = message.content[0]
        if not isinstance(response, anthropic_types.TextBlock):
            raise AnthropicResponseTypeError(type=str(type(response)))

        return response.text

    async def get_response_async(
        self, *, user_prompt: str, character: str | None = None
    ) -> AsyncGenerator[str, None]:
        if character:
            self._add_character_to_system_prompt(character=character)

        async with self._async_client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

        await stream.get_final_message()

    def _add_character_to_system_prompt(self, character: str) -> None:
        self._system_prompt += f"Please assume the persona of {character}."
