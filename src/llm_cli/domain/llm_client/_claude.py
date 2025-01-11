import dataclasses

import anthropic

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


class ClaudeClient(_base.LLMClient):
    def __init__(self) -> None:
        super().__init__()

        try:
            api_key = env.as_str("ANTHROPIC_API_KEY")
        except env.EnvironmentVariableNotSet as exc:
            raise AnthropicAPIKeyNotSet from exc

        self._client = anthropic.Client(api_key=api_key)
        self._model = "claude-3-5-sonnet-20241022"
        self._max_tokens = 1024

    def get_response(self, *, user_prompt: str) -> str:
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIStatusError as exc:
            raise AnthropicAPIError(status_code=exc.status_code)

        return str(message.content[0].text)
