import os

import anthropic

from llm_cli.domain.llm_client import _base


class ClaudeClient(_base.LLMClient):
    def __init__(self) -> None:
        super().__init__()
        self._client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._model = "claude-3-5-sonnet-20241022"
        self._max_tokens = 1024

    def get_response(self, *, user_prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return str(message.content)
