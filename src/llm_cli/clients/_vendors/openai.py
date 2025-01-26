import dataclasses
from collections.abc import AsyncGenerator

import openai

from llm_cli.clients import _base, _models


@dataclasses.dataclass(frozen=True)
class OpenAIAPIError(_base.LLMClientError):
    status_code: int

    def __str__(self) -> str:
        return f"The API responded with status code: {self.status_code}."


class OpenAIClient(_base.LLMClient):
    vendor = _models.Vendor.OPENAI

    def __init__(
        self,
        *,
        system_prompt: str,
        model: _models.Model,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(system_prompt=system_prompt)

        api_key = self._get_api_key(api_key=api_key)

        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = 1024

        # Add the system prompt to `messages`, since this can't be specified separately.
        system_prompt_message = {"role": "system", "content": system_prompt}
        self._messages.append(system_prompt_message)

    async def _stream_response(self) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                messages=self._messages,
                model=self._model.official_name,
                max_tokens=self._max_tokens,
                stream=True,
            )
        except openai.APIStatusError as exc:
            raise OpenAIAPIError(status_code=exc.status_code)

        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""
