import dataclasses
from collections.abc import AsyncGenerator

import mistralai

from llm_cli.clients import _base, _models


@dataclasses.dataclass(frozen=True)
class MistralAPIError(_base.LLMClientError):
    status_code: int

    def __str__(self) -> str:
        return f"The API responded with status code: {self.status_code}."


class MistralClient(_base.LLMClient):
    _api_key_env_var = "MISTRAL_API_KEY"

    def __init__(
        self,
        *,
        system_prompt: str,
        model: _models.Model,
        api_key: str | None = None,
    ) -> None:
        super().__init__(system_prompt=system_prompt)

        api_key = self._get_api_key(api_key=api_key)

        self._client = mistralai.Mistral(api_key=api_key)
        self._model = model.official_name

    async def _stream_response(self) -> AsyncGenerator[str, None]:
        try:
            response = await self._client.chat.stream_async(
                model=self._model,
                messages=self._messages,
            )
        except mistralai.SDKError as exc:
            raise MistralAPIError(status_code=exc.status_code)

        if not response:
            raise MistralAPIError(status_code=503)

        async for chunk in response:
            if (text := chunk.data.choices[0].delta.content) is not None:
                yield text
