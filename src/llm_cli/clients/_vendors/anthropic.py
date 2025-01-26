from collections.abc import AsyncGenerator

import anthropic

from llm_cli.clients import _base, _models


class AnthropicClient(_base.LLMClient):
    vendor = _models.Vendor.ANTHROPIC

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

        self._client = anthropic.AsyncClient(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = 1024

    async def _stream_response(self) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(
                model=self._model.official_name,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=self._messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.APIStatusError as exc:
            raise _base.VendorAPIError(status_code=exc.status_code, vendor=self.vendor)
