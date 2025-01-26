from collections.abc import AsyncGenerator

import mistralai

from llm_cli.clients import _base, _models


class MistralClient(_base.LLMClient):
    vendor = _models.Vendor.MISTRAL

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
        self._model = model
        self._max_tokens = 1024

        # Add the system prompt to `messages`, since this can't be specified separately.
        system_prompt_message = {"role": "system", "content": system_prompt}
        self._messages.append(system_prompt_message)

    async def _stream_response(self) -> AsyncGenerator[str, None]:
        try:
            response = await self._client.chat.stream_async(
                model=self._model.official_name,
                messages=self._messages,
                max_tokens=self._max_tokens,
            )
        except mistralai.SDKError as exc:
            raise _base.VendorAPIError(status_code=exc.status_code, vendor=self.vendor)

        if not response:
            raise _base.VendorAPIError(vendor=self.vendor)

        async for chunk in response:
            if (text := chunk.data.choices[0].delta.content) is not None:
                yield text
