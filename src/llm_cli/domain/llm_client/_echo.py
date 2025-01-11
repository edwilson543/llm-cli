from llm_cli.domain.llm_client import _base


class EchoClient(_base.LLMClient):
    """
    Return the prompt as the response message - an echo.
    """

    def get_response(self, *, user_prompt: str) -> str:
        return user_prompt
