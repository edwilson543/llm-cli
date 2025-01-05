import abc


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def get_response(self, *, user_prompt: str) -> str:
        raise NotImplementedError
