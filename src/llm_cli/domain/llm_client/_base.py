import abc
import dataclasses


@dataclasses.dataclass
class LLMClientError(Exception):
    pass


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def get_response(self, *, user_prompt: str) -> str:
        raise NotImplementedError
