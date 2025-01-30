import dataclasses

from llm_cli import clients


@dataclasses.dataclass(frozen=True)
class ModelParameters(clients.ModelParameters):
    """
    Provide default parameters, to make tests more succinct.
    """

    system_prompt: str = "fake"
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
