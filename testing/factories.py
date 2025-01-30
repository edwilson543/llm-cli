"""
Provide defaults, to make tests more succinct.
"""

import dataclasses

from llm_cli import clients
from llm_cli.commands import question


@dataclasses.dataclass(frozen=True)
class ModelParameters(clients.ModelParameters):
    system_prompt: str = "fake"
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0


@dataclasses.dataclass(frozen=True)
class QuestionCommandArgs(question.QuestionCommandArgs):
    question: str = "question"
    models: list[clients.Model] = dataclasses.field(
        default_factory=lambda: [clients.ECHO]
    )
    persona: str | None = None
    temperature: float = 1.0
    top_p: float = 1.0
