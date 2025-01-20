import argparse
import dataclasses

from llm_cli import clients


@dataclasses.dataclass(frozen=True)
class CommandArgs:
    """
    Base class for command arguments shared by all commands.
    """

    persona: str | None
    model: clients.Model

    @property
    def system_prompt(self) -> str:
        prompt = "Please be as succinct as possible in your answer."
        if self.persona:
            prompt += f" Please assume the persona of {self.persona}."
        return prompt


def add_persona_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-p",
        "--persona",
        type=str,
        required=False,
        help="The persona the model should assume.",
    )


def get_model_from_friendly_name(friendly_name: str) -> clients.Model:
    for model in clients.get_available_models():
        if model.friendly_name == friendly_name:
            return model

    # This is for mypy - argparse pre-validates `friendly_name`.
    raise argparse.ArgumentError(
        argument=None, message=f"Model '{friendly_name}' is not available."
    )
