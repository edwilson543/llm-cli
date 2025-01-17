import argparse
import dataclasses

from llm_cli.domain import llm_client


@dataclasses.dataclass(frozen=True)
class CommandArgs:
    """
    Base class for command arguments shared by all commands.
    """

    persona: str | None
    model: llm_client.Model

    @property
    def system_prompt(self) -> str:
        prompt = "Please be as succinct as possible in your answer."
        if self.persona:
            prompt += f" Please assume the persona of {self.persona}."
        return prompt

    @property
    def interlocutor(self) -> str:
        return self.persona or self.model.friendly_name


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[model.friendly_name for model in llm_client.get_available_models()],
        default=llm_client.get_default_model().friendly_name,
        help="The model that should be used.",
    )


def add_persona_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-p",
        "--persona",
        type=str,
        required=False,
        help="The persona the model should assume.",
    )


def get_model_from_friendly_name(friendly_name: str) -> llm_client.Model:
    for model in llm_client.get_available_models():
        if model.friendly_name == friendly_name:
            return model

    # This is for mypy - argparse pre-validates `friendly_name`.
    raise argparse.ArgumentError(
        argument=None, message=f"Model '{friendly_name}' is not available."
    )
