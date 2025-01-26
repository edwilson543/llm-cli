import argparse

from llm_cli import clients


MODEL_CHOICES_HELP = """The supported models are: 
    - claude-haiku, claude-sonnet, claude-opus
    - llama-3
    - codestral, mistral, ministral
    - gpt-4, gpt-4-mini
    - grok-2"""


def add_persona_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-p",
        "--persona",
        type=str,
        required=False,
        help="The persona the model should assume.",
    )


def get_model_choices() -> list[str]:
    return [model.friendly_name for model in clients.get_available_models()]


def get_model_from_friendly_name(friendly_name: str) -> clients.Model:
    for model in clients.get_available_models():
        if model.friendly_name == friendly_name:
            return model

    # This is for mypy - argparse pre-validates `friendly_name`.
    raise argparse.ArgumentError(
        argument=None, message=f"Model '{friendly_name}' is not available."
    )
