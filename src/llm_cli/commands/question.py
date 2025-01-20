import argparse
import asyncio
import dataclasses
import sys

from llm_cli import clients

from ._utils import parsing as parsing_utils
from ._utils import printing as printing_utils


@dataclasses.dataclass(frozen=True)
class QuestionCommandArgs(parsing_utils.CommandArgs):
    question: str


def main():
    asyncio.run(ask_questions())


async def ask_questions(*, arguments: list[QuestionCommandArgs] | None = None) -> None:
    """
    Command to ask a question to one or more models.
    """
    if arguments is None:
        arguments = _extract_args_from_cli(sys.argv[1:])

    for args in arguments:
        await _ask_question(arguments=args)


async def _ask_question(*, arguments: QuestionCommandArgs) -> None:
    """
    Ask as single question to a single model.
    """
    printing_utils.set_print_colour_to_cyan()

    client = printing_utils.get_llm_client_or_print_error(arguments=arguments)
    if not client:
        return

    with printing_utils.print_block_from_interlocutor(
        interlocutor=arguments.interlocutor
    ):
        try:
            await printing_utils.print_response_stream_to_terminal(
                client.stream_response(user_prompt=arguments.question)
            )
        except clients.LLMClientError as exc:
            print("Error streaming response.", exc, end="")


def _extract_args_from_cli(args: list[str]) -> list[QuestionCommandArgs]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
        type=str,
        help="The question the model should answer.",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="*",
        type=str,
        choices=[model.friendly_name for model in clients.get_available_models()],
        default=[clients.get_default_model().friendly_name],
        help="The model that should be used. Multiple models can be specified, separated by a space.",
    )
    parsing_utils.add_persona_argument(parser)

    parsed_args = parser.parse_args(args)

    return [
        QuestionCommandArgs(
            question=parsed_args.question,
            persona=parsed_args.persona,
            model=parsing_utils.get_model_from_friendly_name(model),
        )
        for model in parsed_args.model
    ]
