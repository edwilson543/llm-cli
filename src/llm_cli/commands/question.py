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
    asyncio.run(ask_question())


async def ask_question(*, arguments: QuestionCommandArgs | None = None) -> None:
    """
    Command to ask a single question to a model.
    """
    printing_utils.set_print_colour_to_cyan()

    if arguments is None:
        arguments = _extract_args_from_cli(sys.argv[1:])

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


def _extract_args_from_cli(args: list[str]) -> QuestionCommandArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
        type=str,
        help="The question the model should answer.",
    )
    parsing_utils.add_model_argument(parser)
    parsing_utils.add_persona_argument(parser)

    parsed_args = parser.parse_args(args)

    return QuestionCommandArgs(
        question=parsed_args.question,
        persona=parsed_args.persona,
        model=parsing_utils.get_model_from_friendly_name(parsed_args.model),
    )
