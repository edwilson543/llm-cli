import argparse
import asyncio
import dataclasses
import re
import sys

from llm_cli.domain import llm_client

from ._utils import parsing as parsing_utils


MAX_LINE_WIDTH = 80


@dataclasses.dataclass(frozen=True)
class CommandArgs(parsing_utils.CommandArgs):
    question: str


def main():
    asyncio.run(ask_question())


async def ask_question(*, arguments: CommandArgs | None = None) -> None:
    """
    Command to ask the model a single question.
    """
    _set_print_colour_to_cyan()

    if arguments is None:
        arguments = _extract_args_from_cli(sys.argv[1:])

    try:
        client = llm_client.get_llm_client(
            model=arguments.model, system_prompt=arguments.system_prompt
        )
    except llm_client.APIKeyNotSet as exc:
        _set_print_colour_to_yellow()
        print(
            f"The {exc.env_var} environment variable must be set to use {arguments.model.vendor.value}'s models!"
        )
        return

    persona = arguments.persona or arguments.model.friendly_name
    print(f"\n{persona}:\n---\n", end="")

    try:
        await _stream_response_and_print_formatted_output(
            client=client, arguments=arguments
        )
    except llm_client.LLMClientError as exc:
        print("Error streaming response.", exc, end="")

    print("\n---\n")


async def _stream_response_and_print_formatted_output(
    *,
    client: llm_client.LLMClient,
    arguments: CommandArgs,
    max_line_width: int = MAX_LINE_WIDTH,
) -> None:
    current_line_width = 0

    async for response_message in client.stream_response(
        user_prompt=arguments.question
    ):
        words_with_spaces = re.split(r"(\s+)", response_message)

        for word in words_with_spaces:
            if len(word) + current_line_width <= max_line_width:
                print(word, end="", flush=True)
                current_line_width += len(word)
            else:
                print("\n", word.lstrip(), sep="", end="", flush=True)
                current_line_width = len(word)

            final_line_break_in_word = word.rfind("\n")
            if final_line_break_in_word > 0:
                current_line_width = len(word) - final_line_break_in_word


def _extract_args_from_cli(args: list[str]) -> CommandArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
        type=str,
        help="The question the model should answer.",
    )
    parsing_utils.add_model_argument(parser)
    parsing_utils.add_persona_argument(parser)

    parsed_args = parser.parse_args(args)

    return CommandArgs(
        question=parsed_args.question,
        persona=parsed_args.persona,
        model=parsing_utils.get_model_from_friendly_name(parsed_args.model),
    )


def _set_print_colour_to_cyan() -> None:
    cyan = "\033[96m"
    print(cyan, end="")


def _set_print_colour_to_yellow() -> None:
    cyan = "\033[93m"
    print(cyan, end="")
