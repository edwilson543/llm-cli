import argparse
import asyncio
import dataclasses
import re
import sys

from llm_cli.domain import llm_client


MAX_LINE_WIDTH = 80


@dataclasses.dataclass(frozen=True)
class CommandArgs:
    question: str
    persona: str | None
    model: llm_client.Model


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
        client = llm_client.get_llm_client(model=arguments.model)
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
    *, client: llm_client.LLMClient, arguments: CommandArgs
) -> None:
    current_width = 0

    async for response_message in client.get_response_async(
        user_prompt=arguments.question, persona=arguments.persona
    ):
        for word in re.split(r"(\s+)", response_message):
            if len(word) + current_width < MAX_LINE_WIDTH:
                print(word, end="", flush=True)
                current_width += len(word)
            else:
                print("\n", word.lstrip(), sep="", end="", flush=True)
                current_width = len(word)

            if (last_line_break := word.rfind("\n")) > 0:
                current_width = len(word) - last_line_break


def _extract_args_from_cli(args: list[str]) -> CommandArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
        type=str,
        help="The question the model should answer.",
    )
    parser.add_argument(
        "-p",
        "--persona",
        type=str,
        required=False,
        help="The persona the model should assume.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[model.friendly_name for model in llm_client.get_available_models()],
        default=llm_client.get_default_model().friendly_name,
        help="The model that should be used.",
    )

    parsed_args = parser.parse_args(args)

    return CommandArgs(
        question=parsed_args.question,
        persona=parsed_args.persona,
        model=_get_model_from_friendly_name(parsed_args.model),
    )


def _get_model_from_friendly_name(friendly_name: str) -> llm_client.Model:
    for model in llm_client.get_available_models():
        if model.friendly_name == friendly_name:
            return model


def _set_print_colour_to_cyan() -> None:
    cyan = "\033[96m"
    print(cyan, end="")


def _set_print_colour_to_yellow() -> None:
    cyan = "\033[93m"
    print(cyan, end="")
