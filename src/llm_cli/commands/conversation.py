import argparse
import asyncio
import sys

from llm_cli import clients

from ._utils import parsing as parsing_utils
from ._utils import printing as printing_utils


EXIT = "exit"


class ConversationCommandArgs(parsing_utils.CommandArgs):
    pass


def main():
    asyncio.run(start_conversation())


async def start_conversation(
    *, arguments: ConversationCommandArgs | None = None
) -> None:
    """
    Command to have a multi-turn conversation with a model.
    """
    printing_utils.set_print_colour_to_cyan()

    if arguments is None:
        arguments = _extract_args_from_cli(sys.argv[1:])

    client = printing_utils.get_llm_client_or_print_error(arguments=arguments)
    if not client:
        return

    print(f"type '{EXIT}' to end conversation\n")

    while True:
        printing_utils.set_print_colour_to_green()

        user_prompt = input("you: \n---\n")
        print("---")

        printing_utils.set_print_colour_to_cyan()

        if user_prompt == EXIT:
            print("\nGoodbye!")
            break

        with printing_utils.print_block_from_interlocutor(
            interlocutor=arguments.interlocutor
        ):
            try:
                await printing_utils.print_response_stream_to_terminal(
                    client.stream_response(user_prompt=user_prompt)
                )
            except clients.LLMClientError as exc:
                print("Error streaming response.", exc, end="")
                break


def _extract_args_from_cli(args: list[str]) -> ConversationCommandArgs:
    parser = argparse.ArgumentParser()

    parsing_utils.add_model_argument(parser)
    parsing_utils.add_persona_argument(parser)

    parsed_args = parser.parse_args(args)

    return ConversationCommandArgs(
        persona=parsed_args.persona,
        model=parsing_utils.get_model_from_friendly_name(parsed_args.model),
    )
