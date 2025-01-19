import contextlib
import re
import typing

from llm_cli.domain import llm_client

from . import parsing


MAX_LINE_WIDTH = 80


def get_llm_client_or_print_error(
    *, arguments: parsing.CommandArgs
) -> llm_client.LLMClient | None:
    try:
        return llm_client.get_llm_client(
            model=arguments.model, system_prompt=arguments.system_prompt
        )
    except llm_client.APIKeyNotSet as exc:
        set_print_colour_to_yellow()
        print(
            f"The {exc.env_var} environment variable must be set to use {arguments.model.vendor.value}'s models!"
        )
        return None


@contextlib.contextmanager
def print_block_from_interlocutor(
    interlocutor: str,
) -> typing.Generator[None, None, None]:
    """
    Wrap anything printed within the context in a block formatted like below.

    grok-2:
    ---
    {anything printed within the context}
    ---
    """
    print(f"\n{interlocutor}: \n---\n", end="")
    yield
    print("\n---\n\n", end="")


async def print_response_stream_to_terminal(
    response_stream: typing.AsyncGenerator[str, None],
    *,
    max_line_width: int = MAX_LINE_WIDTH,
) -> None:
    """
    Print the response stream as a block of text with constant width, without word breaks.
    """

    current_line_width = 0
    is_first_line = True

    async for response_chunk in response_stream:
        words_with_spaces = re.split(r"(\s+)", response_chunk)

        for word in words_with_spaces:
            if len(word) + current_line_width <= max_line_width:
                print(word, end="", flush=True)
                current_line_width += len(word)
            else:
                if not is_first_line:
                    print("\n", end="")
                word = word.lstrip()  # Don't start the line with a space.
                print(word, end="", flush=True)
                current_line_width = len(word)

            # Account for the response stream including its own line breaks.
            final_line_break_in_word = word.rfind("\n")
            if final_line_break_in_word >= 0:
                current_line_width = len(word) - final_line_break_in_word

            is_first_line = False


def set_print_colour_to_green() -> None:
    cyan = "\033[92m"
    print(cyan, end="")


def set_print_colour_to_cyan() -> None:
    cyan = "\033[96m"
    print(cyan, end="")


def set_print_colour_to_yellow() -> None:
    cyan = "\033[93m"
    print(cyan, end="")
