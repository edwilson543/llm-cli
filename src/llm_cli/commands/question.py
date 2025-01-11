import argparse
import asyncio
import dataclasses
import sys

from llm_cli.domain import llm_client


@dataclasses.dataclass
class CommandArgs:
    prompt: str
    character: str | None
    model: llm_client.Model
    stream: bool


def main():
    asyncio.run(ask_question())


async def ask_question(*, arguments: CommandArgs | None = None) -> None:
    if arguments is None:
        arguments = _extract_args_from_cli(sys.argv[1:])

    client = llm_client.get_llm_client(model=arguments.model)

    print("\n", end="")

    if arguments.stream:
        await _ask_question_async(client=client, arguments=arguments)
    else:
        _ask_question_sync(client=client, arguments=arguments)


def _ask_question_sync(*, client: llm_client.LLMClient, arguments: CommandArgs) -> None:
    try:
        response_message = client.get_response(
            user_prompt=arguments.prompt, character=arguments.character
        )
    except llm_client.LLMClientError as exc:
        print(str(exc))
        raise

    print(response_message,"\n")


async def _ask_question_async(
    *, client: llm_client.LLMClient, arguments: CommandArgs
) -> None:
    async for response_message in client.get_response_async(
        user_prompt=arguments.prompt, character=arguments.character
    ):
        print(response_message, end="", flush=True)
    print("\n")


def _extract_args_from_cli(args: list[str]) -> CommandArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "prompt",
        type=str,
        help="The user prompt for the LLM.",
    )
    parser.add_argument(
        "-c",
        "--character",
        type=str,
        required=False,
        help="The character the LLM should assume the persona of.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[model.value for model in llm_client.Model],
        default=llm_client.Model.CLAUDE_3_5_SONNET.value,
        help="The model to ask the question to.",
    )
    parser.add_argument(
        "-s",
        "--stream",
        action="store_true",
        help="Whether to stream the response from the model asynchronously.",
    )

    parsed_args = parser.parse_args(args)

    return CommandArgs(
        prompt=parsed_args.prompt,
        character=parsed_args.character,
        model=llm_client.Model(parsed_args.model),
        stream=parsed_args.stream,
    )
