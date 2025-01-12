import argparse
import asyncio
import dataclasses
import sys

from llm_cli.domain import llm_client


@dataclasses.dataclass
class CommandArgs:
    question: str
    persona: str | None
    model: llm_client.Model
    stream: bool


def main():
    asyncio.run(ask_question())


async def ask_question(*, arguments: CommandArgs | None = None) -> None:
    """
    Command to ask the model a single question.
    """
    if arguments is None:
        arguments = _extract_args_from_cli(sys.argv[1:])

    client = llm_client.get_llm_client(model=arguments.model)

    print("\n", end="")

    if arguments.stream:
        await _ask_question_async(client=client, arguments=arguments)
    else:
        _ask_question_sync(client=client, arguments=arguments)

    print("\n")


def _ask_question_sync(*, client: llm_client.LLMClient, arguments: CommandArgs) -> None:
    try:
        response_message = client.get_response(
            user_prompt=arguments.question, persona=arguments.persona
        )
    except llm_client.LLMClientError as exc:
        print(str(exc))
        raise

    print(response_message, end="")


async def _ask_question_async(
    *, client: llm_client.LLMClient, arguments: CommandArgs
) -> None:
    try:
        response = client.get_response_async(
            user_prompt=arguments.question, persona=arguments.persona
        )
    except llm_client.LLMClientError as exc:
        print(str(exc))
        raise

    async for response_message in response:
        print(response_message, end="", flush=True)


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
        choices=[model.value for model in llm_client.Model.available_models()],
        default=llm_client.Model.CLAUDE_3_5_SONNET.value,
        help="The model that should be used.",
    )
    parser.add_argument(
        "-s",
        "--stream",
        action="store_true",
        help="Whether to stream the response from the model asynchronously.",
    )

    parsed_args = parser.parse_args(args)

    return CommandArgs(
        question=parsed_args.question,
        persona=parsed_args.persona,
        model=llm_client.Model(parsed_args.model),
        stream=parsed_args.stream,
    )
