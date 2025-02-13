import argparse
import asyncio
import dataclasses
import sys

from llm_cli import clients

from ._utils import parsing as parsing_utils
from ._utils import printing as printing_utils


EXIT = "exit"


@dataclasses.dataclass(frozen=True)
class ConversationCommandArgs:
    model: clients.Model
    persona: str | None
    temperature: float
    top_p: float

    @property
    def model_parameters(self) -> clients.ModelParameters:
        return clients.ModelParameters(
            system_prompt=self.system_prompt,
            max_tokens=1024,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    @property
    def system_prompt(self) -> str:
        prompt = "Please be as succinct as possible in your answer."
        if self.persona:
            prompt += f" Please assume the persona of {self.persona}."
        return prompt

    @property
    def interlocutor(self) -> str:
        return printing_utils.get_interlocutor_display_name(
            model=self.model, persona=self.persona
        )


def main():
    asyncio.run(start_conversation())


async def start_conversation(
    *, arguments: ConversationCommandArgs | None = None
) -> None:
    """
    Command to have a multi-turn conversation with a model.
    """
    if arguments is None:
        arguments = _extract_args_from_cli(sys.argv[1:])

    try:
        model_parameters = arguments.model_parameters
    except clients.InvalidModelParameters as exc:
        printing_utils.set_print_colour_to_yellow()
        print(exc)
        return None

    client = printing_utils.get_llm_client_or_print_error(
        model=arguments.model, parameters=model_parameters
    )
    if not client:
        return

    await _conversation_loop(client=client, arguments=arguments)


async def _conversation_loop(
    *, client: clients.LLMClient, arguments: ConversationCommandArgs
) -> None:
    printing_utils.set_print_colour_to_cyan()
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
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=parsing_utils.get_model_choices(),
        default=clients.get_default_model().friendly_name,
        help=f"The model that should be used. {parsing_utils.MODEL_CHOICES_HELP}",
        metavar="",
    )
    parsing_utils.add_persona_argument(parser)
    parsing_utils.add_temperature_argument(parser)
    parsing_utils.add_top_p_argument(parser)

    parsed_args = parser.parse_args(args)

    return ConversationCommandArgs(
        persona=parsed_args.persona,
        model=parsing_utils.get_model_from_friendly_name(parsed_args.model),
        temperature=parsed_args.temperature,
        top_p=parsed_args.top_p,
    )
