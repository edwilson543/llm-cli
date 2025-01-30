import argparse
import asyncio
import dataclasses
import sys

from llm_cli import clients

from ._utils import parsing as parsing_utils
from ._utils import printing as printing_utils


@dataclasses.dataclass(frozen=True)
class QuestionCommandArgs:
    question: str
    models: list[clients.Model]
    persona: str | None

    @property
    def model_parameters(self) -> clients.ModelParameters:
        return clients.ModelParameters(
            system_prompt=self.system_prompt,
            max_tokens=1024,
            temperature=1.0,
            top_p=1.0,
        )

    @property
    def system_prompt(self) -> str:
        prompt = "Please be as succinct as possible in your answer."
        if self.persona:
            prompt += f" Please assume the persona of {self.persona}."
        return prompt


def main():
    asyncio.run(ask_question())


async def ask_question(*, arguments: QuestionCommandArgs | None = None) -> None:
    """
    Command to ask a single question to one or more models.
    """
    if arguments is None:
        arguments = _extract_args_from_cli(sys.argv[1:])

    for model in arguments.models:
        printing_utils.set_print_colour_to_cyan()

        client = printing_utils.get_llm_client_or_print_error(
            model=model, parameters=arguments.model_parameters
        )
        if client is None:
            continue

        interlocutor = printing_utils.get_interlocutor_display_name(
            model=model, persona=arguments.persona
        )
        with printing_utils.print_block_from_interlocutor(interlocutor=interlocutor):
            try:
                await printing_utils.print_response_stream_to_terminal(
                    client.stream_response(user_prompt=arguments.question)
                )
            except clients.LLMClientError as exc:
                print("Error streaming response.", exc, end="")


def _extract_args_from_cli(args: list[str]) -> QuestionCommandArgs:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

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
        choices=parsing_utils.get_model_choices(),
        default=[clients.get_default_model().friendly_name],
        help=f"The model that should be used. Multiple models can be specified, separated by a space. {parsing_utils.MODEL_CHOICES_HELP}",
        metavar="",
    )
    parsing_utils.add_persona_argument(parser)

    parsed_args = parser.parse_args(args)

    models = [
        parsing_utils.get_model_from_friendly_name(model) for model in parsed_args.model
    ]
    return QuestionCommandArgs(
        question=parsed_args.question,
        models=models,
        persona=parsed_args.persona,
    )
