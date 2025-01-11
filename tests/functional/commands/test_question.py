import io
import sys

import pytest

from llm_cli.commands import question
from llm_cli.domain import llm_client


@pytest.mark.asyncio
async def test_asks_question_to_echo_model_and_gets_response_synchronously():
    output = io.StringIO()
    sys.stdout = output

    arguments = question.CommandArgs(
        prompt="What is your speciality?",
        model=llm_client.Model.ECHO,
        character="Eva Perón",
        stream=False,
    )

    await question.ask_question(arguments=arguments)

    assert output.getvalue() == "\nWhat is your speciality?\n\n"


@pytest.mark.asyncio
async def test_asks_question_to_echo_model_and_gets_response_asynchronously():
    output = io.StringIO()
    sys.stdout = output

    arguments = question.CommandArgs(
        prompt="Have you got much on today?",
        model=llm_client.Model.ECHO,
        character=None,
        stream=True,
    )

    await question.ask_question(arguments=arguments)

    assert output.getvalue() == "\nHave you got much on today?\n"
