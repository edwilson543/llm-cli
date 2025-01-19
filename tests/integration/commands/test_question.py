import io
import sys

import pytest

from llm_cli import clients
from llm_cli.commands import question


@pytest.mark.asyncio
async def test_asks_question_to_echo_model_and_gets_response():
    output = io.StringIO()
    sys.stdout = output

    arguments = question.QuestionCommandArgs(
        question="What is your speciality?", model=clients.ECHO, persona="Eva Perón"
    )

    await question.ask_question(arguments=arguments)

    expected_output = "\033[96m\nEva Perón: \n---\nWhat is your speciality?\n---\n\n"
    assert output.getvalue() == expected_output


@pytest.mark.asyncio
async def test_handles_error_raised_while_streaming_response_from_client():
    output = io.StringIO()
    sys.stdout = output

    arguments = question.QuestionCommandArgs(
        question="What is your speciality?", model=clients.BROKEN, persona=None
    )

    await question.ask_question(arguments=arguments)

    expected_output = "\033[96m\nbroken: \n---\nError streaming response. Fake AI is permanently broken.\n---\n\n"
    assert output.getvalue() == expected_output
