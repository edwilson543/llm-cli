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
        question="What is your speciality?", models=[clients.ECHO], persona="Eva Perón"
    )

    await question.ask_question(arguments=arguments)

    expected_output = (
        "\033[96m\necho (Eva Perón): \n---\nWhat is your speciality?\n---\n\n"
    )
    assert output.getvalue() == expected_output


@pytest.mark.asyncio
async def test_asks_question_to_multiple_models_and_gets_response():
    output = io.StringIO()
    sys.stdout = output

    arguments = question.QuestionCommandArgs(
        question="What is your speciality?",
        models=[clients.ECHO, clients.ECHO],
        persona="Eva Perón",
    )

    await question.ask_question(arguments=arguments)

    response = "\033[96m\necho (Eva Perón): \n---\nWhat is your speciality?\n---\n\n"
    assert output.getvalue() == response * 2


@pytest.mark.asyncio
async def test_handles_error_raised_while_streaming_response_from_client():
    output = io.StringIO()
    sys.stdout = output

    arguments = question.QuestionCommandArgs(
        question="What is your speciality?", models=[clients.BROKEN], persona=None
    )

    await question.ask_question(arguments=arguments)

    expected_output = "\033[96m\nbroken: \n---\nError streaming response. The FAKE_AI API responded with status code: 503.\n---\n\n"
    assert output.getvalue() == expected_output
