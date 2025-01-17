import io
import sys
from unittest import mock

import pytest

from llm_cli.commands import question
from llm_cli.domain import llm_client


@pytest.mark.asyncio
async def test_asks_question_to_echo_model_and_gets_response():
    output = io.StringIO()
    sys.stdout = output

    arguments = question.CommandArgs(
        question="What is your speciality?",
        model=llm_client.ECHO,
        persona="Eva Perón",
    )

    await question.ask_question(arguments=arguments)

    expected_output = "\033[96m\nEva Perón:\n---\nWhat is your speciality?\n---\n\n"
    assert output.getvalue() == expected_output


@pytest.mark.asyncio
@mock.patch(
    "llm_cli.env.as_str", side_effect=llm_client.APIKeyNotSet(env_var="XAI_API_KEY")
)
async def test_handles_error_when_api_key_is_not_set(mock_env_vars: mock.Mock):
    output = io.StringIO()
    sys.stdout = output

    arguments = question.CommandArgs(
        question="Do you have any regrets?",
        model=llm_client.GROK_2,
        persona=None,
    )

    await question.ask_question(arguments=arguments)

    expected_output = "\x1b[96m\x1b[93mThe XAI_API_KEY environment variable must be set to use XAI's models!\n"
    assert output.getvalue() == expected_output
    mock_env_vars.assert_called_once_with("XAI_API_KEY")


@pytest.mark.asyncio
async def test_handles_error_raised_while_streaming_response_from_client():
    output = io.StringIO()
    sys.stdout = output

    arguments = question.CommandArgs(
        question="What is your speciality?",
        model=llm_client.BROKEN,
        persona=None,
    )

    await question.ask_question(arguments=arguments)

    expected_output = "\033[96m\nbroken:\n---\nError streaming response. Fake AI is permanently broken.\n---\n\n"
    assert output.getvalue() == expected_output
