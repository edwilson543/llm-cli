import io
import sys
from unittest import mock

import pytest

from llm_cli import clients
from llm_cli.commands import conversation
from testing import factories


@pytest.mark.asyncio
@mock.patch("builtins.input")
async def test_has_brief_conversation_with_echo_model_then_exits(mock_input: mock.Mock):
    output = io.StringIO()
    sys.stdout = output

    mock_input.side_effect = [
        "How many sides does a circle have?",
        "What do you mean 'How do I define what a side is'???",
        "exit",
    ]

    arguments = factories.ConversationCommandArgs(
        model=clients.ECHO, persona="Paul Erdős"
    )

    await conversation.start_conversation(arguments=arguments)

    stdout = output.getvalue()
    assert "echo (Paul Erdős)" in stdout
    assert "How many sides does a circle have?" in stdout
    assert "What do you mean 'How do I define what a side is'???" in stdout
    assert "Goodbye!" in stdout


@pytest.mark.asyncio
@mock.patch("builtins.input", return_value="This is the only Japanese I know.")
async def test_handles_error_raised_while_streaming_response_from_client(
    mock_input: mock.Mock,
):
    output = io.StringIO()
    sys.stdout = output

    arguments = factories.ConversationCommandArgs(model=clients.BROKEN)

    await conversation.start_conversation(arguments=arguments)

    expected_output = (
        "Error streaming response. The FAKE_AI API responded with status code: 503."
    )
    assert expected_output in output.getvalue()


@pytest.mark.asyncio
async def test_handles_error_raised_from_invalid_value_for_top_p():
    output = io.StringIO()
    sys.stdout = output

    arguments = factories.ConversationCommandArgs(model=clients.ECHO, top_p=-31.5)

    await conversation.start_conversation(arguments=arguments)

    expected_output = "\x1b[93mTop p must be in the range [0, 1].\n"
    assert output.getvalue() == expected_output
