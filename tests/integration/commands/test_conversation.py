import io
import sys
from unittest import mock

import pytest

from llm_cli import clients
from llm_cli.commands import conversation


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

    arguments = conversation.ConversationCommandArgs(
        model=clients.ECHO, persona="Paul Erdős"
    )

    await conversation.start_conversation(arguments=arguments)

    stdout = output.getvalue()
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

    arguments = conversation.ConversationCommandArgs(model=clients.BROKEN, persona=None)

    await conversation.start_conversation(arguments=arguments)

    assert (
        "Error streaming response. Fake AI is permanently broken." in output.getvalue()
    )
