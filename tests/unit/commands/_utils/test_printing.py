import io
import sys
from typing import AsyncGenerator
from unittest import mock

import pytest

from llm_cli import clients
from llm_cli.commands._utils import printing
from testing import factories


class TestGetLLMClientOrPrintError:
    def test_gets_client(self):
        output = io.StringIO()
        sys.stdout = output

        client = printing.get_llm_client_or_print_error(
            model=clients.ECHO, parameters=factories.ModelParameters()
        )

        assert client is not None
        assert output.getvalue() == ""

    @mock.patch.object(
        clients,
        "get_llm_client",
        side_effect=clients.APIKeyNotSet(env_var="XAI_API_KEY"),
    )
    def test_prints_error_when_model_not_available(self, mock_get_client: mock.Mock):
        output = io.StringIO()
        sys.stdout = output

        client = printing.get_llm_client_or_print_error(
            model=clients.GROK_2, parameters=factories.ModelParameters()
        )

        assert client is None
        expected_output = (
            "\033[93m"
            + "The XAI_API_KEY environment variable must be set to use XAI's models!\n"
        )
        assert output.getvalue() == expected_output


class TestPrintBlockFromInterlocutor:
    def test_wraps_printed_context_in_block(self):
        content = "Some content"

        output = io.StringIO()
        sys.stdout = output

        with printing.print_block_from_interlocutor(interlocutor="Lex"):
            print(content, end="")

        assert output.getvalue() == "\nLex: \n---\nSome content\n---\n\n"


class TestPrintResponseStreamToTerminal:
    @pytest.mark.asyncio
    async def test_prints_response_on_single_line(self):
        response = "This can all fit on one line."

        async def _response_stream() -> AsyncGenerator[str, None]:
            yield response[: len(response) // 2]
            yield response[len(response) // 2 :]

        output = io.StringIO()
        sys.stdout = output

        await printing.print_response_stream_to_terminal(
            response_stream=_response_stream(), max_line_width=len(response) + 1
        )

        assert output.getvalue() == response

    @pytest.mark.asyncio
    async def test_prints_response_wrapped_over_three_lines(self):
        response = "This does not all fit on one line."

        async def _response_stream() -> AsyncGenerator[str, None]:
            yield response[: len(response) // 2]
            yield response[len(response) // 2 :]

        output = io.StringIO()
        sys.stdout = output

        await printing.print_response_stream_to_terminal(
            response_stream=_response_stream(), max_line_width=10
        )

        assert output.getvalue() == "This does \nnot all \nfit on one\nline."

    @pytest.mark.asyncio
    async def test_resets_current_line_width_when_response_includes_line_break(self):
        multiline_text = """
That's okay! I can:
- Answer questions
- Help solve problems
- Explain topics
- Give suggestions"""

        async def _response_stream() -> AsyncGenerator[str, None]:
            yield multiline_text[: len(multiline_text) // 2]
            yield multiline_text[len(multiline_text) // 2 :]

        output = io.StringIO()
        sys.stdout = output

        await printing.print_response_stream_to_terminal(
            response_stream=_response_stream(), max_line_width=40
        )

        assert output.getvalue() == multiline_text

    @pytest.mark.asyncio
    async def test_does_not_insert_line_break_in_middle_of_word(self):
        all_one_word = "Alloneword"

        async def _response_stream() -> AsyncGenerator[str, None]:
            yield all_one_word

        output = io.StringIO()
        sys.stdout = output

        await printing.print_response_stream_to_terminal(
            response_stream=_response_stream(), max_line_width=1
        )

        assert output.getvalue() == all_one_word
