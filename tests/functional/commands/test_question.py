import io
import sys

import pytest

from llm_cli.commands import question
from llm_cli.domain import llm_client


class TestQuestionNotStreamed:
    @pytest.mark.asyncio
    async def test_asks_question_to_echo_model_and_gets_response(self):
        output = io.StringIO()
        sys.stdout = output

        arguments = question.CommandArgs(
            question="What is your speciality?",
            model=llm_client.ECHO,
            persona="Eva Perón",
            stream=False,
        )

        await question.ask_question(arguments=arguments)

        assert (
            output.getvalue()
            == "\033[96m\nEva Perón:\n---\nWhat is your speciality?\n---\n\n"
        )

    @pytest.mark.asyncio
    async def test_handles_error_raised_by_broken_client(self):
        output = io.StringIO()
        sys.stdout = output

        arguments = question.CommandArgs(
            question="What is your speciality?",
            model=llm_client.BROKEN,
            persona=None,
            stream=False,
        )

        await question.ask_question(arguments=arguments)

        assert (
            output.getvalue()
            == "\033[96m\nbroken:\n---\nUnable to get a response from 'FAKE_AI'. \n---\n\n"
        )


class TestQuestionStreamed:
    @pytest.mark.asyncio
    async def test_asks_question_to_echo_model_and_gets_response(self):
        output = io.StringIO()
        sys.stdout = output

        arguments = question.CommandArgs(
            question="Have you got much on today?",
            model=llm_client.ECHO,
            persona=None,
            stream=True,
        )

        await question.ask_question(arguments=arguments)

        assert (
            output.getvalue()
            == "\033[96m\necho:\n---\nHave you got much on today?\n---\n\n"
        )
