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
            model=llm_client.Model.ECHO,
            persona="Eva Perón",
            stream=False,
        )

        await question.ask_question(arguments=arguments)

        assert output.getvalue() == "\nWhat is your speciality?\n\n"

    @pytest.mark.asyncio
    async def test_handles_error_raised_by_broken_client(self):
        output = io.StringIO()
        sys.stdout = output

        arguments = question.CommandArgs(
            question="What is your speciality?",
            model=llm_client.Model.BROKEN,
            persona=None,
            stream=False,
        )

        with pytest.raises(llm_client.LLMClientError):
            await question.ask_question(arguments=arguments)


class TestQuestionStreamed:
    @pytest.mark.asyncio
    async def test_asks_question_to_echo_model_and_gets_response(self):
        output = io.StringIO()
        sys.stdout = output

        arguments = question.CommandArgs(
            question="Have you got much on today?",
            model=llm_client.Model.ECHO,
            persona=None,
            stream=True,
        )

        await question.ask_question(arguments=arguments)

        assert output.getvalue() == "\nHave you got much on today?\n\n"
