import io
import sys

from llm_cli.commands import question
from llm_cli.domain import llm_client


def test_asks_question_to_echo_model_and_gets_response():
    output = io.StringIO()
    sys.stdout = output

    arguments = question.CommandArgs(
        prompt="What is your speciality?",
        model=llm_client.Model.ECHO,
    )

    question.ask_question(arguments=arguments)

    assert output.getvalue() == "What is your speciality?\n"
