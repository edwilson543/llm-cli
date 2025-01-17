import io
import sys

import pytest

from llm_cli.commands import question
from llm_cli.domain import llm_client


class TestStreamResponseAndPrintFormattedOutput:
    @pytest.mark.asyncio
    async def test_prints_response_on_single_line(self):
        client = llm_client.get_llm_client(model=llm_client.ECHO)
        response = "This can all fit on one line."
        arguments = self._get_arguments(response=response)

        output = io.StringIO()
        sys.stdout = output

        await question._stream_response_and_print_formatted_output(
            client=client, arguments=arguments, max_line_width=len(response) + 1
        )

        assert output.getvalue() == response

    @pytest.mark.asyncio
    async def test_prints_response_wrapped_over_three_lines(self):
        client = llm_client.get_llm_client(model=llm_client.ECHO)
        response = "This does not all fit on one line."
        arguments = self._get_arguments(response=response)

        output = io.StringIO()
        sys.stdout = output

        await question._stream_response_and_print_formatted_output(
            client=client, arguments=arguments, max_line_width=9
        )

        assert output.getvalue() == "This does\nnot all \nfit on \none line."

    @pytest.mark.asyncio
    async def test_resets_current_line_width_when_response_includes_line_break(self):
        client = llm_client.get_llm_client(model=llm_client.ECHO)
        response = "This \nis some multi line text. Thanks!"
        arguments = self._get_arguments(response=response)

        output = io.StringIO()
        sys.stdout = output

        await question._stream_response_and_print_formatted_output(
            client=client, arguments=arguments, max_line_width=13
        )

        assert output.getvalue() == "This \nis some \nmulti line \ntext. Thanks!"

    @staticmethod
    def _get_arguments(response: str) -> question.CommandArgs:
        return question.CommandArgs(
            question=response, persona=None, model=llm_client.ECHO
        )


class TestExtractArgsFromCli:
    def test_gets_prompt_when_specified_as_positional_arg(self):
        raw_args = ["What do you find interesting?"]

        extracted_args = question._extract_args_from_cli(raw_args)

        assert extracted_args.question == "What do you find interesting?"
        # Ensure the defaults are set as expected.
        assert extracted_args.model.friendly_name == "claude-sonnet"
        assert extracted_args.persona is None

    @pytest.mark.parametrize("model_flag", ["-m", "--model"])
    def test_gets_model_specified_via_shorthand_or_longhand_arg(self, model_flag: str):
        raw_args = ["What do you find interesting?", model_flag, "echo"]

        extracted_args = question._extract_args_from_cli(raw_args)

        assert extracted_args.model.friendly_name == "echo"

    @pytest.mark.parametrize("persona_flag", ["-p", "--persona"])
    def test_gets_model_specified_via_shorthand_arg(self, persona_flag: str):
        raw_args = ["What's for breakfast'?", persona_flag, "gandalf"]

        extracted_args = question._extract_args_from_cli(raw_args)

        assert extracted_args.persona == "gandalf"

    def test_raises_when_question_argument_is_specified_incorrectly(self):
        args = ["-q", "What do you find interesting?"]

        with pytest.raises(SystemExit):
            question._extract_args_from_cli(args)
