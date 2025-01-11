import pytest

from llm_cli.commands import question
from llm_cli.domain import llm_client


class TestExtractArgsFromCli:
    def test_gets_prompt_when_specified_as_positional_arg(self):
        raw_args = ["What do you find interesting?"]

        extracted_args = question._extract_args_from_cli(raw_args)

        assert extracted_args.prompt == "What do you find interesting?"
        # The default model should be used, since the model was unspecified.
        assert extracted_args.model == llm_client.Model.CLAUDE_3_5_SONNET

    @pytest.mark.parametrize("model_flag", ["-m", "--model"])
    def test_gets_model_specified_via_shorthand_arg(self, model_flag: str):
        raw_args = ["What do you find interesting?", model_flag, "ECHO"]

        extracted_args = question._extract_args_from_cli(raw_args)

        assert extracted_args.model == llm_client.Model.ECHO

    def test_raises_when_prompt_is_specified_incorrectly(self):
        args = ["--prompt", "What do you find interesting?"]

        with pytest.raises(SystemExit):
            question._extract_args_from_cli(args)
