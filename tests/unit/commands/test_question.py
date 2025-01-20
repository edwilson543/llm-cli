import pytest

from llm_cli.commands import question


class TestQuestionCommandArgs:
    def test_gets_system_prompt_when_persona_is_none(self):
        command_args = question.QuestionCommandArgs(
            question="", persona=None, models=[]
        )

        expected_prompt = "Please be as succinct as possible in your answer."
        assert command_args.system_prompt == expected_prompt

    def test_gets_system_prompt_when_persona_is_not_noe(self):
        command_args = question.QuestionCommandArgs(
            question="", persona="Horatio Nelson", models=[]
        )

        expected_prompt = "Please be as succinct as possible in your answer. Please assume the persona of Horatio Nelson."
        assert command_args.system_prompt == expected_prompt


class TestExtractArgsFromCli:
    def test_gets_prompt_when_specified_as_positional_arg(self):
        raw_args = ["What do you find interesting?"]

        extracted_args = question._extract_args_from_cli(raw_args)

        assert extracted_args.question == "What do you find interesting?"
        # Ensure the defaults are set as expected.
        assert len(extracted_args.models) == 1
        assert extracted_args.models[0].friendly_name == "claude-sonnet"
        assert extracted_args.persona is None

    @pytest.mark.parametrize("model_flag", ["-m", "--model"])
    def test_gets_model_specified_via_shorthand_or_longhand_arg(self, model_flag: str):
        raw_args = ["What do you find interesting?", model_flag, "echo"]

        extracted_args = question._extract_args_from_cli(raw_args)

        assert len(extracted_args.models) == 1
        assert extracted_args.models[0].friendly_name == "echo"

    @pytest.mark.parametrize("model_flag", ["-m", "--model"])
    def test_gets_multiple_models_specified_via_shorthand_or_longhand_arg(
        self, model_flag: str
    ):
        raw_args = [
            "What do you find interesting?",
            model_flag,
            "echo",
            "claude-sonnet",
            "grok-2",
        ]

        extracted_args = question._extract_args_from_cli(raw_args)

        assert len(extracted_args.models) == 3
        models = [model.official_name for model in extracted_args.models]
        assert models == ["echo", "claude-3-5-sonnet-latest", "grok-2-latest"]

    @pytest.mark.parametrize("persona_flag", ["-p", "--persona"])
    def test_gets_model_specified_via_shorthand_arg(self, persona_flag: str):
        raw_args = ["What's for breakfast'?", persona_flag, "gandalf"]

        extracted_args = question._extract_args_from_cli(raw_args)

        assert extracted_args.persona == "gandalf"

    def test_raises_when_question_argument_is_specified_incorrectly(self):
        args = ["-q", "What do you find interesting?"]

        with pytest.raises(SystemExit):
            question._extract_args_from_cli(args)
