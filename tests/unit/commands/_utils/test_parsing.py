import argparse

import pytest

from llm_cli.commands._utils import parsing
from llm_cli.domain import llm_client


class TestCommandArgs__SystemPrompt:
    def test_gets_system_prompt_when_persona_is_none(self):
        command_args = parsing.CommandArgs(persona=None, model=llm_client.ECHO)

        assert (
            command_args.system_prompt
            == "Please be as succinct as possible in your answer."
        )

    def test_gets_system_prompt_when_persona_is_not_noe(self):
        command_args = parsing.CommandArgs(
            persona="Horatio Nelson", model=llm_client.ECHO
        )

        assert (
            command_args.system_prompt
            == "Please be as succinct as possible in your answer. Please assume the persona of Horatio Nelson."
        )


class TestGetModelFromFriendlyName:
    def test_gets_model_when_friendly_name_is_recognised(self):
        echo_model = llm_client.ECHO

        result = parsing.get_model_from_friendly_name(
            friendly_name=echo_model.friendly_name
        )

        assert result == echo_model

    def test_raises_when_friendly_name_is_not_recognised(self):
        with pytest.raises(argparse.ArgumentError) as exc:
            parsing.get_model_from_friendly_name(friendly_name="fake-model")

        assert str(exc.value) == "Model 'fake-model' is not available."
