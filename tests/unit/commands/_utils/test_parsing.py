import argparse

import pytest

from llm_cli import clients
from llm_cli.commands._utils import parsing


class TestGetModelFromFriendlyName:
    def test_gets_model_when_friendly_name_is_recognised(self):
        echo_model = clients.ECHO

        result = parsing.get_model_from_friendly_name(
            friendly_name=echo_model.friendly_name
        )

        assert result == echo_model

    def test_raises_when_friendly_name_is_not_recognised(self):
        with pytest.raises(argparse.ArgumentError) as exc:
            parsing.get_model_from_friendly_name(friendly_name="fake-model")

        assert str(exc.value) == "Model 'fake-model' is not available."
