import pytest

from llm_cli.clients import _base


class TestModelParameters__PostInit:
    def test_does_not_raise_for_valid_parameters(self):
        _base.ModelParameters(
            system_prompt="valid", max_tokens=1024, temperature=1.0, top_p=1.0
        )

    @pytest.mark.parametrize("max_tokens", [0, -1])
    def test_raises_for_invalid_max_tokens_parameter(self, max_tokens: int):
        with pytest.raises(_base.InvalidModelParameters) as exc:
            _base.ModelParameters(
                system_prompt="valid", max_tokens=max_tokens, temperature=1.0, top_p=1.0
            )

        assert str(exc.value) == "Max tokens must be greater than 0."

    @pytest.mark.parametrize("temperature", [-0.5, 1.5])
    def test_raises_for_invalid_temperature_parameter(self, temperature: float):
        with pytest.raises(_base.InvalidModelParameters) as exc:
            _base.ModelParameters(
                system_prompt="valid",
                max_tokens=1024,
                temperature=temperature,
                top_p=1.0,
            )

        assert str(exc.value) == "Temperature must be in the range [0, 1]."

    @pytest.mark.parametrize("top_p", [-0.1, 1.2])
    def test_raises_for_invalid_top_p_parameter(self, top_p: float):
        with pytest.raises(_base.InvalidModelParameters) as exc:
            _base.ModelParameters(
                system_prompt="valid", max_tokens=1024, temperature=0.5, top_p=top_p
            )

        assert str(exc.value) == "Top p must be in the range [0, 1]."
