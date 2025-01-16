from unittest import mock

import pytest

from llm_cli.domain.llm_client import _anthropic, _broken, _config, _echo, _models, _xai


class TestGetLLMClient:
    @mock.patch("llm_cli.env.as_str", return_value="something")
    def test_gets_anthropic_client_for_claude_sonnet_model(
        self, mock_env_vars: mock.Mock
    ):
        claude_sonnet = _models.CLAUDE_SONNET

        client = _config.get_llm_client(model=claude_sonnet)

        assert isinstance(client, _anthropic.AnthropicClient)
        assert client._model == claude_sonnet.official_name
        mock_env_vars.assert_called_once_with("ANTHROPIC_API_KEY")

    @mock.patch("llm_cli.env.as_str", return_value="something")
    def test_gets_xai_client_for_grok_2_model(self, mock_env_vars: mock.Mock):
        grok_2 = _models.GROK_2

        client = _config.get_llm_client(model=grok_2)

        assert isinstance(client, _xai.XAIClient)
        assert client._model == grok_2.official_name
        mock_env_vars.assert_called_once_with("XAI_API_KEY")

    def test_gets_echo_client(self):
        client = _config.get_llm_client(model=_models.ECHO)

        assert isinstance(client, _echo.EchoClient)

    def test_gets_broken_client(self):
        client = _config.get_llm_client(model=_models.BROKEN)

        assert isinstance(client, _broken.BrokenClient)

    def test_raises_when_model_not_configured(self):
        not_configured_model = _models.Model(
            vendor=_models.Vendor.FAKE_AI,
            friendly_name="not-configured",
            official_name="not-configured",
        )

        with pytest.raises(_config.ModelNotConfigured) as exc:
            _config.get_llm_client(model=not_configured_model)

        assert exc.value.model == not_configured_model
        expected_exc_message = (
            "No LLMClient implementation is installed for model 'not-configured'."
        )
        assert str(exc.value) == expected_exc_message
