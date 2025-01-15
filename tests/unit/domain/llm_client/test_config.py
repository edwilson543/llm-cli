from unittest import mock

from llm_cli.domain.llm_client import _anthropic, _broken, _config, _echo, _models


class TestGetLLMClient:
    @mock.patch("llm_cli.env.as_str", return_value="something")
    def test_gets_anthropic_client_for_claude_3_5_sonnet_model(
        self, mock_env_vars: mock.Mock
    ):
        model = _models.Model.CLAUDE_SONNET

        client = _config.get_llm_client(model=model)

        assert isinstance(client, _anthropic.AnthropicClient)
        assert client._model == model
        mock_env_vars.assert_called_once_with("ANTHROPIC_API_KEY")

    def test_gets_echo_client(self):
        client = _config.get_llm_client(model=_models.Model.ECHO)

        assert isinstance(client, _echo.EchoClient)

    def test_gets_broken_client(self):
        client = _config.get_llm_client(model=_models.Model.BROKEN)

        assert isinstance(client, _broken.BrokenClient)
