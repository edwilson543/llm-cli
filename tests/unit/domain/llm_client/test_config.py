from unittest import mock

from llm_cli.domain.llm_client import _anthropic, _broken, _config, _echo


class TestGetLLMClient:
    @mock.patch("llm_cli.env.as_str", return_value="something")
    def test_gets_anthropic_client_for_claude_3_5_sonnet_model(
        self, mock_env_vars: mock.Mock
    ):
        client = _config.get_llm_client(model=_config.Model.CLAUDE_3_5_SONNET)

        assert isinstance(client, _anthropic.AnthropicClient)
        assert client._model == "claude-3-5-sonnet-20241022"
        mock_env_vars.assert_called_once_with("ANTHROPIC_API_KEY")

    def test_gets_echo_client(self):
        client = _config.get_llm_client(model=_config.Model.ECHO)

        assert isinstance(client, _echo.EchoClient)

    def test_gets_broken_client(self):
        client = _config.get_llm_client(model=_config.Model.BROKEN)

        assert isinstance(client, _broken.BrokenClient)
