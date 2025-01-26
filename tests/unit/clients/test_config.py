from unittest import mock

import pytest

from llm_cli import env
from llm_cli.clients import _config, _models
from llm_cli.clients._fakes import broken, echo
from llm_cli.clients._vendors import anthropic, meta, mistral, openai, xai


class TestGetLLMClient:
    @pytest.mark.parametrize(
        "model", [_models.CLAUDE_HAIKU, _models.CLAUDE_SONNET, _models.CLAUDE_OPUS]
    )
    @mock.patch.object(_config.env, "as_str", return_value="something")
    def test_gets_anthropic_client_for_anthropic_models(
        self, mock_env_vars: mock.Mock, model: _models.Model
    ):
        client = _config.get_llm_client(model=model, system_prompt="fake")

        assert isinstance(client, anthropic.AnthropicClient)
        assert client._model == model
        assert client._system_prompt == "fake"
        mock_env_vars.assert_called_once_with("ANTHROPIC_API_KEY")

    @mock.patch.object(_config.env, "as_str", return_value="something")
    def test_gets_meta_client_for_llama_3_model(self, mock_env_vars: mock.Mock):
        llama_3 = _models.LLAMA_3

        client = _config.get_llm_client(model=llama_3, system_prompt="fake")

        assert isinstance(client, meta.MetaClient)
        assert client._model == llama_3
        assert client._messages == [{"role": "system", "content": "fake"}]
        mock_env_vars.assert_called_once_with("META_API_KEY")

    @pytest.mark.parametrize(
        "model", [_models.CODESTRAL, _models.MISTRAL, _models.MINISTRAL]
    )
    @mock.patch.object(_config.env, "as_str", return_value="something")
    def test_gets_mistral_client_for_mistral_models(
        self, mock_env_vars: mock.Mock, model: _models.Model
    ):
        client = _config.get_llm_client(model=model, system_prompt="fake")

        assert isinstance(client, mistral.MistralClient)
        assert client._model == model
        assert client._messages == [{"role": "system", "content": "fake"}]
        mock_env_vars.assert_called_once_with("MISTRAL_API_KEY")

    @pytest.mark.parametrize("model", [_models.GPT_4, _models.GPT_4_MINI])
    @mock.patch.object(_config.env, "as_str", return_value="something")
    def test_gets_openai_client_for_openai_models(
        self, mock_env_vars: mock.Mock, model: _models.Model
    ):
        client = _config.get_llm_client(model=model, system_prompt="fake")

        assert isinstance(client, openai.OpenAIClient)
        assert client._model == model
        assert client._system_prompt == "fake"
        mock_env_vars.assert_called_once_with("OPENAI_API_KEY")

    @mock.patch.object(_config.env, "as_str", return_value="something")
    def test_gets_xai_client_for_grok_2_model(self, mock_env_vars: mock.Mock):
        grok_2 = _models.GROK_2

        client = _config.get_llm_client(model=grok_2, system_prompt="fake")

        assert isinstance(client, xai.XAIClient)
        assert client._model == grok_2
        assert client._system_prompt == "fake"
        mock_env_vars.assert_called_once_with("XAI_API_KEY")

    def test_gets_echo_client(self):
        client = _config.get_llm_client(model=_models.ECHO, system_prompt="fake")

        assert isinstance(client, echo.EchoClient)
        assert client._system_prompt == "fake"

    def test_gets_broken_client(self):
        client = _config.get_llm_client(model=_models.BROKEN, system_prompt="fake")

        assert isinstance(client, broken.BrokenClient)
        assert client._system_prompt == "fake"

    def test_raises_when_model_not_configured(self):
        not_configured_model = _models.Model(
            vendor=_models.Vendor.FAKE_AI,
            friendly_name="not-configured",
            official_name="not-configured",
        )

        with pytest.raises(_config.ModelNotConfigured) as exc:
            _config.get_llm_client(model=not_configured_model, system_prompt="fake")

        assert exc.value.model == not_configured_model
        expected_exc_message = (
            "No LLMClient implementation is installed for model 'not-configured'."
        )
        assert str(exc.value) == expected_exc_message


class TestGetDefaultModel:
    @mock.patch.object(_config.env, "as_str", return_value="grok-2")
    def test_gets_default_model_when_set_via_env_var(self, mock_env_vars: mock.Mock):
        default_model = _config.get_default_model()

        assert default_model == _models.GROK_2
        mock_env_vars.assert_called_once_with("DEFAULT_MODEL")

    @mock.patch.object(
        _config.env,
        "as_str",
        side_effect=env.EnvironmentVariableNotSet(key="DEFAULT_MODEL"),
    )
    def test_gets_claude_sonnet_when_env_var_not_set(self, mock_env_vars: mock.Mock):
        default_model = _config.get_default_model()

        assert default_model == _models.CLAUDE_SONNET
        mock_env_vars.assert_called_once_with("DEFAULT_MODEL")

    @mock.patch.object(_config.env, "as_str", return_value="deep-fake")
    def test_gets_claude_sonnet_when_default_model_not_recognised(
        self, mock_env_vars: mock.Mock
    ):
        default_model = _config.get_default_model()

        assert default_model == _models.CLAUDE_SONNET
        mock_env_vars.assert_called_once_with("DEFAULT_MODEL")
