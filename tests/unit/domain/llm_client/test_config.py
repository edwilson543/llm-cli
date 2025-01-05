from llm_cli.domain import llm_client


class TestGetLLMClient:
    def test_gets_claude_3_5_sonnet(self):
        client = llm_client.get_llm_client(model=llm_client.Model.CLAUDE_3_5_SONNET)

        assert client._model == "claude-3-5-sonnet-20241022"
