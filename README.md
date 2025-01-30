# Vendor agnostic LLM CLI

## Features:
- CLI commands to:
  - Ask a single question to one or more LLM and stream the response
  - Have a multi-turn conversation
- Have the model assume a custom persona, for example Gandalf

https://github.com/user-attachments/assets/b56e9b90-0f71-4f92-85dc-caa7b971020a

https://github.com/user-attachments/assets/5ccd327b-bb7f-4b6d-9fa6-0a5936c00627

## Installation:
- Clone the repository and install the application locally
```bash
git clone git@github.com:edwilson543/llm-cli.git
cd llm-cli
python3.12 -m venv venv
source venv/bin/activate
make install
```

- Set the API key(s) for the vendors you want to use in the template `.env` file
```bash
nano .env
```

- Verify it works
```bash
question 'Have I setup the application correctly?'
```

- To install the CLI system-wide
```bash
brew install pipx # If necessary.
pipx install -e .
```

## Supported models:
- Anthropic: Claude {Haiku, Sonnet, Opus}
- DeepSeek: DeepSeek-V3, DeepSeek-R1
- Meta: Llama 3
- Mistral: Codestral, Mistral, Ministral
- OpenAI: GPT 4, GPT 4 mini
- xAI: Grok 2
- More soon

## Usage:

### Question:
```bash
>>> question --help

usage: q [-h] [-m [{claude-haiku,claude-sonnet,claude-opus,codestral,mistral,ministral,grok-2,echo,broken} ...]] [-p PERSONA] question

positional arguments:
  question              The question the model should answer.

options:
  -h, --help            show this help message and exit
  -m [ ...], --model [ ...]
                        The model that should be used. Multiple models can be specified, separated by a space. The supported models are: 
                            - claude-haiku, claude-sonnet, claude-opus
                            - deepseek-chat, deepseek-reasoner
                            - llama-3
                            - codestral, mistral, ministral
                            - gpt-4, gpt-4-mini
                            - grok-2
  -p PERSONA, --persona PERSONA
                        The persona the model should assume.
```

### Conversation
```bash
>>> conversation --help

usage: conversation [-h] [-m] [-p PERSONA]

options:
  -h, --help            show this help message and exit
  -m , --model          The model that should be used. The supported models are: 
                            - claude-haiku, claude-sonnet, claude-opus
                            - deepseek-chat, deepseek-reasoner
                            - llama-3
                            - codestral, mistral, ministral
                            - gpt-4, gpt-4-mini
                            - grok-2
  -p PERSONA, --persona PERSONA
                        The persona the model should assume.
```
