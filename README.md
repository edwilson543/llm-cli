# Vendor agnostic LLM CLI

## Features:
- Ask a range of LLMs a question via the command line and stream the response
- Have the LLM assume a custom persona, for example Gandalf

https://github.com/user-attachments/assets/b56e9b90-0f71-4f92-85dc-caa7b971020a

- Have a conversation with a range of LLMs, again, optionally with a custom persona

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
- Claude {Haiku, Sonnet, Opus}
- Grok 2
- More soon

## Usage:
#### Question:
```bash
>>> question --help

usage: question [-h] [-p PERSONA] [-m {claude-haiku,claude-sonnet,claude-opus,grok-2,echo,broken}] [-s] question

positional arguments:
  question              The question the model should answer.

options:
  -h, --help            show this help message and exit
  -p PERSONA, --persona PERSONA
                        The persona the model should assume.
  -m {claude-haiku,claude-sonnet,claude-opus,grok-2,echo,broken}, --model {claude-haiku,claude-sonnet,claude-opus,grok-2,echo,broken}
                        The model that should be used.
  -s, --stream          Whether to stream the response from the model asynchronously.
```
