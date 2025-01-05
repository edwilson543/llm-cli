# Installation

install: env_file install_deps

.PHONY:env_file
env_file:
	cp .env.example .env


# Python environment

.PHONY:install_deps
install_deps:
	pip install -r requirements.txt
	pip install -e .


.PHONY:lock_dependencies
lock_dependencies:
	pip-compile pyproject.toml -q --resolver=backtracking --output-file=requirements.txt

# CI checks

local_ci: test lint

.PHONY:test
test:
	pytest .

lint: mypy check

.PHONY:mypy
mypy:
	mypy .

.PHONY:format
format:
	ruff format .
	ruff check . --fix

.PHONY:check
check:
	ruff check .
