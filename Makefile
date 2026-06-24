venv:
	python3 -m venv .venv
	@echo "virtual env created"

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calls.json

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete

lint:
	uv run flake8 . --exclude .venv,llm_sdk
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs --exclude llm_sdk

.PHONY: install run debug clean lint

