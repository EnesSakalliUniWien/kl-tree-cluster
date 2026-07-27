.PHONY: check lint test wiki-lint

check: lint wiki-lint test

lint:
	uv run --no-sync ruff check .

test:
	uv run --no-sync python scripts/run_tests_ordered.py

wiki-lint:
	python3 scripts/wiki/lint.py
