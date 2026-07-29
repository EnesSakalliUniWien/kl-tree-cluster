.PHONY: audit audit-quick audit-test check lint test wiki-lint

check: lint wiki-lint test audit-test

lint:
	uv run --no-sync ruff check .

test:
	uv run --no-sync python scripts/run_tests_ordered.py

wiki-lint:
	python3 scripts/wiki/lint.py

audit:
	tbs-audit --mode map

audit-quick:
	tbs-audit --mode quick

audit-test:
	uv run --project tools/repository_audit pytest -q tools/repository_audit/tests
