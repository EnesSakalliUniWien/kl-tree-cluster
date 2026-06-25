# Setup

This `.mex/` directory is already populated as a compatibility bridge. Do not
run the generic mex population prompt for this repository.

Use the real project setup sources instead:

- `README.md` for the repository overview and implementation map.
- `pyproject.toml` for Python dependencies, optional dependency groups, pytest,
  and ruff configuration.
- `AGENTS.md` for the root memory operating guide.
- `wiki/index.md` for durable project memory.
- `wiki/tools/wiki-search.md` for lookup commands.

After memory edits, run:

```bash
make wiki-lint
pytest tests/wiki/test_memory_contract.py -q
```
