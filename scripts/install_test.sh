#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_DIR=${VENV_DIR:-"$ROOT_DIR/.venv-install-test"}

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    print(f"Python >=3.11 is required, found {sys.version.split()[0]}")
    raise SystemExit(1)
PY

echo "Creating venv at: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

# Ensure build backend is available for PEP 517 builds
python -m pip install "flit_core>=3.2,<4"

# Install the project and its runtime dependencies
python -m pip install "$ROOT_DIR"

# Run the demo pipeline as an installability smoke test
OUTPUT=$(python "$ROOT_DIR/quick_start.py")

printf '%s\n' "$OUTPUT"

if ! printf '%s\n' "$OUTPUT" | grep -q "Adjusted Rand Index"; then
  echo "Install test failed: expected ARI output not found." >&2
  exit 1
fi

echo "Install test passed."
