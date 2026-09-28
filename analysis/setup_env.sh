#!/usr/bin/env bash
# Create the cross-platform analysis environment at .venv-analysis.
#
# Usage:
#   bash analysis/setup_env.sh
#
# Requires uv (https://docs.astral.sh/uv/). If uv is missing:
#   pipx install uv        # or: curl -LsSf https://astral.sh/uv/install.sh | sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv-analysis"
REQS="$REPO_ROOT/analysis/requirements.txt"
PYTHON_VERSION="${ANALYSIS_PYTHON:-3.11}"

uv venv --python "$PYTHON_VERSION" "$VENV"

# Bootstrap build deps for legacy setup.py packages (numpy + Cython for ssm).
uv pip install --python "$VENV" numpy cython

# Prefer the lightweight CPU-only torch wheel so installs stay small.
uv pip install --python "$VENV" torch --index-url https://download.pytorch.org/whl/cpu

uv pip install --python "$VENV" --no-build-isolation -r "$REQS"

echo
echo "Done. Run scripts with:"
echo "  $VENV/bin/python analysis/threat_bins_analysis.py"
