#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
python "${PROJECT_ROOT}/scripts/install_fiji_plugins.py"
