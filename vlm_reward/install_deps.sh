#!/usr/bin/env bash
set -euo pipefail

WITH_PRIMO=0
if [[ "${1:-}" == "--with-primo" ]]; then
  WITH_PRIMO=1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is not installed. Please install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[INFO] Installing required server dependencies via uv extra: sglang-vllm"
uv sync --extra sglang-vllm --active

if [[ "${WITH_PRIMO}" -eq 1 ]]; then
  echo "[INFO] Installing PRIMO-R1 extra dependencies: opencv-python, Pillow"
  uv pip install opencv-python Pillow
fi

echo "[INFO] Dependency installation complete."
