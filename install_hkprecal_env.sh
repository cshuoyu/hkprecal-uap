#!/usr/bin/env bash
set -euo pipefail

# Minimal installer:
# 1) create venv
# 2) install requirements.txt
# 3) install pyrate editable (if default PYRATE_HOME exists)
#
# Usage:
#   bash install_hkprecal_env.sh
#   bash install_hkprecal_env.sh --venv-dir /path/to/venv
#   PY_BIN=python3.8 bash install_hkprecal_env.sh
#   bash install_hkprecal_env.sh --print-only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UAP_HOME="${SCRIPT_DIR}"
HKPRECAL_HOME="$(cd "${UAP_HOME}/.." && pwd)"

DEFAULT_VENV_DIR="${UAP_HOME}/.venv/hkprecal_uap_venv"
REQ_FILE="${UAP_HOME}/requirements.txt"

VENV_DIR="${DEFAULT_VENV_DIR}"
PY_BIN="${PY_BIN:-python3.8}"
PRINT_ONLY=0

# Default pyrate location for this repository layout.
PYRATE_HOME="${PYRATE_HOME:-${HKPRECAL_HOME}/waveformreader/aus_pyrate/darkmatteraustralia-pyrate-3c2e05d64b61}"

usage() {
  cat <<'EOF'
install_hkprecal_env.sh

Options:
  --venv-dir PATH   venv path (default: <UAP_HOME>/.venv/hkprecal_uap_venv)
  --print-only      print resolved values and exit
  -h, --help        show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ENV][ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

echo "[ENV] UAP_HOME=${UAP_HOME}"
echo "[ENV] VENV_DIR=${VENV_DIR}"
echo "[ENV] PY_BIN=${PY_BIN}"
echo "[ENV] REQUIREMENTS=${REQ_FILE}"
echo "[ENV] PYRATE_HOME=${PYRATE_HOME}"

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  echo "[ENV] print-only mode, no installation executed."
  exit 0
fi

if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  echo "[ENV][ERROR] ${PY_BIN} not found."
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[ENV][ERROR] requirements file not found: ${REQ_FILE}"
  exit 1
fi

echo "[ENV] create venv: ${VENV_DIR}"
"${PY_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[ENV] upgrade packaging tools"
python -m pip install --upgrade pip setuptools wheel

echo "[ENV] install requirements"
python -m pip install -r "${REQ_FILE}"

if [[ -f "${PYRATE_HOME}/setup.py" ]]; then
  echo "[ENV] install pyrate (editable): ${PYRATE_HOME}"
  python -m pip install -e "${PYRATE_HOME}"
else
  echo "[ENV][WARN] PYRATE_HOME missing or invalid, skip editable pyrate install."
  echo "[ENV][WARN] expected setup.py at: ${PYRATE_HOME}/setup.py"
fi

echo "[ENV] sanity imports"
python - <<'PY'
import hydra
import omegaconf
import uproot
import tensorflow
import zfit
print("[ENV][OK] import check passed")
PY

echo "[ENV][DONE] venv ready: ${VENV_DIR}"
echo "[ENV][DONE] use: source ${UAP_HOME}/env.sh"
