#!/usr/bin/env bash
# Usage: source env.sh
#


: "${UAP_HOME:=/disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap}"
: "${UAP_ROOT_THISROOT:=/usr/local/sklib_gcc8/root_v6.22.06_python3.8/bin/thisroot.sh}"
: "${PYRATE_HOME:=/disk03/usr8/schen/workspace/HKPrecal/waveformreader/aus_pyrate/darkmatteraustralia-pyrate-3c2e05d64b61}"
export UAP_HOME UAP_ROOT_THISROOT PYRATE_HOME

if [[ ! -d "${UAP_HOME}" ]]; then
  echo "[UAP][ERROR] UAP_HOME not found: ${UAP_HOME}"
  return 1
fi

if [[ ! -f "${UAP_ROOT_THISROOT}" ]]; then
  echo "[UAP][ERROR] UAP_ROOT_THISROOT not found: ${UAP_ROOT_THISROOT}"
  return 1
fi

if [[ ! -d "${PYRATE_HOME}" ]]; then
  echo "[UAP][ERROR] PYRATE_HOME not found: ${PYRATE_HOME}"
  return 1
fi

if [[ ! -f "${PYRATE_HOME}/pyrate_venv/bin/activate" ]]; then
  echo "[UAP][ERROR] Missing: ${PYRATE_HOME}/pyrate_venv/bin/activate"
  return 1
fi

if [[ ! -x "${PYRATE_HOME}/pyrate_venv/bin/pyrate" ]]; then
  echo "[UAP][ERROR] Missing: ${PYRATE_HOME}/pyrate_venv/bin/pyrate"
  return 1
fi

source "${UAP_ROOT_THISROOT}"
source "${PYRATE_HOME}/pyrate_venv/bin/activate"

echo "[UAP] Environment loaded."
echo "[UAP] UAP_HOME=${UAP_HOME}"
echo "[UAP] UAP_ROOT_THISROOT=${UAP_ROOT_THISROOT}"
echo "[UAP] PYRATE_HOME=${PYRATE_HOME}"
