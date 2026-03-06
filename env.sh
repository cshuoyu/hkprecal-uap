#!/usr/bin/env bash
# Usage: source env.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${UAP_HOME:=${SCRIPT_DIR}}"
: "${UAP_ROOT_THISROOT:=/usr/local/sklib_gcc8/root_v6.22.06_python3.8/bin/thisroot.sh}"
: "${PYRATE_HOME:=/disk03/usr8/schen/workspace/HKPrecal/waveformreader/aus_pyrate/darkmatteraustralia-pyrate-3c2e05d64b61}"
: "${KOR_NTP_HOME:=/disk03/usr8/schen/workspace/HKPrecal/waveformreader/kor_ntp}"
: "${UAP_VENV_HOME:=${UAP_HOME}/.venv/hkprecal_uap_venv}"

export UAP_HOME UAP_ROOT_THISROOT PYRATE_HOME UAP_VENV_HOME KOR_NTP_HOME

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

if [[ ! -f "${UAP_VENV_HOME}/bin/activate" ]]; then
  echo "[UAP][ERROR] Missing unified venv activate: ${UAP_VENV_HOME}/bin/activate"
  return 1
fi

if [[ ! -x "${UAP_VENV_HOME}/bin/pyrate" ]]; then
  echo "[UAP][ERROR] Missing pyrate command in unified venv: ${UAP_VENV_HOME}/bin/pyrate"
  return 1
fi

if [[ ! -d "${KOR_NTP_HOME}" ]]; then
  echo "[UAP][ERROR] KOR_NTP_HOME not found: ${KOR_NTP_HOME}"
  return 1
fi

if [[ ! -f "${KOR_NTP_HOME}/prod_ntp_standalone.C" ]]; then
  echo "[UAP][ERROR] Missing: ${KOR_NTP_HOME}/prod_ntp_standalone.C"
  return 1
fi

source "${UAP_ROOT_THISROOT}"
source "${UAP_VENV_HOME}/bin/activate"

echo "[UAP] Environment loaded."
echo "[UAP] UAP_HOME=${UAP_HOME}"
echo "[UAP] UAP_ROOT_THISROOT=${UAP_ROOT_THISROOT}"
echo "[UAP] PYRATE_HOME=${PYRATE_HOME}"
echo "[UAP] UAP_VENV_HOME=${UAP_VENV_HOME}"
echo "[UAP] KOR_NTP_HOME=${KOR_NTP_HOME}"
