# hkprecal-uap

Unified analysis wrapper for AUS/KOR PMT pre-calibration workflows.

## Environment

Source once per shell:

```bash
source /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/env.sh
```

This sets path aliases and tools:

- `UAP_HOME`, `UAP_ROOT_THISROOT`, `PYRATE_HOME`, `KOR_NTP_HOME`

If your pyrate is in a different place, set it before sourcing:

```bash
export PYRATE_HOME=/path/to/darkmatteraustralia-pyrate-3c2e05d64b61
export KOR_NTP_HOME=/path/to/waveformreader/kor_ntp
source /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/env.sh
```

## AUS Runner

Example:

```bash
python3 /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/runners/aus_runner.py \
  --raw-dir /disk03/usr8/schen/workspace/HKPrecal/datastorage/aus/raw/scan_20260203_215759/EL1365-B \
  --template "$(cd "${PYRATE_HOME}/.." && pwd)/R12860_config.yaml" \
  --pyrate-cmd "${PYRATE_HOME}/pyrate_venv/bin/pyrate" \
  --pyrate-cwd "${PYRATE_HOME}"
```

## AUS Cluster Runner

Submit one theta/phi point per cluster job (PJM default), with active-job cap:

```bash
python3 /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/runners/aus_cluster_runner.py \
  --raw-dir /disk03/usr8/schen/workspace/HKPrecal/datastorage/aus/raw/scan_20260203_215759/EL1365-B \
  --out-dir /disk03/usr8/schen/workspace/HKPrecal/datastorage/aus/root/cluster_run \
  --max-active-jobs 100
```

Scheduler `stdout/stderr` default to:

`${UAP_HOME}/logs/scheduler/<scan>/<pmt>/`

## KOR Runner

Run `prod_ntp_standalone.C` over all raw KOR `.root` files in one folder:

```bash
python3 /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/runners/kor_runner.py \
  --raw-dir /disk03/usr8/schen/workspace/HKPrecal/datastorage/kor/raw/20260129 \
  --out-dir /disk03/usr8/schen/workspace/HKPrecal/datastorage/kor/root/20260129 \
  --skip-existing
```

## KOR Cluster Runner

Submit one raw KOR root file per cluster job (PJM default), with active-job cap:

```bash
python3 /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/runners/kor_cluster_runner.py \
  --raw-dir /disk03/usr8/schen/workspace/HKPrecal/datastorage/kor/raw/20260129 \
  --out-dir /disk03/usr8/schen/workspace/HKPrecal/datastorage/kor/root/cluster_run_20260129 \
  --max-active-jobs 100 \
  --skip-existing
```

## ROOT Inspector Tool

Quick inspect/plot helper for output ROOT files:

```bash
# Install dependency once (in pyrate venv)
${PYRATE_HOME}/pyrate_venv/bin/python -m pip install uproot

# List trees/branches/histograms (KOR mode)
${PYRATE_HOME}/pyrate_venv/bin/python /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/tools/root_inspector.py \
  --mode kor \
  --file /path/to/prd_*.root \
  --list

# Plot KOR Pico histogram for ch0
${PYRATE_HOME}/pyrate_venv/bin/python /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/tools/root_inspector.py \
  --mode kor \
  --file /path/to/prd_*.root \
  --preset pico \
  --channel 0 \
  --logy \
  --out pico_ch0.png

# List AUS output trees/branches
${PYRATE_HOME}/pyrate_venv/bin/python /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/tools/root_inspector.py \
  --mode aus \
  --file /path/to/output_theta0_phi0.root \
  --list

# Plot AUS Tree_CH0 PulseCharge
${PYRATE_HOME}/pyrate_venv/bin/python /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/tools/root_inspector.py \
  --mode aus \
  --file /path/to/output_theta0_phi0.root \
  --tree Tree_CH0 \
  --plot-branch PulseCharge \
  --bins 300 \
  --logy \
  --out aus_tree_ch0_pulsecharge.png
```

## Analysis I/O Interface

Unified ROOT variable reader for fitting input:

```python
from analysis.root_io import read_tree_observable, read_features

# AUS: canonical observable name -> branch mapping
r = read_tree_observable(
    "/path/to/output_theta0_phi0.root",
    channel=0,
    observable="charge",   # -> PulseCharge
    system="aus",
)
print(r.valid_count, r.values[:5])

# KOR: read multiple observables for one channel
features = read_features(
    "/path/to/prd_xxx.root",
    channel=1,
    observables=["charge", "pulse_height"],
    system="kor",
    source="tree",
)
```

## Charge Fit Test (2 Gaussian)

Pipeline test module: read charge -> fit 2-Gaussian -> output PNG + JSON.

```bash
python3 /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/analysis/charge_fit_test.py \
  --file /disk03/usr8/schen/workspace/HKPrecal/datastorage/aus/root/cluster_run_20260203_EL1365B/scan_20260203_215759/EL1365-B/outputs/output_theta0_phi0.root \
  --system aus \
  --channel 0 \
  --source tree \
  --observable charge \
  --xmin -10 \
  --xmax 15 \
  --bins 300 \
  --logy \
  --out-dir /tmp \
  --tag aus_theta0_phi0_ch0
```
