# hkprecal-uap

Unified analysis wrapper for AUS/KOR PMT pre-calibration workflows.

## Environment

Source once per shell:

```bash
source /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/scripts/use_uap_env.sh
```

This sets path aliases and tools:

- `UAP_HOME`, `UAP_ROOT_THISROOT`, `PYRATE_HOME`

If your pyrate is in a different place, set it before sourcing:

```bash
export PYRATE_HOME=/path/to/darkmatteraustralia-pyrate-3c2e05d64b61
source /disk03/usr8/schen/workspace/HKPrecal/hkprecal-uap/scripts/use_uap_env.sh
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
