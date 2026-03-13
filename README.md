# HKPrecal UAP

HKPrecal UAP is the unified analysis pipeline for PMT pre-calibration.

The full pre-calibration workflow is:
1. `raw -> root` (waveform processing, done by runners calling pyrate/KOR macro)
2. `root -> csv` (relative quantity calculation)
3. `csv -> plot` (plotting)

This README focuses on step 2 and step 3, and on how the `uap/` package is organized.

## Project Structure

```text
hkprecal-uap/
  main.py
  config/
    aus_root2csv_emg_default.yaml
    kor_root2csv_emg_default.yaml
    csv2plot_default.yaml
  runners/
    aus_cluster_runner.py
    kor_cluster_runner.py
  uap/
    engine/
      root2csv.py
      csv2plot.py
    fit/
      fitter_interface.py
      emg_timing_offset_fit.py
      common_math.py
      plot_utils.py
    scan_reader/
      aus_reader.py
      kor_reader.py
    tool/
      root_io.py
      scan_prepare.py
      window.py
      draw.py
```

## Architecture (uap/)

### 1) `uap.engine`: Pipeline entry
- `root2csv.py`:
  - Builds fitter
  - Resolves output layout (`csv/` + `figures/`)
  - Runs fit pipeline and writes logs/CSV
- `csv2plot.py`:
  - Loads line configs from Hydra YAML
  - Builds each line from CSV + angle selection rule
  - Produces one figure + optional selected points CSV

### 2) `uap.fit`: Fit pipeline + method implementation
- `fitter_interface.py`:
  - Defines common scan workflow (`BaseScanFitter`)
  - Uniform point format, fit execution loop, status logging, CSV writing
  - Method-independent orchestration
- `emg_timing_offset_fit.py`:
  - Method-specific logic (zfit EMG fit)
  - AUS/KOR input preparation
  - Relative quantity postprocessing

### 3) `uap.scan_reader`: Data-source parsing/reading rules
- `aus_reader.py`:
  - Parse `output_theta*_phi*.root`
  - AUS branch read helpers and channel resolution
- `kor_reader.py`:
  - Parse KOR serial/angle info from `prd_*.root` filename
  - KOR branch read helpers and auto channel mapping

### 4) `uap.tool`: Shared utilities
- `root_io.py`: common ROOT branch readers
- `window.py`: window selection
- `scan_prepare.py`: shared point/row/stat builders
- `draw.py`: CSV point selection + line construction + plotting + Hamamatsu angle transform

## Core Data Flow

### A) `root2csv` flow
1. Engine creates fitter.
2. Fitter `prepare_scan(system, args)` builds a list of fit points.
3. Shared analyzer runs main fit (and optional aux fit such as AUS SiPM) for each point.
4. Postprocess computes relative columns.
5. Output CSV is written, and per-point fit diagnostic PNGs go to `figures/`.

### B) `csv2plot` flow
1. Load `lines` from Hydra config.
2. For each line:
   - load CSV
   - select/transform angle points (e.g. `phi_pair`, `single_phi`, `angle_pairs`)
   - optional conversion to Hamamatsu angle
3. Overlay all lines into one figure.

## Quick Start

### 1) Install environment
```bash
bash install_hkprecal_env.sh
source env.sh
```
### 2)Raw -> ROOT (runners/)
The `runners/` scripts submit jobs to the cluster. Keep this layer separate from `uap/` analysis.

Typical usage:
```bash
python3 runners/aus_cluster_runner.py --raw-dir /path/to/aus/raw --out-dir /path/to/aus/root --max-active-jobs 100
python3 runners/kor_cluster_runner.py --raw-dir /path/to/kor/raw --out-dir /path/to/kor/root --max-active-jobs 100
```

### 3) ROOT -> CSV (Hydra mode)
```bash
python3 main.py root2csv --config-name aus_root2csv_emg_default
python3 main.py root2csv --config-name kor_root2csv_emg_default
```

### 4) CSV -> Plot (Hydra mode)
```bash
python3 main.py csv2plot --config-name csv2plot_default
```
