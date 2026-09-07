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
  - Method-specific logic (NumPy/SciPy EMG model, iminuit likelihood fit)
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

## Data File Structures

### AUS System

#### Raw files: `wave0_theta{θ}_phi{φ}.txt`

CAEN digitizer wavedump format. One file per angle point. Each event is a fixed-length block:

```
Record Length: 260
BoardID: 31
Channel: 0
Event Number: 0
Pattern: 0x0000
Trigger Time Stamp: 59443
DC offset (DAC): 0xBFFF
4009       ← ADC sample 0
4001       ← ADC sample 1
...        ← 260 × 14-bit ADC values total
```

#### Processed ROOT files: `output_theta{θ}_phi{φ}.root`

Produced by Pyrate from the raw `.txt` files. One file per angle point. Contains three TTrees (one per digitizer channel), each with ~600k entries:

| Tree | Role |
|------|------|
| `Tree_CH0` | Trigger (laser reference) |
| `Tree_CH1` | SiPM (normalization reference) |
| `Tree_CH2` | PMT under test |

Branches (same layout on all three trees):

| Branch | dtype | Description |
|--------|-------|-------------|
| `PulseStart` | float64 | Pulse leading-edge time (samples). Multiply by 2 ns/sample to get ns. |
| `PulseCharge` | float64 | Integrated charge (ADC·sample units) |
| `PeakHeight` | float64 | Waveform peak amplitude (ADC units) |
| `PeakLocation` | float64 | Sample index of the peak |
| `CFDPulseStart` | float64 | Constant-fraction discriminator timing (samples) |
| `LEDTimes` | object (variable-length array) | LED pulse times |

The analysis uses `PulseStart` only. The timing observable is:
```
delta_PMT  = Tree_CH2.PulseStart − Tree_CH0.PulseStart   (× 2 ns/sample)
delta_SiPM = Tree_CH1.PulseStart − Tree_CH0.PulseStart   (× 2 ns/sample)
```

---

### KOR System

#### Raw files: `{serials}_{date}.root` (no `prd_` prefix)

Produced directly by the digitizer DAQ. One file per angle point, containing all three PMTs. Single TTree `T` with 200,000 events:

| Branch | dtype | Shape | Description |
|--------|-------|-------|-------------|
| `ADC` | uint32 | **(200000, 8, 1024)** | Raw waveforms: events × 8 channels × 1024 samples. Baseline ≈ 15000 ADC counts. |
| `EventNumber` | uint32 | (200000,) | Sequential event index |
| `TriggerTimeTag` | uint32 | (200000,) | Hardware trigger timestamp |
| `RecordLength` | uint32 | fixed = 1024 | Samples per waveform |
| `PostTrigger` | uint32 | fixed = 60 | Samples retained after trigger |
| `OffsetValue0–3` | uint32 | fixed = 7050 | DC offset per channel group |
| `TriggerValue` | uint32 | fixed = 4095 | Trigger threshold |
| `ActiveChannels76543210` | object | (200000,) | Bitmask of active channels per event |

#### Processed ROOT files: `prd_{serials}_{date}.root` (with `prd_` prefix)

Produced by KOR NTP macros from the raw files. One file per angle point, containing all three PMTs. Contains four TTrees and pre-built summary histograms:

TTrees (one per digitizer channel):

| Tree | Role | `diff` mean |
|------|------|-------------|
| `tree_ch0` | PMT #1 | ≈ 11 samples (≈ 22 ns) |
| `tree_ch1` | PMT #2 | ≈ 15 samples (≈ 30 ns) |
| `tree_ch2` | (unpopulated in standard runs) | ≈ 21 samples |
| `tree_ch3` | PMT #3 / trigger reference | ≈ 421 samples (≈ 842 ns) |

Branches (same layout on all four trees):

| Branch | dtype | Description |
|--------|-------|-------------|
| `diff` | float64 | Timing difference relative to trigger (samples). Multiply by 2 ns/sample. |
| `falltime` | float64 | Waveform falling-edge time (samples) |
| `max` | float64 | Waveform peak amplitude |
| `max_time` | int32 | Sample index of the peak (typically ≈ 616) |
| `LiveTime` | float64 | Acquisition live time (ms, 0–200) |
| `pico` | float64 | Picoammeter current reading |

Non-tree summary objects (per channel, `{n}` = 0, 1, 2, 3):

| Object | Type | Description |
|--------|------|-------------|
| `NoiseCount_ch{n}` | TParameter\<long\> | Dark noise event count |
| `NoiseCountRate_ch{n}` | TParameter\<double\> | Dark noise rate (Hz) |
| `Ped_ch{n}` | TH1D | Pedestal charge histogram |
| `Max_ch{n}` | TH1D | Peak amplitude histogram |
| `Time_ch{n}` | TH1D | Timing histogram |
| `Diff_ch{n}` | TH1D | `diff` histogram |
| `Pico_ch{n}` | TH1D | Picoammeter histogram |
| `PicoAbove_ch{n}` | TH1D | Above-threshold picoammeter histogram |

The analysis reads `diff` from `tree_ch{n}`:
```
timing (ns) = tree_ch{n}.diff × 2 ns/sample
```

Channel assignment is derived from the PMT serial order in the filename. The three PMTs map to channels `[0, 1, 3]` by default (channel 2 is skipped). `ch3` exhibits a noise count rate ~500× higher than `ch0`/`ch1`, suggesting it may serve as the laser trigger reference in some configurations.

---

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

To fit one KOR processed ROOT file directly:

```bash
python main.py root2csv --config-name charge_default \
  --input-dir /absolute/path/precal_prd_kor_run_20260815_011.root --serial EM6400
```

`--input-dir` accepts a file or a directory; no copy or symlink is needed.
Relative paths supplied with this flag are resolved from the directory where the
command starts, before Hydra changes directories. Directory inputs should contain
one scan, not repeated angles at different HVs. `--serial` selects the PMT; it may
be omitted if the selected YAML already specifies the correct serial.

The existing defaults save CSV, fit figures, `main.log` and `.hydra/config.yaml`
under `${UAP_HOME:-.}/outputs/<date>/<time>-root2csv-<system>/` (some profiles add
`-charge`). No output-directory argument is needed. Existing `input_dir=...`,
`serial=...` and other Hydra overrides remain supported; the explicit flags take
precedence if both forms are supplied. Single-point results do not establish an
angular dependence or an independently normalized relative QE.

The new `charge_default` and `timing_default` profiles compose four independent
sections: `fit.model`, `fit.constraints`, `fit.statistic`, `fit.optimizer`.
For example, append `fit.constraints.weights=free` to release the Poisson weights,
or `fit.model.backscatter=false` to remove backscatter without changing the
statistic. `system` only selects the reader. Each run also saves the complete
`fit_config.yaml`; see [Fit components](docs/fit_components.md) for definitions,
parameter names, limits and backward compatibility.

### 4) CSV -> Plot (Hydra mode)
```bash
python3 main.py csv2plot --config-name csv2plot_default
```
