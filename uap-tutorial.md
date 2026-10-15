# HKPrecal UAP Tutorial 26/10/15

HKPrecal UAP is a unified analysis pipeline for the two systems. Here is a brief tutorial. If you have any questions please feel free to ask Shuoyu Chen (schen@icrr.u-tokyo.ac.jp)

## 1. Download UAP

### 1.1 Clone the repository

Run these commands in the directory where you want to keep the code:

```bash
git clone https://github.com/cshuoyu/hkprecal-uap.git
cd hkprecal-uap
```

### 1.2 Configure the environment

UAP requires Python 3.8 and ROOT 6.22. Pyrate and the KOR NTP macro are
used only for raw-to-ROOT conversion. Ask the relevant system experts for
access to those repositories. Set their paths before running the installer:

```bash
export UAP_ROOT_THISROOT=/path/to/root/bin/thisroot.sh
export PYRATE_HOME=/path/to/pyrate
export KOR_NTP_HOME=/path/to/kor_ntp
```

Create the virtual environment and load it:

```bash
bash install_hkprecal_env.sh
source env.sh
```

Before each run, execute `source env.sh`. We recommend adding the following lines to `~/.bashrc` so that the
UAP environment is loaded automatically in each new Bash session:

```bash
export UAP_ROOT_THISROOT=/path/to/root/bin/thisroot.sh
export PYRATE_HOME=/path/to/pyrate
export KOR_NTP_HOME=/path/to/kor_ntp
source /path/to/hkprecal-uap/env.sh
```

### 1.3 Check the environment

```bash
command -v python
python --version
root-config --version
command -v pyrate
python -c 'import ROOT, hydra, omegaconf, uproot, numpy, scipy, iminuit, uap; print("[OK] Python imports")'
```

## 2. RAW to ROOT

### 2.1 KOR NTP

Run the following commands from the `hkprecal-uap` directory after loading the
environment with `source env.sh`. The runner calls
`$KOR_NTP_HOME/prod_ntp_standalone.C` for each selected raw file.

#### Process one file

This example processes `precal_raw_kor_run_20261013_000.root`:

```bash
python3 runners/kor_runner.py \
  --raw-dir /media/precalib/rawdata/kor/commissioning \
  --out-dir outputs/kor-ntp-20261013-000 \
  --files precal_raw_kor_run_20261013_000.root
```

The runner reads the raw file directly from `--raw-dir`; it does not copy or
modify the raw file. `--out-dir` is required and specifies where the generated
files are kept. The processed ROOT file and log are written to:

```text
outputs/kor-ntp-20261013-000/outputs/prd_precal_raw_kor_run_20261013_000.root
outputs/kor-ntp-20261013-000/logs/precal_raw_kor_run_20261013_000.log
```

#### Process files 000–045

The following command processes all 46 files sequentially:

```bash
python3 runners/kor_runner.py \
  --raw-dir /media/precalib/rawdata/kor/commissioning \
  --out-dir outputs/kor-ntp-20261013-000-045 \
  --files precal_raw_kor_run_20261013_{000..045}.root
```

The processed ROOT files are written to
`outputs/kor-ntp-20261013-000-045/outputs/`, and the per-file logs are written
to `outputs/kor-ntp-20261013-000-045/logs/`.

After the batch finishes, confirm that 46 processed ROOT files were produced:

```bash
find outputs/kor-ntp-20261013-000-045/outputs \
  -maxdepth 1 -type f -name 'prd_precal_raw_kor_run_20261013_*.root' | wc -l
```

## 3. ROOT to CSV: charge and timing fits

Run these commands from the repository root after `source env.sh`.
Use the processed ROOT file from Section 2, rather than the raw ROOT file.
These examples use the PMT identities saved in this dataset's `RunInfo`:

| Role | PMT serial | Channel | Angles in file 000 |
|---|---|---|---|
| Stationary monitor | EM2740 | 0 | Fixed reference |
| Scanned PMT | EM5730 | 1 | phi = 0, theta = -55 deg |
| Scanned PMT | EL5150 | 2 | phi = 45, theta = -55 deg |

The physical trigger is channel 3. For another dataset, confirm its metadata
and channel mapping before reusing these settings. The converter must preserve
`RunInfo`; the charge prefit also requires `Config_Threshold_mV_ch1` and the
`pico` and `max` branches in `tree_ch1`.

### 3.1 Fit one processed file

Set the input and create a unique analysis output path for this lesson:

```bash
PROCESSED_ROOT="$PWD/outputs/kor-ntp-20261013-000/outputs/prd_precal_raw_kor_run_20261013_000.root"
ANALYSIS_OUT="$PWD/outputs/tutorial-$(date +%Y%m%d-%H%M%S)"
```

Keep using the same terminal so these variables remain available.

#### Charge fit

Fit the charge spectrum of EM5730:

```bash
python3 main.py root2csv --config-name charge_default \
  --input-dir "$PROCESSED_ROOT" \
  --serial EM5730 \
  hydra.run.dir="$ANALYSIS_OUT/charge-EM5730"
```

`charge_default` fits the pedestal and 1-PE response with Poisson-constrained
weights, a chi-square statistic, and no backscatter component. It reads charge
in pC from `pico`. The pulse-height threshold is used to initialize the SPE
prefit, rather than to cut the final charge spectrum.

The output includes:

```text
<ANALYSIS_OUT>/charge-EM5730/
├── csv/kor_charge_results.csv
├── figures/                     # charge spectrum and fitted components
├── main.log
├── fit_config.yaml
└── .hydra/config.yaml
```

The CSV contains SPE parameters, gain, uncertainties, and `fit_status`.
For the standard 0–3 PE model with backscatter and a fitted monitor, use
`--config-name charge_standard` instead. Its backscatter coefficient is fitted
within [0, 0.15]; it is not fixed at 0.15.

#### Timing fit

Fit the stored PMT-to-trigger timing difference of the same PMT:

```bash
python3 main.py root2csv --config-name kor_root2csv_timing_binned \
  --input-dir "$PROCESSED_ROOT" \
  --serial EM5730 \
  hydra.run.dir="$ANALYSIS_OUT/timing-EM5730"
```

This profile fits an EMG signal plus background using a binned likelihood,
integrating the model over the 2 ns timing bins. The input is the converter's
stored `diff` branch multiplied by 2 ns/sample. Check the converter version and
trigger definition when comparing with older processed files.

The CSV is `timing-EM5730/csv/kor_results.csv`; timing diagnostic plots are in
`timing-EM5730/figures/`. The run also saves `main.log`, `fit_config.yaml`, and
`.hydra/config.yaml`. Columns include `mu`, `mu_err`, `sigma`, `lambda`, `FWHM`,
and `fit_status`. `mu` is the fitted EMG location parameter; `FWHM` describes the
fitted timing width.

To analyse EL5150, replace `--serial EM5730` with `--serial EL5150` and use a
different output directory, such as `charge-EL5150` or `timing-EL5150`.

### 3.2 Inspect the single-file fits

The fit commands already generate the charge and timing plots. List them with:

```bash
find "$ANALYSIS_OUT/charge-EM5730/figures" -type f
find "$ANALYSIS_OUT/timing-EM5730/figures" -type f
```

Open the images in VS Code. Check the spectrum, fitted curve and components,
then inspect `fit_status` in each CSV and the prepare/fit counts in `main.log`.
A successful process exit alone does not establish a good fit.

File 000 is a single off-centre point. It is suitable for inspecting spectra
and fit parameters, but does not provide a centre-normalized angular response.
Ignore relative columns inferred from a missing-centre fallback. Use the full
scan below for angular plots.

## 4. Fit and plot the full scan

### 4.1 Fit files 000–045

After the batch conversion in Section 2, fit its processed directory:

```bash
PROCESSED_DIR="$PWD/outputs/kor-ntp-20261013-000-045/outputs"

python3 main.py root2csv --config-name charge_default \
  --input-dir "$PROCESSED_DIR" \
  --serial EM5730 \
  hydra.run.dir="$ANALYSIS_OUT/scan-charge-EM5730"

python3 main.py root2csv --config-name kor_root2csv_timing_binned \
  --input-dir "$PROCESSED_DIR" \
  --serial EM5730 \
  hydra.run.dir="$ANALYSIS_OUT/scan-timing-EM5730"
```

Each CSV should contain 46 rows for EM5730: 23 theta positions from -55 to
+55 deg at phi = 0, and another 23 at phi = 90. The theta = 0 reference runs
are 011 and 034. Check both centre fits and all `fit_status` values before
plotting. Keep only this scan in the input directory; do not mix repeated
measurements or different HV settings into one angular curve.

### 4.2 Plot relative gain

Set the CSV paths produced above:

```bash
CHARGE_CSV="$ANALYSIS_OUT/scan-charge-EM5730/csv/kor_charge_results.csv"
TIMING_CSV="$ANALYSIS_OUT/scan-timing-EM5730/csv/kor_results.csv"
```

Plot the two measured scan directions, normalizing gain to theta = 0 within
each phi group:

```bash
python3 main.py csv2plot \
  --out-dir "$ANALYSIS_OUT/plot-gain/figures" \
  --out-name EM5730_relative_gain.png \
  --points-csv EM5730_relative_gain_points.csv \
  --x-label 'Raw tilt angle (deg)' \
  --y-label 'Relative gain' \
  --title 'EM5730: charge fit' \
  --hline 1 \
  --series-json "[
    {'system':'kor', 'csv':'$CHARGE_CSV', 'mode':'single_phi',
     'phi':0, 'y_col':'relative_gain', 'yerr_col':'relative_gain_err',
     'label':'phi = 0 deg', 'line_color':'C0'},
    {'system':'kor', 'csv':'$CHARGE_CSV', 'mode':'single_phi',
     'phi':90, 'y_col':'relative_gain', 'yerr_col':'relative_gain_err',
     'label':'phi = 90 deg', 'line_color':'C1'}
  ]"
```

The PNG and selected-points CSV are saved in
`<ANALYSIS_OUT>/plot-gain/figures/`. Angles here are raw KOR tilt angles;
no Hamamatsu-angle transformation is applied.

### 4.3 Plot timing versus angle

```bash
python3 main.py csv2plot \
  --out-dir "$ANALYSIS_OUT/plot-timing/figures" \
  --out-name EM5730_timing_mu.png \
  --points-csv EM5730_timing_mu_points.csv \
  --x-label 'Raw tilt angle (deg)' \
  --y-label 'EMG location mu (ns)' \
  --title 'EM5730: timing fit' \
  --series-json "[
    {'system':'kor', 'csv':'$TIMING_CSV', 'mode':'single_phi',
     'phi':0, 'y_col':'mu', 'yerr_col':'mu_err',
     'label':'phi = 0 deg', 'line_color':'C0'},
    {'system':'kor', 'csv':'$TIMING_CSV', 'mode':'single_phi',
     'phi':90, 'y_col':'mu', 'yerr_col':'mu_err',
     'label':'phi = 90 deg', 'line_color':'C1'}
  ]"
```

The PNG and selected-points CSV are saved in
`<ANALYSIS_OUT>/plot-timing/figures/`. This plots the fitted timing location,
including the converter's timing reference and cable/light-path contributions;
it is not an independently calibrated absolute PMT transit time.

To plot the fitted timing width, replace `mu` with `FWHM`, remove the
`yerr_col` entries (the CSV has no FWHM uncertainty column), and change the
y-axis label and output filenames accordingly.

For EL5150, repeat the scan fits with `--serial EL5150`, use its output CSVs,
and change the two plotting phi selections to 45 and 135. Charge-derived
relative gain and timing-fit signal yield are different observables; neither
should be labelled absolute QE.

