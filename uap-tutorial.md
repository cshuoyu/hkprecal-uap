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

