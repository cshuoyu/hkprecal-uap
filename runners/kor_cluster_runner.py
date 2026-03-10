#!/usr/bin/env python3
"""
Submit KOR prod_ntp_standalone.C jobs to cluster with active-job cap.

Default scheduler mode is PJM (pjsub/pjstat), matching SUKAP usage.
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from kor_runner import discover_inputs


def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    return proc.returncode, proc.stdout


def count_active_jobs(scheduler, status_cmd, user):
    if scheduler == "slurm":
        rc, out = run_cmd([status_cmd, "-u", user, "-h"])
        if rc != 0:
            return 0
        return sum(1 for line in out.splitlines() if line.strip())

    # pjm
    rc, out = run_cmd([status_cmd])
    if rc != 0:
        return 0
    n = 0
    for line in out.splitlines():
        txt = line.strip()
        if not txt or txt.startswith("JOB_ID"):
            continue
        if user in txt:
            n += 1
    return n


def submit_job(
    scheduler,
    submit_cmd,
    job_name,
    job_script,
    scheduler_out,
    scheduler_err,
):
    if scheduler == "slurm":
        cmd = [
            submit_cmd,
            "-J",
            job_name,
            "-o",
            str(scheduler_out),
            "-e",
            str(scheduler_err),
            str(job_script),
        ]
    else:
        cmd = [
            submit_cmd,
            "-N",
            job_name,
            "-o",
            str(scheduler_out),
            "-e",
            str(scheduler_err),
            str(job_script),
        ]
    return run_cmd(cmd)


def write_job_script(
    job_script,
    root_thisroot,
    kor_home,
    root_cmd,
    macro,
    input_root,
    point_log,
    out_root,
):
    out_root.parent.mkdir(parents=True, exist_ok=True)
    text = f"""#!/usr/bin/env bash
set -euo pipefail
source "{root_thisroot}"
cd "{kor_home}"
mkdir -p "{out_root.parent}"
macro_call='{macro}("{input_root}","{out_root}")'
"{root_cmd}" -l -b -q "${{macro_call}}" > "{point_log}" 2>&1
if [[ ! -f "{out_root}" ]]; then
  echo "[KOR][ERROR] expected output not found: {out_root}" >> "{point_log}"
  exit 2
fi
"""
    job_script.write_text(text, encoding="utf-8")
    job_script.chmod(0o755)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    uap_home = Path(os.environ.get("UAP_HOME", str(repo_root))).resolve()

    default_kor_home = Path(
        os.environ.get("KOR_NTP_HOME", str(repo_root.parent / "waveformreader" / "kor_ntp"))
    ).resolve()
    default_macro = Path(
        os.environ.get("KOR_PROD_MACRO", str(default_kor_home / "prod_ntp_standalone.C"))
    ).resolve()
    default_root_thisroot = Path(
        os.environ.get(
            "UAP_ROOT_THISROOT",
            "/usr/local/sklib_gcc8/root_v6.22.06_python3.8/bin/thisroot.sh",
        )
    ).resolve()
    default_root_base = Path(
        os.environ.get("UAP_KOR_ROOT_DIR", str(repo_root.parent / "datastorage" / "kor" / "root"))
    ).resolve()
    default_scheduler_log_dir = (uap_home / "logs" / "scheduler" / "kor").resolve()
    default_root_cmd = os.environ.get("ROOT_CMD", "root")

    ap = argparse.ArgumentParser(description="Submit KOR prod_ntp jobs to cluster with active-job cap.")
    ap.add_argument("--raw-dir", required=True, help="Folder containing raw KOR .root files")
    ap.add_argument("--out-dir", required=True, help="Output dir for logs/outputs/jobs")
    ap.add_argument("--kor-home", default=str(default_kor_home), help="kor_ntp directory")
    ap.add_argument("--macro", default=str(default_macro), help="prod_ntp_standalone.C path")
    ap.add_argument("--root-cmd", default=default_root_cmd, help="ROOT executable command")
    ap.add_argument("--root-thisroot", default=str(default_root_thisroot), help="ROOT thisroot.sh path")
    ap.add_argument("--files", nargs="*", help="Optional subset of filenames")
    ap.add_argument("--max-files", type=int, default=0, help="Only submit first N files")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output prd_*.root already exists")
    ap.add_argument("--scheduler", choices=["pjm", "slurm"], default="pjm", help="Scheduler type")
    ap.add_argument("--submit-cmd", default="pjsub", help="Submit command")
    ap.add_argument("--status-cmd", default="pjstat", help="Queue status command")
    ap.add_argument("--max-active-jobs", type=int, default=100, help="Max active jobs for current user")
    ap.add_argument("--poll-seconds", type=int, default=10, help="Wait between queue checks")
    ap.add_argument("--job-prefix", default="korp", help="Job name prefix")
    ap.add_argument(
        "--scheduler-log-dir",
        default=str(default_scheduler_log_dir),
        help="Directory under UAP_HOME to store scheduler stdout/stderr files",
    )
    ap.add_argument("--dry-run", action="store_true", help="Generate scripts only, do not submit")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    kor_home = Path(args.kor_home).resolve()
    macro = Path(args.macro).resolve()
    root_thisroot = Path(args.root_thisroot).resolve()
    scheduler_log_base = Path(args.scheduler_log_dir).resolve()

    if not raw_dir.is_dir():
        raise SystemExit(f"Raw directory not found: {raw_dir}")
    if not kor_home.is_dir():
        raise SystemExit(f"KOR home not found: {kor_home}")
    if not macro.is_file():
        raise SystemExit(f"Macro not found: {macro}")
    if not root_thisroot.is_file():
        raise SystemExit(f"ROOT setup script not found: {root_thisroot}")
    if shutil.which(args.root_cmd) is None:
        raise SystemExit(f"ROOT command not found in PATH: {args.root_cmd}")

    user = os.environ.get("USER", "")
    if not user:
        raise SystemExit("USER environment variable is empty.")

    inputs = discover_inputs(raw_dir)
    if args.files:
        wanted = {x.strip() for x in args.files if x.strip()}
        inputs = [fp for fp in inputs if fp.name in wanted]
    if args.max_files > 0:
        inputs = inputs[: args.max_files]

    if not inputs:
        raise SystemExit(f"No matching input .root files in {raw_dir}")

    run_name = raw_dir.name
    run_base = out_dir / run_name
    scheduler_dir = scheduler_log_base / run_name
    log_dir = run_base / "logs"
    out_root_dir = run_base / "outputs"
    job_dir = run_base / "jobs"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_root_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped = 0

    for idx, inp in enumerate(inputs):
        out_name = f"prd_{inp.name}"
        out_path = out_root_dir / out_name
        point_tag = inp.stem
        point_log = log_dir / f"{point_tag}.log"
        scheduler_out = scheduler_dir / f"{point_tag}.out"
        scheduler_err = scheduler_dir / f"{point_tag}.err"
        job_script = job_dir / f"job_{point_tag}.sh"
        job_name = f"{args.job_prefix}_{idx:04d}"

        if args.skip_existing and out_path.is_file():
            print(f"[SKIP] {inp.name}: output exists")
            skipped += 1
            continue

        write_job_script(
            job_script=job_script,
            root_thisroot=root_thisroot,
            kor_home=kor_home,
            root_cmd=args.root_cmd,
            macro=macro,
            input_root=inp,
            point_log=point_log,
            out_root=out_path,
        )

        if args.dry_run:
            print(f"[DRY] {inp.name} -> {job_script}")
            submitted += 1
            continue

        while True:
            active = count_active_jobs(args.scheduler, args.status_cmd, user)
            if active < args.max_active_jobs:
                break
            print(
                f"[WAIT] active_jobs={active} >= limit={args.max_active_jobs}; "
                f"sleep {args.poll_seconds}s"
            )
            time.sleep(args.poll_seconds)

        rc, out = submit_job(
            scheduler=args.scheduler,
            submit_cmd=args.submit_cmd,
            job_name=job_name,
            job_script=job_script,
            scheduler_out=scheduler_out,
            scheduler_err=scheduler_err,
        )
        if rc == 0:
            submitted += 1
            msg = out.strip().splitlines()[-1] if out.strip() else "submitted"
            print(f"[OK]   {inp.name} -> {msg}")
        else:
            print(f"[FAIL] {inp.name} submit failed (rc={rc})")
            if out.strip():
                print(out.strip())

    print(
        f"[DONE] submitted={submitted}, skipped={skipped}, total_files={len(inputs)}, "
        f"base={run_base}, scheduler_logs={scheduler_dir}"
    )


if __name__ == "__main__":
    main()
