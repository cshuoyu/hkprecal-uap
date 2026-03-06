#!/usr/bin/env python3
"""
Submit AUS Pyrate processing jobs to a cluster scheduler with a max active-job cap.

Default scheduler mode is PJM (pjsub/pjstat), matching SUKAP usage.
"""

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import yaml

from aus_runner import (
    build_config_for_point,
    discover_points,
    extract_scan_and_pmt,
    keep_requested_points,
    parse_channels,
)


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
    pyrate_home,
    pyrate_cmd,
    config_path,
    log_path,
):
    text = f"""#!/usr/bin/env bash
set -euo pipefail
source "{root_thisroot}"
source "{pyrate_home}/pyrate_venv/bin/activate"
"{pyrate_cmd}" -c "{config_path}" > "{log_path}" 2>&1
"""
    job_script.write_text(text, encoding="utf-8")
    job_script.chmod(0o755)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    uap_home = Path(os.environ.get("UAP_HOME", str(repo_root))).resolve()

    default_pyrate_home = Path(
        os.environ.get(
            "PYRATE_HOME",
            str(repo_root / "upstream" / "aus_pyrate" / "darkmatteraustralia-pyrate-3c2e05d64b61"),
        )
    ).resolve()
    default_template = Path(
        os.environ.get("UAP_AUS_TEMPLATE", str(default_pyrate_home.parent / "R12860_config.yaml"))
    ).resolve()
    default_root_thisroot = Path(
        os.environ.get(
            "UAP_ROOT_THISROOT",
            "/usr/local/sklib_gcc8/root_v6.22.06_python3.8/bin/thisroot.sh",
        )
    ).resolve()
    default_pyrate_cmd = Path(
        os.environ.get("PYRATE_CMD", str(default_pyrate_home / "pyrate_venv/bin/pyrate"))
    ).resolve()
    default_scheduler_log_dir = (uap_home / "logs" / "scheduler").resolve()

    ap = argparse.ArgumentParser(description="Submit AUS pyrate point-jobs to cluster with active-job cap.")
    ap.add_argument("--raw-dir", required=True, help="Folder containing wave*_theta*_phi*.txt")
    ap.add_argument("--out-dir", required=True, help="Output dir for configs/logs/outputs/jobs")
    ap.add_argument("--template", default=str(default_template), help="Template YAML")
    ap.add_argument("--pyrate-home", default=str(default_pyrate_home), help="Pyrate home")
    ap.add_argument("--pyrate-cmd", default=str(default_pyrate_cmd), help="Pyrate executable")
    ap.add_argument("--root-thisroot", default=str(default_root_thisroot), help="ROOT thisroot.sh path")
    ap.add_argument("--channels", default="0,1,2", help="Comma-separated channels")
    ap.add_argument("--points", nargs="*", help="Subset points: theta0_phi0 theta10_phi90")
    ap.add_argument("--max-points", type=int, default=0, help="Only submit first N points")
    ap.add_argument("--scheduler", choices=["pjm", "slurm"], default="pjm", help="Scheduler type")
    ap.add_argument("--submit-cmd", default="pjsub", help="Submit command")
    ap.add_argument("--status-cmd", default="pjstat", help="Queue status command")
    ap.add_argument("--max-active-jobs", type=int, default=100, help="Max active jobs for current user")
    ap.add_argument("--poll-seconds", type=int, default=10, help="Wait time between queue checks")
    ap.add_argument("--job-prefix", default="ausp", help="Job name prefix")
    ap.add_argument(
        "--scheduler-log-dir",
        default=str(default_scheduler_log_dir),
        help="Directory under UAP_HOME to store scheduler stdout/stderr files",
    )
    ap.add_argument("--dry-run", action="store_true", help="Generate configs/scripts only, do not submit")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    template_path = Path(args.template).resolve()
    pyrate_home = Path(args.pyrate_home).resolve()
    pyrate_cmd = Path(args.pyrate_cmd).resolve()
    if not pyrate_cmd.is_file():
        auto_cmd = pyrate_home / "pyrate_venv/bin/pyrate"
        if auto_cmd.is_file():
            pyrate_cmd = auto_cmd.resolve()
    root_thisroot = Path(args.root_thisroot).resolve()
    scheduler_log_base = Path(args.scheduler_log_dir).resolve()
    channels = parse_channels(args.channels)

    if not raw_dir.is_dir():
        raise SystemExit(f"Raw directory not found: {raw_dir}")
    if not template_path.is_file():
        raise SystemExit(f"Template YAML not found: {template_path}")
    if not root_thisroot.is_file():
        raise SystemExit(f"ROOT setup script not found: {root_thisroot}")
    if not pyrate_home.is_dir():
        raise SystemExit(f"Pyrate home not found: {pyrate_home}")
    if not pyrate_cmd.is_file():
        raise SystemExit(f"Pyrate executable not found: {pyrate_cmd}")

    user = os.environ.get("USER", "")
    if not user:
        raise SystemExit("USER environment variable is empty.")

    with template_path.open("r", encoding="utf-8") as fi:
        template_cfg = yaml.safe_load(fi)

    points_all = discover_points(raw_dir)
    points = keep_requested_points(points_all, args.points)
    ordered_keys = sorted(points.keys(), key=lambda x: (x[0], x[1]))
    if args.max_points > 0:
        ordered_keys = ordered_keys[: args.max_points]

    if not ordered_keys:
        raise SystemExit(f"No matching points found in {raw_dir}")

    scan_name, pmt_name = extract_scan_and_pmt(raw_dir)
    run_base = out_dir / scan_name / pmt_name
    scheduler_dir = scheduler_log_base / scan_name / pmt_name
    cfg_dir = run_base / "configs"
    log_dir = run_base / "logs"
    out_root_dir = run_base / "outputs"
    job_dir = run_base / "jobs"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_root_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped = 0

    for theta, phi in ordered_keys:
        point_files = points[(theta, phi)]
        missing = [ch for ch in channels if ch not in point_files]
        point_tag = f"theta{theta}_phi{phi}"
        if missing:
            print(f"[SKIP] {point_tag}: missing channels {missing}")
            skipped += 1
            continue

        out_root = out_root_dir / f"output_{point_tag}.root"
        cfg_path = cfg_dir / f"R12860_{point_tag}.yaml"
        log_path = log_dir / f"{point_tag}.log"
        scheduler_out = scheduler_dir / f"{point_tag}.out"
        scheduler_err = scheduler_dir / f"{point_tag}.err"
        job_script = job_dir / f"job_{point_tag}.sh"
        job_name = f"{args.job_prefix}_{theta}_{phi}"

        cfg = build_config_for_point(template_cfg, channels, point_files, out_root)
        with cfg_path.open("w", encoding="utf-8") as fo:
            yaml.safe_dump(cfg, fo, sort_keys=False)

        write_job_script(job_script, root_thisroot, pyrate_home, pyrate_cmd, cfg_path, log_path)

        if args.dry_run:
            print(f"[DRY] {point_tag} -> {job_script}")
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
            args.scheduler,
            args.submit_cmd,
            job_name,
            job_script,
            scheduler_out,
            scheduler_err,
        )
        if rc == 0:
            submitted += 1
            msg = out.strip().splitlines()[-1] if out.strip() else "submitted"
            print(f"[OK]   {point_tag} -> {msg}")
        else:
            print(f"[FAIL] {point_tag} submit failed (rc={rc})")
            if out.strip():
                print(out.strip())

    print(
        f"[DONE] submitted={submitted}, skipped={skipped}, total_points={len(ordered_keys)}, "
        f"base={run_base}, scheduler_logs={scheduler_dir}"
    )


if __name__ == "__main__":
    main()
