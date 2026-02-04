#!/usr/bin/env python3
"""
KOR runner: execute prod_ntp_standalone.C over raw ROOT files.

Expected raw layout example:
  datastorage/kor/raw/20260129/*.root
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def discover_inputs(raw_dir: Path) -> List[Path]:
    files = []
    for fp in sorted(raw_dir.glob("*.root")):
        # avoid re-processing already-produced ntuples
        if fp.name.startswith("prd_"):
            continue
        files.append(fp.resolve())
    return files


def run_with_live_log(cmd: List[str], cwd: Path, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        with log_path.open("w", encoding="utf-8") as fo:
            fo.write("DRY RUN\n")
            fo.write(" ".join(cmd) + "\n")
            fo.write(f"cwd={cwd}\n")
        print(f"[DRY] {' '.join(cmd)}")
        return 0

    with log_path.open("w", encoding="utf-8") as fo:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for ch in iter(lambda: proc.stdout.read(1), ""):
                fo.write(ch)
                sys.stdout.write(ch)
            fo.flush()
            sys.stdout.flush()
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            return proc.wait()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_kor_home = Path(
        os.environ.get("KOR_NTP_HOME", str(repo_root.parent / "waveformreader" / "kor_ntp"))
    ).resolve()
    default_macro = Path(
        os.environ.get("KOR_PROD_MACRO", str(default_kor_home / "prod_ntp_standalone.C"))
    ).resolve()
    default_root_base = Path(
        os.environ.get("UAP_KOR_ROOT_DIR", str(repo_root.parent / "datastorage" / "kor" / "root"))
    ).resolve()
    default_root_cmd = os.environ.get("ROOT_CMD", "root")

    ap = argparse.ArgumentParser(description="Run KOR prod_ntp over all raw ROOT files in one folder.")
    ap.add_argument("--raw-dir", required=True, help="Folder containing raw KOR .root files")
    ap.add_argument("--out-root-base", default=str(default_root_base), help="Base output dir for KOR roots/logs")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Exact output directory for this raw-dir run (contains logs/outputs). Overrides --out-root-base.",
    )
    ap.add_argument("--kor-home", default=str(default_kor_home), help="kor_ntp directory")
    ap.add_argument("--macro", default=str(default_macro), help="prod_ntp_standalone.C path")
    ap.add_argument("--root-cmd", default=default_root_cmd, help="ROOT executable command")
    ap.add_argument("--files", nargs="*", help="Optional subset of filenames to process")
    ap.add_argument("--max-files", type=int, default=0, help="Process only first N files")
    ap.add_argument("--skip-existing", action="store_true", help="Skip files whose output already exists")
    ap.add_argument("--dry-run", action="store_true", help="Generate plan/logs only, do not execute")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_root_base = Path(args.out_root_base).resolve()
    kor_home = Path(args.kor_home).resolve()
    macro = Path(args.macro).resolve()

    if not raw_dir.is_dir():
        raise SystemExit(f"Raw directory not found: {raw_dir}")
    if not kor_home.is_dir():
        raise SystemExit(f"KOR home not found: {kor_home}")
    if not macro.is_file():
        raise SystemExit(f"Macro not found: {macro}")
    if shutil.which(args.root_cmd) is None:
        raise SystemExit(
            f"ROOT command not found in PATH: {args.root_cmd}\n"
            "Did you source ROOT thisroot.sh?"
        )

    inputs = discover_inputs(raw_dir)
    if args.files:
        wanted = {x.strip() for x in args.files if x.strip()}
        inputs = [fp for fp in inputs if fp.name in wanted]
    if args.max_files > 0:
        inputs = inputs[: args.max_files]

    if not inputs:
        print(f"[WARN] No matching input .root files in {raw_dir}")
        return

    run_name = raw_dir.name
    run_base = Path(args.out_dir).resolve() if args.out_dir else (out_root_base / run_name)
    log_dir = run_base / "logs"
    out_dir = run_base / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for inp in inputs:
        total += 1
        out_name = f"prd_{inp.name}"
        out_path = out_dir / out_name
        log_path = log_dir / f"{inp.stem}.log"

        if args.skip_existing and out_path.is_file():
            print(f"[SKIP] {inp.name}: output exists")
            skipped += 1
            continue

        macro_call = f'{macro}("{inp}")'
        cmd = [args.root_cmd, "-l", "-b", "-q", macro_call]
        rc = run_with_live_log(cmd, kor_home, log_path, args.dry_run)
        if rc != 0:
            print(f"[FAIL] {inp.name} (rc={rc}) -> {log_path}")
            failed += 1
            continue

        # Macro writes prd_<basename>.root into kor_home. Move into run outputs.
        produced = kor_home / out_name
        if args.dry_run:
            print(f"[DRY] move {produced} -> {out_path}")
            ok += 1
            continue

        if not produced.is_file():
            print(f"[FAIL] {inp.name}: expected output not found: {produced}")
            failed += 1
            continue

        shutil.move(str(produced), str(out_path))
        print(f"[OK]   {inp.name} -> {out_path.name}")
        ok += 1

    print(
        f"[DONE] total={total}, ok={ok}, skipped={skipped}, failed={failed}, base={run_base}"
    )


if __name__ == "__main__":
    main()
