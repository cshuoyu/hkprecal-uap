#!/usr/bin/env python3
"""
KOR v7 runner: execute prod_ntp_v7.C over raw ROOT files.

Expected raw layout example:
  datastorage/kor/raw/20260129/*.root
"""

import argparse
import os
import re
import shutil
from pathlib import Path

from kor_runner import run_with_live_log


def discover_inputs(raw_dir):
    files = []
    for fp in sorted(raw_dir.glob("*.root")):
        # avoid re-processing legacy and v7 production files
        if fp.name.startswith("prd_") or "_prd_" in fp.name:
            continue
        files.append(fp.resolve())
    return files


def prepare_v7_source(kor_home, macro, runtime_dir, out_dir, figure_dir):
    """Copy the v7 sources locally and replace their site-specific paths."""
    source_files = (
        (macro, runtime_dir / "prod_ntp_v7.C"),
        (kor_home / "path_builder2.h", runtime_dir / "path_builder2.h"),
        (kor_home / "angle_convert.h", runtime_dir / "angle_convert.h"),
        (
            kor_home / "Base/analysisCode/Analysis.cpp",
            runtime_dir / "Base/analysisCode/Analysis.cpp",
        ),
        (
            kor_home / "Base/analysisCode/DrawFormat.cpp",
            runtime_dir / "Base/analysisCode/DrawFormat.cpp",
        ),
        (
            kor_home / "reference-config/config3.h",
            runtime_dir / "config3_local.h",
        ),
    )

    for source, _ in source_files:
        if not source.is_file():
            raise SystemExit(f"v7 source not found: {source}")

    for source, destination in source_files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    include_pattern = r'#include\s+"[^"]*config3\.h"'
    for name in ("prod_ntp_v7.C", "path_builder2.h"):
        path = runtime_dir / name
        text = path.read_text(encoding="utf-8")
        text = re.sub(include_pattern, '#include "config3_local.h"', text)
        path.write_text(text, encoding="utf-8")

    config_path = runtime_dir / "config3_local.h"
    config = config_path.read_text(encoding="utf-8")
    replacements = {
        "ProcessedDataPath": f"{out_dir}/",
        "ImagePath": f"{figure_dir}/",
    }
    for key, value in replacements.items():
        config = re.sub(
            rf"^const std::string {key}\s*=.*;$",
            f'const std::string {key} = "{value}";',
            config,
            flags=re.MULTILINE,
        )
    config_path.write_text(config, encoding="utf-8")

    return runtime_dir / "prod_ntp_v7.C"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_kor_home = Path(
        os.environ.get("KOR_V7_HOME", str(repo_root.parent / "real-run-kor"))
    ).resolve()
    default_macro = Path(
        os.environ.get("KOR_V7_PROD_MACRO", str(default_kor_home / "prod_ntp_v7.C"))
    ).resolve()
    default_root_base = Path(
        os.environ.get(
            "UAP_KOR_ROOT_DIR", str(repo_root.parent / "datastorage" / "kor" / "root")
        )
    ).resolve()
    default_root_cmd = os.environ.get("ROOT_CMD", "root")

    ap = argparse.ArgumentParser(
        description="Run KOR prod_ntp_v7 over all raw ROOT files in one folder."
    )
    ap.add_argument(
        "--raw-dir", required=True, help="Folder containing raw KOR .root files"
    )
    ap.add_argument(
        "--out-root-base",
        default=str(default_root_base),
        help="Base output dir for KOR roots/logs",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Exact output directory for this raw-dir run (contains logs/outputs). Overrides --out-root-base.",
    )
    ap.add_argument(
        "--kor-home", default=str(default_kor_home), help="KOR v7 source directory"
    )
    ap.add_argument("--macro", default=str(default_macro), help="prod_ntp_v7.C path")
    ap.add_argument(
        "--root-cmd", default=default_root_cmd, help="ROOT executable command"
    )
    ap.add_argument(
        "--files", nargs="*", help="Optional subset of filenames to process"
    )
    ap.add_argument(
        "--max-files", type=int, default=0, help="Process only first N files"
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output already exists",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Generate plan/logs only, do not execute"
    )
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
    run_base = (
        Path(args.out_dir).resolve() if args.out_dir else (out_root_base / run_name)
    )
    log_dir = run_base / "logs"
    out_dir = run_base / "outputs"
    figure_dir = run_base / "figures"
    runtime_dir = run_base / "v7_runtime"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    runtime_macro = prepare_v7_source(kor_home, macro, runtime_dir, out_dir, figure_dir)

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for inp in inputs:
        total += 1
        out_name = inp.name.replace("raw", "prd")
        if "prd" not in out_name:
            out_name = f"prd_{out_name}"
        out_path = out_dir / out_name
        log_path = log_dir / f"{inp.stem}.log"

        if args.skip_existing and out_path.is_file():
            print(f"[SKIP] {inp.name}: output exists")
            skipped += 1
            continue

        macro_call = f'{runtime_macro}(0,"{inp}")'
        cmd = [args.root_cmd, "-l", "-b", "-q", macro_call]
        rc = run_with_live_log(cmd, runtime_dir, log_path, args.dry_run)
        if rc != 0:
            print(f"[FAIL] {inp.name} (rc={rc}) -> {log_path}")
            failed += 1
            continue

        if args.dry_run:
            print(f"[DRY] write {out_path}")
            ok += 1
            continue

        if not out_path.is_file():
            print(f"[FAIL] {inp.name}: expected output not found: {out_path}")
            failed += 1
            continue

        print(f"[OK]   {inp.name} -> {out_path.name}")
        ok += 1

    print(
        f"[DONE] total={total}, ok={ok}, skipped={skipped}, failed={failed}, base={run_base}"
    )


if __name__ == "__main__":
    main()
