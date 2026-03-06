#!/usr/bin/env python3
"""
AUS runner: execute pyrate over wave*.txt points under existing datastorage layout.

Expected raw layout example:
  datastorage/aus/raw/scan_YYYYMMDD_HHMMSS/ELxxxx-B/wave0_theta0_phi0.txt
or:
  datastorage/aus/raw/scan_YYYYMMDD_HHMMSS/ELxxxx-B/wave0save_theta0_phi0.txt
"""

import argparse
import copy
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


# Support both naming styles:
#   wave0_theta10_phi90.txt
#   wave0save_theta10_phi90.txt
POINT_RE = re.compile(r"^wave(?P<ch>\d+)(?:save)?_theta(?P<theta>-?\d+)_phi(?P<phi>-?\d+)\.txt$")


def parse_channels(text):
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("No channels parsed from --channels.")
    return out


def discover_points(raw_dir):
    points = {}
    for fp in sorted(raw_dir.glob("wave*_theta*_phi*.txt")):
        m = POINT_RE.match(fp.name)
        if not m:
            continue
        ch = int(m.group("ch"))
        theta = int(m.group("theta"))
        phi = int(m.group("phi"))
        key = (theta, phi)
        if key not in points:
            points[key] = {}
        points[key][ch] = fp.resolve()
    return points


def keep_requested_points(
    points,
    requested,
):
    if not requested:
        return points
    keep = {}
    for item in requested:
        m = re.match(r"^theta(-?\d+)_phi(-?\d+)$", item.strip())
        if not m:
            raise ValueError(f"Bad --points item: {item} (expected thetaX_phiY)")
        key = (int(m.group(1)), int(m.group(2)))
        if key in points:
            keep[key] = points[key]
    return keep


def extract_scan_and_pmt(raw_dir):
    # .../datastorage/aus/raw/<scan>/<pmt>
    pmt = raw_dir.name
    scan = raw_dir.parent.name
    return scan, pmt


def build_config_for_point(
    template_cfg,
    channels,
    point_files,
    output_root,
):
    cfg = copy.deepcopy(template_cfg)
    cfg["<channels>"] = channels

    if "InputName" not in cfg:
        raise KeyError("Template config missing top-level key: InputName")

    input_block = cfg["InputName"]
    # Remove pre-existing per-channel blocks and rebuild from requested channels
    for k in list(input_block.keys()):
        if re.match(r"^ch\d+$", str(k)):
            del input_block[k]

    for ch in channels:
        if ch not in point_files:
            raise KeyError(f"Missing waveform file for channel {ch} at this theta/phi point.")
        input_block[f"ch{ch}"] = {
            "reader": "ReaderWaveDump",
            "files": [str(point_files[ch])],
        }

    # Update every TreeMaker output file path to this point's output root
    for obj_name, obj_cfg in cfg.items():
        if not isinstance(obj_cfg, dict):
            continue
        if obj_cfg.get("algorithm") == "TreeMaker":
            obj_cfg["file"] = str(output_root)

    return cfg


def run_pyrate(config_path, pyrate_cmd, pyrate_cwd, log_path, dry_run):
    cmd = [pyrate_cmd, "-c", str(config_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        with log_path.open("w", encoding="utf-8") as fo:
            fo.write("DRY RUN\n")
            fo.write(" ".join(cmd) + "\n")
            fo.write(f"cwd={pyrate_cwd}\n")
        return 0

    with log_path.open("w", encoding="utf-8") as fo:
        proc = subprocess.Popen(
            cmd,
            cwd=str(pyrate_cwd),
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


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_template_from_pyrate = None
    if os.environ.get("PYRATE_HOME"):
        default_template_from_pyrate = str(Path(os.environ["PYRATE_HOME"]).resolve().parent / "R12860_config.yaml")

    default_template = Path(
        os.environ.get(
            "UAP_AUS_TEMPLATE",
            default_template_from_pyrate
            if default_template_from_pyrate
            else str(repo_root / "upstream" / "aus_pyrate" / "R12860_config.yaml"),
        )
    )
    default_pyrate_cwd = Path(
        os.environ.get(
            "PYRATE_CWD",
            os.environ.get(
                "PYRATE_HOME",
                str(repo_root / "upstream" / "aus_pyrate" / "darkmatteraustralia-pyrate-3c2e05d64b61"),
            ),
        )
    )
    default_root_base = Path(
        os.environ.get("UAP_AUS_ROOT_DIR", str(repo_root.parent / "datastorage" / "aus" / "root"))
    )
    default_pyrate_cmd = os.environ.get("PYRATE_CMD", "pyrate")

    ap = argparse.ArgumentParser(description="Run AUS pyrate over all theta/phi points in one raw folder.")
    ap.add_argument("--raw-dir", required=True, help="Folder containing wave*_theta*_phi*.txt")
    ap.add_argument("--template", default=str(default_template), help="Template YAML (R12860_config.yaml)")
    ap.add_argument("--pyrate-cwd", default=str(default_pyrate_cwd), help="Directory where pyrate is runnable")
    ap.add_argument("--pyrate-cmd", default=default_pyrate_cmd, help="pyrate executable name/path")
    ap.add_argument("--channels", default="0,1,2", help="Comma-separated channels, e.g. 0,1,2")
    ap.add_argument("--points", nargs="*", help="Optional subset, e.g. theta0_phi0 theta10_phi90")
    ap.add_argument("--max-points", type=int, default=0, help="Optional limit for quick tests")
    ap.add_argument("--out-root-base", default=str(default_root_base), help="Base output dir for AUS roots/logs/configs")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Exact output directory for this raw-dir run (contains configs/logs/outputs). Overrides --out-root-base.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Generate configs/log plan only, do not execute pyrate")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    template_path = Path(args.template).resolve()
    pyrate_cwd = Path(args.pyrate_cwd).resolve()
    out_root_base = Path(args.out_root_base).resolve()
    channels = parse_channels(args.channels)

    if not raw_dir.is_dir():
        raise SystemExit(f"Raw directory not found: {raw_dir}")
    if not template_path.is_file():
        raise SystemExit(
            f"Template YAML not found: {template_path}\n"
            "Pass --template, or set UAP_AUS_TEMPLATE in your environment."
        )
    if "/" in args.pyrate_cmd and not Path(args.pyrate_cmd).is_file():
        raise SystemExit(f"pyrate command path does not exist: {args.pyrate_cmd}")
    if "/" not in args.pyrate_cmd and shutil.which(args.pyrate_cmd) is None:
        raise SystemExit(
            f"pyrate command not found in PATH: {args.pyrate_cmd}\n"
            "Pass --pyrate-cmd, or set PYRATE_CMD in your environment."
        )
    if not pyrate_cwd.is_dir():
        raise SystemExit(
            f"pyrate working directory not found: {pyrate_cwd}\n"
            "Pass --pyrate-cwd, or set PYRATE_CWD in your environment."
        )

    with template_path.open("r", encoding="utf-8") as fi:
        template_cfg = yaml.safe_load(fi)

    points_all = discover_points(raw_dir)
    points = keep_requested_points(points_all, args.points)
    ordered_keys = sorted(points.keys(), key=lambda x: (x[0], x[1]))
    if args.max_points > 0:
        ordered_keys = ordered_keys[: args.max_points]

    scan_name, pmt_name = extract_scan_and_pmt(raw_dir)
    run_base = Path(args.out_dir).resolve() if args.out_dir else (out_root_base / scan_name / pmt_name)
    cfg_dir = run_base / "configs"
    log_dir = run_base / "logs"
    out_dir = run_base / "outputs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ordered_keys:
        print(f"[WARN] No matching points found in {raw_dir}")
        return

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for theta, phi in ordered_keys:
        total += 1
        point_files = points[(theta, phi)]
        missing = [ch for ch in channels if ch not in point_files]
        point_tag = f"theta{theta}_phi{phi}"
        if missing:
            print(f"[SKIP] {point_tag}: missing channels {missing}")
            skipped += 1
            continue

        out_root = out_dir / f"output_{point_tag}.root"
        cfg_path = cfg_dir / f"R12860_{point_tag}.yaml"
        log_path = log_dir / f"{point_tag}.log"

        cfg = build_config_for_point(template_cfg, channels, point_files, out_root)
        with cfg_path.open("w", encoding="utf-8") as fo:
            yaml.safe_dump(cfg, fo, sort_keys=False)

        rc = run_pyrate(cfg_path, args.pyrate_cmd, pyrate_cwd, log_path, args.dry_run)
        if rc == 0:
            ok += 1
            print(f"[OK]   {point_tag}")
        else:
            failed += 1
            print(f"[FAIL] {point_tag} (rc={rc}) -> {log_path}")

    print(
        f"[DONE] total={total}, ok={ok}, skipped={skipped}, failed={failed}, "
        f"base={run_base}"
    )


if __name__ == "__main__":
    main()
