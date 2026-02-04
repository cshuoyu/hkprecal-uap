#!/usr/bin/env python3
"""
Inspect and quickly plot variables from ROOT files.

Current focus:
  - mode=kor : KOR output files produced by prod_ntp_standalone.C
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


def strip_cycle(name: str) -> str:
    return name.split(";")[0]


def classify_keys(root_file: uproot.ReadOnlyDirectory) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {"trees": [], "hists": [], "params": [], "other": []}
    for key in root_file.keys(cycle=False):
        obj = root_file[key]
        cls = getattr(obj, "classname", "Unknown")
        entry = (key, cls)
        if "TTree" in cls:
            out["trees"].append(entry)
        elif "TH1" in cls or "TH2" in cls:
            out["hists"].append(entry)
        elif "TParameter" in cls:
            out["params"].append(entry)
        else:
            out["other"].append(entry)
    return out


def print_summary(root_file: uproot.ReadOnlyDirectory, mode: str) -> None:
    groups = classify_keys(root_file)
    print(f"[INFO] Objects in file: {sum(len(v) for v in groups.values())}")
    for label in ["trees", "hists", "params", "other"]:
        print(f"\n[{label.upper()}] {len(groups[label])}")
        for key, cls in groups[label]:
            print(f"  - {key} ({cls})")
            if label == "trees":
                tree = root_file[key]
                branches = list(tree.keys())
                print(f"      entries={tree.num_entries}, branches={branches}")

    if mode == "kor":
        channels = []
        for key, _ in groups["trees"]:
            m = re.match(r"tree_ch(\d+)$", strip_cycle(key))
            if m:
                channels.append(int(m.group(1)))
        if channels:
            channels = sorted(set(channels))
            print(f"\n[KOR] detected channels: {channels}")
            print("[KOR] common trees: tree_ch<ch>")
            print("[KOR] common histograms: Ped_ch<ch>, Max_ch<ch>, Time_ch<ch>, Diff_ch<ch>, Pico_ch<ch>, PicoAbove_ch<ch>")


def resolve_hist_key(
    root_file: uproot.ReadOnlyDirectory,
    mode: str,
    plot_hist: Optional[str],
    preset: Optional[str],
    channel: Optional[int],
) -> Optional[str]:
    if plot_hist:
        return plot_hist

    if preset is None:
        return None
    if channel is None:
        raise SystemExit("--preset requires --channel.")

    if mode == "kor":
        table = {
            "pico": f"Pico_ch{channel}",
            "pico_above": f"PicoAbove_ch{channel}",
            "ped": f"Ped_ch{channel}",
            "max": f"Max_ch{channel}",
            "diff": f"Diff_ch{channel}",
            "time": f"Time_ch{channel}",
        }
        return table[preset]

    return None


def plot_hist_object(root_file: uproot.ReadOnlyDirectory, hist_key: str, out_png: Path, logy: bool) -> None:
    if hist_key not in root_file:
        raise SystemExit(f"Histogram not found: {hist_key}")

    hist_obj = root_file[hist_key]
    values, edges = hist_obj.to_numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.step(centers, values, where="mid", linewidth=1.3)
    ax.set_xlabel(strip_cycle(hist_key))
    ax.set_ylabel("Counts")
    ax.set_title(strip_cycle(hist_key))
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[OK] Saved histogram plot: {out_png}")


def flatten_array(arr: np.ndarray) -> np.ndarray:
    # KOR tree branches are scalar, but keep this robust for jagged/object arrays.
    if arr.dtype == object:
        chunks: List[np.ndarray] = []
        for item in arr:
            item_arr = np.asarray(item)
            if item_arr.size > 0:
                chunks.append(item_arr.reshape(-1))
        if not chunks:
            return np.array([], dtype=float)
        return np.concatenate(chunks)
    return np.asarray(arr).reshape(-1)


def plot_tree_branch(
    root_file: uproot.ReadOnlyDirectory,
    tree_name: str,
    branch_name: str,
    out_png: Path,
    bins: int,
    xmin: Optional[float],
    xmax: Optional[float],
    logy: bool,
) -> None:
    if tree_name not in root_file:
        raise SystemExit(f"Tree not found: {tree_name}")
    tree = root_file[tree_name]
    if branch_name not in tree.keys():
        raise SystemExit(f"Branch '{branch_name}' not found in {tree_name}. Available: {list(tree.keys())}")

    arr = tree[branch_name].array(library="np")
    vals = flatten_array(arr)
    if vals.size == 0:
        raise SystemExit(f"Branch has no values: {tree_name}.{branch_name}")

    hist_range = None
    if xmin is not None and xmax is not None:
        hist_range = (xmin, xmax)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.hist(vals, bins=bins, range=hist_range, histtype="step", linewidth=1.3)
    ax.set_xlabel(f"{tree_name}.{branch_name}")
    ax.set_ylabel("Counts")
    ax.set_title(f"{tree_name}.{branch_name}")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[OK] Saved branch plot: {out_png}")
    print(f"[INFO] entries={vals.size}, mean={vals.mean():.6g}, std={vals.std():.6g}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect and plot ROOT files (KOR/AUS helper).")
    ap.add_argument("--file", required=True, help="Input ROOT file path")
    ap.add_argument("--mode", choices=["kor", "aus"], default="kor", help="Layout hint for presets")
    ap.add_argument("--list", action="store_true", help="Print key/tree/branch summary")
    ap.add_argument("--plot-hist", default="", help="Plot an existing histogram object by key name")
    ap.add_argument(
        "--preset",
        choices=["pico", "pico_above", "ped", "max", "diff", "time"],
        help="KOR histogram preset (requires --channel)",
    )
    ap.add_argument("--channel", type=int, help="Channel index for --preset")
    ap.add_argument("--plot-branch", default="", help="Plot a branch from a TTree")
    ap.add_argument("--tree", default="", help="Tree name for --plot-branch (default: tree_ch<channel> for KOR)")
    ap.add_argument("--bins", type=int, default=200, help="Bins for branch histogram")
    ap.add_argument("--xmin", type=float, default=None, help="xmin for branch histogram")
    ap.add_argument("--xmax", type=float, default=None, help="xmax for branch histogram")
    ap.add_argument("--logy", action="store_true", help="Use log y-axis")
    ap.add_argument("--out", default="", help="Output PNG path")
    args = ap.parse_args()

    in_path = Path(args.file).resolve()
    if not in_path.is_file():
        raise SystemExit(f"ROOT file not found: {in_path}")

    do_plot_hist = bool(args.plot_hist) or bool(args.preset)
    do_plot_branch = bool(args.plot_branch)
    if not args.list and not do_plot_hist and not do_plot_branch:
        args.list = True

    with uproot.open(in_path) as root_file:
        if args.list:
            print_summary(root_file, args.mode)

        if do_plot_hist:
            hist_key = resolve_hist_key(root_file, args.mode, args.plot_hist, args.preset, args.channel)
            assert hist_key is not None
            out_png = Path(args.out).resolve() if args.out else (Path.cwd() / f"{strip_cycle(hist_key)}.png")
            plot_hist_object(root_file, hist_key, out_png, args.logy)

        if do_plot_branch:
            tree_name = args.tree
            if not tree_name:
                if args.mode == "kor" and args.channel is not None:
                    tree_name = f"tree_ch{args.channel}"
                else:
                    tree_name = "tree_ch0"
            out_png = (
                Path(args.out).resolve()
                if args.out
                else (Path.cwd() / f"{strip_cycle(tree_name)}_{args.plot_branch}.png")
            )
            plot_tree_branch(
                root_file,
                tree_name=tree_name,
                branch_name=args.plot_branch,
                out_png=out_png,
                bins=args.bins,
                xmin=args.xmin,
                xmax=args.xmax,
                logy=args.logy,
            )


if __name__ == "__main__":
    main()

