#!/usr/bin/env python3
"""
Test analysis module:
  - read charge from ROOT via unified I/O
  - fit with two Gaussians (pedestal + signal)
  - save fit plot and fit result JSON
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    from analysis.root_io import detect_system, read_hist_observable, read_tree_observable
except Exception:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from analysis.root_io import detect_system, read_hist_observable, read_tree_observable


def gauss(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def two_gauss(
    x: np.ndarray,
    amp0: float,
    mu0: float,
    sigma0: float,
    amp1: float,
    mu1: float,
    sigma1: float,
) -> np.ndarray:
    return gauss(x, amp0, mu0, sigma0) + gauss(x, amp1, mu1, sigma1)


@dataclass
class FitSummary:
    status: str
    message: str
    system: str
    channel: int
    source: str
    observable: str
    n_raw: int
    n_used: int
    xmin: float
    xmax: float
    bins: int
    params: Dict[str, float]
    errors: Dict[str, float]
    chi2: float
    ndf: int
    chi2_ndf: float
    signal_fraction: float
    gain_proxy: float


def initial_guess(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # Basic robust guess for a two-peak fit.
    vstd = float(np.std(values))
    if not np.isfinite(vstd) or vstd <= 0:
        vstd = max((x.max() - x.min()) / 20.0, 1e-3)

    mu0 = float(np.quantile(values, 0.20))
    mu1 = float(np.quantile(values, 0.80))
    if mu1 <= mu0:
        mu1 = mu0 + max(vstd, 1e-3)

    amp0 = float(max(y.max(), 1.0))
    amp1 = float(max(y.max() * 0.5, 1.0))
    sig0 = max(vstd * 0.5, 1e-3)
    sig1 = max(vstd * 0.5, 1e-3)

    p0 = np.array([amp0, mu0, sig0, amp1, mu1, sig1], dtype=float)
    lo = np.array([0.0, x.min(), 1e-4, 0.0, x.min(), 1e-4], dtype=float)
    hi = np.array([np.inf, x.max(), (x.max() - x.min()), np.inf, x.max(), (x.max() - x.min())], dtype=float)
    return p0, (lo, hi)


def reorder_by_mean(p: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # enforce mu0 <= mu1
    if p[1] <= p[4]:
        return p, cov

    idx = np.array([3, 4, 5, 0, 1, 2], dtype=int)
    p2 = p[idx]
    cov2 = cov[np.ix_(idx, idx)]
    return p2, cov2


def make_result(
    status: str,
    message: str,
    system: str,
    channel: int,
    source: str,
    observable: str,
    n_raw: int,
    n_used: int,
    xmin: float,
    xmax: float,
    bins: int,
    p: np.ndarray,
    perr: np.ndarray,
    chi2: float,
    ndf: int,
) -> FitSummary:
    y0 = p[0]
    m0 = p[1]
    s0 = p[2]
    y1 = p[3]
    m1 = p[4]
    s1 = p[5]

    # Use discrete component sums as event-fraction proxy.
    # (good enough for this pipeline test module)
    comp0 = max(y0, 0.0) * max(s0, 1e-12)
    comp1 = max(y1, 0.0) * max(s1, 1e-12)
    denom = comp0 + comp1
    signal_fraction = (comp1 / denom) if denom > 0 else float("nan")
    gain_proxy = m1 - m0

    params = {
        "amp0": float(y0),
        "mu0": float(m0),
        "sigma0": float(s0),
        "amp1": float(y1),
        "mu1": float(m1),
        "sigma1": float(s1),
    }
    errors = {
        "amp0": float(perr[0]),
        "mu0": float(perr[1]),
        "sigma0": float(perr[2]),
        "amp1": float(perr[3]),
        "mu1": float(perr[4]),
        "sigma1": float(perr[5]),
    }

    return FitSummary(
        status=status,
        message=message,
        system=system,
        channel=channel,
        source=source,
        observable=observable,
        n_raw=n_raw,
        n_used=n_used,
        xmin=float(xmin),
        xmax=float(xmax),
        bins=bins,
        params=params,
        errors=errors,
        chi2=float(chi2),
        ndf=int(ndf),
        chi2_ndf=float(chi2 / ndf) if ndf > 0 else float("nan"),
        signal_fraction=float(signal_fraction),
        gain_proxy=float(gain_proxy),
    )


def fit_charge(values: np.ndarray, bins: int, xmin: float, xmax: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    hist, edges = np.histogram(values, bins=bins, range=(xmin, xmax))
    centers = 0.5 * (edges[:-1] + edges[1:])

    p0, bounds = initial_guess(centers, hist, values)
    sigma_y = np.sqrt(np.maximum(hist.astype(float), 1.0))

    popt, pcov = curve_fit(
        two_gauss,
        centers,
        hist.astype(float),
        p0=p0,
        bounds=bounds,
        sigma=sigma_y,
        absolute_sigma=True,
        maxfev=100000,
    )
    popt, pcov = reorder_by_mean(popt, pcov)

    model = two_gauss(centers, *popt)
    chi2 = float(np.sum(((hist - model) / sigma_y) ** 2))
    ndf = int(len(hist) - len(popt))

    return centers, hist, popt, pcov, chi2, ndf


def save_plot(out_png: Path, centers: np.ndarray, hist: np.ndarray, p: np.ndarray, xmin: float, xmax: float, logy: bool) -> None:
    xx = np.linspace(xmin, xmax, 1200)
    comp0 = gauss(xx, p[0], p[1], p[2])
    comp1 = gauss(xx, p[3], p[4], p[5])
    total = comp0 + comp1

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.step(centers, hist, where="mid", linewidth=1.2, label="data")
    ax.plot(xx, total, linewidth=1.8, label="2G total")
    ax.plot(xx, comp0, linestyle="--", linewidth=1.2, label="G0")
    ax.plot(xx, comp1, linestyle="--", linewidth=1.2, label="G1")
    ax.set_xlabel("charge")
    ax.set_ylabel("counts / bin")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Test: read charge and fit with two Gaussians.")
    ap.add_argument("--file", required=True, help="Input ROOT file")
    ap.add_argument("--channel", type=int, required=True, help="Channel index")
    ap.add_argument("--system", choices=["aus", "kor", "auto"], default="auto", help="Input file system")
    ap.add_argument("--source", choices=["tree", "hist"], default="tree", help="Read source for charge")
    ap.add_argument("--observable", default="charge", help="Canonical observable name (default: charge)")
    ap.add_argument("--bins", type=int, default=250)
    ap.add_argument("--xmin", type=float, default=-10.0)
    ap.add_argument("--xmax", type=float, default=15.0)
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--out-dir", default=".", help="Output directory")
    ap.add_argument("--tag", default="", help="Optional output tag")
    args = ap.parse_args()

    in_file = str(Path(args.file).resolve())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    system = detect_system(in_file) if args.system == "auto" else args.system

    if args.source == "tree":
        read = read_tree_observable(in_file, args.channel, args.observable, system=system)
    else:
        read = read_hist_observable(in_file, args.channel, args.observable, system=system)

    values = read.values
    in_range = values[(values >= args.xmin) & (values <= args.xmax)]
    if in_range.size < 100:
        raise SystemExit(f"Too few entries in fit range: {in_range.size}")

    centers, hist, popt, pcov, chi2, ndf = fit_charge(in_range, args.bins, args.xmin, args.xmax)
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))

    summary = make_result(
        status="ok",
        message="fit converged",
        system=system,
        channel=args.channel,
        source=read.source,
        observable=args.observable,
        n_raw=read.raw_count,
        n_used=int(in_range.size),
        xmin=args.xmin,
        xmax=args.xmax,
        bins=args.bins,
        p=popt,
        perr=perr,
        chi2=chi2,
        ndf=ndf,
    )

    tag = args.tag.strip() if args.tag else f"{Path(in_file).stem}_ch{args.channel}"
    out_png = out_dir / f"{tag}_charge_2g.png"
    out_json = out_dir / f"{tag}_charge_2g.json"

    save_plot(out_png, centers, hist, popt, args.xmin, args.xmax, args.logy)
    out_json.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    print(
        "[FIT] "
        f"status={summary.status} system={summary.system} ch={summary.channel} "
        f"n_used={summary.n_used} chi2/ndf={summary.chi2_ndf:.3f} "
        f"gain_proxy={summary.gain_proxy:.6g} signal_fraction={summary.signal_fraction:.6g}"
    )
    print(f"[OUT] plot={out_png}")
    print(f"[OUT] json={out_json}")


if __name__ == "__main__":
    main()

