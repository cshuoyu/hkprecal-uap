"""Shared plotting helpers for fit diagnostics."""

from pathlib import Path

import matplotlib as mpl
import numpy as np


FIT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "axes.unicode_minus": False,  # This Times font lacks the Unicode minus glyph.
}


def charge_fit_text(out):
    """Display peak gain, absolute SPE center and total Gaussian sigma (pC)."""
    gain = float(out.get("gain_pedestal_subtracted", np.nan))
    if np.isfinite(gain) and gain > 0:
        mantissa, exponent = "{:.3e}".format(gain).split("e")
        gain_text = r"{}\times10^{{{}}}".format(mantissa, int(exponent))
    else:
        gain_text = r"\mathrm{N/A}"
    lines = [
        r"$\mathrm{Gain}=" + gain_text + "$",
        r"$\mu_{\mathrm{SPE}}=" + "{:.3f}".format(out.get("spe_mean", np.nan)) + r"\,\mathrm{pC}$",
        r"$\sigma_{\mathrm{SPE}}=" + "{:.3f}".format(out.get("spe_sigma", np.nan)) + r"\,\mathrm{pC}$",
    ]
    chi2, ndf = out.get("chi2", np.nan), out.get("ndf", np.nan)
    if np.isfinite(chi2) and np.isfinite(ndf) and ndf > 0:
        lines.append(r"$\chi^2/\mathrm{ndf}=" + "{:.2f}".format(chi2 / ndf) + "$")
    return lines


@mpl.rc_context(FIT_STYLE)
def save_fit_with_pull_plot(
    plt,
    out_png,
    centers,
    counts,
    yerr,
    x_model,
    y_model,
    pull,
    xlim,
    text_lines=None,
    y_label="Events",
    x_label="Time (ns)",
    pull_ylim=(-5, 5),
):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [4, 1]}, figsize=(10, 10)
    )
    ax1.errorbar(centers, counts, yerr=yerr, fmt="ok", label="Data")
    ax1.plot(x_model, y_model, linewidth=2, label="Total")
    ax1.set_xlim([float(xlim[0]), float(xlim[1])])
    ax1.set_ylabel(y_label)
    ax1.legend(loc="best")

    if text_lines:
        ax1.text(
            0.72,
            0.95,
            "\n".join([str(x) for x in text_lines]),
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.6},
        )

    ax2.plot(
        np.linspace(float(xlim[0]), float(xlim[1]), 300),
        np.zeros(300),
        "--",
        color="black",
    )
    ax2.errorbar(centers, pull, fmt="ok")
    ax2.set_xlim([float(xlim[0]), float(xlim[1])])
    ax2.set_ylim([float(pull_ylim[0]), float(pull_ylim[1])])
    ax2.set_yticks([float(pull_ylim[0]), 0, float(pull_ylim[1])])
    ax2.set_ylabel("Pull")
    ax2.set_xlabel(x_label)

    fig.tight_layout()
    target = Path(out_png).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(target))
    plt.close(fig)
