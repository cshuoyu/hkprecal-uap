"""Window selection for analyses."""

import numpy as np


def _sanitize(values, positive_only=True, drop_zero=False):
    data = np.asarray(values).reshape(-1)
    data = data[np.isfinite(data)]
    if drop_zero:
        data = data[data != 0]
    if positive_only:
        data = data[data > 0]
    return data


def _percentile_range(data, qlow=1.0, qhigh=99.0):
    lo = np.percentile(data, qlow)
    hi = np.percentile(data, qhigh)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(data))
        hi = float(np.max(data))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def select_window(
    values,
    method,
    tmin=None,
    tmax=None,
    half_width=20.0,
    bin_width=None,
    positive_only=True,
    drop_zero=False,
    qlow=1.0,
    qhigh=99.0,
):
    data = _sanitize(values, positive_only=positive_only, drop_zero=drop_zero)
    if data.size == 0:
        raise RuntimeError("No valid entries for window selection.")

    method = str(method or "peak_percentile").strip().lower()
    half_width = float(half_width)
    if half_width <= 0:
        raise RuntimeError("half_width must be > 0.")

    if method == "manual":
        raise RuntimeError("manual window is deprecated; use peak-based window.")

    if method in ("peak_percentile", "peak", "peak_center"):
        if tmin is not None and tmax is not None and tmax > tmin:
            peak_src = data[(data > float(tmin)) & (data < float(tmax))]
            if peak_src.size == 0:
                peak_src = data
            lo, hi = float(tmin), float(tmax)
            if hi <= lo:
                lo, hi = _percentile_range(peak_src, qlow=qlow, qhigh=qhigh)
        else:
            peak_src = data
            lo, hi = _percentile_range(peak_src, qlow=qlow, qhigh=qhigh)

        if bin_width is not None and float(bin_width) > 0:
            bw = float(bin_width)
            edges = np.arange(lo, hi + bw, bw)
            if edges.size < 2:
                edges = np.array([lo, lo + bw], dtype=float)
            counts, edges = np.histogram(peak_src, bins=edges)
        else:
            bins = int(max(50, min(600, hi - lo)))
            counts, edges = np.histogram(peak_src, bins=bins, range=(lo, hi))

        centers = 0.5 * (edges[:-1] + edges[1:])
        peak = float(centers[int(np.argmax(counts))])
        return peak - half_width, peak + half_width, peak

    return select_window(
        values,
        method="peak_percentile",
        tmin=tmin,
        tmax=tmax,
        half_width=half_width,
        bin_width=bin_width,
        positive_only=positive_only,
        drop_zero=drop_zero,
        qlow=qlow,
        qhigh=qhigh,
    )
