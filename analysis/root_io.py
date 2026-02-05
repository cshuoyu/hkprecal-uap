#!/usr/bin/env python3
"""
Unified ROOT input interface for fitting tasks.

This module provides one API for reading observables from AUS/KOR output files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import uproot


# Canonical observable names used by analysis code.
KOR_TREE_BRANCH_MAP = {
    "charge": "pico",
    "pulse_height": "max",
    "pulse_start": "falltime",
    "time_diff": "diff",
}

AUS_TREE_BRANCH_MAP = {
    "charge": "PulseCharge",
    "pulse_height": "PeakHeight",
    "pulse_start": "PulseStart",
    "peak_location": "PeakLocation",
    "cfd_start": "CFDPulseStart",
}

KOR_HIST_MAP = {
    "charge_hist": "Pico_ch{ch}",
    "charge_above_hist": "PicoAbove_ch{ch}",
    "pulse_height_hist": "Max_ch{ch}",
    "pedestal_hist": "Ped_ch{ch}",
}


@dataclass
class ReadResult:
    system: str
    file: str
    channel: int
    observable: str
    source: str
    raw_count: int
    valid_count: int
    values: np.ndarray


def _flatten(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != object:
        return np.asarray(arr).reshape(-1)
    chunks: List[np.ndarray] = []
    for item in arr:
        v = np.asarray(item).reshape(-1)
        if v.size > 0:
            chunks.append(v)
    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate(chunks)


def _clean_numeric(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    return vals[np.isfinite(vals)]


def detect_system(root_path: str) -> str:
    with uproot.open(root_path) as f:
        keys = set(f.keys(cycle=False))
    if any(k.startswith("Tree_CH") for k in keys):
        return "aus"
    if any(k.startswith("tree_ch") for k in keys):
        return "kor"
    raise ValueError(f"Cannot detect system for file: {root_path}")


def list_channels(root_path: str, system: Optional[str] = None) -> List[int]:
    sys_name = system or detect_system(root_path)
    channels: List[int] = []
    with uproot.open(root_path) as f:
        keys = f.keys(cycle=False)
        if sys_name == "aus":
            prefix = "Tree_CH"
        elif sys_name == "kor":
            prefix = "tree_ch"
        else:
            raise ValueError(f"Unsupported system: {sys_name}")
        for k in keys:
            if k.startswith(prefix):
                try:
                    channels.append(int(k.split(prefix)[1]))
                except Exception:
                    continue
    return sorted(set(channels))


def _tree_name(system: str, channel: int) -> str:
    if system == "aus":
        return f"Tree_CH{channel}"
    if system == "kor":
        return f"tree_ch{channel}"
    raise ValueError(f"Unsupported system: {system}")


def _branch_name(system: str, observable: str) -> str:
    if system == "aus":
        table = AUS_TREE_BRANCH_MAP
    elif system == "kor":
        table = KOR_TREE_BRANCH_MAP
    else:
        raise ValueError(f"Unsupported system: {system}")

    if observable in table:
        return table[observable]
    # allow direct branch name pass-through
    return observable


def read_tree_observable(
    root_path: str,
    channel: int,
    observable: str,
    system: Optional[str] = None,
) -> ReadResult:
    sys_name = system or detect_system(root_path)
    tree_name = _tree_name(sys_name, channel)
    branch_name = _branch_name(sys_name, observable)

    with uproot.open(root_path) as f:
        if tree_name not in f:
            raise KeyError(f"Tree not found: {tree_name}")
        tree = f[tree_name]
        if branch_name not in tree.keys():
            raise KeyError(f"Branch not found: {tree_name}.{branch_name}")
        arr = tree[branch_name].array(library="np")

    flat = _flatten(arr)
    clean = _clean_numeric(flat)
    return ReadResult(
        system=sys_name,
        file=str(Path(root_path).resolve()),
        channel=channel,
        observable=observable,
        source=f"{tree_name}.{branch_name}",
        raw_count=int(flat.size),
        valid_count=int(clean.size),
        values=clean,
    )


def read_hist_observable(
    root_path: str,
    channel: int,
    observable: str,
    system: str = "kor",
) -> ReadResult:
    if system != "kor":
        raise ValueError("Histogram-mode observable is currently supported only for KOR.")
    if observable not in KOR_HIST_MAP:
        raise KeyError(f"Unknown KOR histogram observable: {observable}")

    hist_name = KOR_HIST_MAP[observable].format(ch=channel)
    with uproot.open(root_path) as f:
        if hist_name not in f:
            raise KeyError(f"Histogram not found: {hist_name}")
        h = f[hist_name]
        counts, edges = h.to_numpy()
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Expand histogram to event-like values for a shared fit API.
        # Keep this as an option; tree-based read is still preferred when available.
        expanded = np.repeat(centers, counts.astype(np.int64))

    clean = _clean_numeric(expanded)
    return ReadResult(
        system=system,
        file=str(Path(root_path).resolve()),
        channel=channel,
        observable=observable,
        source=hist_name,
        raw_count=int(expanded.size),
        valid_count=int(clean.size),
        values=clean,
    )


def read_features(
    root_path: str,
    channel: int,
    observables: Iterable[str],
    system: Optional[str] = None,
    source: str = "tree",
) -> Dict[str, ReadResult]:
    out: Dict[str, ReadResult] = {}
    for obs in observables:
        if source == "tree":
            out[obs] = read_tree_observable(root_path, channel, obs, system=system)
        elif source == "hist":
            out[obs] = read_hist_observable(root_path, channel, obs, system=(system or "kor"))
        else:
            raise ValueError("source must be 'tree' or 'hist'")
    return out


__all__ = [
    "ReadResult",
    "detect_system",
    "list_channels",
    "read_tree_observable",
    "read_hist_observable",
    "read_features",
]

