"""KOR scan ROOT readers and angle parsing."""

from functools import lru_cache
from pathlib import Path
import re
import numpy as np
import uproot
from uap.tool.root_io import (
    list_tree_channels as list_tree_channels_common,
    read_channel_branch,
)


# Raw-data channel mapping (verified from 20260129 waveforms):
# - ch0/1/2 hold the three PMTs listed in the filename (in order),
#   each showing sparse single-photon hits in the PMT signal window.
# - ch3 is the laser-monitor / physical trigger: 100% hit rate at a fixed
#   early sample, never in the PMT signal window.
# The NTP C++ (prod_ntp_standalone.C) hardcodes TriggerCh = 2, which is
# actually the third PMT (EL9590B). This causes diff = falltime + 1 in the
# prd_*.root files; UAP downstream still reads the stored diff as-is.
DEFAULT_SERIAL_ORDER_CHANNELS = [0, 1, 2]
DEFAULT_TRIGGER_CH = 3
ADC_TO_MV = 2000.0 / 16384.0  # KOR v7 Config::ADC_to_mV


@lru_cache(maxsize=2048)
def read_run_info(source):
    """Metadata fallback for run-number-only processed v7 filenames."""
    path = Path(source)
    if not path.is_file():
        return {}
    with uproot.open(path) as root_file:
        if "RunInfo" not in root_file:
            return {}
        tree = root_file["RunInfo"]
        if tree.num_entries != 1:
            raise ValueError("Expected exactly one RunInfo entry: {}".format(path))
        fields = ["SN1", "SN2", "SN3", "RawRotateAngle2", "RawTiltAngle2",
                  "RawRotateAngle3", "RawTiltAngle3"]
        return {name: tree[name].array(library="np")[0] for name in fields if name in tree}


def read_charge_seed(root_path, channel):
    """Use the saved threshold for the SPE prefit, not for the final charge fit."""
    with uproot.open(root_path) as root_file:
        tree = root_file["tree_ch{}".format(channel)]
        charge = tree["pico"].array(library="np")
        height = tree["max"].array(library="np")
        threshold = float(root_file["Config_Threshold_mV_ch{}".format(channel)].member("fVal"))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid saved charge-seed threshold: {}".format(root_path))
    selected = np.isfinite(charge) & np.isfinite(height) & (height * ADC_TO_MV > threshold)
    return charge[selected], threshold


# KOR scan filenames have a more complex structure
# e.g.:prd_EM2740A_hv1670_R00_T00_EL1635B_hv1840_RP45_TM20_EL9590B_hv1770_RP45_TM20_laser134_20260129.0001.root
def parse_kor_signed_angle(tag):
    if not tag.startswith("T"):
        raise ValueError("Invalid T tag: {}".format(tag))
    body = tag[1:]
    if body.startswith("P"):
        return int(body[1:])
    if body.startswith("M"):
        return -int(body[1:])
    return int(body)


# Extract phi and raw theta from a KOR scan filename based on the serial number pattern.
def extract_serial_block_angles(name, serial):
    pattern = re.compile(
        r"{}_hv\d+_R(?P<r>[PM]?\d+)_T(?P<t>[PM]?\d+)".format(re.escape(serial))
    )
    match = pattern.search(Path(name).name)
    if not match:
        info = read_run_info(name)
        for index in (1, 2, 3):
            if str(info.get("SN{}".format(index), "")).upper() == str(serial).upper():
                if index == 1:
                    return 0, 0  # stationary monitor
                return int(info["RawRotateAngle{}".format(index)]), int(info["RawTiltAngle{}".format(index)])
        return None
    rtag = match.group("r")
    ttag = "T" + match.group("t")
    if rtag.startswith("P"):
        phi = int(rtag[1:])
    elif rtag.startswith("M"):
        phi = -int(rtag[1:])
    else:
        phi = int(rtag)
    theta_raw = parse_kor_signed_angle(ttag)
    return phi, theta_raw


# List tree channels.
def list_tree_channels(root_path):
    return list_tree_channels_common(root_path, tree_prefix="tree_ch")


# Read one branch from one KOR channel tree.
def read_tree_branch(root_path, channel, branch, tree_prefix="tree_ch"):
    return read_channel_branch(root_path, channel, branch, tree_prefix=tree_prefix)


# Extract serial block order from filename, e.g. [EM2740A, EL1635B, EL9590B].
def extract_serial_order(name):
    pattern = re.compile(r"([A-Za-z0-9]+)_hv\d+_R[PM]?\d+_T[PM]?\d+")
    out = []
    for m in pattern.finditer(Path(name).name):
        out.append(m.group(1))
    if not out:
        info = read_run_info(name)
        if info:
            out = [str(info["SN{}".format(index)]) for index in (1, 2, 3)]
    return out


# Check whether serial block order is consistent across all files.
def check_serial_order_consistency(files):
    reference = None
    mismatches = []
    for fp in files:
        name = getattr(fp, "name", str(fp))
        order = [x.upper() for x in extract_serial_order(fp)]
        if not order:
            continue
        if reference is None:
            reference = order
            continue
        if order != reference:
            mismatches.append((name, order))
    return reference, mismatches


# Auto pick trigger channel.
def auto_pick_trigger_channel(serial_order_channels=None):
    return int(DEFAULT_TRIGGER_CH)


# Auto pick PMT channel from target serial position in filename order.
def auto_pick_channel(files, serial, trigger_ch=None, serial_order_channels=None):
    order_map = list(serial_order_channels or DEFAULT_SERIAL_ORDER_CHANNELS)
    if not order_map:
        order_map = list(DEFAULT_SERIAL_ORDER_CHANNELS)
    serial = str(serial).upper()
    for fp in files:
        serial_order = [
            x.upper() for x in extract_serial_order(fp)
        ]
        if not serial_order:
            continue
        if serial not in serial_order:
            continue
        idx = serial_order.index(serial)
        if idx >= len(order_map):
            continue
        ch = int(order_map[idx])
        if trigger_ch is not None and ch == int(trigger_ch):
            continue
        return ch
    return int(order_map[0]) if order_map else 0
