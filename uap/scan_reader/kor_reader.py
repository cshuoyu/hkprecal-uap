"""KOR scan ROOT readers and angle parsing."""

import re
from uap.tool.root_io import (
    list_tree_channels as list_tree_channels_common,
    read_channel_branch,
)


DEFAULT_SERIAL_ORDER_CHANNELS = [0, 1, 3]


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
    match = pattern.search(name)
    if not match:
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
    for m in pattern.finditer(str(name)):
        out.append(m.group(1))
    return out


# Check whether serial block order is consistent across all files.
def check_serial_order_consistency(files):
    reference = None
    mismatches = []
    for fp in files:
        name = getattr(fp, "name", str(fp))
        order = [x.upper() for x in extract_serial_order(name)]
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
    order_map = list(serial_order_channels or DEFAULT_SERIAL_ORDER_CHANNELS)
    if not order_map:
        order_map = list(DEFAULT_SERIAL_ORDER_CHANNELS)
    return int(order_map[0])


# Auto pick PMT channel from target serial position in filename order.
def auto_pick_channel(files, serial, trigger_ch=None, serial_order_channels=None):
    order_map = list(serial_order_channels or DEFAULT_SERIAL_ORDER_CHANNELS)
    if not order_map:
        order_map = list(DEFAULT_SERIAL_ORDER_CHANNELS)
    serial = str(serial).upper()
    for fp in files:
        serial_order = [
            x.upper() for x in extract_serial_order(getattr(fp, "name", str(fp)))
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
