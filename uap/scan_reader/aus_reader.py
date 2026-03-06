"""AUS scan ROOT readers and angle parsing."""

import re
from pathlib import Path
from uap.tool.root_io import read_branch


FILE_RE_AUS = re.compile(r"output_theta(?P<theta>-?\d+)_phi(?P<phi>-?\d+)\.root$")
DEFAULT_AUS_CHANNELS = {"trigger": 0, "sipm": 1, "pmt": 2}


# Extract (theta, phi) from AUS output filename.
def parse_theta_phi_aus(path):
    path = Path(path)
    match = FILE_RE_AUS.search(path.name)
    if not match:
        return None
    return int(match.group("theta")), int(match.group("phi"))


# Read one branch from one AUS ROOT tree.
def load_branch(root_path, tree_name, branch):
    return read_branch(root_path, tree_name, branch)


# Check whether channel setting is "auto".
def is_auto_channel(value):
    return str(value).strip().lower() == "auto"


# Parse channel config into int or "auto".
def parse_channel_value(value, arg_name):
    if is_auto_channel(value):
        return "auto"
    try:
        return int(value)
    except Exception:
        raise SystemExit(
            "{} must be integer or 'auto'. got: {}".format(arg_name, value)
        )


# Resolve one AUS channel from value/default.
def resolve_aus_channel(value, arg_name, default_value):
    parsed = parse_channel_value(value, arg_name)
    if parsed == "auto":
        return int(default_value)
    return int(parsed)
