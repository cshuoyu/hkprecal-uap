"""Shared ROOT branch readers for AUS/KOR."""

import numpy as np
import uproot


def strip_cycle(name):
    return str(name).split(";")[0]


def read_branch(root_path, tree_name, branch):
    with uproot.open(root_path) as root_file:
        if tree_name not in root_file:
            raise KeyError("Tree not found: {} in {}".format(tree_name, root_path))
        tree = root_file[tree_name]
        if branch not in tree.keys():
            raise KeyError("Branch not found: {} in {}".format(branch, tree_name))
        arr = tree[branch].array(library="np")
    return np.asarray(arr).reshape(-1)


def read_channel_branch(root_path, channel, branch, tree_prefix):
    tree_name = "{}{}".format(tree_prefix, int(channel))
    return read_branch(root_path, tree_name, branch)


def list_tree_channels(root_path, tree_prefix):
    out = []
    with uproot.open(root_path) as root_file:
        for key in root_file.keys():
            name = strip_cycle(key)
            if not name.startswith(tree_prefix):
                continue
            suffix = name[len(tree_prefix) :]
            try:
                out.append(int(suffix))
            except ValueError:
                continue
    return sorted(set(out))

