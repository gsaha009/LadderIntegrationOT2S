# ---------------------------------- #
#   Env: conda activate PyROOTEnv    #
# ---------------------------------- #
import os
import uproot
import awkward as ak
import numpy as np


def processor(filename: str, treename: str):
    def build_record(prefix):
        prefix_keys = [k for k in keys if k.startswith(prefix)]
        arrays = {k[len(prefix):]: tree[k].array() for k in prefix_keys}
        return ak.zip(arrays, depth_limit=1)
    
    fptr = uproot.open(filename)
    tree = fptr[treename]

    keys = list(tree.keys())
    if "event" not in keys:
        raise RuntimeError("No event branch in the TTree")
    event = tree["event"].array()

    board   = build_record("board_")
    hybrid  = build_record("hybrid_")
    cluster = build_record("cluster_")
    stub    = build_record("stub_")

    # Combine into single record
    events = ak.zip({
        "event": event,
        "board": board,
        "hybrid": hybrid,
        "cluster": cluster,
        "stub": stub
    }, depth_limit=1)

    return events
    


file = "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/OTPhysicsTest_Data/flat.root"

events = processor(file, "Events")
from IPython import embed; embed()
