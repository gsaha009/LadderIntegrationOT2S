import uproot
import awkward as ak
import logging
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
hep.style.use("CMS")

logger = logging.getLogger('main')

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[1;32m',    # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.msg = f"{color}{record.msg}{self.RESET}"

        fmt = "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
        formatter = logging.Formatter(fmt, '%Y-%m-%d:%H:%M:%S')
        return formatter.format(record)


def setup_logger():
    # Create a logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Use the custom ColorFormatter
    formatter = ColorFormatter()
    console_handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


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


def plot_basic_settings():
    return {"size": (8.9, 6.5),
            "heplogo": "Internal",
            "logoloc": 0,
            "histtype": "step",
            "linewidth": 0,
            "marker": "o",
            "markersize": 1.2,
            "capsize": 0.2,
            "ylim": None,
            "xlim": None,
            "markeredgewidth": 1.5,
            "markerstyles": ['o', 's', 'D', '*', '^', 'v', 'P', 'X', '<', '>'],
            "colors": ["#165a86","#cc660b","#217a21","#a31e1f","#6e4e92","#6b443e","#b85fa0","#666666","#96971b","#1294a6","#8c1c62","#144d3a"],
            "linestyles": ["-","--","-.",":","(0, (3, 1))", "(0, (3, 1, 1, 1))","(0, (5, 5))","(0, (1, 1))","(0, (6, 2))","(0, (4, 2, 1, 2))"]}



def plot_heatmap(data = None, title = 'default', name = 'default', **kwargs):
    basics = plot_basic_settings()

    xticklabels = kwargs.get("xticklabels", None)
    yticklabels = kwargs.get("yticklabels", None)
    colmap = kwargs.get("colmap", "viridis")
    outdir = kwargs.get("outdir", "../Output")
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    cbar_label = kwargs.get("cbar_label", "CMNoise fraction")
    
    fig, ax = plt.subplots(figsize=basics["size"])
    hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

    if vmin is None:
        vmin = float(np.min(data))
        
    if vmax is None:
        vmax=float(np.max(data))
            
    im = ax.imshow(data, cmap=colmap, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        
    ax.set_title(f"{title}", fontsize=14, loc='right')
    ax.tick_params(direction="in", top=False, right=False, labelsize=12, length=3)
    plt.tight_layout()
    fig.savefig(f"{outdir}/{name}.png", dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved : {outdir}/{name}.png")
    plt.close()
    
