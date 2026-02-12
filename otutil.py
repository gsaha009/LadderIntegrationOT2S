import time
import uproot
import awkward as ak
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np
hep.style.use("CMS")

from Fitter import Fitter

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


def plot_basic(x = None, data_dict = None,
               title = 'default', name = 'default',
               **kwargs):
    basics = plot_basic_settings()

    xticklabels = kwargs.get("xticklabels", None)
    yticklabels = kwargs.get("yticklabels", None)
    outdir = kwargs.get("outdir", "../Output")
    ylim = kwargs.get("ylim", basics["ylim"])
    xlim = kwargs.get("xlim", basics["xlim"])
    linewidth = kwargs.get("linewidth", basics["linewidth"]) 
    xlabel = kwargs.get("xlabel", "var")
    ylabel = kwargs.get("ylabel", "var")
    marker = kwargs.get("marker", basics["marker"])
    markersize = kwargs.get("markersize", basics["markersize"])
    markerfacecolor = kwargs.get("markerfacecolor", None)
    markeredgewidth = kwargs.get("markeredgewidth", basics['markeredgewidth'])
    capsize = kwargs.get("capsize", basics["capsize"])
    elinewidth = kwargs.get("elinewidth", 0.5)
    dofit = kwargs.get("fit", False)
    fitfunc = kwargs.get("fitfunc", None)
    fitmodel = kwargs.get("fitmodel", None)
    dograd = kwargs.get("dograd", False)
    colors = kwargs.get("colors", basics["colors"])
    markerstyles = kwargs.get("markerstyles", basics["markerstyles"])
    linestyles = kwargs.get("linestyles", basics["linestyles"])
    fitlinewidth = kwargs.get("fitlinewidth", 1.2)
    mean_init = kwargs.get("mean_init", 500.0)
    sigma_init = kwargs.get("sigma_init", 100.0)
    nticks = kwargs.get("nticks", None)
    tick_offset = kwargs.get("tick_offset", 0.1)
    legend_labels = kwargs.get("legs", None)
    density = kwargs.get("density", False)
    
    fig, ax = plt.subplots(figsize=basics["size"])
    hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS


    legend_handles = []
    x = np.array(x)
    for i,(key,data) in enumerate(data_dict.items()):
        data = np.array(data)
        _val = data if data.ndim == 1 else data[:,0]
        if density == True:
            _val = _val/float(np.sum(_val))
        err  = np.zeros_like(_val) if data.ndim == 1 else data[:,1]
        val  = np.gradient(_val) if dograd else _val 
        ax.errorbar(x,
                    val,
                    yerr=err,
                    fmt = markerstyles[i],
                    elinewidth=elinewidth,
                    linewidth=linewidth,
                    linestyle='-',
                    #marker=marker,
                    markersize=markersize,
                    markerfacecolor=colors[i] if markerfacecolor is not None else 'none',
                    markeredgewidth = markeredgewidth,
                    color=colors[i],
                    label=key if legend_labels == None else legend_labels[i],
                    capsize=capsize)

        if dofit:
            #mask = val > 0.0
            #x_filtered = x[mask]
            #val_filtered = val[mask]
            #err_filtered = err[mask]

            fitobj = Fitter(x, val, err, modeltype=fitmodel)
            result = fitobj.result
            #from IPython import embed; embed(); exit()
                

            if result is not None:
                fit_val, fit_label_text = result
                ax.plot(x, fit_val, color=colors[i], linewidth=fitlinewidth)
                label_text = f"{key}: {fit_label_text}"

                legend_handles.append(
                    Line2D([0], [0], color=colors[i], linewidth=fitlinewidth, label=label_text)
                )



    if dofit:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=8.0,
            ncol=2,
            loc="upper left",
            #bbox_to_anchor=(1, 1)
        )
    else:
        ax.legend(fontsize=10, framealpha=1, facecolor='white', ncol=2)

    
    ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)        

    if xticklabels:
        if nticks is not None:
            tick_count = nticks
            tick_locs = np.linspace(0, x.shape[0]-1, tick_count, dtype=int)
            tick_labels = [xticklabels[i] for i in tick_locs]
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labels, rotation=90, ha="right", rotation_mode="anchor", fontsize=13)
        else:                
            ticks_ = [x-tick_offset for x in list(range(len(xticklabels)))]
            ax.set_xticks(ticks_)
            ax.set_xticklabels(xticklabels, rotation=90, ha='right', rotation_mode='anchor', fontsize=13)
    else:
        ax.set_xlabel(xlabel)


    ax.set_ylabel(ylabel)

    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
        
    ax.set_title(f"{title}", fontsize=14, loc='right')
    
    plt.tight_layout()
    fig.savefig(f"{outdir}/{name}.png", dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved : {outdir}/{name}.png")
    plt.close()



def hist_basic(bins = None, data_dict = None,
               title = 'default', name = 'default',
               **kwargs):
    basics = plot_basic_settings()

    outdir = kwargs.get("outdir", "../Output")
    ylim = kwargs.get("ylim", basics["ylim"])
    xlim = kwargs.get("xlim", basics["xlim"])
    linewidth = kwargs.get("linewidth", basics["linewidth"]) 
    xlabel = kwargs.get("xlabel", "var")
    ylabel = kwargs.get("ylabel", "var")
    colors = kwargs.get("colors", basics["colors"])
    linestyles = kwargs.get("linestyles", basics["linestyles"])
    legend_labels = kwargs.get("legs", None)
    density = kwargs.get("density", False)
    logy = kwargs.get("logy", False)
    
    fig, ax = plt.subplots(figsize=basics["size"])
    hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

    for i,(key,data) in enumerate(data_dict.items()):
        data = np.array(data)
        val = data if data.ndim == 1 else data[:,0]
        err  = np.zeros_like(val) if data.ndim == 1 else data[:,1]

        ax.hist(val,
                bins=np.linspace(bins[0], bins[1], bins[2]),
                histtype='step',
                linewidth=linewidth,
                linestyle=linestyles[i],
                color=colors[i],
                density=density,
                label=key,
                log=logy)

    ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)
    
    ax.legend(fontsize=13, framealpha=1, facecolor='white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_title(f"{title}", fontsize=14, loc='right')
    
    plt.tight_layout()
    fig.savefig(f"{outdir}/{name}.png", dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved : {outdir}/{name}.png")
    plt.close()
    
    


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


    neg_cmap = plt.cm.gray(np.linspace(0.05, 0.75, 64)) # black to gray

    #pos_colors = [
    #    "#f7fcb9",  # pale yellow
    #    "#41ab5d",  # green
    #    "#6a51a3",  # purple
    #    "#7f2704",  # dark brown
    #]
    #pos_colors = [
    #    "#ffffb2",  # light yellow
    #    "#66c2a5",  # green-teal
    #    "#3288bd",  # blue-teal
    #    "#084081",  # dark blue
    #]
    #pos_cmap = mcolors.LinearSegmentedColormap.from_list(
    #    "pos_custom",
    #    pos_colors,
    #    N=64
    #)
    #pos_cmap = pos_cmap(np.linspace(0, 1, 64))  # (64, 4)
    
    #pos_cmap = plt.cm.YlGnBu(np.linspace(0.1, 0.9, 64)) # light orange to brown
    pos_cmap = plt.cm.YlOrBr(np.linspace(0.1, 0.9, 64)) # light orange to brown

    cmap = ListedColormap(np.vstack((neg_cmap, pos_cmap)))
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)//2, vmax = vmax)
    
    #im = ax.imshow(data, cmap=colmap, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", origin="lower")

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
    


def plot_colormesh(x = None,
                   y = None,
                   z = None,
                   title = 'default',
                   name = 'default',
                   **kwargs):
    basics = plot_basic_settings()

    outdir  = kwargs.get("outdir", "../Output")
    shading = kwargs.get("shading", "auto")
    xlabel  = kwargs.get("xlabel", None)
    ylabel  = kwargs.get("ylabel", None)
    ylim    = kwargs.get("ylim", None)
    
    fig, ax = plt.subplots(figsize=basics["size"])
    hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

    pcm = ax.pcolormesh(
        x,
        y,
        z,
        shading=shading
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, loc='right')

    # Optional: limit frequency range
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # Colorbar
    #fig.colorbar(pcm, ax=ax, label="amplitude")

    plt.tight_layout()
    fig.savefig(f"{outdir}/{name}.png", dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved : {outdir}/{name}.png")
    plt.close()
    
