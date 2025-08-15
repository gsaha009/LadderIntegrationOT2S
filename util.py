import os
import sys
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import mplhep as hep
hep.style.use("CMS")


def basic_settings():
    return {"size": (8.9, 6.5),
            "heplogo": "Internal",
            "logoloc": 0,
            "histtype": "step",
            "linewidth": 0,
            "marker": "o",
            "markersize": 1.2,
            "capsize": 0.2,
            "ylim": [2.0,10.0],
            "xlim": None,
            "colors": ["#165a86","#cc660b","#217a21","#a31e1f","#6e4e92","#6b443e","#b85fa0","#666666","#96971b","#1294a6","#8c1c62","#144d3a"],
            "linestyles": ["-","--","-.",":","(0, (3, 1))", "(0, (3, 1, 1, 1))","(0, (5, 5))","	(0, (1, 1))","(0, (6, 2))","(0, (4, 2, 1, 2))"]}


def plot_basic(x, data_dict,
               title="", name="",
               **kwargs):

    basics = basic_settings()

    ylim = kwargs.get("ylim", basics["ylim"])
    xlim = kwargs.get("xlim", basics["xlim"])
    linewidth = kwargs.get("linewidth", basics["linewidth"]) 
    xlabel = kwargs.get("xlabel", "var")
    ylabel = kwargs.get("ylabel", "var")
    outdir = kwargs.get("outdir", "../Output")
    marker = kwargs.get("marker", basics["marker"])
    markersize = kwargs.get("markersize", basics["markersize"])
    capsize = kwargs.get("capsize", basics["capsize"])
    xticklabels = kwargs.get("xticklabels", None)
    group = kwargs.get("group", False)
    group_labels = kwargs.get("group_labels", [])
    elinewidth = kwargs.get("elinewidth", 0.5)
    dofit = kwargs.get("fit", False)
    dograd = kwargs.get("dograd", False)
    
    
    fig, ax = plt.subplots(figsize=basics["size"])
    hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

    for i,(leg,arr) in enumerate(data_dict.items()):
        _val = arr[:,0]
        err  = arr[:,1]
        val  = np.gradient(_val) if dograd else _val 

        if not group:        
            ax.errorbar(x,
                        val,
                        yerr=err,
                        elinewidth=elinewidth,
                        linewidth=linewidth,
                        marker=marker,
                        markersize=markersize,
                        color=basics["colors"][i],
                        label=leg,
                        capsize=capsize)
        else:
            group_len = int(len(val)/len(x))
            h_val = np.array(val[:,0]).reshape(len(x), group_len)
            h_err = np.array(val[:,1]).reshape(len(x), group_len)
            offset = 0.1

            #print(f"val : {h_val}")
            #print(f"err : {h_err}")

            for j in range(group_len):  # Two groups per x   
                plt.errorbar(np.array(x) + (j - 0.5) * offset * 2,  # offset: left and right
                             h_val[:, j],
                             yerr=h_err[:, j],
                             elinewidth=elinewidth,
                             linewidth=linewidth,
                             marker=marker,
                             markersize=markersize,
                             color=basics["colors"][2*i+j],
                             capsize=capsize,
                             label=group_labels[i][j])
            

            
    ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)
    ax.legend(fontsize=17,framealpha=1, facecolor='white')
        
    if xticklabels:
        ax.set_xticks(range(len(xticklabels)), labels=xticklabels,
                      rotation=45, ha="right", rotation_mode="anchor",
                      fontsize=15)
    else:
        ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
        
    ax.set_title(f"{title}", fontsize=18, loc='right')

    plt.tight_layout()
    fig.savefig(f"{outdir}/{name}.pdf", dpi=300)
    plt.close()

    

def hist_basic(data_dict, bins=None, title="", name="", **kwargs):
    basics = basic_settings()

    ylim = kwargs.get("ylim", basics["ylim"])
    lw = kwargs.get("linewidth", basics["linewidth"]) 
    xlabel = kwargs.get("xlabel", "var")
    ylabel = kwargs.get("ylabel", "a.u.")
    outdir = kwargs.get("outdir", "../Output")
    colors = kwargs.get("colors", basics["colors"])
    styles = kwargs.get("linestyles", basics["linestyles"])
    
    fig, ax = plt.subplots(figsize=basics["size"])
    hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS    

    for i,(leg,arr) in enumerate(data_dict.items()):
        val  = arr[:,0]
        err  = arr[:,1]    
        mean = np.mean(val)
        std  = np.std(val)
        label = f'{leg}\n(μ={mean:.2f}, σ={std:.2f})'
        ax.hist(val,
                bins=bins,
                histtype='step',
                linewidth=lw,
                color=colors[i],
                linestyle=styles[i],
                label=label)

    ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)
        
    ax.legend(fontsize=15,framealpha=1, facecolor='white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_ylim(ylim[0],ylim[1])
    ax.set_title(f"{title}", fontsize=18, loc='right')

    plt.tight_layout()
    fig.savefig(f"{outdir}/{name}.pdf", dpi=300)
    plt.close()




    
def plot_basic_2(x, hist_arrs, labels, title="", tag="", **kwargs):
    basics = basic_settings()

    ylim = kwargs.get("ylim", basics["ylim"])
    xlim = kwargs.get("xlim", basics["xlim"])
    linewidth = kwargs.get("linewidth", basics["linewidth"]) 
    xlabel = kwargs.get("xlabel", "var")
    ylabel = kwargs.get("ylabel", "var")
    outdir = kwargs.get("out", "../Output")
    marker = kwargs.get("marker", basics["marker"])
    markersize = kwargs.get("markersize", basics["markersize"])
    capsize = kwargs.get("capsize", basics["capsize"])
    xticklabels = kwargs.get("xticklabels", None)
    group = kwargs.get("group", False)
    group_labels = kwargs.get("group_labels", [])
    elinewidth = kwargs.get("elinewidth", 0.5)
    dofit = kwargs.get("fit", False)

    #model = lambda mu, sigma: 

    
    
    fig, ax = plt.subplots(figsize=basics["size"])
    #fig.subplots_adjust(left    = 0.12,
    #                    right   = 0.95,
    #                    top     = 0.89,
    #                    bottom  = 0.13,
    #                    hspace  = 0.5,
    #                    wspace  = 0.4)
    hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

    for i,h in enumerate(hist_arrs):
        # hists list has dictionaries
        # one with value and other with error
        if not group:
            ax.plot(x,
                    h,
                    #yerr=np.array(h["err"]),
                    #elinewidth=elinewidth,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize,
                    color=basics["colors"][i],
                    label=labels[i])
    
    ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)

    ax.legend(ncol=2, fontsize=17,framealpha=1, facecolor='white')
        
    if xticklabels:
        ax.set_xticks(range(len(xticklabels)), labels=xticklabels,
                      rotation=45, ha="right", rotation_mode="anchor",
                      fontsize=15)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
        
    ax.set_title(f"{title}", fontsize=18, loc='right')


    plt.tight_layout()
    fig.savefig(f"{outdir}/{title}_{tag}.pdf", dpi=300)
    plt.close()






    

def save_2d(hist, title="", tag="", **kwargs):
    outdir = kwargs.get("out", "../Output")
    c = ROOT.TCanvas("c", "c", 1200, 800)
    hist.Draw("COLZ")
    c.SaveAs(f"{outdir}/{title}_{tag}.pdf")


def save_1d(hist, title="", tag="", **kwargs):
    outdir = kwargs.get("out", "../Output")
    c = ROOT.TCanvas("c", "c", 1200, 800)
    hist.Draw("hist")
    c.SaveAs(f"{outdir}/{title}_{tag}.pdf")
    

def hist_2d(hist_arr, title="", tag="", **kwargs):
    basics = basic_settings()

    ylim = kwargs.get("ylim", basics["ylim"])
    xlabel = kwargs.get("xlabel", "var")
    ylabel = kwargs.get("ylabel", "a.u.")
    outdir = kwargs.get("out", "../Output")
    
    fig, ax = plt.subplots(figsize=basics["size"])
    hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

    cmap = plt.cm.viridis.copy()
    cmap.set_under('white')
    norm = Normalize(vmin=0.0001, vmax=1)
    
    #cax = ax.imshow(hist_arr, origin='lower', aspect='auto', cmap=cmap, interpolation='none', norm=norm)
    cax = ax.pcolormesh(hist_arr, cmap=cmap, norm=norm)
    fig.colorbar(cax, ax=ax, label='Occupancy')

    ax.set_ylim(ylim[0], ylim[1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('2D Histogram from ROOT File')

    plt.tight_layout()
    fig.savefig(f"{outdir}/{title}_{tag}.pdf", dpi=300)
    plt.close()
    

def fill_basic(x, hist_arrs, labels, title="", tag="", **kwargs):
    basics = basic_settings()

    ylim = kwargs.get("ylim", basics["ylim"])
    xlim = kwargs.get("xlim", basics["xlim"])
    linewidth = kwargs.get("linewidth", basics["linewidth"]) 
    xlabel = kwargs.get("xlabel", "var")
    ylabel = kwargs.get("ylabel", "var")
    outdir = kwargs.get("out", "../Output")
    marker = kwargs.get("marker", basics["marker"])
    markersize = kwargs.get("markersize", basics["markersize"])
    capsize = kwargs.get("capsize", basics["capsize"])
    xticklabels = kwargs.get("xticklabels", None)
    group = kwargs.get("group", False)
    group_labels = kwargs.get("group_labels", [])
    elinewidth = kwargs.get("elinewidth", 0.5)
    dofit = kwargs.get("fit", False)

    #model = lambda mu, sigma: 

    
    
    fig, ax = plt.subplots(figsize=basics["size"])
    #fig.subplots_adjust(left    = 0.12,
    #                    right   = 0.95,
    #                    top     = 0.89,
    #                    bottom  = 0.13,
    #                    hspace  = 0.5,
    #                    wspace  = 0.4)
    hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

    for i,h in enumerate(hist_arrs):
        y1 = np.array(h["begin"])
        y2 = np.array(h["end"])
        y2 = np.where(y2 > (y1+10.0), y1+0.1, y2)
        
        ax.fill_between(np.array(x),
                        y1,
                        y2,
                        linewidth=linewidth,
                        color=basics["colors"][i],
                        label=labels[i],
                        alpha=0.3)
        
    ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)
    ax.legend(ncol=2, fontsize=17,framealpha=1, facecolor='white')
        
    if xticklabels:
        ax.set_xticks(range(len(xticklabels)), labels=xticklabels,
                      rotation=45, ha="right", rotation_mode="anchor",
                      fontsize=15)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
        
    ax.set_title(f"{title}", fontsize=18, loc='right')


    plt.tight_layout()
    fig.savefig(f"{outdir}/{title}_{tag}.pdf", dpi=300)
    plt.close()
