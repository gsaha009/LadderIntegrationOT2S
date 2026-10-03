# Plotter function
# Ladder Integration at IPHC
# Author: G.Saha

import os
import sys
import yaml
import importlib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import mplhep as hep
hep.style.use("CMS")

import logging
logger = logging.getLogger('main')

import ROOT
ROOT.gROOT.SetBatch(True)
import CMS_lumi
from tdrstyle import *

ROOT.gStyle.SetImageScaling(3.0)



class Plotter:
    def __init__(self,
                 testinfo: dict,
                 data: dict,
                 outdirP: str,
                 outdirF: str,
                 ladpos: str,
                 ladid: str,
                 **kwargs):
        self.testinfo = testinfo
        self.data     = data
        self.outdir   = outdirP
        self.outdirF  = outdirF
        self.markerStyle  = 20
        self.markerSize   = 0.6
        self.markerSizeIV = 1
        self.lineWidth    = 2
        self.timeDivision = 503
        self.linecolor    = ROOT.kBlue
        self.ladpos   = ladpos
        self.ladid    = ladid

    def __basic_settings(self):
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
            "markerstyles": [
                'o', 's', 'D', '*', '^', 'v', 'P', 'X', '<', '>',
                'h', '8', 'p', 'H', '+', 'x', 'd', '|', '_', '.'
            ],
            "colors": [
                "#165a86","#cc660b","#217a21","#a31e1f","#6e4e92","#6b443e",
                "#b85fa0","#666666","#96971b","#1294a6","#8c1c62","#144d3a",
                "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33",
                "#a65628","#f781bf"
            ],
            "linestyles": [
                "-", "--", "-.", ":",
                (0, (3, 1)),
                (0, (3, 1, 1, 1)),
                (0, (5, 5)),
                (0, (1, 1)),
                (0, (6, 2)),
                (0, (4, 2, 1, 2)),
                (0, (2, 2)),
                (0, (7, 3)),
                (0, (5, 1)),
                (0, (3, 2, 1, 2)),
                (0, (8, 2)),
                (0, (2, 1)),
                (0, (1, 2)),
                (0, (9, 3)),
                (0, (4, 4)),
                (0, (6, 1, 2, 1))
            ]}

        
    def __setGraphStyle(self, graph, **kwargs):
        graph.SetMarkerStyle(self.markerStyle)
        graph.SetMarkerSize(self.markerSize)
        graph.SetLineColor(kwargs.get('lincol'))
        graph.SetLineWidth(self.lineWidth)
        graph.SetLineStyle(ROOT.kSolid)
        # Configure x-axis to display time correctly
        #graph.GetXaxis().SetTimeDisplay(1)
        #graph.GetXaxis().SetNdivisions(self.timeDivision)
        #graph.GetXaxis().SetTimeFormat(self.TIME_FORMAT)
        #graph.GetXaxis().SetTimeOffset(0)

    
    def draw_pad_title(self, text,pos=[0.35, 0.92]):
        latex = ROOT.TLatex()
        latex.SetNDC(True)
        latex.SetTextAlign(13)     # left-top
        latex.SetTextFont(42)      # CMS font
        latex.SetTextSize(0.045)
        latex.DrawLatex(pos[0], pos[1], text)
            

    def store_modIDs(self,
                     modIDlist = None,
                     tdir  = None):
        labels = ROOT.TObjString(",".join(modIDlist))
        tdir.WriteObject(labels, f"OpticalGroups_ModuleIDs")

        
    def to_root_temp(self,
                     x = None,
                     y = None,
                     yerr = None,
                     title = "Default",
                     name  = "Default",
                     tdir  = None,
                     **kwargs):
        xlabel = kwargs.get("xlabel", "var")
        ylabel = kwargs.get("ylabel", "var")
        lincol = kwargs.get("lincol", ROOT.kBlue)
        
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        yerr = np.ascontiguousarray(yerr if yerr is not None else np.zeros_like(y), dtype=np.float64)
        #x = np.asarray(x, dtype=np.float64)
        #y = np.asarray(y, dtype=np.float64)
        #yerr = np.asarray(yerr, dtype=np.float64) if yerr is not None else np.zeros_like(y)
        xerr = np.zeros_like(x)

        # Create graph
        gr = ROOT.TGraphErrors(len(x), x, y, xerr, yerr)
        gr.SetName(f"{name}")
        gr.SetTitle(f"{title};{xlabel};{ylabel}")

        gr.GetXaxis().SetLimits(-0.5, 11.5)
        
        if not tdir:
            raise Exception("ROOT Dir <tdir> must be valid")

        self.__setGraphStyle(gr, lincol=lincol)
        
        tdir.cd()
        gr.Write()

        
    def plot_basic(self,
                   x = None,
                   data_list = None,
                   legends = None,
                   title = "Default",
                   name = "Default",
                   **kwargs):

        basics = self.__basic_settings()

        ylim = kwargs.get("ylim", basics["ylim"])
        xlim = kwargs.get("xlim", basics["xlim"])
        linewidth = kwargs.get("linewidth", basics["linewidth"]) 
        xlabel = kwargs.get("xlabel", "var")
        ylabel = kwargs.get("ylabel", "var")
        outdir = kwargs.get("outdir", "../Output")
        marker = kwargs.get("marker", basics["marker"])
        markersize = kwargs.get("markersize", basics["markersize"])
        markerfacecolor = kwargs.get("markerfacecolor", None)
        markeredgewidth = kwargs.get("markeredgewidth", basics['markeredgewidth'])
        capsize = kwargs.get("capsize", basics["capsize"])
        xticklabels = kwargs.get("xticklabels", None)
        group = kwargs.get("group", False)
        group_labels = kwargs.get("group_labels", [])
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
        ncols = kwargs.get("ncols", 2)

        errlow = kwargs.get("errlow", False)
        errhigh = kwargs.get("errhigh", False)
        
        #inROOTDir = kwargs.get("inROOTDir", None)
        #if inROOTDir:
            
        
        fig, ax = plt.subplots(figsize=basics["size"])
        hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

        for i,data in enumerate(data_list):
            data = np.array(data)
            _val = data[:,0]
            err  = data[:,1]
            if errlow:
                err = np.array([err,np.zeros_like(err)])
            elif errhigh:
                err = np.array([np.zeros_like(err),err])
            
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
                        label=legends[i],
                        capsize=capsize)

            
            if dofit:
                params = self.__get_mean_std_for_cmn(x, val)
                mean = params[0]
                sigma = params[1]
                label_text = f"{legends[i]} (µ = {round(mean,1)}, σ = {round(sigma,1)})"
                

                mask = val > 0.0
                x_filtered = x[mask]
                val_filtered = val[mask]
                err_filtered = err[mask]
                
                fitobj = Fitter(x_filtered, val_filtered, err_filtered)
                result = fitobj.result
                #from IPython import embed; embed(); exit()
                

                if result is not None:
                    fit_val, fit_label_text = result
                    ax.plot(x_filtered, fit_val, color=colors[i], linewidth=fitlinewidth)
                    label_text = f"{label_text}\n{fit_label_text}"

                leg_handle = Line2D([0], [0], color=colors[i], label=label_text) #label=f"{legends[i]}")
                leg = ax.legend(handles=[leg_handle],
                                #title=":".join(fit_info),
                                frameon=False,
                                loc=1, bbox_to_anchor=(1, 1 - (i*0.11)), fontsize=11, title_fontsize=12)
                plt.gca().add_artist(leg)
            else:
                ax.legend(fontsize=12, framealpha=1, facecolor='white', ncols=ncols)
                
        ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)
        
        if xticklabels:
            if nticks is not None:
                tick_count = nticks
                tick_locs = np.linspace(0, x.shape[0]-1, tick_count, dtype=int)
                tick_labels = [xticklabels[i] for i in tick_locs]
                #print(tick_locs, tick_labels)
                ax.set_xticks(tick_locs)
                ax.set_xticklabels(tick_labels, rotation=90, ha="right", rotation_mode="anchor", fontsize=13)
            else:                
                ticks_ = [x-tick_offset for x in list(range(len(xticklabels)))]
                ax.set_xticks(ticks_)
                ax.set_xticklabels(xticklabels, rotation=90, ha='right', rotation_mode='anchor', fontsize=13)
                #ax.set_xticks(range(len(xticklabels)), labels=xticklabels,
                #              rotation=90, ha="right", rotation_mode="anchor",
                #              fontsize=13)
        else:
            ax.set_xlabel(xlabel)

        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim[0],ylim[1])
        if xlim:
            ax.set_xlim(xlim[0],xlim[1])
        
        ax.set_title(f"{title}", fontsize=14, loc='right')

        plt.tight_layout()
        fig.savefig(f"{outdir}/{name}.{self.testinfo.get('plot_extn')}", dpi=self.testinfo.get('plot_dpi'))
        plt.close()


    def plot_basic_from_dict(self,
                             data = None,
                             yerr = None,
                             legends = None,
                             title = "Default",
                             name = "Default",
                             **kwargs):

        basics = self.__basic_settings()

        ylim = kwargs.get("ylim", basics["ylim"])
        xlim = kwargs.get("xlim", basics["xlim"])
        linewidth = kwargs.get("linewidth", basics["linewidth"])
        xlabel = kwargs.get("xlabel", "var")
        ylabel = kwargs.get("ylabel", "var")
        outdir = kwargs.get("outdir", "../Output")
        marker = kwargs.get("marker", basics["marker"])
        markersize = kwargs.get("markersize", basics["markersize"])
        markerfacecolor = kwargs.get("markerfacecolor", None)
        markeredgewidth = kwargs.get("markeredgewidth", basics['markeredgewidth'])
        capsize = kwargs.get("capsize", basics["capsize"])
        xticklabels = kwargs.get("xticklabels", None)
        elinewidth = kwargs.get("elinewidth", 0.5)
        colors = kwargs.get("colors", basics["colors"])
        markerstyles = kwargs.get("markerstyles", basics["markerstyles"])
        linestyles = kwargs.get("linestyles", basics["linestyles"])
        nticks = kwargs.get("nticks", None)
        tick_offset = kwargs.get("tick_offset", 0.1)


        fig, ax = plt.subplots(figsize=basics["size"])
        hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS
        
        for i,(key,val) in enumerate(data.items()):
            err  = np.zeros_like(val) if yerr is None else yerr[key]

            x = np.arange(val.shape[0])
            ax.errorbar(x,
                        val,
                        yerr=err,
                        fmt = 'o',
                        elinewidth=elinewidth,
                        linewidth=linewidth,
                        linestyle='-',
                        #marker=marker,
                        markersize=markersize,
                        markerfacecolor=colors[i] if markerfacecolor is not None else 'none',
                        markeredgewidth = markeredgewidth,
                        color=colors[i],
                        label=key,
                        capsize=capsize)

        ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)
        ax.legend(fontsize=12, framealpha=1, facecolor='white', ncols=3)

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
        fig.savefig(f"{outdir}/{name}.{self.testinfo.get('plot_extn')}", dpi=self.testinfo.get('plot_dpi'))
        plt.close()
            
            

    def plot_group(self,
                   x = None,
                   data_list = None,
                   legends = None,
                   title = "Default",
                   name = "Default",
                   **kwargs):

        basics = self.__basic_settings()

        ylim = kwargs.get("ylim", basics["ylim"])
        xlim = kwargs.get("xlim", basics["xlim"])
        linewidth = kwargs.get("linewidth", basics["linewidth"]) 
        xlabel = kwargs.get("xlabel", "var")
        ylabel = kwargs.get("ylabel", "var")
        outdir = kwargs.get("outdir", "../Output")
        marker = kwargs.get("marker", basics["marker"])
        markerfacecolor = kwargs.get("markerfacecolor", None)
        markeredgewidth = kwargs.get("markeredgewidth", basics['markeredgewidth'])
        markersize = kwargs.get("markersize", basics["markersize"])
        capsize = kwargs.get("capsize", basics["capsize"])
        xticklabels = kwargs.get("xticklabels", None)
        elinewidth = kwargs.get("elinewidth", 0.5)
        dofit = kwargs.get("fit", False)
        dograd = kwargs.get("dograd", False)
        markerstyles = kwargs.get("markerstyles", basics["markerstyles"])
        linestyles = kwargs.get("linestyles", basics["linestyles"])
        colors = kwargs.get("colors", basics["colors"])
        tick_offset = kwargs.get("tick_offset", 0.1)
        
        fig, ax = plt.subplots(figsize=basics["size"])
        hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

        #from IPython import embed; embed()
        
        for i, group_data in enumerate(data_list):
            group_data = np.array(group_data)
            group_val = group_data[:,:,0]
            group_err = group_data[:,:,1]
            offset = 0.1
            for j in range(group_data.shape[0]):
                ax.errorbar(x + (j - 0.5) * offset * 2,
                            group_val[j],
                            yerr=group_err[j],
                            fmt = markerstyles[i],
                            elinewidth=elinewidth,
                            linewidth=linewidth,
                            #marker=marker,
                            markersize=markersize,
                            markerfacecolor=colors[i] if markerfacecolor is not None else 'none',
                            color=colors[2*i+j],
                            label=legends[i][j],
                            capsize=capsize)
            
        ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)
        ax.legend(fontsize=13, framealpha=1, facecolor='white', ncols=2)
        
        if xticklabels:
            #from IPython import embed; embed(); exit()
            ticks_ = [x-tick_offset for x in list(range(len(xticklabels)))]
            ax.set_xticks(ticks_)
            #ax.set_xticks(range(len(xticklabels)), labels=xticklabels,
            #              rotation=90, ha="right", rotation_mode="anchor",
            #              fontsize=13)
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
        fig.savefig(f"{outdir}/{name}.{self.testinfo.get('plot_extn')}", dpi=self.testinfo.get('plot_dpi'))
        plt.close()



    """
    def plot_group(self,
                   x = None,
                   data_list = None,
                   legends = None,
                   title = "Default",
                   name = "Default",
                   **kwargs):

        basics = self.__basic_settings()

        ylim = kwargs.get("ylim", basics["ylim"])
        xlim = kwargs.get("xlim", basics["xlim"])
        linewidth = kwargs.get("linewidth", basics["linewidth"]) 
        xlabel = kwargs.get("xlabel", "var")
        ylabel = kwargs.get("ylabel", "var")
        outdir = kwargs.get("outdir", "../Output")
        marker = kwargs.get("marker", basics["marker"])
        markerfacecolor = kwargs.get("markerfacecolor", None)
        markeredgewidth = kwargs.get("markeredgewidth", basics['markeredgewidth'])
        markersize = kwargs.get("markersize", basics["markersize"])
        capsize = kwargs.get("capsize", basics["capsize"])
        xticklabels = kwargs.get("xticklabels", None)
        elinewidth = kwargs.get("elinewidth", 0.5)
        dofit = kwargs.get("fit", False)
        dograd = kwargs.get("dograd", False)
        markerstyles = kwargs.get("markerstyles", basics["markerstyles"])
        linestyles = kwargs.get("linestyles", basics["linestyles"])
        colors = kwargs.get("colors", basics["colors"])
        tick_offset = kwargs.get("tick_offset", 0.1)
        
        fig, ax = plt.subplots(figsize=basics["size"])
        hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

        for i, group_data in enumerate(data_list):
            group_data = np.array(group_data)
            group_val = group_data[:,:,0]
            group_err = group_data[:,:,1]
            offset = 0.1
            for j in range(group_data.shape[0]):
                ax.errorbar(x + (j - 0.5) * offset * 2,
                            group_val[j],
                            yerr=group_err[j],
                            fmt = markerstyles[i],
                            elinewidth=elinewidth,
                            linewidth=linewidth,
                            #marker=marker,
                            markersize=markersize,
                            markerfacecolor=colors[i] if markerfacecolor is not None else 'none',
                            color=colors[2*i+j],
                            label=legends[i][j],
                            capsize=capsize)
            
        ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)
        ax.legend(fontsize=13, framealpha=1, facecolor='white', ncols=2)
        
        if xticklabels:
            #from IPython import embed; embed(); exit()
            ticks_ = [x-tick_offset for x in list(range(len(xticklabels)))]
            ax.set_xticks(ticks_)
            #ax.set_xticks(range(len(xticklabels)), labels=xticklabels,
            #              rotation=90, ha="right", rotation_mode="anchor",
            #              fontsize=13)
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
        fig.savefig(f"{outdir}/{name}.{self.testinfo.get('plot_extn')}", dpi=self.testinfo.get('plot_dpi'))
        plt.close()
    """
        
        
    def plot_box(self,
                 x = None,
                 data_list_1 = None,
                 data_list_2 = None,
                 legends = None,
                 title = "Default",
                 name = "Default",
                 **kwargs):

        xticklabels = kwargs.get("xticklabels", None)
        ylabel      = kwargs.get("ylabel", "var")
        outdir      = kwargs.get("outdir", "../Output")
        offset      = kwargs.get("box_offset", 0.1)
        
        if xticklabels is None:
            xticklabels = list(range(len(x)))
        
        noise_1 = np.array(data_list_1)
        noise_val_1 = noise_1[:,:,0] if noise_1.ndim == 3 else noise_1[:,0]
        noise_2 = np.array(data_list_2)
        noise_val_2 = noise_2[:,:,0] if	noise_2.ndim ==	3 else noise_2[:,0]

        basics = self.__basic_settings()

        color1 = basics["colors"][0]
        color2 = basics["colors"][1]

        boxprops1 = dict(linestyle='-', linewidth=1.3, color=color1)
        boxprops2 = dict(linestyle='-', linewidth=1.3, color=color2)
        
        positions1 = np.arange(1, len(xticklabels) + 1) * 2    # positions for first set, spaced by 2 units
        positions2 = positions1 + 0.8                          # positions for second set, shifted by 0.8

        fig, ax = plt.subplots(figsize=basics["size"])
        hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS
        
        bp1 = ax.boxplot(noise_val_1.tolist(),
                         boxprops=dict(color=color1, linewidth=1.5),
                         medianprops=dict(color=color1),
                         whiskerprops=dict(color=color1),
                         capprops=dict(color=color1),
                         flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor=color1, markersize=5),
                         positions=positions1,
                         widths=0.6,
                         patch_artist=True)
        for box in bp1['boxes']:
            box.set(facecolor='none') 
            
        bp2 = ax.boxplot(noise_val_2.tolist(),
                         boxprops=dict(color=color2, linewidth=1.5),
                         medianprops=dict(color=color2),
                         whiskerprops=dict(color=color2),
                         capprops=dict(color=color2),
                         flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor=color2, markersize=5),
                         positions=positions2,
                         widths=0.6,
                         patch_artist=True)
        for box in bp2['boxes']:
            box.set(facecolor='none') 

        middle_positions = (positions1 + positions2) / 2
        middle_positions = middle_positions - offset
        ax.set_xticks(middle_positions)
        ax.set_xticklabels(xticklabels, rotation=90, ha='right', rotation_mode='anchor', fontsize=15)

        ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)
        
        # Custom legend
        legend_patches = [
            Patch(facecolor='none', edgecolor=color1, linewidth=1.5, label=legends[0]),
            Patch(facecolor='none', edgecolor=color2, linewidth=1.5, label=legends[1])
        ]
        ax.legend(handles=legend_patches, fontsize=13, framealpha=1, facecolor='white')
        ax.set_ylabel(ylabel)

        ax.set_title(f"{title}", fontsize=14, loc='right')
        plt.tight_layout()
        fig.savefig(f"{outdir}/{name}.{self.testinfo.get('plot_extn')}", dpi=self.testinfo.get('plot_dpi'))
        plt.close()



    def plot_heatmap(self,
                     data = None,
                     title = "Default",
                     name = "Default",
                     **kwargs):

        annotint = kwargs.get('annotint', False)
        norm = kwargs.get('norm', None)
        
        if isinstance(data, np.ndarray):
            data = np.abs(data).T
        elif isinstance(data, list):
            data = np.abs(np.array(data)).T
        else:
            raise RuntimeError("Wrong data format ...")
            
        #from IPython import embed; embed(); exit() 
        basics = self.__basic_settings()

        xticklabels = kwargs.get("xticklabels", None)
        yticklabels = kwargs.get("yticklabels", None)
        colmap = kwargs.get("colmap", "viridis")
        outdir = kwargs.get("outdir", "../Output")
        vmin = kwargs.get("vmin", None)
        vmax = kwargs.get("vmax", None)
        cbar_label = kwargs.get("cb_label", "CMNoise fraction")
        
        fig, ax = plt.subplots(figsize=basics["size"])
        hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

        if vmin is None:
            vmin = float(np.min(data))
                
        #threshold = np.percentile(data, 99)
        #masked_data = np.ma.masked_greater(data, threshold)
        #from IPython import embed; embed(); exit()
        
        if data.shape[0] == 17:
            mask = np.zeros_like(data, dtype=bool)
            mask[8, :] = True  # exclude SEH row from scaling
            #masked_data = np.ma.masked_array(masked_data, mask)
            masked_data = np.ma.masked_array(data, mask)
        else:
            masked_data = data
            
        cmap=plt.cm.get_cmap(colmap).copy()
        cmap.set_bad(color="#FFFFFF") 
        
        #vmax = float(np.percentile(data, 95))
        if vmax is None:
            #vmax=threshold
            vmax=float(np.max(data))

        if norm:
            im = ax.imshow(masked_data, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        else:
            im = ax.imshow(masked_data, cmap=cmap, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)  
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(xticklabels)
        if yticklabels is None:
            yticklabels = [f'CBC_{i}' for i in range(8)]
        ax.set_yticklabels(yticklabels)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        # Annotate each cell with the value
        """
        threshold2 = np.percentile(data, 50)
        for i in range(data.shape[0]):
            if i == 8: continue
            for j in range(data.shape[1]):
                if data[i, j] <= threshold:
                    ax.text(
                        j, i, f"{data[i, j]:.2f}",  # format with 2 decimals
                        ha="center", va="center",
                        color="white" if data[i, j] < threshold2 else "black",
                        fontsize=10
                    )
                else:
                    ax.text(
                        j, i, f"{data[i, j]:.2f}",
                        ha="center", va="center",
                        color="black", fontsize=9, fontweight="bold"  # highlight outlier text
                    )
        """
        #threshold2 = np.percentile(data, 50)
        threshold2 = np.median(data)
        for i in range(data.shape[0]):
            if (data.shape[0] == 17) & (i == 8): continue
            for j in range(data.shape[1]):
                num = float(data[i, j])
                if annotint:
                    num = int(num)
                else:
                    num = round(num, 2)
                ax.text(
                    j, i, f"{num}",  # format with 2 decimals
                    ha="center", va="center",
                    #color="black" if data[i, j] < threshold2 else "white",
                    color="black",
                    fontsize=8,
                )
        
        ax.set_title(f"{title}", fontsize=14, loc='right')
        ax.tick_params(direction="in", top=False, right=False, labelsize=12, length=3)
        plt.tight_layout()
        fig.savefig(f"{outdir}/{name}.{self.testinfo.get('plot_extn')}",
                    dpi=self.testinfo.get('plot_dpi'),
                    bbox_inches="tight")
        plt.close()


    def plot_fitted_result(self,
                           x : list[np.array],
                           y : list[np.array],
                           labels : list[str],
                           title = "default",
                           name = "default",
                           **kwargs):
        

        assert len(x) == 3, "max x entries must be 3"
        assert len(y) == 3, "max y entries must be 3"
        
        basics = self.__basic_settings()
        outdir = kwargs.get("outdir", "../Output")
        h3_w = kwargs.get("w", None)
        
        fig, axes = plt.subplots(1,3, figsize=(14,8))
        #hep.cms.text(basics["heplogo"], loc=0) # CMS
        fig.text(0.05, 0.97, "CMS", fontsize=40, fontweight='bold', ha='left', va='top')
        fig.text(0.14, 0.97, "Internal", fontsize=30, style='italic', ha='left', va='top')

        fig.text(0.95, 0.97, f"{title}", fontsize=20, ha='right', va='top')
        #plt.title(f"{title}", fontsize=13, loc='right')

        plt.subplots_adjust(
            left=0.08,   # reduce left margin
            right=0.98,  # reduce right margin
            top=0.80,    # reduce top margin
            bottom=0.08, # reduce bottom margin
            wspace=0.25, # horizontal spacing between subplots
            hspace=0.35  # vertical spacing
        )
        
        axes[0].bar(x[0][0], y[0][0], alpha=0.6, label="Observed")
        axes[0].plot(x[0][1], y[0][1], 'r-', lw=2, label="Fitted")
        axes[0].set_title(labels[0])
        axes[0].legend()

        axes[1].bar(x[1][0], y[1][0], alpha=0.6, label="Observed", color="orange")
        axes[1].plot(x[1][1], y[1][1], 'r-', lw=2, label="Fitted")
        axes[1].set_title(labels[1])
        axes[1].set_xlim((0,10))
        axes[1].legend()

        axes[2].bar(x[2][0], y[2][0], width=h3_w, alpha=0.5, label="Fitted")
        axes[2].set_title(labels[2])
        #axes[2].set_yscale('log')
        axes[2].legend()
        
        
        #plt.tight_layout(pad=0.7, w_pad=0.8, h_pad=0.8)
        fig.savefig(f"{outdir}/{name}.{self.testinfo.get('plot_extn')}",
                    dpi=self.testinfo.get('plot_dpi'))
        plt.close()


        
    def hist_basic(self,
                   bins = None,
                   data_list = None,
                   legends = None,
                   title = "Default",
                   name = "Default",
                   **kwargs):
        
        basics = self.__basic_settings()

        lw = kwargs.get("linewidth", basics["linewidth"]) 
        xlabel = kwargs.get("xlabel", "var")
        ylabel = kwargs.get("ylabel", "a.u.")
        outdir = kwargs.get("outdir", "../Output")
        colors = kwargs.get("colors", basics["colors"])
        styles = kwargs.get("linestyles", basics["linestyles"])
        
        fig, ax = plt.subplots(figsize=basics["size"])
        hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS    

        #bins = np.array(bins)
        for i,data in enumerate(data_list):
            data = np.array(data)
            val  = data[:,0]
            err  = data[:,1]    
            leg  = legends[i]
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
        
        ax.legend(fontsize=13, framealpha=1, facecolor='white')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_title(f"{title}", fontsize=14, loc='right')
        
        plt.tight_layout()
        fig.savefig(f"{outdir}/{name}.{self.testinfo.get('plot_extn')}", dpi=self.testinfo.get('plot_dpi'))
        plt.close()



    """
    def plot_ROOT_2Dhist(self, name="default", title="default",
                         hist=None, **kwargs):

        outdir = kwargs.get("outdir", "../Output")
        
        
        #ROOT.gROOT.LoadMacro("tdrstyle.C")
        #setTDRStyle()
        #setCMSText()
        
        #ROOT.gROOT.LoadMacro("CMS_lumi.C")

        ROOT.gStyle.SetPalette(ROOT.kViridis)

        CANVAS_W = 800
        CANVAS_H = 700

        c = ROOT.TCanvas(f"c_{name}", name, CANVAS_W, CANVAS_H)
        
        # CMS margins (important!)
        c.SetLeftMargin(0.12)
        c.SetRightMargin(0.16)   # space for COLZ
        c.SetBottomMargin(0.12)
        c.SetTopMargin(0.08)
        
        #hist.SetTitle("")           # CMS uses external labels
        hist.Draw("COLZ")

        # Add CMS lumi/label on top-left
        #ROOT.CMS_lumi.writeExtraText = True
        #ROOT.CMS_lumi.extraText = "Private"

        #c.Update()
        c.SaveAs(f"{outdir}/{name}.png")
        c.Close()
    """
        

    def plot_ROOT_2Dhist(self,
                         hist_dict=None,
                         name="default",
                         title="default",
                         **kwargs):

        outdir = kwargs.get("outdir", "../Output")
        nrows  = kwargs.get("nRows", None)
        ncols  = kwargs.get("nCols", None)
        ptype  = kwargs.get("pType", "COLZ TEXT")
        
        #ROOT.gROOT.LoadMacro("tdrstyle.C")
        setTDRStyle()
        #setCMSText()
        
        #ROOT.gROOT.LoadMacro("CMS_lumi.C")
        CMS_lumi.writeExtraText = False   # just "CMS"
        CMS_lumi.lumi_sqrtS = ""          # no lumi text
        
        #ROOT.gStyle.SetPalette(ROOT.kViridis)

        #CANVAS_W = 800
        #CANVAS_H = 700

        #nHists = len(hist_dict)

        #if nHists > 3:
        #    ncols = nHists // 2
        #    nrows = nHists - ncols
        #else:
        #    ncols = nHists
        #    nrows = 1
            
        # base pad size (tune this)
        pad_w = 800
        pad_h = 700
        
        canvas_w = pad_w * ncols
        canvas_h = pad_h * nrows
        

        c = ROOT.TCanvas("c", "", canvas_w, canvas_h)
        
        c.SetLeftMargin(0.12)
        c.SetRightMargin(0.16)
        c.SetBottomMargin(0.18)
        c.SetTopMargin(0.18)
        
        c.Divide(ncols, nrows, 0.01, 0.01)

            
        for i, (key, hist) in enumerate(hist_dict.items()):
        
            c.cd(i+1)
            ROOT.gPad.SetRightMargin(0.15)
            #hist.Draw("COLZ") if i > 0 else hist.Draw("COLZ TEXT")
            hist.Draw(ptype)
            self.draw_pad_title(key,pos=[0.15,0.99])



        # ---- CMS label tuning ----
        CMS_lumi.cmsTextSize = 0.4
        CMS_lumi.extraTextSize = 0.2
        CMS_lumi.relPosX = 0.01
        CMS_lumi.relPosY = 0.01
        CMS_lumi.writeExtraText = True
        CMS_lumi.extraText = "Internal"

            
        #c.cd()  # go back to canvas (not a pad)
        #CMS_lumi.CMS_lumi(c, 0, 0)
        
        c.SaveAs(f"{outdir}/{name}.png")
        c.Close()

        
        

    def plot_ROOT_2Dhist_AllCBCs(self, name="default", title="default",
                                 hist_dict=None, **kwargs):

        outdir = kwargs.get("outdir", "../Output")
        
        
        #ROOT.gROOT.LoadMacro("tdrstyle.C")
        setTDRStyle()
        #setCMSText()
        
        #ROOT.gROOT.LoadMacro("CMS_lumi.C")
        CMS_lumi.writeExtraText = False   # just "CMS"
        CMS_lumi.lumi_sqrtS = ""          # no lumi text
        
        #ROOT.gStyle.SetPalette(ROOT.kViridis)

        #CANVAS_W = 800
        #CANVAS_H = 700

        c = ROOT.TCanvas(f"c", "", 3200, 920)
        
        # CMS margins (important!)
        #c.SetLeftMargin(0.12)
        #c.SetRightMargin(0.16)   # space for COLZ
        #c.SetBottomMargin(0.12)
        #c.SetTopMargin(0.52)

        c.Divide(8,2,0.01,0.01)

        for i in range(8):
            c.cd(i+1)
            ROOT.gPad.SetRightMargin(0.15)
            hist_dict['Hybrid_0'][f'CBC_{i}'].Draw("COLZ")
            self.draw_pad_title(f"Hybrid 0 - CBC {i}")
        for i in range(8):
            c.cd(i+9)
            ROOT.gPad.SetRightMargin(0.15)
            hist_dict['Hybrid_1'][f'CBC_{7-i}'].Draw("COLZ")
            self.draw_pad_title(f"Hybrid 1 - CBC {7-i}")



        # ---- CMS label tuning ----
        CMS_lumi.cmsTextSize = 0.4
        CMS_lumi.extraTextSize = 0.2
        CMS_lumi.relPosX = 0.01
        CMS_lumi.relPosY = 0.01
        CMS_lumi.writeExtraText = True
        CMS_lumi.extraText = "Internal"

            
        c.cd()  # go back to canvas (not a pad)
        CMS_lumi.CMS_lumi(c, 0, 0)
        
        c.SaveAs(f"{outdir}/{name}.png")
        c.Close()
