# Plotter function
# Ladder Integration at IPHC
# Author: G.Saha

import os
import sys
import yaml
import importlib
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import mplhep as hep
hep.style.use("CMS")

from Fitter import Fitter
#from modules.2SLadderCMNAna.CMNFitter import *
CMNmod = importlib.import_module("modules.2SLadderCMNAna.CMNFitter")

import logging
logger = logging.getLogger('main')


import ROOT
#from CMSStyle import setTDRStyle,setCMSText
ROOT.gROOT.SetBatch(True)
import CMS_lumi, tdrstyle

tdrstyle.setTDRStyle()
CMS_lumi.writeExtraText = 1
CMS_lumi.extraText = "Preliminary"




class Plotter:
    def __init__(self,
                 testinfo: dict,
                 data: dict,
                 outdir: str,
                 **kwargs):
        self.testinfo = testinfo
        self.data     = data
        self.outdir   = outdir

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
                "markerstyles": ['o', 's', 'D', '*', '^', 'v', 'P', 'X', '<', '>'],
                "colors": ["#165a86","#cc660b","#217a21","#a31e1f","#6e4e92","#6b443e","#b85fa0","#666666","#96971b","#1294a6","#8c1c62","#144d3a"],
                "linestyles": ["-","--","-.",":","(0, (3, 1))", "(0, (3, 1, 1, 1))","(0, (5, 5))","(0, (1, 1))","(0, (6, 2))","(0, (4, 2, 1, 2))"]}


    def __get_mean_std_for_cmn(self, x, val, axis=0):
        mean = np.average(x, weights=val, axis=axis)
        sigma = np.sqrt(np.average((x - mean)**2, weights=val, axis=axis))
        return [round(mean,2), round(sigma,2)]
    
    def _get_mean_std_for_cmn_per_module(self, arr):
        x = np.arange(arr.shape[-1])
        arr = arr.tolist()
        outlist = []
        for item in arr:
            _arr = np.array(item)
            mean_sigma = self.__get_mean_std_for_cmn(x, _arr)
            mean = mean_sigma[0]
            sigma = mean_sigma[1]
            outlist.append([float(mean), float(sigma)])

        return np.array(outlist)

    
    def	__extractCMN_iphc(self, nchannels=None, mean=None, std=None):
        """
        Ref: https://indico.cern.ch/event/1465528/contributions/6170035/attachments/2949439/5184080/systemtest_1710_JT.pdf
        """
        alpha = 2*np.pi*(std**2-mean*(1-mean/nchannels))/(nchannels*(nchannels-1))
        cmn = np.sqrt(np.sin(alpha)/(1-np.sin(alpha)))
        return cmn
        
    def __extractCMN_giovanni(self, nchannels=None, mean=None, std=None):
        """
        Ref: CMNoiseFraction ( Giovanni's Note )
        """
        cmn = np.sqrt((1/(nchannels - 1)) * (std**2/(mean*(1 - (mean/nchannels))) - 1))
        return cmn
    
    def __extractCMN(self, nchannels=None, mean=None, std=None):
        cmn = self.__extractCMN_iphc(nchannels=nchannels, mean=mean, std=std)*100
        return np.concatenate((cmn[:,None], np.zeros_like(cmn)[:,None]), axis=1)

    def __extractCMN_crude(self, nchannels=None, mean=None, std=None):
        std_expected = np.sqrt(mean)/2.0
        cmn = (std - std_expected)/std
        return cmn

    def __extractCMN_potato(self, hitsarr):
        val = np.array(hitsarr)[:,0]
        nch = float(val.shape[0])
        ch  = np.arange(nch)
        mask = ((ch < int(nch*0.2)) | (ch > int(nch*0.8)))
        val_pass = val[mask]
        cmn_frac = float(np.sum(val_pass)/np.sum(val))
        return cmn_frac

    
    def __mask_cmn_frac(self, cmn):
        cmn = np.abs(np.array(cmn))
        cmn_min = np.min(cmn, axis=1)
        #cmn_min = np.broadcast()
        pass
        
    
    def plot_heatmap(self,
                     data = None,
                     title = "Default",
                     name = "Default",
                     **kwargs):

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
            if i == 8: continue
            for j in range(data.shape[1]):
                ax.text(
                    j, i, f"{data[i, j]:.2f}",  # format with 2 decimals
                    ha="center", va="center",
                    #color="black" if data[i, j] < threshold2 else "white",
                    color="black",
                    fontsize=8,
                )
        
        ax.set_title(f"{title}", fontsize=14, loc='right')
        ax.tick_params(direction="in", top=False, right=False, labelsize=12, length=3)
        plt.tight_layout()
        fig.savefig(f"{outdir}/{name}.{self.testinfo.get('plot_extn')}", dpi=self.testinfo.get('plot_dpi'))
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
        fig.savefig(f"{outdir}/{name}.{self.testinfo.get('plot_extn')}", dpi=self.testinfo.get('plot_dpi'))
        plt.close()



        
    
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
        
        fig, ax = plt.subplots(figsize=basics["size"])
        hep.cms.text(basics["heplogo"], loc=basics["logoloc"]) # CMS

        for i,data in enumerate(data_list):
            data = np.array(data)
            _val = data[:,0]
            err  = data[:,1]
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
                ax.legend(fontsize=12, framealpha=1, facecolor='white', ncols=2)
                
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

        #print(f"Final number of xticks: {len(ax.get_xticks())}")
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


    def __get_mean_std_per_cbc(self, noise_dict):
        out_list = []
        for i in range(8):
            key = f"CBC_{i}"
            temp = np.array(noise_dict[key])
            temp_mean = np.mean(temp[:,0])
            temp_std  =	np.std(temp[:,0])
            temp_list =	[temp_mean, temp_std]
            out_list.append(temp_list)
            
        return out_list
    
    def __get_cmn_mean_std_per_cbc(self, noise_dict):
        out_list = []
        for i in range(8):
            key = f"CBC_{i}"
            temp = np.array(noise_dict[key])
            x = np.arange(temp.shape[0])
            mean_std = self.__get_mean_std_for_cmn(x, temp[:,0])
            #from IPython import embed; embed()
            #exit()            
            temp_mean = float(mean_std[0])
            temp_std  =	float(mean_std[1])
            temp_list =	[temp_mean, temp_std]
            out_list.append(temp_list)
            
        return out_list


    def __get_cmn_mean_sigma_for_plotting(self, arr):
        arr_mean_sigma = self._get_mean_std_for_cmn_per_module(arr)
        arr_mean = np.concatenate((arr_mean_sigma[:,0:1],
                                   np.zeros_like(arr_mean_sigma[:,0:1])), axis=1)
        arr_sigma = np.concatenate((arr_mean_sigma[:,1:2],
                                    np.zeros_like(arr_mean_sigma[:,1:2])), axis=1)
        return arr_mean, arr_sigma


    def __get_noise_mean_sigma_for_plotting(self, arr):
        return np.concatenate((np.mean(arr, axis=1)[:,None], np.std(arr, axis=1)[:,None]), axis=1)
    
    def __rearrange_arrs(self, arr):
        arr = np.array(arr).reshape(2,-1).T.reshape(-1)
        return arr


    def __get_noisy_and_dead_channels(self, noise_array):
        noise = np.array(noise_array)[:,0]
        channels = np.arange(noise.shape[0])
        median = np.median(noise)
        mad = np.median(np.abs(noise - median))
        modified_z_scores = 0.6745 * (noise - median) / mad
        pos_mask = modified_z_scores > 3.5
        #neg_mask = modified_z_scores < -2*3.5
        neg_mask = noise < 3.0
        pos_outliers = channels[pos_mask]
        neg_outliers = channels[neg_mask]
        return pos_outliers.tolist(), neg_outliers.tolist()

    def get_noisy_and_dead_channels_cbc(self, strip_noise_dict):
        noisy_ch_list = []
        dead_ch_list = []
        for i in range(8):
            hb0_noise_ = strip_noise_dict[f"CBC_{i}"]
            #from IPython import embed; embed(); exit()
            noisy_channels, dead_channels = self.__get_noisy_and_dead_channels(hb0_noise_)
            noisy_ch_list.append(len(noisy_channels))
            dead_ch_list.append(len(dead_channels)+(254-len(hb0_noise_)))
        return noisy_ch_list, dead_ch_list



    def __simfit_and_analyse(self, noise_0_dict, noise_3_dict, labels: list, title: str, name: str, outdir: str):
        fracs = []
        # loop over CBCs
        for i in range(8):
            nHits_0sigma = np.array(noise_0_dict[f'CBC_{i}'])[:,0]
            sigmaHits_0sigma = np.array(noise_0_dict[f'CBC_{i}'])[:,1]
            nHits_3sigma = np.array(noise_3_dict[f'CBC_{i}'])[:,0]
            sigmaHits_3sigma = np.array(noise_3_dict[f'CBC_{i}'])[:,1]

            logger.info(f"fitting nHits for CBC_{i}")
            sigma_fit, k_probs_fit, res = CMNmod.fit_k_and_sigma_from_hists(nHits_0sigma,
                                                                            nHits_3sigma,
                                                                            sigmaHits_0sigma,
                                                                            sigmaHits_3sigma)
                                
            P_A, P_B, R_A, R_B = CMNmod.predict_distributions(sigma_fit, k_probs_fit)
            expA = 10000 * P_A
            expB = 10000 * P_B
            #from IPython import embed; embed(); exit()
            
            N_BINS_K = 20
            K_MIN, K_MAX = -2.0, 2.0
            BIN_EDGES = np.linspace(K_MIN, K_MAX, N_BINS_K + 1)
            BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
            BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]
            
            
            x = [ [np.arange(nHits_0sigma.shape[0]), np.arange(len(expA))] ,
                  [np.arange(nHits_3sigma.shape[0]), np.arange(len(expB))] ,
                  [BIN_CENTERS] ]
            y = [ [nHits_0sigma/10000, P_A] ,
                  [nHits_3sigma/10000, P_B] ,
                  [k_probs_fit] ]
            h3_width = BIN_WIDTH*0.9
            
            self.plot_fitted_result(x = x,
                                    y = y,
                                    w = h3_width,
                                    labels=labels,
                                    title = f"{title}_CBC_{i}",
                                    name = f"{name}_CBC_{i}",
                                    outdir = outdir)
            
            #fracs.append(float(R_A['frac_common']))

            mu_k = np.sum(BIN_CENTERS * k_probs_fit)
            var_k = np.sum((BIN_CENTERS - mu_k)**2 * k_probs_fit)
            sigma_k = np.sqrt(var_k)

            logger.info(f"sigma : {float(sigma_k)}")
            frac_sigma = float((sigma_k**2)/(sigma_k**2 + sigma_fit**2))
            logger.info(f"sigma_k / sigma_g: {frac_sigma}")
            #fracs.append(float(sigma_k))
            fracs.append(frac_sigma)
            
            #from IPython import embed; embed(); exit()
            
        return fracs


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
        
        
    
    def plotEverything(self):

        sensor_temps_setup = {}
        # channel noise
        strip_noise_hb0_setup = {}
        strip_noise_hb0_bot_setup = {}
        strip_noise_hb0_top_setup = {}
        strip_noise_hb1_setup = {}
        strip_noise_hb1_bot_setup = {}
        strip_noise_hb1_top_setup = {}


        num_noisy_channels_hb0_setup = {}
        num_noisy_channels_hb1_setup = {}
        num_dead_channels_hb0_setup = {}
        num_dead_channels_hb1_setup = {}

        num_noisy_channels_hb0_cbc_setup  = {}
        num_noisy_channels_hb1_cbc_setup  = {}
        num_dead_channels_hb0_cbc_setup  = {}
        num_dead_channels_hb1_cbc_setup  = {}

        # common noise
        common_noise_setup = {}
        common_noise_bot_setup = {}
        common_noise_top_setup = {}
        common_noise_hb0_setup = {}
        common_noise_hb0_bot_setup = {}
        common_noise_hb0_top_setup = {}
        common_noise_hb1_setup = {}
        common_noise_hb1_bot_setup = {}
        common_noise_hb1_top_setup = {}

        common_noise_fit_hb0_cbc_setup = {}
        common_noise_fit_hb0_cbc_top_sensor_setup = {}
        common_noise_fit_hb0_cbc_bot_sensor_setup = {}

        common_noise_fit_hb1_cbc_setup = {}
        common_noise_fit_hb1_cbc_top_sensor_setup = {}
        common_noise_fit_hb1_cbc_bot_sensor_setup = {}

        common_noise_giovanni_hb0_cbc_setup = {}
        common_noise_giovanni_hb0_cbc_top_sensor_setup = {}
        common_noise_giovanni_hb0_cbc_bot_sensor_setup = {}

        common_noise_giovanni_hb1_cbc_setup = {}
        common_noise_giovanni_hb1_cbc_top_sensor_setup = {}
        common_noise_giovanni_hb1_cbc_bot_sensor_setup = {}
        
        common_noise_iphc_hb0_cbc_setup = {}
        common_noise_iphc_hb0_cbc_top_sensor_setup = {}
        common_noise_iphc_hb0_cbc_bot_sensor_setup = {}

        common_noise_iphc_hb1_cbc_setup = {}
        common_noise_iphc_hb1_cbc_top_sensor_setup = {}
        common_noise_iphc_hb1_cbc_bot_sensor_setup = {}

        common_noise_crude_hb0_cbc_setup = {}
        common_noise_crude_hb1_cbc_setup = {}

        common_noise_frac_potato_hb0_cbc_setup = {}
        common_noise_frac_potato_hb1_cbc_setup = {}        

        common_noise_frac_simfit_hb0_cbc_setup = {}
        common_noise_frac_simfit_hb1_cbc_setup = {}

        
        allModuleIDs = []

        for datakey, dataval in self.data.items():
            logger.info(f"Setup : {datakey}")
            """
            data:
            
            IPHC_coldBox:
              2S_18_6_KIT-10019:
                RoomTemp:
                  Run1:
                    strip_noise_hb0:
                      ... 
            """
            
            # Create different output dirs for different setup
            _outdir = f"{self.outdir}/{datakey}"
            if not os.path.exists(_outdir): os.mkdir(_outdir)
            _outdirCBC = f"{_outdir}/CBCLevel"
            if not os.path.exists(_outdirCBC): os.mkdir(_outdirCBC)

            sensor_temps_mod = {}
            # initialize the dictionaries to prepare module level data
            strip_noise_hb0_mod = {}
            strip_noise_hb0_bot_mod = {}
            strip_noise_hb0_top_mod = {}
            strip_noise_hb1_mod = {}
            strip_noise_hb1_bot_mod = {}
            strip_noise_hb1_top_mod = {}

            num_noisy_channels_hb0_mod = {}
            num_noisy_channels_hb1_mod = {}
            num_dead_channels_hb0_mod = {}
            num_dead_channels_hb1_mod = {}
            
            num_noisy_channels_hb0_cbc_mod  = {}
            num_noisy_channels_hb1_cbc_mod  = {}
            num_dead_channels_hb0_cbc_mod  = {}
            num_dead_channels_hb1_cbc_mod  = {}
            
            # common noise
            common_noise_mod = {}
            common_noise_bot_mod = {}
            common_noise_top_mod = {}
            common_noise_hb0_mod = {}
            common_noise_hb0_bot_mod = {}
            common_noise_hb0_top_mod = {}
            common_noise_hb1_mod = {}
            common_noise_hb1_bot_mod = {}
            common_noise_hb1_top_mod = {}

            common_noise_fit_hb0_cbc_mod = {}
            common_noise_fit_hb0_cbc_top_mod = {}
            common_noise_fit_hb0_cbc_bot_mod = {}
            
            common_noise_fit_hb1_cbc_mod = {}
            common_noise_fit_hb1_cbc_top_mod = {}
            common_noise_fit_hb1_cbc_bot_mod = {}


            common_noise_giovanni_hb0_cbc_mod = {}
            common_noise_giovanni_hb0_cbc_top_mod = {}
            common_noise_giovanni_hb0_cbc_bot_mod = {}
            
            common_noise_giovanni_hb1_cbc_mod = {}
            common_noise_giovanni_hb1_cbc_top_mod = {}
            common_noise_giovanni_hb1_cbc_bot_mod = {}

            common_noise_iphc_hb0_cbc_mod = {}
            common_noise_iphc_hb0_cbc_top_mod = {}
            common_noise_iphc_hb0_cbc_bot_mod = {}
            
            common_noise_iphc_hb1_cbc_mod = {}
            common_noise_iphc_hb1_cbc_top_mod = {}
            common_noise_iphc_hb1_cbc_bot_mod = {}

            common_noise_crude_hb0_cbc_mod = {}
            common_noise_crude_hb1_cbc_mod = {}

            common_noise_frac_potato_hb0_cbc_mod = {}
            common_noise_frac_potato_hb1_cbc_mod = {}

            common_noise_frac_simfit_hb0_cbc_mod = {}
            common_noise_frac_simfit_hb1_cbc_mod = {}
            
            
            # define dict to save info per module level
            moduleIDs = []
            for moduleID, moduleDict in dataval.items():
                logger.info(f"Module ID : {moduleID}")
                moduleIDs.append(moduleID)

                _outdirMod = f"{_outdir}/{moduleID}"
                if not os.path.exists(_outdirMod): os.mkdir(_outdirMod)
                
                _outdirModCBC = f"{_outdirMod}/CBCLevel"
                if not os.path.exists(_outdirModCBC): os.mkdir(_outdirModCBC)

                
                for martaTemp, _noiseDict in moduleDict.items():
                    logger.info(f"Temperature : {martaTemp}")
                    if martaTemp not in strip_noise_hb0_bot_mod.keys():
                        sensor_temps_mod[martaTemp] = []
                        # channel noise
                        strip_noise_hb0_mod[martaTemp] = []
                        strip_noise_hb0_bot_mod[martaTemp] = []
                        strip_noise_hb0_top_mod[martaTemp] = []
                        strip_noise_hb1_mod[martaTemp] = []                        
                        strip_noise_hb1_bot_mod[martaTemp] = []
                        strip_noise_hb1_top_mod[martaTemp] = []


                        num_noisy_channels_hb0_mod[martaTemp]  = []
                        num_noisy_channels_hb1_mod[martaTemp]  = []
                        num_dead_channels_hb0_mod[martaTemp]  = []
                        num_dead_channels_hb1_mod[martaTemp]  = []
                        
                        num_noisy_channels_hb0_cbc_mod[martaTemp]  = []
                        num_noisy_channels_hb1_cbc_mod[martaTemp]  = []
                        num_dead_channels_hb0_cbc_mod[martaTemp]  = []
                        num_dead_channels_hb1_cbc_mod[martaTemp]  = []

                        # common noise
                        common_noise_mod[martaTemp] = []
                        common_noise_bot_mod[martaTemp] = []
                        common_noise_top_mod[martaTemp] = []
                        common_noise_hb0_mod[martaTemp] = []
                        common_noise_hb0_bot_mod[martaTemp] = []
                        common_noise_hb0_top_mod[martaTemp] = []
                        common_noise_hb1_mod[martaTemp] = []
                        common_noise_hb1_bot_mod[martaTemp] = []
                        common_noise_hb1_top_mod[martaTemp] = []

                        common_noise_fit_hb0_cbc_mod[martaTemp] = []
                        common_noise_fit_hb0_cbc_top_mod[martaTemp] = []
                        common_noise_fit_hb0_cbc_bot_mod[martaTemp] = []
                        
                        common_noise_fit_hb1_cbc_mod[martaTemp] = []
                        common_noise_fit_hb1_cbc_top_mod[martaTemp] = []
                        common_noise_fit_hb1_cbc_bot_mod[martaTemp] = []


                        common_noise_giovanni_hb0_cbc_mod[martaTemp] = []
                        common_noise_giovanni_hb0_cbc_top_mod[martaTemp] = []
                        common_noise_giovanni_hb0_cbc_bot_mod[martaTemp] = []
                        
                        common_noise_giovanni_hb1_cbc_mod[martaTemp] = []
                        common_noise_giovanni_hb1_cbc_top_mod[martaTemp] = []
                        common_noise_giovanni_hb1_cbc_bot_mod[martaTemp] = []

                        common_noise_iphc_hb0_cbc_mod[martaTemp] = []
                        common_noise_iphc_hb0_cbc_top_mod[martaTemp] = []
                        common_noise_iphc_hb0_cbc_bot_mod[martaTemp] = []
                        
                        common_noise_iphc_hb1_cbc_mod[martaTemp] = []
                        common_noise_iphc_hb1_cbc_top_mod[martaTemp] = []
                        common_noise_iphc_hb1_cbc_bot_mod[martaTemp] = []
                        
                        common_noise_crude_hb0_cbc_mod[martaTemp] = []
                        common_noise_crude_hb1_cbc_mod[martaTemp] = []

                        common_noise_frac_potato_hb0_cbc_mod[martaTemp] = []
                        common_noise_frac_potato_hb1_cbc_mod[martaTemp] = []

                        common_noise_frac_simfit_hb0_cbc_mod[martaTemp] = []
                        common_noise_frac_simfit_hb1_cbc_mod[martaTemp] = []

                        
                    noiseDict = _noiseDict["Run1"] # right now, only one run is allowed

                    # Now we have the access to the noise dict
                    """
                    strip_noise_hb0:
                      allCBC:
                        - [6.766048431396484, 0.05587056279182434]
                        - [5.9948577880859375, 0.04917684197425842]
                        - ...
                      CBC_0:
                        - ...
                      ... 

                    """
                    # Remarks:
                    # for now, I am not looping over the keys
                    # access those individually to prepare proper input for plotting

                    if self.testinfo.get("check_extra") == True:
                        scurve_dict = noiseDict['SCurve']
                        for key_hb, cbc_scurve in scurve_dict.items():
                            for key_cbc, hist_scurve in cbc_scurve.items():
                                self.plot_ROOT_2Dhist(hist=hist_scurve,
                                                      name=f'SCurve_{moduleID}_{key_hb}_{key_cbc}',
                                                      title=f'SCurve_{moduleID}_{key_hb}_{key_cbc}',
                                                      outdir=_outdirModCBC)
                                
















                                
                    

                    # Sensor Temperature
                    if self.testinfo.get("check_sensor_temperature") == True:
                        time_stamps = np.array(noiseDict['time_stamps'])
                        sensor_temps = np.array(noiseDict['sensor_temps'])
                        sensor_temps_mod[martaTemp].append(float(sensor_temps[-1]))

                        #from IPython import embed; embed(); exit()

                        self.plot_basic(x          = np.arange(time_stamps.shape[0]),
                                        data_list  = [np.concatenate((sensor_temps[:,None], np.zeros_like(sensor_temps)[:,None]), axis=1)],
                                        legends    = ["sensor temp"],
                                        title      = f"Sensor Temperature: {moduleID}",
                                        name       = f"Plot_SensorTemp_{martaTemp}_{moduleID}_{datakey}",
                                        #xlabel     = "Time Stamps",
                                        ylabel     = "Sensor Temperature (deg C)",
                                        xticklabels=time_stamps.tolist(),
                                        marker     = "o",
                                        linewidth  = 1.2,
                                        markersize = 2.5,
                                        #ylim       = [20.0,27.0],
                                        outdir     = _outdirMod,
                                        nticks     = 40 if time_stamps.shape[0] > 40 else None)

                    else:
                        logger.warning("skip plotting sensor temperature")
                        
                    strip_noise_hb0_dict = noiseDict['strip_noise_hb0']
                    strip_noise_hb0 = strip_noise_hb0_dict['allCBC']
                    strip_noise_hb0_mod[martaTemp].append(strip_noise_hb0)
                    
                    strip_noise_hb0_bot_dict = noiseDict['strip_noise_hb0_bot']
                    strip_noise_hb0_bot = strip_noise_hb0_bot_dict['allCBC']
                    strip_noise_hb0_bot_mod[martaTemp].append(strip_noise_hb0_bot)
                    
                    strip_noise_hb0_top_dict = noiseDict['strip_noise_hb0_top']
                    strip_noise_hb0_top = strip_noise_hb0_top_dict['allCBC']
                    strip_noise_hb0_top_mod[martaTemp].append(strip_noise_hb0_top)

                    strip_noise_hb1_dict = noiseDict['strip_noise_hb1']
                    strip_noise_hb1 = strip_noise_hb1_dict['allCBC']
                    strip_noise_hb1_mod[martaTemp].append(strip_noise_hb1)                    
                    
                    strip_noise_hb1_bot_dict = noiseDict['strip_noise_hb1_bot']
                    strip_noise_hb1_bot = strip_noise_hb1_bot_dict['allCBC']
                    strip_noise_hb1_bot_mod[martaTemp].append(strip_noise_hb1_bot)
                    
                    strip_noise_hb1_top_dict = noiseDict['strip_noise_hb1_top']
                    strip_noise_hb1_top = strip_noise_hb1_top_dict['allCBC']
                    strip_noise_hb1_top_mod[martaTemp].append(strip_noise_hb0_top)                    
                    
                    
                    self.plot_basic(x          = np.arange(len(strip_noise_hb0)),
                                    data_list  = [strip_noise_hb0, strip_noise_hb1],
                                    legends    = ["hybrid 0", "hybrid 1"],
                                    title      = f"StripNoise_{moduleID}: ({martaTemp})",
                                    name       = f"Plot_StripNoise_bothHybrids_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Channel",
                                    ylabel     = "Noise [VcTh]",
                                    ylim       = [0.0,12.0],
                                    outdir     = _outdirMod)

                    
                    # Plotting strip noise channel wise : [hb0_bot, hb1_bot]
                    self.plot_basic(x          = np.arange(len(strip_noise_hb0_top)),
                                    data_list  = [strip_noise_hb0_bot, strip_noise_hb1_bot],
                                    legends    = ["hybrid 0", "hybrid 1"],
                                    title      = f"StripNoise_Bottom_{moduleID}: ({martaTemp})",
                                    name       = f"Plot_StripNoise_bothHybrids_BottomSensor_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Channel",
                                    ylabel     = "Noise [VcTh]",
                                    ylim       = [0.0,12.0],
                                    outdir     = _outdirMod)

                    # Plotting strip noise channel wise : [hb0_top, hb1_top]
                    self.plot_basic(x          = np.arange(len(strip_noise_hb0_top)),
                                    data_list  = [strip_noise_hb0_top, strip_noise_hb1_top],
                                    legends    = ["hybrid 0", "hybrid 1"],
                                    title      = f"StripNoise_Top_{moduleID}: ({martaTemp})",
                                    name       = f"Plot_StripNoise_bothHybrids_TopSensor_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Channel",
                                    ylabel     = "Noise [VcTh]",
                                    ylim       = [0.0,12.0],
                                    outdir     = _outdirMod)


                    # Plotting hist for stripNoise : [hb0_bot, hb0_top, hb1_bot, hb1_top]
                    self.hist_basic(bins       = np.linspace(2,10,80),
                                    data_list  = [strip_noise_hb0_bot, strip_noise_hb0_top, strip_noise_hb1_bot, strip_noise_hb1_top],
                                    legends    = ["hybrid 0 bottom", "hybrid 0 top", "hybrid 1 bottom", "hybrid 1 top"],
                                    title      = f"StripNoise_{moduleID}: ({martaTemp})",
                                    name       = f"Hist_StripNoise_bothHybrids_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Noise [VcTh]",
                                    ylabel     = "Entries",
                                    linewidth  = 2,
                                    outdir     = _outdirMod,
                                    colors     = ["#165a86","#cc660b","#165a86","#cc660b"],
                                    linestyles = ["-","-","--","--"])
                    
                    
                    # Plotting hist for stripNoise per CBC : [hb0 cbc level]
                    self.hist_basic(bins       = np.linspace(2,10,80),
                                    data_list  = [strip_noise_hb0_dict['CBC_0'],
                                                  strip_noise_hb0_dict['CBC_1'],
                                                  strip_noise_hb0_dict['CBC_2'],
                                                  strip_noise_hb0_dict['CBC_3'],
                                                  strip_noise_hb0_dict['CBC_4'],
                                                  strip_noise_hb0_dict['CBC_5'],
                                                  strip_noise_hb0_dict['CBC_6'],
                                                  strip_noise_hb0_dict['CBC_7']],
                                    legends    = ["Chip 0", "Chip 1", "Chip 2", "Chip 3", "Chip 4", "Chip 5", "Chip 6", "Chip 7"],
                                    title      = f"StripNoise_Hb0_{moduleID}: ({martaTemp})",
                                    name       = f"Hist_StripNoiseCBC_Hybrid0_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Noise [VcTh]",
                                    ylabel     = "Entries",
                                    linewidth  = 1.2,
                                    outdir     = _outdirModCBC,
                                    colors     = ["#0B1A2F", "#152C4D", "#1F3E6C", "#29508A", "#3362A9", "#3D74C7", "#4796E6", "#61B4FF"],
                                    linestyles = 8*["-"])


                    
                    # ------------------------------------------------------------------ #
                    #                Get Noisy & Dead channel fractions                  #
                    # ------------------------------------------------------------------ #

                    # per hybrid
                    hb0_noisy_channels, hb0_dead_channels = self.__get_noisy_and_dead_channels(strip_noise_hb0)
                    num_noisy_channels_hb0_mod[martaTemp].append(len(hb0_noisy_channels))
                    num_dead_channels_hb0_mod[martaTemp].append(len(hb0_dead_channels))
                                        
                    hb1_noisy_channels, hb1_dead_channels = self.__get_noisy_and_dead_channels(strip_noise_hb1)
                    num_noisy_channels_hb1_mod[martaTemp].append(len(hb1_noisy_channels))
                    num_dead_channels_hb1_mod[martaTemp].append(len(hb1_dead_channels))
                    
                    # per CBC
                    noisy_ch_list_hb0, dead_ch_list_hb0 = self.get_noisy_and_dead_channels_cbc(strip_noise_hb0_dict)
                    noisy_ch_list_hb1, dead_ch_list_hb1 = self.get_noisy_and_dead_channels_cbc(strip_noise_hb1_dict)

                    num_noisy_channels_hb0_cbc_mod[martaTemp].append(noisy_ch_list_hb0)
                    num_dead_channels_hb0_cbc_mod[martaTemp].append(dead_ch_list_hb0)
                    num_noisy_channels_hb1_cbc_mod[martaTemp].append(noisy_ch_list_hb1)
                    num_dead_channels_hb1_cbc_mod[martaTemp].append(dead_ch_list_hb1)
                    
                    #from IPython import embed; embed()

                    
                    # Plotting hist for stripNoise per CBC : [hb1 cbc level]
                    self.hist_basic(bins       = np.linspace(2,10,80),
                                    data_list  = [strip_noise_hb1_dict['CBC_0'],
                                                  strip_noise_hb1_dict['CBC_1'],
                                                  strip_noise_hb1_dict['CBC_2'],
                                                  strip_noise_hb1_dict['CBC_3'],
                                                  strip_noise_hb1_dict['CBC_4'],
                                                  strip_noise_hb1_dict['CBC_5'],
                                                  strip_noise_hb1_dict['CBC_6'],
                                                  strip_noise_hb1_dict['CBC_7']],
                                    legends    = ["Chip 0", "Chip 1", "Chip 2", "Chip 3", "Chip 4", "Chip 5", "Chip 6", "Chip 7"],
                                    title      = f"StripNoise_Hb1_{moduleID}: ({martaTemp})",
                                    name       = f"Hist_StripNoiseCBC_Hybrid1_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Noise [VcTh]",
                                    ylabel     = "Entries",
                                    linewidth  = 1.2,
                                    outdir     = _outdirModCBC,
                                    colors     = ["#2E0000", "#4A0A0A", "#661515", "#821F1F", "#9E2A2A", "#BA3535", "#D65050", "#F26B6B"],
                                    linestyles = 8*["-"])


                    # Get the mean and std of noise per CBC
                    strip_noise_hb0_bot_per_cbc_mean_std_list = self.__get_mean_std_per_cbc(strip_noise_hb0_bot_dict)
                    strip_noise_hb0_top_per_cbc_mean_std_list = self.__get_mean_std_per_cbc(strip_noise_hb0_top_dict)
                    #
                    self.plot_group(x           = np.arange(8),
                                    data_list   = [[strip_noise_hb0_bot_per_cbc_mean_std_list,
                                                    strip_noise_hb0_top_per_cbc_mean_std_list]],
                                    legends     = [["Bottom Sensor",
                                                    "Top Sensor"]],
                                    title       = f"StripNoise_Hb0_{martaTemp}_{moduleID}",
                                    name        = f"Plot_StripNoise_perCBC_Hybrid0_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                    xticklabels = [f"CBC_{i}" for i in range(8)],
                                    ylim        = [2.0,10.0],
                                    ylabel      = "Noise [VcTh]",
                                    outdir      = _outdirModCBC,
                                    marker      = "o",
                                    markersize  = 2.5,
                                    capsize     = 1.5,
                                    elinewidth  = 1.0)
                    
                    strip_noise_hb1_bot_per_cbc_mean_std_list = self.__get_mean_std_per_cbc(strip_noise_hb1_bot_dict)
                    strip_noise_hb1_top_per_cbc_mean_std_list = self.__get_mean_std_per_cbc(strip_noise_hb1_top_dict)
                    self.plot_group(x           = np.arange(8),
                                    data_list   = [[strip_noise_hb1_bot_per_cbc_mean_std_list,
                                                    strip_noise_hb1_top_per_cbc_mean_std_list]],
                                    legends     = [["Bottom Sensor",
                                                    "Top Sensor"]],
                                    title       = f"StripNoise_Hb1_{martaTemp}_{moduleID}",
                                    name        = f"Plot_StripNoise_perCBC_Hybrid1_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                    xticklabels = [f"CBC_{i}" for i in range(8)],
                                    ylim        = [2.0,10.0],
                                    ylabel      = "Noise [VcTh]",
                                    outdir      = _outdirModCBC,
                                    marker      = "o",
                                    markersize  = 2.5,
                                    capsize     = 1.5,
                                    elinewidth  = 1.0)

                    
                    strip_noise_hb0_per_cbc_mean_std_list = self.__get_mean_std_per_cbc(strip_noise_hb0_dict)
                    strip_noise_hb1_per_cbc_mean_std_list = self.__get_mean_std_per_cbc(strip_noise_hb1_dict)
                    self.plot_group(x           = np.arange(8),
                                    data_list   = [[strip_noise_hb0_per_cbc_mean_std_list,
                                                    strip_noise_hb1_per_cbc_mean_std_list]],
                                    legends     = [["Hybrid 0",
                                                    "Hybrid 1"]],
                                    title       = f"StripNoise_{martaTemp}_{moduleID}",
                                    name        = f"Plot_StripNoise_perCBC_bothHybrids_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                    xticklabels = [f"CBC_{i}" for i in range(8)],
                                    ylim        = [2.0,10.0],
                                    ylabel      = "Noise [VcTh]",
                                    outdir      = _outdirModCBC,
                                    marker      = "o",
                                    markersize  = 2.5,
                                    capsize     = 1.5,
                                    elinewidth  = 1.0)




                    
                    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                    #       Common Mode Noise      #
                    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                    if self.testinfo.get("check_common_noise") == True:
                        # module level
                        common_noise_module = noiseDict['common_noise_module']
                        common_noise_mod[martaTemp].append(common_noise_module)
                        # module (bottom sensor)
                        common_noise_module_bot = noiseDict['common_noise_module_bot']
                        common_noise_bot_mod[martaTemp].append(common_noise_module_bot)
                        # module (top sensor)
                        common_noise_module_top = noiseDict['common_noise_module_top']
                        common_noise_top_mod[martaTemp].append(common_noise_module_top)
                        
                        # hb-0
                        common_noise_hb0_dict = noiseDict['common_noise_hb0']
                        common_noise_hb0 = common_noise_hb0_dict['allCBC']
                        common_noise_hb0_mod[martaTemp].append(common_noise_hb0)
                        # hb-0 (bottom)
                        common_noise_hb0_bot_dict = noiseDict['common_noise_hb0_bot']
                        common_noise_hb0_bot = common_noise_hb0_bot_dict['allCBC']
                        common_noise_hb0_bot_mod[martaTemp].append(common_noise_hb0_bot)
                        # hb-0 (top)
                        common_noise_hb0_top_dict = noiseDict['common_noise_hb0_top']
                        common_noise_hb0_top = common_noise_hb0_top_dict['allCBC']
                        common_noise_hb0_top_mod[martaTemp].append(common_noise_hb0_top)
                        # hb-1
                        common_noise_hb1_dict = noiseDict['common_noise_hb1']
                        common_noise_hb1 = common_noise_hb1_dict['allCBC']
                        common_noise_hb1_mod[martaTemp].append(common_noise_hb1)
                        # hb-1 (bottom)
                        common_noise_hb1_bot_dict = noiseDict['common_noise_hb1_bot']
                        common_noise_hb1_bot = common_noise_hb1_bot_dict['allCBC']
                        common_noise_hb1_bot_mod[martaTemp].append(common_noise_hb1_bot)
                        # hb-1 (top)
                        common_noise_hb1_top_dict = noiseDict['common_noise_hb1_top']
                        common_noise_hb1_top = common_noise_hb1_top_dict['allCBC']
                        common_noise_hb1_top_mod[martaTemp].append(common_noise_hb1_top)
                    

                        # Plotting common noise (for whole module)
                        self.plot_basic(x          = np.arange(len(common_noise_module)),
                                        data_list  = [common_noise_module],
                                        legends    = ["Module"],
                                        title      = f"CMNoise_{martaTemp}_{moduleID}",
                                        name       = f"Plot_CommonModeNoise_module_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Number of hits",
                                        ylabel     = "Number of events",
                                        #ylim       = [0.0,12.0],
                                        outdir     = _outdirMod,
                                        linewidth  = 0.0,
                                        markersize = 0.5,
                                        capsize    = 0.0,
                                        elinewidth = 0.05,
                                        fitlinewidth = 2.0,
                                        fit        = self.testinfo.get("fit_common_noise"),
                                        #fitmodel   = self.__gauss_model,
                                        #fitfunc    = self.__gauss_fit,
                                        #mean_init  = 1500.0,
                                        #sigma_init = 40.0
                                        )
                        # Plotting common noise (top and bot sensors at module level)
                        self.plot_basic(x          = np.arange(len(common_noise_module_bot)),
                                        data_list  = [common_noise_module_bot, common_noise_module_top],
                                        legends    = ["Bottom Sensor", "Top Sensor"],
                                        title      = f"CMNoise_{martaTemp}_{moduleID}",
                                        name       = f"Plot_CommonModeNoise_module_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Number of hits",
                                        ylabel     = "Number of events",
                                        #ylim       = [0.0,12.0],
                                        outdir     = _outdirMod,
                                        linewidth  = 0.0,
                                        markersize = 0.5,
                                        capsize    = 0.0,
                                        elinewidth = 0.05,
                                        fitlinewidth = 2.0,
                                        fit        = self.testinfo.get("fit_common_noise"),
                                        #fitmodel   = self.__gauss_model,
                                        #fitfunc    = self.__gauss_fit,
                                        #mean_init  = 800.0,
                                        #sigma_init = 30.0
                                        )
                        # Plotting common noise (hybrids at module level)
                        self.plot_basic(x          = np.arange(len(common_noise_hb0)),
                                        data_list  = [common_noise_hb0, common_noise_hb1],
                                        legends    = ["Hybrid 0", "Hybrid 1"],
                                        title      = f"CMNoise_{martaTemp}_{moduleID}",
                                        name       = f"Plot_CommonModeNoise_module_bothHybrids_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Number of hits",
                                        ylabel     = "Number of events",
                                        #ylim       = [0.0,12.0],
                                        outdir     = _outdirMod,
                                        linewidth  = 0.0,
                                        markersize = 0.5,
                                        capsize    = 0.0,
                                        elinewidth = 0.05,
                                        fitlinewidth = 2.0,
                                        fit        = self.testinfo.get("fit_common_noise"),
                                        #fitmodel   = self.__gauss_model,
                                        #fitfunc    = self.__gauss_fit,
                                        #mean_init  = 800.0,
                                        #sigma_init = 100.0
                                        )
                        # Plotting common noise (Top sensor)
                        self.plot_basic(x          = np.arange(len(common_noise_hb0_top)),
                                        data_list  = [common_noise_hb0_top, common_noise_hb1_top],
                                        legends    = ["Hybrid 0", "Hybrid 1"],
                                        title      = f"CMNoise_Top_{martaTemp}_{moduleID}",
                                        name       = f"Plot_CommonModeNoise_bothHybrids_topSensor_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Number of hits",
                                        ylabel     = "Number of events",
                                        #ylim       = [0.0,12.0],
                                        linewidth  = 0.0,
                                        markersize = 0.5,
                                        capsize    = 0.0,
                                        elinewidth = 0.05,
                                        fitlinewidth = 2.0,
                                        outdir     = _outdirMod,
                                        fit        = self.testinfo.get("fit_common_noise"),
                                        #fitmodel   = self.__gauss_model,
                                        #fitfunc    = self.__gauss_fit
                                        )
                        # Plotting common noise (Bottom sensor)
                        self.plot_basic(x          = np.arange(len(common_noise_hb0_bot)),
                                        data_list  = [common_noise_hb0_bot, common_noise_hb1_bot],
                                        legends    = ["Hybrid 0", "Hybrid 1"],
                                        title      = f"CMNoise_Bot_{martaTemp}_{moduleID}",
                                        name       = f"Plot_CommonModeNoise_bothHybrids_bottomSensor_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Number of hits",
                                        ylabel     = "Number of events",
                                        #ylim       = [0.0,12.0],
                                        outdir     = _outdirMod,
                                        linewidth  = 0.0,
                                        markersize = 0.5,
                                        capsize    = 0.0,
                                        elinewidth = 0.05,
                                        fitlinewidth = 2.0,
                                        fit        = self.testinfo.get("fit_common_noise"),
                                        #fitmodel   = self.__gauss_model,
                                        #fitfunc    = self.__gauss_fit
                                        )


                        # Plotting hist for stripNoise per CBC : [hb0 cbc level]
                        self.plot_basic(x          = np.arange(len(common_noise_hb0_dict["CBC_0"])),
                                        data_list  = [common_noise_hb0_dict['CBC_0'],
                                                      common_noise_hb0_dict['CBC_1'],
                                                      common_noise_hb0_dict['CBC_2'],
                                                      common_noise_hb0_dict['CBC_3'],
                                                      common_noise_hb0_dict['CBC_4'],
                                                      common_noise_hb0_dict['CBC_5'],
                                                      common_noise_hb0_dict['CBC_6'],
                                                      common_noise_hb0_dict['CBC_7']],
                                        legends    = [f"CBC{i}" for i in range(8)],
                                        title      = f"CMNoiseCBC_hb0_{martaTemp}_{moduleID}",
                                        name       = f"Plot_CommonNoiseDistributionCBC_Hybrid0_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Number of hits",
                                        ylabel     = "Number of events",
                                        outdir     = _outdirModCBC,
                                        colors     = ["#0B1A2F", "#152C4D", "#1F3E6C", "#29508A", "#3362A9", "#3D74C7", "#4796E6", "#61B4FF"],
                                        linestyles = 8*["-"],
                                        linewidth  = 1.0,
                                        markersize = 0.3,
                                        capsize    = 0.0,
                                        elinewidth = 0.2,
                                        fit        = self.testinfo.get("fit_common_noise"),
                                        #fitmodel   = self.__gauss_model,
                                        #fitfunc    = self.__gauss_fit,
                                        fitlinewidth = 0.5,
                                        #mean_init  = 100.0,
                                        #sigma_init = 20.0
                                        )
                        
                        self.plot_basic(x          = np.arange(len(common_noise_hb1_dict["CBC_0"])),
                                        data_list  = [common_noise_hb1_dict['CBC_0'],
                                                      common_noise_hb1_dict['CBC_1'],
                                                      common_noise_hb1_dict['CBC_2'],
                                                      common_noise_hb1_dict['CBC_3'],
                                                      common_noise_hb1_dict['CBC_4'],
                                                      common_noise_hb1_dict['CBC_5'],
                                                      common_noise_hb1_dict['CBC_6'],
                                                      common_noise_hb1_dict['CBC_7']],
                                        legends    = [f"CBC{i}" for i in range(8)],
                                        title      = f"CMNoiseCBC_hb1_{martaTemp}_{moduleID}",
                                        name       = f"Plot_CommonNoiseDistributionCBC_Hybrid1_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Number of hits",
                                        ylabel     = "Number of events",
                                        outdir     = _outdirModCBC,
                                        colors     = ["#2E0000", "#4A0A0A", "#661515", "#821F1F", "#9E2A2A", "#BA3535", "#D65050", "#F26B6B"],
                                        linestyles = 8*["-"],
                                        linewidth  = 1.0,
                                        markersize = 0.3,
                                        capsize    = 0.0,
                                        elinewidth = 0.2,
                                        fit        = self.testinfo.get("fit_common_noise"),
                                        #fitmodel   = self.__gauss_model,
                                        #fitfunc    = self.__gauss_fit,
                                        fitlinewidth = 0.5,
                                        #mean_init  = 100.0,
                                        #sigma_init = 20.0
                                        )
                    

                    
                        # Get the mean and std of noise per CBC
                        common_noise_hb0_per_cbc_mean_std_list = self.__get_cmn_mean_std_per_cbc(common_noise_hb0_dict)
                        common_noise_hb1_per_cbc_mean_std_list = self.__get_cmn_mean_std_per_cbc(common_noise_hb1_dict)
                        self.plot_group(x           = np.arange(8),
                                        data_list   = [[common_noise_hb0_per_cbc_mean_std_list,
                                                        common_noise_hb1_per_cbc_mean_std_list]],
                                        legends     = [["hb0", "hb1"]],
                                        title       = f"CMNoise_{martaTemp}_{moduleID}",
                                        name        = f"Plot_CommonNoiseCBC_bothHybrids_{martaTemp}_{moduleID}_{datakey}",
                                        xticklabels = [f"CBC_{i}" for i in range(8)],
                                        ylim        = [50, 200],
                                        ylabel      = "Common Noise",
                                        outdir      = _outdirModCBC,
                                        marker      = "o",
                                        markersize  = 2.5,
                                        capsize     = 1.5,
                                        elinewidth  = 1.0)
                        common_noise_hb0_bot_per_cbc_mean_std_list = self.__get_cmn_mean_std_per_cbc(common_noise_hb0_bot_dict)
                        common_noise_hb0_top_per_cbc_mean_std_list = self.__get_cmn_mean_std_per_cbc(common_noise_hb0_top_dict)                    
                        common_noise_hb1_bot_per_cbc_mean_std_list = self.__get_cmn_mean_std_per_cbc(common_noise_hb1_bot_dict)
                        common_noise_hb1_top_per_cbc_mean_std_list = self.__get_cmn_mean_std_per_cbc(common_noise_hb1_top_dict)
                        self.plot_group(x           = np.arange(8),
                                        data_list   = [[common_noise_hb0_bot_per_cbc_mean_std_list, common_noise_hb1_bot_per_cbc_mean_std_list,
                                                        common_noise_hb0_top_per_cbc_mean_std_list, common_noise_hb1_top_per_cbc_mean_std_list]],
                                        legends     = [["hybrid 0 bottom", "hybrid 1 bottom",
                                                        "hybrid 0 top", "hybrid 1 top"]],
                                        title       = f"CMNoise_{martaTemp}_{moduleID}",
                                        name        = f"Plot_CommonNoiseCBC_bothHybrids_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                        xticklabels = [f"CBC_{i}" for i in range(8)],
                                        ylim        = [20, 100],
                                        ylabel      = "Common Noise",
                                        outdir      = _outdirModCBC,
                                        marker      = "o",
                                        markersize  = 2.5,
                                        capsize     = 1.5,
                                        elinewidth  = 1.0)



                        common_noise_giovanni_hb0_cbc_mod[martaTemp].append(self.__extractCMN_giovanni(nchannels=254,
                                                                                                       mean=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,0],
                                                                                                       std=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,1]).tolist())
                        common_noise_giovanni_hb1_cbc_mod[martaTemp].append(self.__extractCMN_giovanni(nchannels=254,
                                                                                                       mean=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,0],
                                                                                                       std=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,1]).tolist())

                        common_noise_iphc_hb0_cbc_mod[martaTemp].append(self.__extractCMN_iphc(nchannels=254,
                                                                                               mean=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,0],
                                                                                               std=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,1]).tolist())
                        common_noise_iphc_hb1_cbc_mod[martaTemp].append(self.__extractCMN_iphc(nchannels=254,
                                                                                               mean=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,0],
                                                                                               std=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,1]).tolist())

                        common_noise_crude_hb0_cbc_mod[martaTemp].append(self.__extractCMN_crude(nchannels=254,
                                                                                                 mean=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,0],
                                                                                                 std=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,1]).tolist())
                        common_noise_crude_hb1_cbc_mod[martaTemp].append(self.__extractCMN_crude(nchannels=254,
                                                                                                 mean=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,0],
                                                                                                 std=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,1]).tolist())



                        #from IPython import embed; embed(); exit()


                        append_fit_result = lambda target, source: target.append(
                            [
                                source[f'CBC_{i}_fit_params']['cmnFraction']
                                for i in range(8)
                                if f'CBC_{i}_fit_params' in source.keys()
                            ]
                        )
                        # saving the cmn from fit per CBC
                        #common_noise_fit_hb0_cbc_mod[martaTemp].append([common_noise_hb0_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                        append_fit_result(common_noise_fit_hb0_cbc_mod[martaTemp], common_noise_hb0_dict)
                        append_fit_result(common_noise_fit_hb0_cbc_bot_mod[martaTemp], common_noise_hb0_bot_dict)
                        append_fit_result(common_noise_fit_hb0_cbc_top_mod[martaTemp], common_noise_hb0_top_dict)
                        append_fit_result(common_noise_fit_hb1_cbc_mod[martaTemp], common_noise_hb1_dict)
                        append_fit_result(common_noise_fit_hb1_cbc_bot_mod[martaTemp], common_noise_hb1_bot_dict)
                        append_fit_result(common_noise_fit_hb1_cbc_top_mod[martaTemp], common_noise_hb1_top_dict)
                        
                        append_cmn_potato_result = lambda target, source: target.append(
                            [
                                self.__extractCMN_potato(source[f'CBC_{i}'])
                                for i in range(8)
                            ]
                        )
                        append_cmn_potato_result(common_noise_frac_potato_hb0_cbc_mod[martaTemp], common_noise_hb0_dict)
                        append_cmn_potato_result(common_noise_frac_potato_hb1_cbc_mod[martaTemp], common_noise_hb1_dict)
                        
                        #common_noise_fit_hb0_cbc_bot_mod[martaTemp].append([common_noise_hb0_bot_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                        #common_noise_fit_hb0_cbc_bot_mod[martaTemp].append(
                        #    [
                        #        common_noise_hb0_bot_dict[f'CBC_{i}_fit_params']['cmnFraction']
                        #        for i in range(8)
                        #        
                        #common_noise_fit_hb0_cbc_top_mod[martaTemp].append([common_noise_hb0_top_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                        #common_noise_fit_hb1_cbc_mod[martaTemp].append([common_noise_hb1_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                        #common_noise_fit_hb1_cbc_bot_mod[martaTemp].append([common_noise_hb1_bot_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                        #common_noise_fit_hb1_cbc_top_mod[martaTemp].append([common_noise_hb1_top_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])


                        if self.testinfo.get("fit_simultaneous_common_noise") == True:
                            common_noise_3sigma_hb0_dict = noiseDict['common_3sigma_noise_hb0']
                            common_noise_3sigma_hb1_dict = noiseDict['common_3sigma_noise_hb1']

                            cmn_res_hb0 = self.__simfit_and_analyse(common_noise_hb0_dict,
                                                                    common_noise_3sigma_hb0_dict,
                                                                    labels=["nHits : 0 sigma", "nHits : 3 sigma", "CMNoise"],
                                                                    title = f"CMN_fitted_hb0_{martaTemp}_{moduleID}",
                                                                    name = f"Plot_CommonNoiseCBC_hb0_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                                                    outdir = _outdirModCBC)
                            common_noise_frac_simfit_hb0_cbc_mod[martaTemp].append(cmn_res_hb0)
                            
                            cmn_res_hb1 = self.__simfit_and_analyse(common_noise_hb1_dict,
                                                                    common_noise_3sigma_hb1_dict,
                                                                    labels=["nHits : 0 sigma", "nHits : 3 sigma", "CMNoise"],
                                                                    title = f"CMN_fitted_hb1_{martaTemp}_{moduleID}",
                                                                    name = f"Plot_CommonNoiseCBC_hb1_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                                                    outdir = _outdirModCBC)

                            common_noise_frac_simfit_hb1_cbc_mod[martaTemp].append(cmn_res_hb1)

                            #from IPython import embed; embed()
                            
                            """
                            # loop over CBCs
                            for i in range(8):
                                fracs = []
                                nHits_0sigma = np.array(common_noise_hb0_dict[f'CBC_{i}'])[:,0]
                                sigmaHits_0sigma = np.array(common_noise_hb0_dict[f'CBC_{i}'])[:,1]
                                nHits_3sigma = np.array(common_noise_3sigma_hb0_dict[f'CBC_{i}'])[:,0]
                                sigmaHits_3sigma = np.array(common_noise_3sigma_hb0_dict[f'CBC_{i}'])[:,1]

                                logger.info(f"fitting nHits for CBC_{i}")
                                sigma_fit, k_probs_fit, res = CMNmod.fit_k_and_sigma_from_hists(nHits_0sigma,
                                                                                                nHits_3sigma,
                                                                                                sigmaHits_0sigma,
                                                                                                sigmaHits_3sigma)
                                
                                P_A, P_B, R_A, R_B = CMNmod.predict_distributions(sigma_fit, k_probs_fit)
                                expA = 10000 * P_A
                                expB = 10000 * P_B
                                #from IPython import embed; embed(); exit()

                                N_BINS_K = 20
                                K_MIN, K_MAX = -2.0, 2.0
                                BIN_EDGES = np.linspace(K_MIN, K_MAX, N_BINS_K + 1)
                                BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
                                BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]

                                
                                x = [ [np.arange(nHits_0sigma.shape[0]), np.arange(len(expA))] ,
                                      [np.arange(nHits_3sigma.shape[0]), np.arange(len(expB))] ,
                                      [BIN_CENTERS] ]
                                y = [ [nHits_0sigma/10000, P_A] ,
                                      [nHits_3sigma/10000, P_B] ,
                                      [k_probs_fit] ]
                                h3_width = BIN_WIDTH*0.9
                                
                                self.plot_fitted_result(x = x,
                                                        y = y,
                                                        w = h3_width,
                                                        labels=["nHits : 0 sigma", "nHits : 3 sigma", "CMNoise"],
                                                        title = f"CMN_fitted_hb0_{martaTemp}_{moduleID}_CBC{i}",
                                                        name = f"Plot_CommonNoiseCBC_hb0_bothSensors_{martaTemp}_{moduleID}_{datakey}_CBC_{i}",
                                                        outdir = _outdirCBC)

                                from IPython import embed; embed()
                                fracs.append(R_A['frac_common'])

                            common_noise_frac_simfit_hb0_cbc_mod[martaTemp].append(fracs)
                            """
                        else:
                            logger.warning("Skip common noise extraction by fitting nHits simultaneously for 0 & 3 sigma noise")

                            
                    else:
                        logger.warning("skip plotting common mode noise")



                    if self.testinfo.get("check_pedestal") == True:
                        pede_hb0_dict = noiseDict['pedestal_hb0']
                        pede_hb1_dict = noiseDict['pedestal_hb1']
                        # Total pede
                        pede_hb0 = pede_hb0_dict['CBC_0'] + pede_hb0_dict['CBC_1'] + pede_hb0_dict['CBC_2'] + pede_hb0_dict['CBC_3'] + pede_hb0_dict['CBC_4'] + pede_hb0_dict['CBC_5'] + pede_hb0_dict['CBC_6'] + pede_hb0_dict['CBC_7']
                        pede_hb1 = pede_hb1_dict['CBC_0'] + pede_hb1_dict['CBC_1'] + pede_hb1_dict['CBC_2'] + pede_hb1_dict['CBC_3'] + pede_hb1_dict['CBC_4'] + pede_hb1_dict['CBC_5'] + pede_hb1_dict['CBC_6'] + pede_hb1_dict['CBC_7']
                        self.plot_basic(x          = np.arange(len(pede_hb0)),
                                        data_list  = [pede_hb0, pede_hb1],
                                        legends    = ["hybrid 0", "hybrid 1"],
                                        title      = f"Pedestal_{moduleID}_{martaTemp}",
                                        name       = f"Plot_Pedestal_bothHybrids_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Channel",
                                        ylabel     = "Pedestal [VcTh]",
                                        ylim       = [595.0,605.0],
                                        outdir     = _outdirMod)
                    
                        self.hist_basic(bins       = np.linspace(590,610,80),
                                        data_list  = [pede_hb0_dict['CBC_0'],
                                                      pede_hb0_dict['CBC_1'],
                                                      pede_hb0_dict['CBC_2'],
                                                      pede_hb0_dict['CBC_3'],
                                                      pede_hb0_dict['CBC_4'],
                                                      pede_hb0_dict['CBC_5'],
                                                      pede_hb0_dict['CBC_6'],
                                                      pede_hb0_dict['CBC_7']],
                                        legends    = ["Chip 0", "Chip 1", "Chip 2", "Chip 3", "Chip 4", "Chip 5", "Chip 6", "Chip 7"],
                                        title      = f"PedestalCBC_Hb0_{moduleID}_{martaTemp}",
                                        name       = f"Hist_PedestalCBC_Hybrid0_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Pedestal [VcTh]",
                                        ylabel     = "Entries",
                                        linewidth  = 1.2,
                                        outdir     = _outdirModCBC,
                                        colors     = ["#0B1A2F", "#152C4D", "#1F3E6C", "#29508A", "#3362A9", "#3D74C7", "#4796E6", "#61B4FF"],
                                        linestyles = 8*["-"])
                        self.hist_basic(bins       = np.linspace(590,610,80),
                                        data_list  = [pede_hb1_dict['CBC_0'],
                                                      pede_hb1_dict['CBC_1'],
                                                      pede_hb1_dict['CBC_2'],
                                                      pede_hb1_dict['CBC_3'],
                                                      pede_hb1_dict['CBC_4'],
                                                      pede_hb1_dict['CBC_5'],
                                                      pede_hb1_dict['CBC_6'],
                                                      pede_hb1_dict['CBC_7']],
                                        legends    = ["Chip 0", "Chip 1", "Chip 2", "Chip 3", "Chip 4", "Chip 5", "Chip 6", "Chip 7"],
                                        title      = f"PedestalCBC_Hb1_{moduleID}_{martaTemp}",
                                        name       = f"Hist_PedestalCBC_Hybrid1_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Pedestal [VcTh]",
                                        ylabel     = "Entries",
                                        linewidth  = 1.2,
                                        outdir     = _outdirModCBC,
                                        colors     = ["#2E0000", "#4A0A0A", "#661515", "#821F1F", "#9E2A2A", "#BA3535", "#D65050", "#F26B6B"],
                                        linestyles = 8*["-"])

                        
                        
                        # Get the mean and std of noise per CBC
                        pede_hb0_per_cbc_mean_std_list = self.__get_mean_std_per_cbc(pede_hb0_dict)
                        pede_hb1_per_cbc_mean_std_list = self.__get_mean_std_per_cbc(pede_hb1_dict)
                        
                        self.plot_group(x           = np.arange(8),
                                        data_list   = [[pede_hb0_per_cbc_mean_std_list,
                                                        pede_hb1_per_cbc_mean_std_list]],
                                        legends     = [["hybrid 0",
                                                        "hybrid 1"]],
                                        title       = f"Pedestal_{martaTemp}_{moduleID}",
                                        name        = f"Plot_PedestalCBC_bothHybrids_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                        xticklabels = [f"CBC_{i}" for i in range(8)],
                                        ylim        = [595, 605],
                                        ylabel      = "Pedestal [VcTh]",
                                        outdir      = _outdirModCBC,
                                        marker      = "o",
                                        markersize  = 2.5,
                                        capsize     = 1.5,
                                        elinewidth  = 1.0)
                        
                        
                        
                # Loop over temp ends here
            # Loop over modules end here
            allModuleIDs += moduleIDs

            
            temps = list(strip_noise_hb0_bot_mod.keys())
            logger.info("Plotting Noise per Module ...")
            for temp in temps:
                logger.info(temp)
                if self.testinfo.get("check_sensor_temperature") == True:
                    sensor_temps = sensor_temps_mod[temp]
                    sensor_temps = np.array(sensor_temps)
                    #from IPython import embed; embed(); exit()
                
                    self.plot_basic(x          = np.arange(sensor_temps.shape[0]),
                                    data_list  = [np.concatenate((sensor_temps[:,None], np.zeros_like(sensor_temps)[:,None]), axis=1)],
                                    legends    = ["sensor temp"],
                                    title      = f"Sensor Temperature [{temp}]",
                                    name       = f"Plot_SensorTemp_allMods_{temp}_{datakey}",
                                    xticklabels= moduleIDs,
                                    ylabel     = "Sensor Temperature (deg C)",
                                    marker     = "o",
                                    linewidth  = 1.5,
                                    markersize = 4.5,
                                    ylim       = [18.0,29.0],
                                    outdir     = _outdir)
                else:
                    logger.warning("skip plotting sensor temperature")

                
                noise_hb0 = strip_noise_hb0_mod[temp]
                noise_hb1 = strip_noise_hb1_mod[temp]          
                self.plot_box(data_list_1 = noise_hb0,
                              data_list_2 = noise_hb1,
                              legends     = ["hybrid 0", "hybrid 1"],
                              title       = f"StripNoise_{temp}",
                              name        = f"Plot_StripNoiseBox_allModules_bothHybrids_{temp}_{datakey}",
                              xticklabels = moduleIDs,
                              ylabel      = "Noise [VcTh]",
                              outdir      = _outdir)
                
                bot_noise_hb0 = strip_noise_hb0_bot_mod[temp]
                bot_noise_hb1 = strip_noise_hb1_bot_mod[temp]                
                self.plot_box(data_list_1 = bot_noise_hb0,
                              data_list_2 = bot_noise_hb1,
                              legends     = ["hybrid 0", "hybrid 1"],
                              title       = f"StripNoise_bottomSensor_{temp}",
                              name        = f"Plot_StripNoiseBox_allModules_bothHybrids_bottomSensor_{temp}_{datakey}",
                              xticklabels = moduleIDs,
                              ylabel      = "Noise [VcTh]",
                              outdir      = _outdir)
                top_noise_hb0 = strip_noise_hb0_top_mod[temp]
                top_noise_hb1 = strip_noise_hb1_top_mod[temp]                
                self.plot_box(data_list_1 = top_noise_hb0,
                              data_list_2 = top_noise_hb1,
                              legends     = ["hybrid 0", "hybrid 1"],
                              title       = f"StripNoise_topSensor_{temp}",
                              name        = f"Plot_StripNoiseBox_allModules_bothHybrids_topSensor_{temp}_{datakey}",
                              xticklabels = moduleIDs,
                              ylabel      = "Noise [VcTh]",
                              outdir      = _outdir)
                # Preparing to plot the mean and std
                bot_noise_hb0 = np.array(bot_noise_hb0)[:,:,0]
                bot_noise_hb0_mean_std = np.concatenate((np.mean(bot_noise_hb0, axis=1)[:,None],
                                                         np.std(bot_noise_hb0, axis=1)[:,None]), axis=1)
                bot_noise_hb1 = np.array(bot_noise_hb1)[:,:,0]
                bot_noise_hb1_mean_std = np.concatenate((np.mean(bot_noise_hb1, axis=1)[:,None],
                                                         np.std(bot_noise_hb1, axis=1)[:,None]), axis=1)
                top_noise_hb0 = np.array(top_noise_hb0)[:,:,0]
                top_noise_hb0_mean_std = np.concatenate((np.mean(top_noise_hb0, axis=1)[:,None],
                                                         np.std(top_noise_hb0, axis=1)[:,None]), axis=1)
                top_noise_hb1 = np.array(top_noise_hb1)[:,:,0]
                top_noise_hb1_mean_std = np.concatenate((np.mean(top_noise_hb1, axis=1)[:,None],
                                                         np.std(top_noise_hb1, axis=1)[:,None]), axis=1)


                self.plot_group(x           = np.arange(len(moduleIDs)),
                                data_list   = [[bot_noise_hb0_mean_std,
                                                bot_noise_hb1_mean_std],
                                               [top_noise_hb0_mean_std,
                                                top_noise_hb1_mean_std]],
                                legends     = [["hb0 (bottom sensor)", "hb1 (bottom sensor)"],
                                               ["hb0 (top sensor)",    "hb1 (top sensor)"]],
                                title       = f"StripNoise_{temp}",
                                name        = f"Plot_StripNoise_allModules_bothHybrids_bothSensors_{temp}_{datakey}",
                                xticklabels = moduleIDs,
                                ylim        = [3.5,8.5],
                                ylabel      = "Noise [VcTh]",
                                outdir      = _outdir,
                                marker      = "o",
                                markersize  = 2.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0)


                noisy_channels_hb0_cbc = np.array(num_noisy_channels_hb0_cbc_mod[temp])
                noisy_channels_hb1_cbc = np.array(num_noisy_channels_hb1_cbc_mod[temp])
                noisy_channels_cbc = np.concatenate((noisy_channels_hb0_cbc,
                                                     np.zeros_like(noisy_channels_hb0_cbc[:,:1]),
                                                     noisy_channels_hb1_cbc), axis=1)

                #from IPython import embed; embed()                
                self.plot_heatmap(noisy_channels_cbc.astype(int),
                                  title       = f"nNoisyCh: {temp}",
                                  name        = f"Plot_NumNoisyCh_bothHybrids_{temp}_{datakey}",
                                  xticklabels = moduleIDs,
                                  yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                  colmap      = "coolwarm",
                                  cb_label    = "nChannels",
                                  #vmin        = 0.5,
                                  #vmax        = 1.5,
                                  outdir      = _outdirCBC)

                dead_channels_hb0_cbc = np.array(num_dead_channels_hb0_cbc_mod[temp])
                dead_channels_hb1_cbc = np.array(num_dead_channels_hb1_cbc_mod[temp])
                dead_channels_cbc = np.concatenate((dead_channels_hb0_cbc,
                                                     np.zeros_like(dead_channels_hb0_cbc[:,:1]),
                                                     dead_channels_hb1_cbc), axis=1)

                self.plot_heatmap(dead_channels_cbc.astype(int),
                                  title       = f"nDeadCh: {temp}",
                                  name        = f"Plot_NumDeadCh_bothHybrids_{temp}_{datakey}",
                                  xticklabels = moduleIDs,
                                  yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                  colmap      = "coolwarm",
                                  cb_label    =	"nChannels",
                                  #vmin        = 0.5,
                                  #vmax        = 1.5,
                                  outdir      = _outdirCBC)
                
                

                # same for common mode noise
                if self.testinfo.get("check_common_noise") == True:


                    if self.testinfo.get("fit_simultaneous_common_noise") == True:
                        cmn_frac_hb0_cbc = np.array(common_noise_frac_simfit_hb0_cbc_mod[temp])
                        cmn_frac_hb1_cbc = np.array(common_noise_frac_simfit_hb1_cbc_mod[temp])
                        cmn_frac_cbc = np.concatenate((cmn_frac_hb0_cbc,
                                                       np.zeros_like(cmn_frac_hb0_cbc[:,:1]),
                                                       cmn_frac_hb1_cbc), axis=1)
                        
                        self.plot_heatmap(cmn_frac_cbc,
                                          title       = f"CMNFrac_SimFit: {temp}",
                                          name        = f"Plot_CMNFrac_SimFit_bothHybrids_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                          colmap      = "coolwarm",
                                          cb_label    = "nChannels",
                                          vmin        = 0.0,
                                          vmax        = 1.0,
                                          outdir      = _outdirCBC)
                        
                    
                    cmn_noise_hb0_mean_std, cmn_noise_hb0_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_mod[temp])[:,:,0])
                    cmn_noise_hb1_mean_std, cmn_noise_hb1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_mod[temp])[:,:,0])
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_hb0_mean_std, cmn_noise_hb1_mean_std],
                                    legends    = ["Hybrid 0", "Hybrid 1"],
                                    title      = f"#Hits (50% Occ) (µ) {temp}",
                                    name       = f"Plot_nHitsMean_bothHybrids_{temp}_{datakey}",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (µ)",
                                    markersize = 10,
                                    ylim       = [500.0,1200.0],
                                    outdir     = _outdir,
                                    fit        = False)
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_hb0_sigma_std, cmn_noise_hb1_sigma_std],
                                    legends    = ["Hybrid 0", "Hybrid 1"],
                                    title      = f"#Hits (50% Occ) (σ) {temp}",
                                    name       = f"Plot_nHitsStd_bothHybrids_{temp}_{datakey}",
                                    xticklabels = moduleIDs,
                                    ylabel     = "CM Noise",
                                    markersize = 10,
                                    ylim       = [50,250],
                                    outdir     = _outdir,
                                    fit        = False)
                    

                    cmn_noise_hb0_bot_mean_std, cmn_noise_hb0_bot_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_bot_mod[temp])[:,:,0])
                    cmn_noise_hb1_bot_mean_std, cmn_noise_hb1_bot_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_bot_mod[temp])[:,:,0])
                    
                    cmn_noise_hb0_top_mean_std, cmn_noise_hb0_top_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_top_mod[temp])[:,:,0])                
                    cmn_noise_hb1_top_mean_std, cmn_noise_hb1_top_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_top_mod[temp])[:,:,0])
                    
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_hb0_bot_mean_std, cmn_noise_hb1_bot_mean_std,
                                                  cmn_noise_hb0_top_mean_std, cmn_noise_hb1_top_mean_std],
                                    legends    = ["Hybrid 0 bottom", "Hybrid 1 bottom",
                                                  "Hybrid 0 top", "Hybrid 1 top"],
                                    title      = f"#Hits (50% Occ) (µ){temp}",
                                    name       = f"Plot_nHitsMean_bothHybrids_bothSensors_{temp}_{datakey}",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (µ)",
                                    markersize = 10,
                                    ylim       = [100,700],
                                    outdir     = _outdir,
                                    fit        = False)
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_hb0_bot_sigma_std, cmn_noise_hb1_bot_sigma_std,
                                                  cmn_noise_hb0_top_sigma_std, cmn_noise_hb1_top_sigma_std],
                                    legends    = ["Hybrid 0 bottom", "Hybrid 1 bottom",
                                                  "Hybrid 0 top", "Hybrid 1 top"],
                                    title      = f"#Hits (50% Occ) (σ) {temp}",
                                    name       = f"Plot_nHitsStd_bothHybrids_bothSensors_{temp}_{datakey}",
                                    xticklabels = moduleIDs,
                                    ylabel     = "CM Noise",
                                    markersize = 10,
                                    ylim       = [20,160],
                                    outdir     = _outdir,
                                    fit        = False)



                    common_noise_giovanni_hb0_cbc = np.array(common_noise_giovanni_hb0_cbc_mod[temp])
                    common_noise_giovanni_hb1_cbc = np.array(common_noise_giovanni_hb1_cbc_mod[temp])
                    common_noise_giovanni_cbc = np.concatenate((common_noise_giovanni_hb0_cbc,
                                                                np.zeros_like(common_noise_giovanni_hb0_cbc[:,:1]),
                                                                common_noise_giovanni_hb1_cbc), axis=1)
                    
                    self.plot_heatmap(common_noise_giovanni_cbc,
                                      title       = f"CMN frac: {temp}",
                                      name        = f"Plot_CMN_Fraction_Giovanni_bothHybrids_{temp}_{datakey}",
                                      xticklabels = moduleIDs,
                                      yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                      colmap      = "coolwarm",
                                      #vmin        = 0.5,
                                      #vmax        = 1.5,
                                      outdir      = _outdirCBC) 
                    
                    common_noise_iphc_hb0_cbc = np.array(common_noise_iphc_hb0_cbc_mod[temp])
                    common_noise_iphc_hb1_cbc = np.array(common_noise_iphc_hb1_cbc_mod[temp])
                    common_noise_iphc_cbc = np.concatenate((common_noise_iphc_hb0_cbc,
                                                            np.zeros_like(common_noise_iphc_hb0_cbc[:,:1]),
                                                            common_noise_iphc_hb1_cbc), axis=1)

                    self.plot_heatmap(common_noise_iphc_cbc,
                                      title       = f"CMN frac: {temp}",
                                      name        = f"Plot_CMN_Fraction_IPHC_bothHybrids_{temp}_{datakey}",
                                      xticklabels = moduleIDs,
                                      yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                      colmap      = "coolwarm",
                                      vmin        = 0.0,
                                      vmax        = 0.5,
                                      outdir      = _outdirCBC)
                    
                    
                    common_noise_potato_hb0_cbc = np.array(common_noise_frac_potato_hb0_cbc_mod[temp])
                    common_noise_potato_hb1_cbc = np.array(common_noise_frac_potato_hb1_cbc_mod[temp])
                    common_noise_potato_cbc = np.concatenate((common_noise_potato_hb0_cbc,
                                                              np.zeros_like(common_noise_potato_hb0_cbc[:,:1]),
                                                              common_noise_potato_hb1_cbc), axis=1)

                    self.plot_heatmap(common_noise_potato_cbc,
                                      title       = f"CMN frac: {temp}",
                                      name        = f"Plot_CMN_Fraction_Potato_bothHybrids_{temp}_{datakey}",
                                      xticklabels = moduleIDs,
                                      yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                      colmap      = "coolwarm",
                                      #vmin        = 0.5,
                                      #vmax        = 1.5,
                                      outdir      = _outdirCBC)


    
                    #from IPython import embed; embed(); exit()
                    common_noise_fit_hb0_cbc = common_noise_fit_hb0_cbc_mod[temp]
                    is_empty = all(not x for x in common_noise_fit_hb0_cbc)
                    if not is_empty:
                        self.plot_heatmap(common_noise_fit_hb0_cbc,
                                          title       = f"CMN-hb0: {temp}",
                                          name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb0_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          colmap      = "coolwarm",
                                          outdir      = _outdirCBC) 
                        common_noise_fit_hb0_cbc_top = common_noise_fit_hb0_cbc_top_mod[temp]
                        self.plot_heatmap(common_noise_fit_hb0_cbc_top,
                                          title       = f"CMN-hb0-top: {temp}",
                                          name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb0_top_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          colmap      = "coolwarm",
                                          outdir      = _outdirCBC)
                        common_noise_fit_hb0_cbc_bot = common_noise_fit_hb0_cbc_bot_mod[temp]
                        self.plot_heatmap(common_noise_fit_hb0_cbc_bot,
                                          title       = f"CMN-hb0-bot: {temp} ",
                                          name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb0_bot_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          colmap      = "coolwarm",
                                          outdir      = _outdirCBC)

                        common_noise_fit_hb1_cbc = common_noise_fit_hb1_cbc_mod[temp]
                        self.plot_heatmap(common_noise_fit_hb1_cbc,
                                          title       = f"CMN-hb1: {temp}",
                                          name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb1_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          colmap      = "coolwarm",
                                          outdir      = _outdirCBC) 
                        common_noise_fit_hb1_cbc_top = common_noise_fit_hb1_cbc_top_mod[temp]
                        self.plot_heatmap(common_noise_fit_hb1_cbc_top,
                                          title       = f"CMN-hb1-top: {temp}",
                                          name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb1_top_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          colmap      = "coolwarm",
                                          outdir      = _outdirCBC)
                        common_noise_fit_hb1_cbc_bot = common_noise_fit_hb1_cbc_bot_mod[temp]
                        self.plot_heatmap(common_noise_fit_hb1_cbc_bot,
                                          title       = f"CMN-hb1-bot: {temp}",
                                          name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb1_bot_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          colmap      = "coolwarm",
                                          outdir      = _outdirCBC)




                        self.plot_heatmap((np.array(common_noise_giovanni_hb0_cbc)/np.array(common_noise_fit_hb0_cbc)).tolist(),
                                          title       = f"CMN-hb0-ratio: {temp}",
                                          name        = f"Plot_CMN_Fraction_Giovanni_by_fit_hb0_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          colmap      = "coolwarm",
                                          vmin        = 0.5, vmax = 1.5,
                                          cb_label    = "Giovanni_Form/Ph2ACF_Fit",
                                          outdir      = _outdirCBC) 
                        self.plot_heatmap((np.array(common_noise_giovanni_hb1_cbc)/np.array(common_noise_fit_hb1_cbc)).tolist(),
                                          title       = f"CMN-hb1-ratio: {temp}",
                                          name        = f"Plot_CMN_Fraction_Giovanni_by_fit_hb1_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          colmap      = "coolwarm",
                                          vmin        = 0.5, vmax = 1.5,
                                          cb_label    = "Giovanni_Form/Ph2ACF_Fit",
                                          outdir      = _outdirCBC) 
                        self.plot_heatmap((np.array(common_noise_iphc_hb0_cbc)/np.array(common_noise_fit_hb0_cbc)).tolist(),
                                          title       = f"CMN-hb0-ratio: {temp}",
                                          name        = f"Plot_CMN_Fraction_Iphc_by_fit_hb0_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          colmap      = "coolwarm",
                                          vmin        = 0.5, vmax = 1.5,
                                          cb_label    = "IPHC_Form/Ph2ACF_Fit",
                                          outdir      = _outdirCBC) 
                        self.plot_heatmap((np.array(common_noise_iphc_hb1_cbc)/np.array(common_noise_fit_hb1_cbc)).tolist(),
                                          title       = f"CMN-hb1-ratio: {temp}",
                                          name        = f"Plot_CMN_Fraction_Iphc_by_fit_hb1_{temp}_{datakey}",
                                          xticklabels = moduleIDs,
                                          colmap      = "coolwarm",
                                          vmin        = 0.5, vmax = 1.5,
                                          cb_label	  = "IPHC_Form/Ph2ACF_Fit",
                                          outdir      = _outdirCBC) 
                        
                    common_noise_crude_hb0_cbc = common_noise_crude_hb0_cbc_mod[temp]
                    #self.plot_heatmap(common_noise_crude_hb0_cbc,
                    #                  title       = f"CMN-hb0: {temp}",
                    #                  name        = f"Plot_CMN_Fraction_Crude_hb0_{temp}_{datakey}",
                    #                  xticklabels = moduleIDs,
                    #                  colmap      = "coolwarm",
                    #                  vmin        = 0.5, vmax = +1.0,
                    #                  cb_label	  = "(σ - 0.5x√µ)/σ",
                    #                  outdir      = _outdirCBC) 
                    common_noise_crude_hb1_cbc = common_noise_crude_hb1_cbc_mod[temp]
                    #self.plot_heatmap(common_noise_crude_hb1_cbc,
                    #                  title       = f"CMN-hb1: {temp}",
                    #                  name        = f"Plot_CMN_Fraction_Crude_hb1_{temp}_{datakey}",
                    #                  xticklabels = moduleIDs,
                    #                  colmap      = "coolwarm",
                    #                  vmin        = 0.5, vmax = +1.0,
                    #                  cb_label    = "(σ - 0.5x√µ)/σ",
                    #                  outdir      = _outdirCBC) 

                    
                else:
                    logger.warning("skipping module wise common mode noise comparison")

            sensor_temps_setup[datakey] = sensor_temps_mod
            strip_noise_hb0_setup[datakey] = strip_noise_hb0_mod
            strip_noise_hb0_bot_setup[datakey] = strip_noise_hb0_bot_mod
            strip_noise_hb0_top_setup[datakey] = strip_noise_hb0_top_mod
            strip_noise_hb1_setup[datakey] = strip_noise_hb1_mod
            strip_noise_hb1_bot_setup[datakey] = strip_noise_hb1_bot_mod
            strip_noise_hb1_top_setup[datakey] = strip_noise_hb1_top_mod

            num_noisy_channels_hb0_setup[datakey] = num_noisy_channels_hb0_mod
            num_noisy_channels_hb1_setup[datakey] = num_noisy_channels_hb1_mod
            num_dead_channels_hb0_setup[datakey] = num_dead_channels_hb0_mod
            num_dead_channels_hb1_setup[datakey] = num_dead_channels_hb1_mod
            
            common_noise_setup[datakey] = common_noise_mod
            common_noise_bot_setup[datakey] = common_noise_bot_mod
            common_noise_top_setup[datakey] = common_noise_top_mod
            common_noise_hb0_setup[datakey] = common_noise_hb0_mod
            common_noise_hb0_bot_setup[datakey] = common_noise_hb0_bot_mod
            common_noise_hb0_top_setup[datakey] = common_noise_hb0_top_mod
            common_noise_hb1_setup[datakey] = common_noise_hb1_mod
            common_noise_hb1_bot_setup[datakey] = common_noise_hb1_bot_mod
            common_noise_hb1_top_setup[datakey] = common_noise_hb1_top_mod

            common_noise_fit_hb0_cbc_setup[datakey] = common_noise_fit_hb0_cbc_mod
            common_noise_fit_hb0_cbc_top_sensor_setup[datakey] = common_noise_fit_hb0_cbc_top_mod
            common_noise_fit_hb0_cbc_bot_sensor_setup[datakey] = common_noise_fit_hb0_cbc_bot_mod
            common_noise_fit_hb1_cbc_setup[datakey] = common_noise_fit_hb1_cbc_mod
            common_noise_fit_hb1_cbc_top_sensor_setup[datakey] = common_noise_fit_hb1_cbc_top_mod
            common_noise_fit_hb1_cbc_bot_sensor_setup[datakey] = common_noise_fit_hb1_cbc_bot_mod

            common_noise_giovanni_hb0_cbc_setup[datakey] = common_noise_giovanni_hb0_cbc_mod
            common_noise_giovanni_hb1_cbc_setup[datakey] = common_noise_giovanni_hb1_cbc_mod

            common_noise_iphc_hb0_cbc_setup[datakey] = common_noise_iphc_hb0_cbc_mod
            common_noise_iphc_hb1_cbc_setup[datakey] = common_noise_iphc_hb1_cbc_mod

            common_noise_crude_hb0_cbc_setup[datakey] = common_noise_crude_hb0_cbc_mod
            common_noise_crude_hb1_cbc_setup[datakey] = common_noise_crude_hb1_cbc_mod

            common_noise_frac_potato_hb0_cbc_setup[datakey] = common_noise_frac_potato_hb0_cbc_mod
            common_noise_frac_potato_hb1_cbc_setup[datakey] = common_noise_frac_potato_hb1_cbc_mod
            
            
        # Loop over setup ends here    
        #from IPython import embed; embed(); exit()

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
        #                                          Comparing two setup                                               #
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
        if self.testinfo.get("compare_two_setup") == True:
            logger.info(" ===> Comparing two setup ...")
            setup_keys = list(strip_noise_hb0_bot_setup.keys())
            setup_1 = setup_keys[0]
            setup_2 = setup_keys[1]
        
            logger.info(f"Setup-1 : {setup_1}, Setup-2 : {setup_2}")
        
            tick_offset = 0.1
            box_offset = 0.1
            if self.testinfo.get("are_same_modules") == False:
                moduleIDs = np.array(allModuleIDs).reshape(2,-1).T.tolist()
                moduleIDs = [f"{mid[0]}\n{mid[1]}" for mid in moduleIDs]
                tick_offset = 0.22
                box_offset = 0.52

            cool_temps_setup_1 = list(strip_noise_hb0_bot_setup[setup_1].keys())
            cool_temps_setup_2 = list(strip_noise_hb0_bot_setup[setup_2].keys())

            cool_temps = [temp for temp in cool_temps_setup_1 if temp in cool_temps_setup_2]
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
            #                                     WARNING: FEW THIGS ARE HARDCODED                                         #
            #                        Here, one can use hardcoding to compare different setup                               #
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
            #from IPython import embed; embed(); exit()

            for temp in cool_temps:
                logger.info(f"Cooling temperature : {temp}")

                #from IPython import embed; embed()
                n_noisy_channels_hb0_set_1 = np.array(num_noisy_channels_hb0_setup[setup_1][temp])
                n_noisy_channels_hb1_set_1 = np.array(num_noisy_channels_hb1_setup[setup_1][temp])
                n_noisy_channels_hb0_set_2 = np.array(num_noisy_channels_hb0_setup[setup_2][temp])
                n_noisy_channels_hb1_set_2 = np.array(num_noisy_channels_hb1_setup[setup_2][temp])
                
                self.plot_group(x           = np.arange(len(moduleIDs)),
                                data_list   = [[np.concatenate((n_noisy_channels_hb0_set_1[:,None], np.zeros_like(n_noisy_channels_hb0_set_1)[:,None]), axis=1).tolist(),
                                                np.concatenate((n_noisy_channels_hb1_set_1[:,None], np.zeros_like(n_noisy_channels_hb1_set_1)[:,None]), axis=1).tolist()],
                                               [np.concatenate((n_noisy_channels_hb0_set_2[:,None], np.zeros_like(n_noisy_channels_hb0_set_2)[:,None]), axis=1).tolist(),
                                                np.concatenate((n_noisy_channels_hb1_set_2[:,None], np.zeros_like(n_noisy_channels_hb1_set_2)[:,None]), axis=1).tolist()]],
                                legends     = [[f"hb0_{setup_1}", f"hb1_{setup_1}"], [f"hb0_{setup_2}", f"hb1_{setup_2}"]],
                                title       = f"n_noisy_ch ({temp})",
                                name        = f"Plot_NoisyChannels_allModules_{temp}_compare",
                                xticklabels = moduleIDs,
                                #ylim        = [4.0,8.0],
                                ylabel      = "nNoisyChannels",
                                outdir      = self.outdir,
                                marker      = "o",
                                markerfacecolor=None,
                                markersize  = 7.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0,
                                tick_offset = tick_offset)



                n_dead_channels_hb0_set_1 = np.array(num_dead_channels_hb0_setup[setup_1][temp])
                n_dead_channels_hb1_set_1 = np.array(num_dead_channels_hb1_setup[setup_1][temp])
                n_dead_channels_hb0_set_2 = np.array(num_dead_channels_hb0_setup[setup_2][temp])
                n_dead_channels_hb1_set_2 = np.array(num_dead_channels_hb1_setup[setup_2][temp])
                
                self.plot_group(x           = np.arange(len(moduleIDs)),
                                data_list   = [[np.concatenate((n_dead_channels_hb0_set_1[:,None], np.zeros_like(n_dead_channels_hb0_set_1)[:,None]), axis=1).tolist(),
                                                np.concatenate((n_dead_channels_hb1_set_1[:,None], np.zeros_like(n_dead_channels_hb1_set_1)[:,None]), axis=1).tolist()],
                                               [np.concatenate((n_dead_channels_hb0_set_2[:,None], np.zeros_like(n_dead_channels_hb0_set_2)[:,None]), axis=1).tolist(),
                                                np.concatenate((n_dead_channels_hb1_set_2[:,None], np.zeros_like(n_dead_channels_hb1_set_2)[:,None]), axis=1).tolist()]],
                                legends     = [[f"hb0_{setup_1}", f"hb1_{setup_1}"], [f"hb0_{setup_2}", f"hb1_{setup_2}"]],
                                title       = f"n_Dead_ch ({temp})",
                                name        = f"Plot_DeadChannels_allModules_{temp}_compare",
                                xticklabels = moduleIDs,
                                #ylim        = [4.0,8.0],
                                ylabel      = "nDeadChannels",
                                outdir      = self.outdir,
                                marker      = "o",
                                markerfacecolor=None,
                                markersize  = 7.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0,
                                tick_offset = tick_offset)

                
                
                if self.testinfo.get("check_sensor_temperature") == True:
                    # ===>> Sensor temperature
                    sensor_temps_setup_1 = np.array(sensor_temps_setup[setup_1][temp])
                    sensor_temps_setup_2 = np.array(sensor_temps_setup[setup_2][temp])
                    #from IPython import embed; embed(); exit()
                    
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [np.concatenate((sensor_temps_setup_1[:,None], np.zeros_like(sensor_temps_setup_1)[:,None]), axis=1).tolist(),
                                                  np.concatenate((sensor_temps_setup_2[:,None], np.zeros_like(sensor_temps_setup_2)[:,None]), axis=1).tolist()],
                                    legends    = [f"{setup_1}", f"{setup_2}"],
                                    title      = f"Sensor Temperature ({temp})",
                                    name       = f"Plot_SensorTemp_allModules_{temp}_compare",
                                    xticklabels= moduleIDs,
                                    ylabel     = "Sensor Temperature (deg C)",
                                    marker     = "o",
                                    linewidth  = 1.5,
                                    markersize = 4.5,
                                    ylim       = [18.0,29.0],
                                    outdir     = self.outdir,
                                    tick_offset = tick_offset)
                else:
                    logger.warning("skip comparing sensor temperature")

            
                # ===>> strip noise hybrid 0
                strip_noise_hb0_setup_1 = strip_noise_hb0_setup[setup_1][temp]
                strip_noise_hb0_setup_2 = strip_noise_hb0_setup[setup_2][temp]
                # box plot
                self.plot_box(data_list_1 = strip_noise_hb0_setup_1,
                              data_list_2 = strip_noise_hb0_setup_2,
                              legends     = [f"{setup_1}", f"{setup_2}"],
                              title       = f"StripNoise_hb0 ({temp})",
                              name        = f"Plot_StripNoiseBox_allModules_Hybrid0_{temp}_compare",
                              xticklabels = moduleIDs,
                              ylabel      = "Noise [VcTh]",
                              offset      = box_offset,
                              outdir      = self.outdir,
                              box_offset  = box_offset)
                # group plot
                strip_noise_hb0_setup_1_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_setup_1)[:,:,0])
                strip_noise_hb0_setup_2_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_setup_2)[:,:,0])
                self.plot_group(x           = np.arange(len(moduleIDs)),
                                data_list   = [[strip_noise_hb0_setup_1_mean_std,
                                                strip_noise_hb0_setup_2_mean_std]],
                                legends     = [[f"{setup_1}", f"{setup_2}"]],
                                title       = f"StripNoise_hb0 ({temp})",
                                name        = f"Plot_StripNoise_allModules_hybrid0_{temp}_compare",
                                xticklabels = moduleIDs,
                                ylim        = [4.0,8.0],
                                ylabel      = "Noise [VcTh]",
                                outdir      = self.outdir,
                                marker      = "o",
                                markerfacecolor=None,
                                markersize  = 5.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0,
                                tick_offset = tick_offset)


                # ===>> strip noise hybrid 1
                # box plot
                strip_noise_hb1_setup_1 = strip_noise_hb1_setup[setup_1][temp]
                strip_noise_hb1_setup_2 = strip_noise_hb1_setup[setup_2][temp]
                self.plot_box(data_list_1 = strip_noise_hb1_setup_1,
                              data_list_2 = strip_noise_hb1_setup_2,
                              legends     = [f"{setup_1}", f"{setup_2}"],
                              title       = f"StripNoise_hb1 ({temp})",
                              name        = f"Plot_StripNoiseBox_allModules_Hybrid1_{temp}_compare",
                              xticklabels = moduleIDs,
                              ylabel      = "Noise [VcTh]",
                              offset      = box_offset,
                              outdir      = self.outdir,
                              box_offset  = box_offset)
                # group plot
                strip_noise_hb1_setup_1_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_setup_1)[:,:,0])
                strip_noise_hb1_setup_2_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_setup_2)[:,:,0])
                self.plot_group(x           = np.arange(len(moduleIDs)),
                                data_list   = [[strip_noise_hb1_setup_1_mean_std,
                                                strip_noise_hb1_setup_2_mean_std]],
                                legends     = [[f"{setup_1}", f"{setup_2}"]],
                                title       = f"StripNoise_hb1 ({temp})",
                                name        = f"Plot_StripNoise_allModules_hybrid1_{temp}_compare",
                                xticklabels = moduleIDs,
                                ylim        = [4.0,8.0],
                                ylabel      = "Noise [VcTh]",
                                outdir      = self.outdir,
                                marker      = "o",
                                markerfacecolor=None,
                                markersize  = 5.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0,
                                tick_offset = tick_offset)
                
                
                # ===>> strip noise hybrid 0 (bottom)
                # box plot
                strip_noise_hb0_bot_setup_1 = strip_noise_hb0_bot_setup[setup_1][temp]
                strip_noise_hb0_bot_setup_2 = strip_noise_hb0_bot_setup[setup_2][temp]
                self.plot_box(data_list_1 = strip_noise_hb0_bot_setup_1,
                              data_list_2 = strip_noise_hb0_bot_setup_2,
                              legends     = [f"{setup_1}", f"{setup_2}"],
                              title       = f"StripNoise_hb0_bottom ({temp})",
                              name        = f"Plot_StripNoiseBox_allModules_Hybrid0_bottomSensor_{temp}_compare",
                              xticklabels = moduleIDs,
                              ylabel      = "Noise [VcTh]",
                              offset      =	box_offset,
                              outdir      = self.outdir,
                              box_offset  = box_offset)
                # group plot
                strip_noise_hb0_bot_setup_1_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_bot_setup_1)[:,:,0])
                strip_noise_hb0_bot_setup_2_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_bot_setup_2)[:,:,0])
                self.plot_group(x           = np.arange(len(moduleIDs)),
                                data_list   = [[strip_noise_hb0_bot_setup_1_mean_std,
                                                strip_noise_hb0_bot_setup_2_mean_std]],
                                legends     = [[f"{setup_1}", f"{setup_2}"]],
                                title       = f"StripNoise_hb0_bottom ({temp})",
                                name        = f"Plot_StripNoise_allModules_hybrid0_bottomSensor_{temp}_compare",
                                xticklabels = moduleIDs,
                                ylim        = [4.0,8.0],
                                ylabel      = "Noise [VcTh]",
                                outdir      = self.outdir,
                                marker      = "o",
                                markerfacecolor=None,
                                markersize  = 5.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0,
                                tick_offset = tick_offset)
            
                # ===>> strip noise hybrid 1 (bottom)
                # box plot        
                strip_noise_hb1_bot_setup_1 = strip_noise_hb1_bot_setup[setup_1][temp]
                strip_noise_hb1_bot_setup_2 = strip_noise_hb1_bot_setup[setup_2][temp]
                self.plot_box(data_list_1 = strip_noise_hb1_bot_setup_1,
                              data_list_2 = strip_noise_hb1_bot_setup_2,
                              legends     = [f"{setup_1}", f"{setup_2}"],
                              title       = f"StripNoise_hb1_bottom ({temp})",
                              name        = f"Plot_StripNoiseBox_allModules_Hybrid1_bottomSensor_{temp}_compare",
                              xticklabels = moduleIDs,
                              ylabel      = "Noise [VcTh]",
                              offset      = box_offset,
                              outdir      = self.outdir,
                              box_offset  = box_offset)
                # group plot
                strip_noise_hb1_bot_setup_1_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_bot_setup_1)[:,:,0])
                strip_noise_hb1_bot_setup_2_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_bot_setup_2)[:,:,0])
                self.plot_group(x           = np.arange(len(moduleIDs)),
                                data_list   = [[strip_noise_hb1_bot_setup_1_mean_std, strip_noise_hb1_bot_setup_2_mean_std]],
                                legends     = [[f"{setup_1}", f"{setup_2}"]],
                                title       = f"StripNoise_hb1_bottom ({temp})",
                                name        = f"Plot_StripNoise_allModules_hybrid1_bottomSensor_{temp}_compare",
                                xticklabels = moduleIDs,
                                ylim        = [4.0,8.0],
                                ylabel      = "Noise [VcTh]",
                                outdir      = self.outdir,
                                marker      = "o",
                                markerfacecolor=None,
                                markersize  = 5.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0,
                                tick_offset = tick_offset)


                # ===>> strip noise hybrid 0 (top)
                # box plot
                strip_noise_hb0_top_setup_1 = strip_noise_hb0_top_setup[setup_1][temp]
                strip_noise_hb0_top_setup_2 = strip_noise_hb0_top_setup[setup_2][temp]
                self.plot_box(data_list_1 = strip_noise_hb0_top_setup_1,
                              data_list_2 = strip_noise_hb0_top_setup_2,
                              legends     = [f"{setup_1}", f"{setup_2}"],
                              title       = f"StripNoise_hb0_top ({temp})",
                              name        = f"Plot_StripNoiseBox_allModules_Hybrid0_topSensor_{temp}_compare",
                              xticklabels = moduleIDs,
                              ylabel      = "Noise [VcTh]",
                              offset      = box_offset,
                              outdir      = self.outdir,
                              box_offset  = box_offset)
                # group plot
                strip_noise_hb0_top_setup_1_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_top_setup_1)[:,:,0])
                strip_noise_hb0_top_setup_2_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_top_setup_2)[:,:,0])
                self.plot_group(x           = np.arange(len(moduleIDs)),
                                data_list   = [[strip_noise_hb0_top_setup_1_mean_std, strip_noise_hb0_top_setup_2_mean_std]],
                                legends     = [[f"{setup_1}", f"{setup_2}"]],
                                title       = f"StripNoise_hb0_top ({temp})",
                                name        = f"Plot_StripNoise_allModules_hybrid0_topSensor_{temp}_compare",
                                xticklabels = moduleIDs,
                                ylim        = [4.0,8.0],
                                ylabel      = "Noise [VcTh]",
                                outdir      = self.outdir,
                                marker      = "o",
                                markerfacecolor=None,
                                markersize  = 5.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0,
                                tick_offset = tick_offset)
            
        
                # ===>> strip noise hybrid 1 (top)
                # box plot
                strip_noise_hb1_top_setup_1 = strip_noise_hb1_top_setup[setup_1][temp]
                strip_noise_hb1_top_setup_2 = strip_noise_hb1_top_setup[setup_2][temp]
                self.plot_box(data_list_1 = strip_noise_hb1_top_setup_1,
                              data_list_2 = strip_noise_hb1_top_setup_2,
                              legends     = [f"{setup_1}", f"{setup_2}"],
                              title       = f"StripNoise_hb1_top ({temp})",
                              name        = f"Plot_StripNoiseBox_allModules_Hybrid1_topSensor_{temp}_compare",
                              xticklabels = moduleIDs,
                              ylabel      = "Noise [VcTh]",
                              offset      = box_offset,
                              outdir      = self.outdir,
                              box_offset  = box_offset)
                # group plot
                strip_noise_hb1_top_setup_1_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_top_setup_1)[:,:,0])
                strip_noise_hb1_top_setup_2_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_top_setup_2)[:,:,0])
                self.plot_group(x           = np.arange(len(moduleIDs)),
                                data_list   = [[strip_noise_hb1_top_setup_1_mean_std,
                                                strip_noise_hb1_top_setup_2_mean_std]],
                                legends     = [[f"{setup_1}", f"{setup_2}"]],
                                title       = f"StripNoise_hb1_top ({temp})",
                                name        = f"Plot_StripNoise_allModules_hybrid1_topSensor_{temp}_compare",
                                xticklabels = moduleIDs,
                                ylim        = [4.0,8.0],
                                ylabel      = "Noise [VcTh]",
                                outdir      = self.outdir,
                                marker      = "o",
                                markerfacecolor=None,
                                markersize  = 5.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0,
                                tick_offset = tick_offset)

        
                # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                #        Comparing common mode noise             #
                # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                if self.testinfo.get("check_common_noise") == True:
                    # ===>> nHits mean and std
                    cmn_noise_setup_1_mean_std, cmn_noise_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_setup[setup_1][temp])[:,:,0])
                    cmn_noise_setup_2_mean_std, cmn_noise_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_setup[setup_2][temp])[:,:,0])
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_setup_1_mean_std, cmn_noise_setup_2_mean_std],
                                    legends    = [f"{setup_1}", f"{setup_2}"],
                                    title      = f"#hits (50% Occ) (µ) : {temp}",
                                    name       = f"Plot_nHitsMean_module_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (µ)",
                                    markersize = 10,
                                    ylim       = [1200.0,2200.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_setup_1_sigma_std, cmn_noise_setup_2_sigma_std],
                                    legends    = [f"{setup_1}", f"{setup_2}"],
                                    title      = f"#hits (50% Occ) (σ) : {temp}",
                                    name       = f"Plot_nHitsStd_module_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (σ)",
                                    markersize = 10,
                                    ylim       = [50.0,250.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset=tick_offset)
                    # ===>> Extract CMN
                    CMN_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_setup[setup_1][temp]).shape[1],
                                                    mean = cmn_noise_setup_1_mean_std[:,0],
                                                    std = cmn_noise_setup_1_sigma_std[:,0])
                    CMN_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_setup[setup_2][temp]).shape[1],
                                                    mean = cmn_noise_setup_2_mean_std[:,0],
                                                    std = cmn_noise_setup_2_sigma_std[:,0])
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [CMN_setup_1, CMN_setup_2],
                                    legends    = [f"{setup_1}", f"{setup_2}"],
                                    title      = f"CMNoise: ({temp})",
                                    name       = f"Plot_CMN_module_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "CMN (%)",
                                    markersize = 10,
                                    ylim       = [0.0,10.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
            

                    # ===>>> nHits top and bottom
                    cmn_noise_bot_setup_1_mean_std, cmn_noise_bot_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_bot_setup[setup_1][temp])[:,:,0])
                    cmn_noise_bot_setup_2_mean_std, cmn_noise_bot_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_bot_setup[setup_2][temp])[:,:,0])
                    cmn_noise_top_setup_1_mean_std, cmn_noise_top_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_top_setup[setup_1][temp])[:,:,0]) 
                    cmn_noise_top_setup_2_mean_std, cmn_noise_top_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_top_setup[setup_2][temp])[:,:,0])
            
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_bot_setup_1_mean_std, cmn_noise_bot_setup_2_mean_std,
                                                  cmn_noise_top_setup_1_mean_std, cmn_noise_top_setup_2_mean_std],
                                    legends    = [f"bot: {setup_1}", f"bot: {setup_2}",
                                                  f"top: {setup_1}", f"top: {setup_2}"],
                                    title      = f"#hits (50%Occ) (µ) : sensors ({temp})",
                                    name       = f"Plot_nHitsMean_module_bothSensors_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (µ)",
                                    markersize = 10,
                                    ylim       = [500.0,1200.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_bot_setup_1_sigma_std, cmn_noise_bot_setup_2_sigma_std,
                                                  cmn_noise_top_setup_1_sigma_std, cmn_noise_top_setup_2_sigma_std],
                                    legends    = [f"bot: {setup_1}", f"bot: {setup_2}",
                                                  f"top: {setup_1}", f"top: {setup_2}"],
                                    title      = f"#hits (50% Occ) (σ) : sensors ({temp})",
                                    name       = f"Plot_nHitsStd_module_bothSensors_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (σ)",
                                    markersize = 10,
                                    ylim       = [0.0,200.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
                    # ===>> Extract CMN
                    CMN_bot_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_bot_setup[setup_1][temp]).shape[1],
                                                        mean = cmn_noise_bot_setup_1_mean_std[:,0],
                                                        std = cmn_noise_bot_setup_1_sigma_std[:,0])
                    CMN_bot_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_bot_setup[setup_2][temp]).shape[1],
                                                        mean = cmn_noise_bot_setup_2_mean_std[:,0],
                                                        std = cmn_noise_bot_setup_2_sigma_std[:,0])
                    CMN_top_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_top_setup[setup_1][temp]).shape[1],
                                                        mean = cmn_noise_top_setup_1_mean_std[:,0],
                                                        std = cmn_noise_top_setup_1_sigma_std[:,0])
                    CMN_top_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_top_setup[setup_2][temp]).shape[1],
                                                        mean = cmn_noise_top_setup_2_mean_std[:,0],
                                                        std = cmn_noise_top_setup_2_sigma_std[:,0])
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [CMN_bot_setup_1, CMN_bot_setup_2,
                                                  CMN_top_setup_1, CMN_top_setup_2],
                                    legends    = [f"bot: {setup_1}", f"bot: {setup_2}",
                                                  f"top: {setup_1}", f"top: {setup_2}"],
                                    title      = f"CMNoise (sensors): {temp}",
                                    name       = f"Plot_CMN_module_bothSensors_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "CMN (%)",
                                    markersize = 10,
                                    ylim       = [0.0,40.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
                

                    # nHits both hybrids
                    cmn_noise_hb0_setup_1_mean_std, cmn_noise_hb0_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_setup[setup_1][temp])[:,:,0])
                    cmn_noise_hb0_setup_2_mean_std, cmn_noise_hb0_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_setup[setup_2][temp])[:,:,0])
                    cmn_noise_hb1_setup_1_mean_std, cmn_noise_hb1_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_setup[setup_1][temp])[:,:,0]) 
                    cmn_noise_hb1_setup_2_mean_std, cmn_noise_hb1_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_setup[setup_2][temp])[:,:,0])
                    
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_hb0_setup_1_mean_std, cmn_noise_hb0_setup_2_mean_std,
                                                  cmn_noise_hb1_setup_1_mean_std, cmn_noise_hb1_setup_2_mean_std],
                                    legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                                  f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                    title      = f"#hits (50% Occ) (µ) : hybrids ({temp})",
                                    name       = f"Plot_nHitsMean_module_bothHybrids_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (µ)",
                                    markersize = 10,
                                    ylim       = [500.0,1200.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_hb0_setup_1_sigma_std, cmn_noise_hb0_setup_2_sigma_std,
                                                  cmn_noise_hb1_setup_1_sigma_std, cmn_noise_hb1_setup_2_sigma_std],
                                    legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                                  f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                    title      = f"#hits (50% Occ) (σ) : hybrids ({temp})",
                                    name       = f"Plot_nHitsStd_module_bothHybrids_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (σ)",
                                    markersize = 10,
                                    ylim       = [50.0,300.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
                    
                    CMN_hb0_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb0_setup[setup_1][temp]).shape[1],
                                                        mean = cmn_noise_hb0_setup_1_mean_std[:,0],
                                                        std = cmn_noise_hb0_setup_1_sigma_std[:,0])
                    CMN_hb0_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb0_setup[setup_2][temp]).shape[1],
                                                        mean = cmn_noise_hb0_setup_2_mean_std[:,0],
                                                        std = cmn_noise_hb0_setup_2_sigma_std[:,0])
                    CMN_hb1_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb1_setup[setup_1][temp]).shape[1],
                                                        mean = cmn_noise_hb1_setup_1_mean_std[:,0],
                                                        std = cmn_noise_hb1_setup_1_sigma_std[:,0])
                    CMN_hb1_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb1_setup[setup_2][temp]).shape[1],
                                                        mean = cmn_noise_hb1_setup_2_mean_std[:,0],
                                                        std = cmn_noise_hb1_setup_2_sigma_std[:,0])
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [CMN_hb0_setup_1, CMN_hb0_setup_2,
                                                  CMN_hb1_setup_1, CMN_hb1_setup_2],
                                    legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                                  f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                    title      = f"CMNoise (hybrids): {temp}",
                                    name       = f"Plot_CMN_module_bothHybrids_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "CMN (%)",
                                    markersize = 10,
                                    ylim       = [0.0,40.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
                    
                
                
                    cmn_noise_hb0_bot_setup_1_mean_std, cmn_noise_hb0_bot_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                        np.array(common_noise_hb0_bot_setup[setup_1][temp])[:,:,0]
                    )
                    cmn_noise_hb0_bot_setup_2_mean_std, cmn_noise_hb0_bot_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                        np.array(common_noise_hb0_bot_setup[setup_2][temp])[:,:,0]
                    )
                    
                    cmn_noise_hb1_bot_setup_1_mean_std, cmn_noise_hb1_bot_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                        np.array(common_noise_hb1_bot_setup[setup_1][temp])[:,:,0]
                    )
                    cmn_noise_hb1_bot_setup_2_mean_std, cmn_noise_hb1_bot_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                        np.array(common_noise_hb1_bot_setup[setup_2][temp])[:,:,0]
                    )
                    
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_hb0_bot_setup_1_mean_std, cmn_noise_hb0_bot_setup_2_mean_std,
                                                  cmn_noise_hb1_bot_setup_1_mean_std, cmn_noise_hb1_bot_setup_2_mean_std],
                                    legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                                  f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                    title      = f"#hits (50% Occ) botSensor (µ): {temp}",
                                    name       = f"Plot_nHitsMean_module_bothHybrids_bottomSensor_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (µ)",
                                    markersize = 10,
                                    ylim       = [200.0,600.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_hb0_bot_setup_1_sigma_std, cmn_noise_hb0_bot_setup_2_sigma_std,
                                                  cmn_noise_hb1_bot_setup_1_sigma_std, cmn_noise_hb1_bot_setup_2_sigma_std],
                                    legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                                  f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                    title      = f"#hits (50% Occ) botSensor (σ): {temp}",
                                    name       = f"Plot_nHitsStd_module_bothHybrids_bottomSensor_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (σ)",
                                    markersize = 10,
                                    ylim       = [10.0,160.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset=tick_offset)
                    
                    CMN_hb0_bot_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb0_bot_setup[setup_1][temp]).shape[1],
                                                            mean = cmn_noise_hb0_bot_setup_1_mean_std[:,0],
                                                            std = cmn_noise_hb0_bot_setup_1_sigma_std[:,0])
                    CMN_hb0_bot_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb0_bot_setup[setup_2][temp]).shape[1],
                                                            mean = cmn_noise_hb0_bot_setup_2_mean_std[:,0],
                                                            std = cmn_noise_hb0_bot_setup_2_sigma_std[:,0])
                    CMN_hb1_bot_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb1_bot_setup[setup_1][temp]).shape[1],
                                                            mean = cmn_noise_hb1_bot_setup_1_mean_std[:,0],
                                                            std = cmn_noise_hb1_bot_setup_1_sigma_std[:,0])
                    CMN_hb1_bot_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb1_bot_setup[setup_2][temp]).shape[1],
                                                            mean = cmn_noise_hb1_bot_setup_2_mean_std[:,0],
                                                            std = cmn_noise_hb1_bot_setup_2_sigma_std[:,0])
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [CMN_hb0_bot_setup_1, CMN_hb0_bot_setup_2,
                                                  CMN_hb1_bot_setup_1, CMN_hb1_bot_setup_2],
                                    legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                                  f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                    title      = f"CMNoise (bottom sensor): {temp}",
                                    name       = f"Plot_CMN_module_bothHybrids_bottomSensor_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "CMN (%)",
                                    markersize = 10,
                                    ylim       = [0.0,40.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
                
                    cmn_noise_hb0_top_setup_1_mean_std, cmn_noise_hb0_top_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                        np.array(common_noise_hb0_top_setup[setup_1][temp])[:,:,0]
                    )
                    cmn_noise_hb0_top_setup_2_mean_std, cmn_noise_hb0_top_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                        np.array(common_noise_hb0_top_setup[setup_2][temp])[:,:,0]
                    )
                    
                    cmn_noise_hb1_top_setup_1_mean_std, cmn_noise_hb1_top_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                        np.array(common_noise_hb1_top_setup[setup_1][temp])[:,:,0]
                    )
                    cmn_noise_hb1_top_setup_2_mean_std, cmn_noise_hb1_top_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                        np.array(common_noise_hb1_top_setup[setup_2][temp])[:,:,0]
                    )
                    
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_hb0_top_setup_1_mean_std, cmn_noise_hb0_top_setup_2_mean_std,
                                                  cmn_noise_hb1_top_setup_1_mean_std, cmn_noise_hb1_top_setup_2_mean_std],
                                    legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                                  f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                    title      = f"#hits (50% Occ) topSensor (µ): {temp}",
                                    name       = f"Plot_nHitsMean_module_bothHybrids_topSensor_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (µ)",
                                    markersize = 10,
                                    ylim       = [200.0,600.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset = tick_offset)
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [cmn_noise_hb0_top_setup_1_sigma_std, cmn_noise_hb0_top_setup_2_sigma_std,
                                                  cmn_noise_hb1_top_setup_1_sigma_std, cmn_noise_hb1_top_setup_2_sigma_std],
                                    legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                                  f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                    title      = f"#hits (50% Occ) topSensor (σ): {temp}",
                                    name       = f"Plot_nHitsStd_module_bothHybrids_topSensor_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "#hits (σ)",
                                    markersize = 10,
                                    ylim       = [10.0,160.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset = tick_offset)
                
                    CMN_hb0_top_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb0_top_setup[setup_1][temp]).shape[1],
                                                            mean = cmn_noise_hb0_top_setup_1_mean_std[:,0],
                                                            std = cmn_noise_hb0_top_setup_1_sigma_std[:,0])
                    CMN_hb0_top_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb0_top_setup[setup_2][temp]).shape[1],
                                                            mean = cmn_noise_hb0_top_setup_2_mean_std[:,0],
                                                            std = cmn_noise_hb0_top_setup_2_sigma_std[:,0])
                    CMN_hb1_top_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb1_top_setup[setup_1][temp]).shape[1],
                                                            mean = cmn_noise_hb1_top_setup_1_mean_std[:,0],
                                                            std = cmn_noise_hb1_top_setup_1_sigma_std[:,0])
                    CMN_hb1_top_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb1_top_setup[setup_2][temp]).shape[1],
                                                            mean = cmn_noise_hb1_top_setup_2_mean_std[:,0],
                                                            std = cmn_noise_hb1_top_setup_2_sigma_std[:,0])
                    self.plot_basic(x          = np.arange(len(moduleIDs)),
                                    data_list  = [CMN_hb0_top_setup_1, CMN_hb0_top_setup_2,
                                                  CMN_hb1_top_setup_1, CMN_hb1_top_setup_2],
                                    legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                                  f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                    title      = f"CMNoise (top sensor): {temp}",
                                    name       = f"Plot_CMN_module_bothHybrids_topSensor_{temp}_compare",
                                    xticklabels = moduleIDs,
                                    ylabel     = "CMN (%)",
                                    markersize = 10,
                                    ylim       = [0.0,40.0],
                                    outdir     = self.outdir,
                                    fit        = False,
                                    tick_offset= tick_offset)
                
                else:
                    logger.warning(f"skip comparing common mode noise between {setup_1} and {setup_2}")

        else:
            logger.warning("skip comparing two setup")
