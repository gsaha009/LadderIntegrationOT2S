# Plotter function
# Ladder Integration at IPHC
# Author: G.Saha

import os
import sys
import yaml
import numpy as np
import pandas as pd

#from util import plot_basic, hist_basic
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import mplhep as hep
hep.style.use("CMS")
import seaborn as sns

from Fitter import Fitter

import logging
logger = logging.getLogger('main')



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
        #from IPython import embed; embed(); exit()
        arr = arr.tolist()
        outlist = []
        for item in arr:
            #print(len(item))
            #print(x.shape)
            _arr = np.array(item)
            #print(arr.shape)
            #mean = np.average(x, weights=_arr, axis=0)
            #sigma = np.sqrt(np.average((x - mean)**2, weights=_arr, axis=0))
            mean_sigma = self.__get_mean_std_for_cmn(x, _arr)
            #print(mean, sigma)
            mean = mean_sigma[0]
            sigma = mean_sigma[1]
            outlist.append([float(mean), float(sigma)])
        #x = np.repeat(x[None,:], arr.shape[0], axis=0)
        #mean_sigma = self.__get_mean_std_for_cmn(x, arr, axis=1)
        #return (mean_sigma[0], mean_sigma[1])
        #from IPython import embed; embed(); exit()
        return np.array(outlist)

    def __extractCMN(self, nchannels=None, mean=None, std=None):
        alpha = 2*np.pi*(std**2-mean*(1-mean/nchannels))/(nchannels*(nchannels-1))
        cmn = np.sqrt(np.sin(alpha)/(1-np.sin(alpha)))*100
        return np.concatenate((cmn[:,None], np.zeros_like(cmn)[:,None]), axis=1)
        
    
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
        fig.savefig(f"{outdir}/{name}.pdf", dpi=300)
        #fig.savefig(f"{outdir}/{name}.png", dpi=250)
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
        fig.savefig(f"{outdir}/{name}.pdf", dpi=300)
        #fig.savefig(f"{outdir}/{name}.png", dpi=250)
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
        fig.savefig(f"{outdir}/{name}.pdf", dpi=300)
        #fig.savefig(f"{outdir}/{name}.png", dpi=250)
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
        fig.savefig(f"{outdir}/{name}.pdf", dpi=300)
        #fig.savefig(f"{outdir}/{name}.png", dpi=250)
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

    
    def plotEverything(self):

        sensor_temps_setup = {}
        # channel noise
        strip_noise_hb0_setup = {}
        strip_noise_hb0_bot_setup = {}
        strip_noise_hb0_top_setup = {}
        strip_noise_hb1_setup = {}
        strip_noise_hb1_bot_setup = {}
        strip_noise_hb1_top_setup = {}
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
            
            # define dict to save info per module level
            moduleIDs = []
            for moduleID, moduleDict in dataval.items():
                logger.info(f"Module ID : {moduleID}")
                moduleIDs.append(moduleID)
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
                                        outdir     = _outdir,
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
                                    title      = f"StripNoise_{moduleID}",
                                    name       = f"Plot_StripNoise_bothHybrids_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Channel",
                                    ylabel     = "Noise [VcTh]",
                                    ylim       = [0.0,12.0],
                                    outdir     = _outdir)

                    
                    # Plotting strip noise channel wise : [hb0_bot, hb1_bot]
                    self.plot_basic(x          = np.arange(len(strip_noise_hb0_top)),
                                    data_list  = [strip_noise_hb0_bot, strip_noise_hb1_bot],
                                    legends    = ["hybrid 0", "hybrid 1"],
                                    title      = f"StripNoise_Bottom_{moduleID}",
                                    name       = f"Plot_StripNoise_bothHybrids_BottomSensor_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Channel",
                                    ylabel     = "Noise [VcTh]",
                                    ylim       = [0.0,12.0],
                                    outdir     = _outdir)

                    # Plotting strip noise channel wise : [hb0_top, hb1_top]
                    self.plot_basic(x          = np.arange(len(strip_noise_hb0_top)),
                                    data_list  = [strip_noise_hb0_top, strip_noise_hb1_top],
                                    legends    = ["hybrid 0", "hybrid 1"],
                                    title      = f"StripNoise_Top_{moduleID}",
                                    name       = f"Plot_StripNoise_bothHybrids_TopSensor_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Channel",
                                    ylabel     = "Noise [VcTh]",
                                    ylim       = [0.0,12.0],
                                    outdir     = _outdir)


                    # Plotting hist for stripNoise : [hb0_bot, hb0_top, hb1_bot, hb1_top]
                    self.hist_basic(bins       = np.linspace(2,10,80),
                                    data_list  = [strip_noise_hb0_bot, strip_noise_hb0_top, strip_noise_hb1_bot, strip_noise_hb1_top],
                                    legends    = ["hybrid 0 bottom", "hybrid 0 top", "hybrid 1 bottom", "hybrid 1 top"],
                                    title      = f"StripNoise_{moduleID}",
                                    name       = f"Hist_StripNoise_bothHybrids_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Noise [VcTh]",
                                    ylabel     = "Entries",
                                    linewidth  = 2,
                                    outdir     = _outdir,
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
                                    title      = f"StripNoise_Hb0_{moduleID}",
                                    name       = f"Hist_StripNoiseCBC_Hybrid0_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Noise [VcTh]",
                                    ylabel     = "Entries",
                                    linewidth  = 1.2,
                                    outdir     = _outdirCBC,
                                    colors     = ["#0B1A2F", "#152C4D", "#1F3E6C", "#29508A", "#3362A9", "#3D74C7", "#4796E6", "#61B4FF"],
                                    linestyles = 8*["-"])
                    
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
                                    title      = f"StripNoise_Hb1_{moduleID}",
                                    name       = f"Hist_StripNoiseCBC_Hybrid1_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                    xlabel     = "Noise [VcTh]",
                                    ylabel     = "Entries",
                                    linewidth  = 1.2,
                                    outdir     = _outdirCBC,
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
                                    outdir      = _outdirCBC,
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
                                    outdir      = _outdirCBC,
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
                                    outdir      = _outdirCBC,
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
                                        outdir     = _outdir,
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
                                        outdir     = _outdir,
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
                                        outdir     = _outdir,
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
                                        outdir     = _outdir,
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
                                        outdir     = _outdir,
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
                                        outdir     = _outdirCBC,
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
                                        outdir     = _outdirCBC,
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
                                        title       = f"CMNoise_{moduleID}",
                                        name        = f"Plot_CommonNoiseCBC_bothHybrids_{martaTemp}_{moduleID}_{datakey}",
                                        xticklabels = [f"CBC_{i}" for i in range(8)],
                                        ylim        = [50, 200],
                                        ylabel      = "Common Noise",
                                        outdir      = _outdirCBC,
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
                                        outdir      = _outdirCBC,
                                        marker      = "o",
                                        markersize  = 2.5,
                                        capsize     = 1.5,
                                        elinewidth  = 1.0)
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
                                        title      = f"Pedestal_{moduleID}",
                                        name       = f"Plot_Pedestal_bothHybrids_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Channel",
                                        ylabel     = "Pedestal [VcTh]",
                                        ylim       = [595.0,605.0],
                                        outdir     = _outdir)
                    
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
                                        title      = f"PedestalCBC_Hybrid0_{moduleID}",
                                        name       = f"Hist_PedestalCBC_Hybrid0_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Pedestal [VcTh]",
                                        ylabel     = "Entries",
                                        linewidth  = 1.2,
                                        outdir     = _outdirCBC,
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
                                        title      = f"PedestalCBC_Hybrid1_{moduleID}",
                                        name       = f"Hist_PedestalCBC_Hybrid1_bothSensors_{martaTemp}_{moduleID}_{datakey}",
                                        xlabel     = "Pedestal [VcTh]",
                                        ylabel     = "Entries",
                                        linewidth  = 1.2,
                                        outdir     = _outdirCBC,
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
                                        outdir      = _outdirCBC,
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
                                    title      = f"Sensor Temperature",
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

                # same for common mode noise
                if self.testinfo.get("check_common_noise") == True:
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
                else:
                    logger.warning("skipping module wise common mode noise comparison")

            sensor_temps_setup[datakey] = sensor_temps_mod
            strip_noise_hb0_setup[datakey] = strip_noise_hb0_mod
            strip_noise_hb0_bot_setup[datakey] = strip_noise_hb0_bot_mod
            strip_noise_hb0_top_setup[datakey] = strip_noise_hb0_top_mod
            strip_noise_hb1_setup[datakey] = strip_noise_hb1_mod
            strip_noise_hb1_bot_setup[datakey] = strip_noise_hb1_bot_mod
            strip_noise_hb1_top_setup[datakey] = strip_noise_hb1_top_mod
            common_noise_setup[datakey] = common_noise_mod
            common_noise_bot_setup[datakey] = common_noise_bot_mod
            common_noise_top_setup[datakey] = common_noise_top_mod
            common_noise_hb0_setup[datakey] = common_noise_hb0_mod
            common_noise_hb0_bot_setup[datakey] = common_noise_hb0_bot_mod
            common_noise_hb0_top_setup[datakey] = common_noise_hb0_top_mod
            common_noise_hb1_setup[datakey] = common_noise_hb1_mod
            common_noise_hb1_bot_setup[datakey] = common_noise_hb1_bot_mod
            common_noise_hb1_top_setup[datakey] = common_noise_hb1_top_mod

        # Loop over setup ends here    


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

            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
            #                                     WARNING: FEW THIGS ARE HARDCODED                                         #
            #                        Here, one can use hardcoding to compare different setup                               #
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
            if self.testinfo.get("check_sensor_temperature") == True:        
                # ===>> Sensor temperature
                sensor_temps_setup_1 = np.array(sensor_temps_setup[setup_1]['RoomTemp'])
                sensor_temps_setup_2 = np.array(sensor_temps_setup[setup_2]['RoomTemp'])
                #from IPython import embed; embed(); exit()
                
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [np.concatenate((sensor_temps_setup_1[:,None], np.zeros_like(sensor_temps_setup_1)[:,None]), axis=1).tolist(),
                                              np.concatenate((sensor_temps_setup_2[:,None], np.zeros_like(sensor_temps_setup_2)[:,None]), axis=1).tolist()],
                                legends    = [f"{setup_1}", f"{setup_2}"],
                                title      = f"Sensor Temperature (RoomTemp)",
                                name       = f"Plot_SensorTemp_allModules_RoomTemp_compare",
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
            strip_noise_hb0_setup_1 = strip_noise_hb0_setup[setup_1]['RoomTemp']
            strip_noise_hb0_setup_2 = strip_noise_hb0_setup[setup_2]['RoomTemp']
            # box plot
            self.plot_box(data_list_1 = strip_noise_hb0_setup_1,
                          data_list_2 = strip_noise_hb0_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hybrid0_RoomTemp",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid0_RoomTemp_compare",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          offset      =	box_offset,
                          outdir      = self.outdir,
                          box_offset  = box_offset)
            # group plot
            strip_noise_hb0_setup_1_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_setup_1)[:,:,0])
            strip_noise_hb0_setup_2_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_setup_2)[:,:,0])
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[strip_noise_hb0_setup_1_mean_std,
                                            strip_noise_hb0_setup_2_mean_std]],
                            legends     = [[f"{setup_1}", f"{setup_2}"]],
                            title       = f"StripNoise_hb0 (RoomTemp)",
                            name        = f"Plot_StripNoise_allModules_hybrid0_RoomTemp_compare",
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
            strip_noise_hb1_setup_1 = strip_noise_hb1_setup[setup_1]['RoomTemp']
            strip_noise_hb1_setup_2 = strip_noise_hb1_setup[setup_2]['RoomTemp']
            self.plot_box(data_list_1 = strip_noise_hb1_setup_1,
                          data_list_2 = strip_noise_hb1_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hybrid1_RoomTemp",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid1_RoomTemp_compare",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          offset      =	box_offset,
                          outdir      = self.outdir,
                          box_offset  = box_offset)
            # group plot
            strip_noise_hb1_setup_1_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_setup_1)[:,:,0])
            strip_noise_hb1_setup_2_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_setup_2)[:,:,0])
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[strip_noise_hb1_setup_1_mean_std,
                                            strip_noise_hb1_setup_2_mean_std]],
                            legends     = [[f"{setup_1}", f"{setup_2}"]],
                            title       = f"StripNoise_hb1 (RoomTemp)",
                            name        = f"Plot_StripNoise_allModules_hybrid1_RoomTemp_compare",
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
            strip_noise_hb0_bot_setup_1 = strip_noise_hb0_bot_setup[setup_1]['RoomTemp']
            strip_noise_hb0_bot_setup_2 = strip_noise_hb0_bot_setup[setup_2]['RoomTemp']
            self.plot_box(data_list_1 = strip_noise_hb0_bot_setup_1,
                          data_list_2 = strip_noise_hb0_bot_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hybrid0_bottom_RoomTemp",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid0_bottomSensor_RoomTemp_compare",
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
                            title       = f"StripNoise_hb0_bottom (RoomTemp)",
                            name        = f"Plot_StripNoise_allModules_hybrid0_bottomSensor_RoomTemp_compare",
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
            strip_noise_hb1_bot_setup_1 = strip_noise_hb1_bot_setup[setup_1]['RoomTemp']
            strip_noise_hb1_bot_setup_2 = strip_noise_hb1_bot_setup[setup_2]['RoomTemp']
            self.plot_box(data_list_1 = strip_noise_hb1_bot_setup_1,
                          data_list_2 = strip_noise_hb1_bot_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hybrid1_bottom_RoomTemp",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid1_bottomSensor_RoomTemp_compare",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          offset      =	box_offset,
                          outdir      = self.outdir,
                          box_offset  = box_offset)
            # group plot
            strip_noise_hb1_bot_setup_1_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_bot_setup_1)[:,:,0])
            strip_noise_hb1_bot_setup_2_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_bot_setup_2)[:,:,0])
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[strip_noise_hb1_bot_setup_1_mean_std, strip_noise_hb1_bot_setup_2_mean_std]],
                            legends     = [[f"{setup_1}", f"{setup_2}"]],
                            title       = f"StripNoise_hb1_bottom (RoomTemp)",
                            name        = f"Plot_StripNoise_allModules_hybrid1_bottomSensor_RoomTemp_compare",
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
            strip_noise_hb0_top_setup_1 = strip_noise_hb0_top_setup[setup_1]['RoomTemp']
            strip_noise_hb0_top_setup_2 = strip_noise_hb0_top_setup[setup_2]['RoomTemp']
            self.plot_box(data_list_1 = strip_noise_hb0_top_setup_1,
                          data_list_2 = strip_noise_hb0_top_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hybrid0_top_RoomTemp",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid0_topSensor_RoomTemp_compare",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          offset      =	box_offset,
                          outdir      = self.outdir,
                          box_offset  = box_offset)
            # group plot
            strip_noise_hb0_top_setup_1_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_top_setup_1)[:,:,0])
            strip_noise_hb0_top_setup_2_mean_std = self.__get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_top_setup_2)[:,:,0])
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[strip_noise_hb0_top_setup_1_mean_std, strip_noise_hb0_top_setup_2_mean_std]],
                            legends     = [[f"{setup_1}", f"{setup_2}"]],
                            title       = f"StripNoise_hb0_top (RoomTemp)",
                            name        = f"Plot_StripNoise_allModules_hybrid0_topSensor_RoomTemp_compare",
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
            strip_noise_hb1_top_setup_1 = strip_noise_hb1_top_setup[setup_1]['RoomTemp']
            strip_noise_hb1_top_setup_2 = strip_noise_hb1_top_setup[setup_2]['RoomTemp']
            self.plot_box(data_list_1 = strip_noise_hb1_top_setup_1,
                          data_list_2 = strip_noise_hb1_top_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hybrid1_top_RoomTemp",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid1_topSensor_RoomTemp_compare",
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
                            title       = f"StripNoise_hb1_top (RoomTemp)",
                            name        = f"Plot_StripNoise_allModules_hybrid1_topSensor_RoomTemp_compare",
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
                cmn_noise_setup_1_mean_std, cmn_noise_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_setup[setup_1]['RoomTemp'])[:,:,0])
                cmn_noise_setup_2_mean_std, cmn_noise_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_setup[setup_2]['RoomTemp'])[:,:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_setup_1_mean_std, cmn_noise_setup_2_mean_std],
                                legends    = [f"{setup_1}", f"{setup_2}"],
                                title      = f"#hits (50% Occ) (µ)",
                                name       = f"Plot_nHitsMean_module_compare",
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
                                title      = f"#hits (50% Occ) (σ)",
                                name       = f"Plot_nHitsStd_module_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (σ)",
                                markersize = 10,
                                ylim       = [50.0,250.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset=tick_offset)
                # ===>> Extract CMN
                CMN_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_setup[setup_1]['RoomTemp']).shape[1],
                                                mean = cmn_noise_setup_1_mean_std[:,0],
                                                std = cmn_noise_setup_1_sigma_std[:,0])
                CMN_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_setup[setup_2]['RoomTemp']).shape[1],
                                                mean = cmn_noise_setup_2_mean_std[:,0],
                                                std = cmn_noise_setup_2_sigma_std[:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [CMN_setup_1, CMN_setup_2],
                                legends    = [f"{setup_1}", f"{setup_2}"],
                                title      = f"CMNoise",
                                name       = f"Plot_CMN_module_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "CMN (%)",
                                markersize = 10,
                                ylim       = [0.0,20.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
            

                # ===>>> nHits top and bottom
                cmn_noise_bot_setup_1_mean_std, cmn_noise_bot_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_bot_setup[setup_1]['RoomTemp'])[:,:,0])
                cmn_noise_bot_setup_2_mean_std, cmn_noise_bot_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_bot_setup[setup_2]['RoomTemp'])[:,:,0])
                cmn_noise_top_setup_1_mean_std, cmn_noise_top_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_top_setup[setup_1]['RoomTemp'])[:,:,0]) 
                cmn_noise_top_setup_2_mean_std, cmn_noise_top_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_top_setup[setup_2]['RoomTemp'])[:,:,0])
            
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_bot_setup_1_mean_std, cmn_noise_bot_setup_2_mean_std,
                                              cmn_noise_top_setup_1_mean_std, cmn_noise_top_setup_2_mean_std],
                                legends    = [f"bot: {setup_1}", f"bot: {setup_2}",
                                              f"top: {setup_1}", f"top: {setup_2}"],
                                title      = f"#hits (50%Occ) (µ) : sensors",
                                name       = f"Plot_nHitsMean_module_bothSensors_compare",
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
                                title      = f"#hits (50% Occ) (σ) : sensors",
                                name       = f"Plot_nHitsStd_module_bothSensors_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (σ)",
                                markersize = 10,
                                ylim       = [0.0,200.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                # ===>> Extract CMN
                CMN_bot_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_bot_setup[setup_1]['RoomTemp']).shape[1],
                                                    mean = cmn_noise_bot_setup_1_mean_std[:,0],
                                                    std = cmn_noise_bot_setup_1_sigma_std[:,0])
                CMN_bot_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_bot_setup[setup_2]['RoomTemp']).shape[1],
                                                    mean = cmn_noise_bot_setup_2_mean_std[:,0],
                                                    std = cmn_noise_bot_setup_2_sigma_std[:,0])
                CMN_top_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_top_setup[setup_1]['RoomTemp']).shape[1],
                                                    mean = cmn_noise_top_setup_1_mean_std[:,0],
                                                    std = cmn_noise_top_setup_1_sigma_std[:,0])
                CMN_top_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_top_setup[setup_2]['RoomTemp']).shape[1],
                                                    mean = cmn_noise_top_setup_2_mean_std[:,0],
                                                    std = cmn_noise_top_setup_2_sigma_std[:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [CMN_bot_setup_1, CMN_bot_setup_2,
                                              CMN_top_setup_1, CMN_top_setup_2],
                                legends    = [f"bot: {setup_1}", f"bot: {setup_2}",
                                              f"top: {setup_1}", f"top: {setup_2}"],
                                title      = f"CMNoise (sensors)",
                                name       = f"Plot_CMN_module_bothSensors_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "CMN (%)",
                                markersize = 10,
                                ylim       = [0.0,40.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                

                # nHits both hybrids
                cmn_noise_hb0_setup_1_mean_std, cmn_noise_hb0_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_setup[setup_1]['RoomTemp'])[:,:,0])
                cmn_noise_hb0_setup_2_mean_std, cmn_noise_hb0_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_setup[setup_2]['RoomTemp'])[:,:,0])
                cmn_noise_hb1_setup_1_mean_std, cmn_noise_hb1_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_setup[setup_1]['RoomTemp'])[:,:,0]) 
                cmn_noise_hb1_setup_2_mean_std, cmn_noise_hb1_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_setup[setup_2]['RoomTemp'])[:,:,0])
                
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_setup_1_mean_std, cmn_noise_hb0_setup_2_mean_std,
                                              cmn_noise_hb1_setup_1_mean_std, cmn_noise_hb1_setup_2_mean_std],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"#hits (50% Occ) (µ) : hybrids",
                                name       = f"Plot_nHitsMean_module_bothHybrids_compare",
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
                                title      = f"#hits (50% Occ) (σ) : hybrids",
                                name       = f"Plot_nHitsStd_module_bothHybrids_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (σ)",
                                markersize = 10,
                                ylim       = [50.0,300.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                
                CMN_hb0_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb0_setup[setup_1]['RoomTemp']).shape[1],
                                                    mean = cmn_noise_hb0_setup_1_mean_std[:,0],
                                                    std = cmn_noise_hb0_setup_1_sigma_std[:,0])
                CMN_hb0_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb0_setup[setup_2]['RoomTemp']).shape[1],
                                                    mean = cmn_noise_hb0_setup_2_mean_std[:,0],
                                                    std = cmn_noise_hb0_setup_2_sigma_std[:,0])
                CMN_hb1_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb1_setup[setup_1]['RoomTemp']).shape[1],
                                                    mean = cmn_noise_hb1_setup_1_mean_std[:,0],
                                                    std = cmn_noise_hb1_setup_1_sigma_std[:,0])
                CMN_hb1_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb1_setup[setup_2]['RoomTemp']).shape[1],
                                                    mean = cmn_noise_hb1_setup_2_mean_std[:,0],
                                                    std = cmn_noise_hb1_setup_2_sigma_std[:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [CMN_hb0_setup_1, CMN_hb0_setup_2,
                                              CMN_hb1_setup_1, CMN_hb1_setup_2],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"CMNoise (hybrids)",
                                name       = f"Plot_CMN_module_bothHybrids_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "CMN (%)",
                                markersize = 10,
                                ylim       = [0.0,40.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                
                
                
                cmn_noise_hb0_bot_setup_1_mean_std, cmn_noise_hb0_bot_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb0_bot_setup[setup_1]['RoomTemp'])[:,:,0]
                )
                cmn_noise_hb0_bot_setup_2_mean_std, cmn_noise_hb0_bot_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb0_bot_setup[setup_2]['RoomTemp'])[:,:,0]
                )
                
                cmn_noise_hb1_bot_setup_1_mean_std, cmn_noise_hb1_bot_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb1_bot_setup[setup_1]['RoomTemp'])[:,:,0]
                )
                cmn_noise_hb1_bot_setup_2_mean_std, cmn_noise_hb1_bot_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb1_bot_setup[setup_2]['RoomTemp'])[:,:,0]
                )
                
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_bot_setup_1_mean_std, cmn_noise_hb0_bot_setup_2_mean_std,
                                              cmn_noise_hb1_bot_setup_1_mean_std, cmn_noise_hb1_bot_setup_2_mean_std],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"#hits (50% Occ) botSensor (µ)",
                                name       = f"Plot_nHitsMean_module_bothHybrids_bottomSensor_compare",
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
                                title      = f"#hits (50% Occ) botSensor (σ)",
                                name       = f"Plot_nHitsStd_module_bothHybrids_bottomSensor_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (σ)",
                                markersize = 10,
                                ylim       = [10.0,160.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset=tick_offset)
            
                CMN_hb0_bot_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb0_bot_setup[setup_1]['RoomTemp']).shape[1],
                                                        mean = cmn_noise_hb0_bot_setup_1_mean_std[:,0],
                                                        std = cmn_noise_hb0_bot_setup_1_sigma_std[:,0])
                CMN_hb0_bot_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb0_bot_setup[setup_2]['RoomTemp']).shape[1],
                                                        mean = cmn_noise_hb0_bot_setup_2_mean_std[:,0],
                                                        std = cmn_noise_hb0_bot_setup_2_sigma_std[:,0])
                CMN_hb1_bot_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb1_bot_setup[setup_1]['RoomTemp']).shape[1],
                                                        mean = cmn_noise_hb1_bot_setup_1_mean_std[:,0],
                                                        std = cmn_noise_hb1_bot_setup_1_sigma_std[:,0])
                CMN_hb1_bot_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb1_bot_setup[setup_2]['RoomTemp']).shape[1],
                                                        mean = cmn_noise_hb1_bot_setup_2_mean_std[:,0],
                                                        std = cmn_noise_hb1_bot_setup_2_sigma_std[:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [CMN_hb0_bot_setup_1, CMN_hb0_bot_setup_2,
                                              CMN_hb1_bot_setup_1, CMN_hb1_bot_setup_2],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"CMNoise (bottom sensor)",
                                name       = f"Plot_CMN_module_bothHybrids_bottomSensor_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "CMN (%)",
                                markersize = 10,
                                ylim       = [0.0,40.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                
                cmn_noise_hb0_top_setup_1_mean_std, cmn_noise_hb0_top_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb0_top_setup[setup_1]['RoomTemp'])[:,:,0]
                )
                cmn_noise_hb0_top_setup_2_mean_std, cmn_noise_hb0_top_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb0_top_setup[setup_2]['RoomTemp'])[:,:,0]
                )
                
                cmn_noise_hb1_top_setup_1_mean_std, cmn_noise_hb1_top_setup_1_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb1_top_setup[setup_1]['RoomTemp'])[:,:,0]
                )
                cmn_noise_hb1_top_setup_2_mean_std, cmn_noise_hb1_top_setup_2_sigma_std = self.__get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb1_top_setup[setup_2]['RoomTemp'])[:,:,0]
                )
            
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_top_setup_1_mean_std, cmn_noise_hb0_top_setup_2_mean_std,
                                              cmn_noise_hb1_top_setup_1_mean_std, cmn_noise_hb1_top_setup_2_mean_std],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"#hits (50% Occ) topSensor (µ)",
                                name       = f"Plot_nHitsMean_module_bothHybrids_topSensor_compare",
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
                                title      = f"#hits (50% Occ) topSensor (σ)",
                                name       = f"Plot_nHitsStd_module_bothHybrids_topSensor_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (σ)",
                                markersize = 10,
                                ylim       = [10.0,160.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset = tick_offset)
                
                CMN_hb0_top_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb0_top_setup[setup_1]['RoomTemp']).shape[1],
                                                        mean = cmn_noise_hb0_top_setup_1_mean_std[:,0],
                                                        std = cmn_noise_hb0_top_setup_1_sigma_std[:,0])
                CMN_hb0_top_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb0_top_setup[setup_2]['RoomTemp']).shape[1],
                                                        mean = cmn_noise_hb0_top_setup_2_mean_std[:,0],
                                                        std = cmn_noise_hb0_top_setup_2_sigma_std[:,0])
                CMN_hb1_top_setup_1 = self.__extractCMN(nchannels = np.array(common_noise_hb1_top_setup[setup_1]['RoomTemp']).shape[1],
                                                        mean = cmn_noise_hb1_top_setup_1_mean_std[:,0],
                                                        std = cmn_noise_hb1_top_setup_1_sigma_std[:,0])
                CMN_hb1_top_setup_2 = self.__extractCMN(nchannels = np.array(common_noise_hb1_top_setup[setup_2]['RoomTemp']).shape[1],
                                                        mean = cmn_noise_hb1_top_setup_2_mean_std[:,0],
                                                        std = cmn_noise_hb1_top_setup_2_sigma_std[:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [CMN_hb0_top_setup_1, CMN_hb0_top_setup_2,
                                              CMN_hb1_top_setup_1, CMN_hb1_top_setup_2],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"CMNoise (top sensor)",
                                name       = f"Plot_CMN_module_bothHybrids_topSensor_compare",
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
