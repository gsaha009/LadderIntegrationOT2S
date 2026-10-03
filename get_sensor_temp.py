#

import os
import ROOT
import numpy as np
from prettytable import PrettyTable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import mplhep as hep
hep.style.use("CMS")


def basic_plot_settings():
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


def clean_outliers(sensor_temps):
    sensor_temp_median = np.median(sensor_temps)
    sensor_temps_sel = sensor_temps[np.abs(sensor_temps-sensor_temp_median) < 1.0]
        
    sensor_temps_sel_grd = np.gradient(sensor_temps_sel)
    sensor_temps_sel = sensor_temps_sel[np.abs(sensor_temps_sel_grd) < 0.5]

    return sensor_temps_sel



def plot_basic(data = None,
               yerr = None,
               legends = None,
               title = "Default",
               name = "Default",
               **kwargs):
    
    basics = basic_plot_settings()

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
            #print(tick_locs, tick_labels)
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
    fig.savefig(f"{outdir}/{name}.png", dpi=300)
    plt.close()
    


def plot_sensor_temp(sensor_temp_dict,
                     outdir = None,
                     cumul = False,
                     **kwargs):

    pname = kwargs.get('pname', 'default')
    cool_temp = kwargs.get('cool', 15.0)
    ylim      = kwargs.get("ylim", [-50.0, 50.0])
    
    temp_val_dict = {'SensorTemp': []}
    temp_err_dict = {'SensorTemp': []}
        
    plot_basic(data       = sensor_temp_dict,
               title      = f"Sensor_Temperature_CO2_{cool_temp}",
               name       = f"{pname}_CO2_{cool_temp}",
               outdir     = outdir,
               xlabel     = 'time stamps',
               ylabel     = 'sensor temperature (deg C)',
               linewidth  = 0.5,
               ylim       = ylim)
    
    if cumul == True:
        for setup,temp in sensor_temp_dict.items():
            temp_val_dict['SensorTemp'].append(float(np.mean(temp)))
            temp_err_dict['SensorTemp'].append(float(np.std(temp)))
        
        temp_val_dict = {key: np.array(val) for key,val in temp_val_dict.items()}
        temp_err_dict = {key: np.array(val) for key,val in temp_err_dict.items()}

        print(temp_val_dict)
        
        #from IPython import embed; embed()
        plot_basic(data        = temp_val_dict,
                   yerr        = temp_err_dict,
                   title       = f"Sensor_Temperature_{cool_temp}",
                   name        = f"{pname}_cumul_CO2_{cool_temp}",
                   outdir      = outdir,
                   xticklabels = list(sensor_temp_dict.keys()),
                   ylabel      = 'sensor temperature (deg C)',
                   linewidth   = 1.2,
                   ylim        = ylim)
        
    


def main(files, **kwargs):

    out = kwargs.get('out', os.getcwd())
    table_name = kwargs.get('tabname', 'senT')
    OG = kwargs.get('OG')

    cool = kwargs.get('cool', 15.0)
    
    table = PrettyTable()
    table.field_names = ["Setup", "Value", "StdDev", "DeltaT"]
    table.title = f"Sensor Temperature, CO2: {cool} deg"
    table.align = "r"
    table.border = True
    
    get_point_x = lambda graph_temp : np.array([graph_temp.GetPointX(i) for i in range(graph_temp.GetN())])
    get_point_y = lambda graph_temp : np.array([graph_temp.GetPointY(i) for i in range(graph_temp.GetN())])

    temp_dict_raw = {}
    temp_dict_sel = {}

    
    for key, file_path in files.items():
        #print(key)
        
        dqm_file = os.path.join(file_path, 'MonitorDQM.root')
        if not os.path.exists(dqm_file):
            raise RuntimeError(f"{dqm_file} not found")
        
        ptr = ROOT.TFile(dqm_file, "READ")

        gr_sensor_temp = ptr.Get(f'Detector/Board_0/OpticalGroup_{OG}/D_B(0)_LpGBT_DQM_SensorTemp_OpticalGroup({OG})')
        
        #time_stamps  = get_point_x(gr_sensor_temp)
        sensor_temps = get_point_y(gr_sensor_temp)
        temp_dict_raw[key] = sensor_temps
        
        sensor_temps_sel = clean_outliers(sensor_temps)        
        temp_dict_sel[key] = sensor_temps_sel
        
        temp = np.max(sensor_temps_sel)
        temp_std = np.std(sensor_temps_sel)
        
        
        table.add_row([
            key,
            round(temp, 3),
            round(temp_std, 3),
            round(temp-cool, 3)
        ])
                    
    print(str(table))
    with open(f"{out}/{table_name}.txt", "w") as ftxt:
        ftxt.write(table.get_string())
    with open(f"{out}/{table_name}.csv", "w") as fcsv:
        fcsv.write(table.get_csv_string())
        
    plot_sensor_temp(temp_dict_raw,
                     outdir = out,
                     pname = "Sensor_Temp_Raw",
                     cool = cool,
                     ylim = [17.0, 33.0] if cool > 0.0 else [-22.0, -12.0])
    plot_sensor_temp(temp_dict_sel,
                     outdir = out,
                     cumul = True,
                     pname = "Sensor_Temp_Clean",
                     cool = cool,
                     ylim = [17.0, 33.0] if cool > 0.0 else [-22.0, -12.0])


        
if __name__ == "__main__":


    
    # single module on ladder
    files1 = {
        "all_connected"     : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_89__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_none",
        "pos_1_missing"     : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_88__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_1",
        "pos_2_missing"     : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_90__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_2",
        "pos_3_missing"     : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_91__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_3",
        "pos_4_missing"     : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_92__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_4",
        "pos_5_missing"     : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_94__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_5",
        "pos_6_missing"     : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_95__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_6",
        #"pos_65_missing"    : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_96__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_65",
        #"pos_654_missing"   : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_97__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_654",
        #"pos_6543_missing"  : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_98__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_6543",
        #"pos_65432_missing" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_99__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_65432",
        #"pos_12_missing"    : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_101__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_12",
        #"pos_123_missing"   : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_102__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_123",
        #"pos_1234_missing"  : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_103__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_1234",
        #"pos_12345_missing" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_104__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_12345",
        "pos_all_missing"   : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_100__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_654321",
    }
    #files2 = {
    #    "all_connected"     : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_89__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_none",
    #    "pos_6_missing"     : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_95__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_6",
    #    "pos_65_missing"    : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_96__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_65",
    #    "pos_654_missing"   : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_97__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_654",
    #    "pos_6543_missing"  : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_98__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_6543",
    #    "pos_65432_missing" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_99__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_65432",
    #    "pos_all_missing"   : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_100__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_654321",
    #    "pos_12_missing"    : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_101__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_12",
    #    "pos_123_missing"   : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_102__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_123",
    #    "pos_1234_missing"  : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_103__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_1234",
    #    "pos_12345_missing" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_104__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_12345",
    #}


    out = "/Users/gsaha/Work/IPHC/TrackerUpgrade/Output/ThermalResults"
    if not os.path.exists(out):
        os.mkdir(out)

    out_single_wok = f'{out}/Single_p15_without_kapton_tape'
    if not os.path.exists(out_single_wok):
        os.mkdir(out_single_wok)
    
    main(files1, tabname='sensor_temp_one_module_without_kapton', OG=2, cool=15.0, out = out_single_wok)
    #main(files2, tabname='sensor_temp_setup_2', OG=2, cool=15.0)

    files2 = {
        "all_connected" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_89__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_none",        
        "pos_1_missing" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_115__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_kapton_1",
        "pos_2_missing" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_114__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_kapton_2",
        "pos_3_missing" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_113__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_kapton_3",
        "pos_4_missing" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_112__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_kapton_4",
        "pos_5_missing" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_111__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_kapton_5",
        "pos_6_missing" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_110__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_kapton_6",
        "pos_all_missing"   : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/Run_100__minus_6CP_8__pos2__OG2__p15__box_open__all_mod_on__missing_screw_654321",        
    }    

    out_single_wk = f'{out}/Single_p15_with_kapton_tape'
    if not os.path.exists(out_single_wk):
        os.mkdir(out_single_wk)
    
    main(files2, tabname='sensor_temp_one_module_with_kapton', OG=2, cool=15.0, out = out_single_wk)
    
