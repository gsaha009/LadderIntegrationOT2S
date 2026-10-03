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
            "markerstyles": ['o', 's', 'D', '*', '^', 'v', 'P', 'X', '<', '>', 'h', '8'],
            "colors": [
                "#165a86","#cc660b","#217a21","#a31e1f","#6e4e92","#6b443e",
                "#b85fa0","#666666","#96971b","#1294a6","#8c1c62","#144d3a"
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
                (0, (7, 3))
            ]}
    


def clean_outliers(sensor_temps):
    sensor_temp_median = np.median(sensor_temps)
    sensor_temps_sel = sensor_temps[np.abs(sensor_temps-sensor_temp_median) < 1.0]
        
    sensor_temps_sel_grd = np.gradient(sensor_temps_sel)
    sensor_temps_sel = sensor_temps_sel[np.abs(sensor_temps_sel_grd) < 0.5]

    return sensor_temps_sel


def get_chunk(temps):
    n = len(temps)    
    first = np.arange(0, 10)
    last = np.arange(n - 10, n)
    middle_indices = np.arange(10, n - 10)
    middle_sample = np.random.choice(middle_indices, 80, replace=False)
    all_indices = np.concatenate([first, middle_sample, last])
    all_indices = np.sort(all_indices)
    temps_selected = temps[all_indices]

    return temps_selected


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
                    fmt = markerstyles[i],
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
                     ladderlevel = False,
                     **kwargs):

    pname = kwargs.get('pname', 'default')
    cool_temp = kwargs.get('cool', 15.0)
    ylim      = kwargs.get("ylim", [-50.0, 50.0])


    
    temp_val_dict = {key: [] for key in sensor_temp_dict}
    temp_err_dict = {key: [] for key in sensor_temp_dict}
        
    for ladder_id, og_temp_dict in sensor_temp_dict.items():
        print(f"Ladder : {ladder_id}")

        plot_basic(data       = og_temp_dict,
                   title      = f"Sensor_Temperature_CO2_{cool_temp}_{ladder_id}",
                   name       = f"{pname}_{ladder_id}_CO2_{cool_temp}",
                   outdir     = outdir,
                   xlabel     = 'time stamps',
                   ylabel     = 'sensor temperature (deg C)',
                   linewidth  = 0.5,
                   ylim       = ylim)

        if ladderlevel == True:
            for og,temp in og_temp_dict.items():
                temp_val_dict[ladder_id].append(float(np.mean(temp)))
                temp_err_dict[ladder_id].append(float(np.std(temp)))


    if ladderlevel == True:
        temp_val_dict = {key: np.array(val) for key,val in temp_val_dict.items()}
        temp_err_dict = {key: np.array(val) for key,val in temp_err_dict.items()}
        
        #from IPython import embed; embed()
        plot_basic(data        = temp_val_dict,
                   yerr        = temp_err_dict,
                   title       = f"Sensor_Temperature_{cool_temp}",
                   name        = f"{pname}_CO2_{cool_temp}",
                   outdir      = outdir,
                   xticklabels = [f'OG_{i}' for i in range(12)],
                   ylabel      = 'sensor temperature (deg C)',
                   linewidth   = 1.2,
                   ylim        = ylim)
                


def main(files, **kwargs):

    out = kwargs.get('out', os.getcwd())
    table_name = kwargs.get('tabname', 'senT')
    OGs = kwargs.get('OGs', [])
    
    if len(OGs) == 0:
        raise RuntimeError("No OpticalGroup is specified")
    cool = kwargs.get('cool', 15.0)
    
    get_point_x = lambda graph_temp : np.array([graph_temp.GetPointX(i) for i in range(graph_temp.GetN())])
    get_point_y = lambda graph_temp : np.array([graph_temp.GetPointY(i) for i in range(graph_temp.GetN())])

    temp_dict_raw = {key: {} for key in files}
    temp_dict_sel = {key: {} for key in files}
    
    for key, file_path in files.items():
        #print(key)
        
        dqm_file = file_path
        if not os.path.exists(dqm_file):
            raise RuntimeError(f"{dqm_file} not found")
        
        ptr = ROOT.TFile(dqm_file, "READ")

        table = PrettyTable()
        table.field_names = ["OpticalGroup", "Value", "StdDev", "DeltaT"]
        table.title = f"Sensor Temperature : {key}, CO2: {cool} deg"
        table.align = "r"
        table.border = True
    

        for OG in OGs:
            
            gr_sensor_temp = ptr.Get(f'Detector/Board_0/OpticalGroup_{OG}/D_B(0)_LpGBT_DQM_SensorTemp_OpticalGroup({OG})')
        
            #time_stamps  = get_point_x(gr_sensor_temp)
            sensor_temps = get_point_y(gr_sensor_temp)
            temp_dict_raw[key][f'OpticalGroup_{OG}'] = sensor_temps            

            sensor_temps_sel = clean_outliers(sensor_temps)
            temp_dict_sel[key][f'OpticalGroup_{OG}'] = sensor_temps_sel

            
            temp = np.mean(sensor_temps_sel)
            temp_std = np.std(sensor_temps_sel)


            
            table.add_row([
                OG,
                round(temp, 3),
                round(temp_std, 3),
                round(temp-cool, 3)
            ])
                    
        print(str(table))
        with open(f"{out}/{table_name}_{key}.txt", "w") as ftxt:
            ftxt.write(table.get_string())
        with open(f"{out}/{table_name}_{key}.csv", "w") as fcsv:
            fcsv.write(table.get_csv_string())
        
    #from IPython import embed; embed()
    plot_sensor_temp(temp_dict_raw,
                     outdir = out,
                     ladderlevel = False,
                     pname = "Sensor_Temp_Raw",
                     cool = cool,
                     ylim = [18.0, 26.0] if cool > 0.0 else [-22.0, -12.0])
    plot_sensor_temp(temp_dict_sel,
                     outdir = out,
                     ladderlevel = True,
                     pname = "Sensor_Temp_Clean",
                     cool = cool,
                     ylim = [18.0, 26.0] if cool > 0.0 else [-22.0, -12.0])
    
        
if __name__ == "__main__":


    
    # full ladder
    files_15 = {
        "Ladder_minus_6CP_1" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_1__p15.root",
        "Ladder_minus_6CP_4" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_4__p15.root",
        "Ladder_minus_6CP_5" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_5__p15.root",
        "Ladder_minus_6CP_6" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_6__p15.root",
        "Ladder_minus_6CP_7" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_7__p15.root",
        "Ladder_minus_6CP_8" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_8__p15.root",
        "Ladder_minus_6CP_9" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_9__p15.root",
    }
    
    
    out = "/Users/gsaha/Work/IPHC/TrackerUpgrade/Output/ThermalResults"
    if not os.path.exists(out):
        os.mkdir(out)

    out15 = f'{out}/CO2_p15'
    if not os.path.exists(out15):
        os.mkdir(out15)
    
    main(files_15, tabname='sensor_temp_ladders_p15', OGs=list(range(12)),
         cool=15.0, out = out15)



    files_30 = {
        "Ladder_minus_6CP_1" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_1__m30.root",
        "Ladder_minus_6CP_4" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_4__m30.root",
        "Ladder_minus_6CP_5" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_5__m30.root",
        "Ladder_minus_6CP_6" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_6__m30.root",
        "Ladder_minus_6CP_7" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_7__m30.root",
        "Ladder_minus_6CP_8" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_8__m30.root",
        "Ladder_minus_6CP_9" : "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/ThermalResults/MonitorDQM__Ladder_minus_6CP_9__m30.root",
    }

    
    out30 = f'{out}/CO2_m30'
    if not os.path.exists(out30):
        os.mkdir(out30)

    main(files_30, tabname='sensor_temp_ladders_m30', OGs=list(range(12)),
         cool=-30.0, out = out30)
