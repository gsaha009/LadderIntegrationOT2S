# ---------------------------------------- #
#   Env: conda activate PyROOTEnv (Mac)    #
#   Env:    source setup.sh (LxPlus)       #
# ---------------------------------------- #
import os
import yaml
import time
import uproot
import argparse
import datetime
import awkward as ak
import numpy as np
from scipy import stats
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from otutil import setup_logger, processor
from otutil import plot_basic, hist_basic, plot_heatmap, plot_colormesh


def get_correlation(cls):
    cls_ogid = cls.opticalGroupId
    bincount_per_og = ak.sum(cls_ogid == 0, axis=1)[:,None] # for ogId = 0 
    for iog in range(1,12): # 1 to 11
        temp = ak.sum(cls_ogid == iog, axis=1)[:,None]
        bincount_per_og = ak.concatenate([bincount_per_og, temp], axis=1) 
    cls_ogId_encoded = ak.to_numpy(bincount_per_og)
        
    cls_hbid = cls.hybridId
    bincount_per_hb = ak.sum(cls_hbid == 0, axis=1)[:,None] # for hbId = 0
    for ihb in range(1,24): # 1 to 23
        temp = ak.sum(cls_hbid == ihb, axis=1)[:,None]
        bincount_per_hb = ak.concatenate([bincount_per_hb, temp], axis=1) 
    cls_hbId_encoded = ak.to_numpy(bincount_per_hb)
    
    corr_og = ak.to_numpy(np.corrcoef(cls_ogId_encoded, rowvar=False))
    corr_hb = ak.to_numpy(np.corrcoef(cls_hbId_encoded, rowvar=False))

    return corr_og, corr_hb


def clean_white_noise(noise):
    #white_noise = np.median(noise) # 50%
    white_noise = np.percentile(noise, 90) # 90%
    noise_clean = noise - white_noise
    noise_clean[noise_clean < 0] = 0
    return noise_clean


def main(args):
    real_start = time.perf_counter()
    cpu_start  = time.process_time()
    
    dttag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger()
    logger.info(f"date-time: {dttag}")

    config = None
    if not os.path.exists(args.config):
        raise RuntimeError(f"{args.config} does not exist")
    else:
        with open(args.config, 'r') as _c:
            config = yaml.safe_load(_c)
    
    infile = config.get('INPUT_FILE')
    if os.path.exists(infile):
        logger.info(f"Input file {infile} found")
    else:
        raise RuntimeError(f"Input file {infile} not found")

    if args.tag != "":
        output = f"{config.get('OUTPUT')}__{args.tag}"
    else:
        output = f"{config.get('OUTPUT')}__{config.get('OUTTAG')}"

    if os.path.exists(output):
        logger.info(f"Output dir {output} found")
    else:
        logger.info(f"Creating output dir {output}")
        os.mkdir(output)

    
    # ==>>> Process Events ==>>>
    events = processor(infile, "Events")


    #OG = int(np.sort(np.unique(ak.to_numpy(ak.flatten(events.cluster.opticalGroupId))))[-1])
    OGs = np.arange(12).tolist()
    #from IPython import embed; embed(); exit()
    #hb0_id = 2*OG
    #hb1_id = 2*OG + 1


    # =================================== #
    # ==>>> nClusters at different levels #
    # =================================== #
    nClusters_per_OG = {
        "All_OptGroups": [ak.sum(events.cluster.opticalGroupId == iog) for iog in OGs]
    }

    plot_basic(x         = [i for i in range(12)],
               data_dict = nClusters_per_OG,
               title     = "nClusters_per_OG",
               name      = "nClusters_per_OG",
               xlabel    = "",
               ylabel    = "nClusters",
               xticklabels = [f'OG_{i}' for i in range(12)],
               outdir    = output,
               linewidth = 1.5,
               #ylim      = [0, 14000.0],
               markersize = 5.0,
               fit       = False)
    
    nClusters_per_OG_HB = {
        "HB0": [ak.sum(events.cluster.hybridId == 2*iog) for iog in OGs],
        "HB1": [ak.sum(events.cluster.hybridId == 2*iog+1) for iog in OGs]
    }
    plot_basic(x         = [i for i in range(12)],
               data_dict = nClusters_per_OG_HB,
               title     = "nClusters_per_OG_HB",
               name      = "nClusters_per_OG_HB",
               xlabel    = "",
               ylabel    = "nClusters",
               xticklabels = [f'OG_{i}' for i in range(12)],
               outdir    = output,
               linewidth = 1.5,
               #ylim      = [0, 14000.0],
               markersize = 5.0,
               fit       = False)


    nClusters_per_OG_Sensor = {"BottomS":[], "TopS":[]}
    for iog in OGs:
        _cls = events.cluster.fromWhichSensor[events.cluster.opticalGroupId == iog]
        nClusters_per_OG_Sensor["BottomS"].append(ak.sum(_cls == 0))
        nClusters_per_OG_Sensor["TopS"].append(ak.sum(_cls == 1))

    
    plot_basic(x         = [i for i in range(12)],
               data_dict = nClusters_per_OG_Sensor,
               title     = "nClusters_per_OG_Sensor",
               name      = "nClusters_per_OG_Sensor",
               xlabel    = "",
               ylabel    = "nClusters",
               xticklabels = [f'OG_{i}' for i in range(12)],
               outdir    = output,
               linewidth = 1.5,
               #ylim      = [0, 14000.0],
               markersize = 5.0,
               fit       = False)


    nclusters = ak.num(events.cluster.width, axis=1)


    get_ncls = lambda mask: ak.to_numpy(ak.sum(mask, axis=1))
    ncls_dict_sensors = {
        "OGs": {
            "bottom": get_ncls(events.cluster.fromWhichSensor == 0),
            "top": get_ncls(events.cluster.fromWhichSensor == 1),
        }
    }
    
    hist_basic(bins      = [0, 3000, 3000], # [begin, end, nbins]
               data_dict = ncls_dict_sensors['OGs'],
               title     = f"nClusters_per_sensor",
               name      = f"nClusters_per_sensor",
               xlabel    = "number of clusters",
               ylabel    = "number of events",
               outdir    = output,
               linewidth = 1.5,
               #ylim      = [0, 0.5],
               density   = False,
               logy      = True)



    ncls_dict_sensors_OG = {
        f"OG_{OG}": {
            f"bottom-OG{OG}": get_ncls((events.cluster.opticalGroupId == OG) & (events.cluster.fromWhichSensor == 0)),
            f"top-OG{OG}": get_ncls((events.cluster.opticalGroupId == OG) & (events.cluster.fromWhichSensor == 1)),
        } for OG in OGs
    }

    for OG in OGs:
        hist_basic(bins      = [0.,256.,256], # [begin, end, nbins]
                   data_dict = ncls_dict_sensors_OG[f'OG_{OG}'],
                   title     = f"nClusters_per_sensor_OG_{OG}",
                   name      = f"nClusters_per_sensor_OG_{OG}",
                   xlabel    = "number of clusters",
                   ylabel    = "number of events",
                   outdir    = output,
                   linewidth = 1.5,
                   #ylim      = [0, 0.5],
                   density   = False,
                   logy      = True)


        
    # ========================================================= #
    # ==>>> Get time evolution wrt nHits i.e. sum(clusterWidth) #
    # ========================================================= #
    n_events = len(events)
    #step = 200000
    step = int(0.04*n_events)
    boundaries = [min(i, n_events) for i in range(0, n_events + step, step)]
    pairs = list(zip(boundaries[:-1], boundaries[1:]))
    cluster_per_cbc_with_time = {
        f"OG_{OG}": {
            f"hb_{2*OG}": {f'CBC_{i}' : [] for i in range(8)},
            f"hb_{1+2*OG}": {f'CBC_{i}' : [] for i in range(8)}
        } for OG in OGs
    }
    for pair in pairs:
        cls = events.cluster[pair[0]:pair[1], :]
        for iopt in OGs:
            cls_opt = cls[cls.opticalGroupId == iopt]
            for ihb in [2*iopt, 2*iopt+1]:
                cls_hb = cls_opt[cls_opt.hybridId == ihb]
                for icbc in range(8):
                    cls_cbc = cls_hb[cls_hb.chipId == icbc]
                    nHits = ak.to_numpy(ak.sum(cls_cbc.width, axis=1))

                    cluster_per_cbc_with_time[f'OG_{iopt}'][f'hb_{ihb}'][f'CBC_{icbc}'].append(
                        [float(np.mean(nHits)),
                         float(np.std(nHits)/np.sqrt(pair[1]-pair[0]))]
                    )

    for OG in OGs:
        plot_basic(x         = [i for i in range(len(pairs))],
                   data_dict = cluster_per_cbc_with_time[f'OG_{OG}'][f'hb_{2*OG}'],
                   title     = f"nHits_with_time_OG{OG}_hb{2*OG}",
                   name      = f"nHits_with_time_OG{OG}_hb{2*OG}",
                   xlabel    = f"time stamps (every {step} events)",
                   ylabel    = "nHits",
                   outdir    = output,
                   linewidth = 1.2,
                   #ylim      = [0, 0.0045],
                   fit       = False,
                   fitmodel  = 'poly2')
        plot_basic(x         = [i for i in range(len(pairs))],
                   data_dict = cluster_per_cbc_with_time[f'OG_{OG}'][f'hb_{1+2*OG}'],
                   title     = f"nHits_with_time_OG{OG}_hb{1+2*OG}",
                   name      = f"nHits_with_time_OG{OG}_hb{1+2*OG}",
                   xlabel    = f"time stamps (every {step} events)",
                   ylabel    = "nHits",
                   outdir    = output,
                   linewidth = 1.2,
                   #ylim      = [0, 0.0045],
                   fit       = False,
                   fitmodel  = 'poly2')




    # ========================================================= #
    # ==>>> Extract hidden frequencies from Clusters            #
    # ========================================================= #
    nclusters = ak.num(events.cluster.opticalGroupId, axis=1)
    #nclusters_m1p1 = ak.to_numpy(ak.where(nclusters > 0, 1, -1))
    nclusters_m1p1 = ak.to_numpy(2*nclusters/np.max(nclusters) - 1)
    
    fs = config.get("FS") #244000  # Hz
    logger.warning(f"Check carefully ==> fs is {fs} Hz")
    f, t, Sxx = spectrogram(
        nclusters_m1p1,
        fs=fs,
        window='hann',
        nperseg=10240, #300000,
        noverlap=int(0.85*10240), #30000,
        scaling='density',
        mode='magnitude'
    )

    #from IPython import embed; embed(); exit()
    
    plot_colormesh(x = t,
                   y = f,
                   z = Sxx,
                   shading='gouraud',
                   xlabel = "time (s)",
                   ylabel = "frquency (Hz)",
                   title  = "hidden_noise_freq",
                   name   = "hidden_noise_freq_with_time",
                   ylim   = [0,300],
                   outdir = output)

    # ==>>> Sum over time
    Sxx_sum_time = np.sum(Sxx, axis=1)
    # use gaussian filter
    #Sxx_sum_smooth = gaussian_filter1d(Sxx_sum_time, sigma=2)

    Sxx_max = np.max(Sxx_sum_time)
    f_max_peak = f[np.argmax(Sxx_sum_time)]

    print(f"Sxx_max    : {Sxx_max}")
    print(f"f_max_peak : {f_max_peak}")
    
    Sxx_err_time = np.zeros_like(Sxx_sum_time)
    Sxx_sum = np.concatenate((Sxx_sum_time[:,None], Sxx_err_time[:,None]), axis=1)

    #Sxx_sum_smooth_incl = np.concat((Sxx_sum_smooth[:,None], Sxx_err_time[:,None]), axis=1)


    # get white noise
    #Sxx_white_noise = np.percentile(Sxx_sum_time, 95)
    Sxx_clean = clean_white_noise(Sxx_sum_time)
    #Sxx_white_noise = np.median(Sxx_sum_time)
    #Sxx_clean = Sxx_sum_time - Sxx_white_noise
    #Sxx_clean[Sxx_clean < 0] = 0
    
    Sxx_clean_incl = np.concatenate((Sxx_clean[:,None], Sxx_err_time[:,None]), axis=1)
    
    plot_basic(x         = f, #[i for i in range(Sxx_sum_time.shape[0])],
               data_dict = {"raw": Sxx_sum, "subtract_white_noise": Sxx_clean_incl},
               title     = f"hidden_noise_freq_incl_time",
               name      = f"hidden_noise_freq_incl_time",
               xlabel    = f"freq (Hz)",
               ylabel    = "amplitude (arbitrary)",
               outdir    = output,
               linewidth = 1.5,
               xlim      = [0, 3000],
               fit       = False)
    plot_basic(x         = f, #[i for i in range(Sxx_sum_time.shape[0])],
               data_dict = {"raw": Sxx_sum, "subtract_white_noise": Sxx_clean_incl},
               title     = f"hidden_noise_freq_incl_time",
               name      = f"hidden_noise_freq_incl_time_zoom",
               xlabel    = f"freq (Hz)",
               ylabel    = "amplitude (arbitrary)",
               outdir    = output,
               linewidth = 1.5,
               xlim      = [0, 300],
               fit       = False)
    



    #peaks, properties = find_peaks(Sxx_sum_smooth, prominence=0.2 * Sxx_sum_smooth.max())
    peaks_pos, properties = find_peaks(Sxx_clean, prominence=0.25 * Sxx_clean.max())
    peaks_pos = peaks_pos[:10]
    peaks = f[peaks_pos]
    #peaks, properties = find_peaks(Sxx_sum_time, prominence=0.2 * Sxx_sum_time.max())
    peaks_pos_all = peaks_pos.tolist()
    peaks_all = peaks.tolist()
    print(peaks_all)



    
    # ====================================================== #
    # ==>>> Keep only those events with at least one cluster #
    # ====================================================== #
    no_cluster_mask = ak.num(events.cluster.opticalGroupId, axis=1) == 0
    events = events[~no_cluster_mask]

    # clusters from hb0
    #cls_hb0 = events.cluster[events.cluster.hybridId == hb0_id]
    # clusters from hb1
    #cls_hb1 = events.cluster[events.cluster.hybridId == hb1_id]

    # ==>>> pearson correl coeff among all channels
    corr_og, corr_hb = get_correlation(events.cluster)
    plot_heatmap(data = corr_og,
                 title = "Pearson Corr-Coeff / OG",
                 name = "corr_coeff_per_og",
                 xticklabels = [f'OG-{i}' for i in range(12)],
                 yticklabels = [f'OG-{i}' for i in range(12)],
                 outdir = output,
                 vmin = -1, vmax = 1,
                 colmap = 'Spectral',
                 cbar_label = "Correlation Coefficient (-1 to +1)")
    plot_heatmap(data = corr_hb,
                 title = "Pearson Corr-Coeff / Hb",
                 name = "corr_coeff_per_hb",
                 xticklabels = [f'OG{i}-hb{j}' for i in range(12) for j in [2*i, 1+2*i]],
                 yticklabels = [f'OG{i}-hb{j}' for i in range(12) for j in [2*i, 1+2*i]],
                 outdir = output,
                 vmin = -1, vmax = 1,
                 colmap = 'Spectral',
                 cbar_label = "Correlation Coefficient (-1 to +1)")


    # cluster hb0 from top/bot sensor
    cls_top = events.cluster[events.cluster.fromWhichSensor == 1]
    cls_bot = events.cluster[events.cluster.fromWhichSensor == 0]

    # ==>>> pearson correl for channels from top sensor
    corr_top_og, corr_top_hb = get_correlation(cls_top)
    plot_heatmap(data = corr_top_og,
                 title = "Pearson Corr-Coeff / OG (TopS)",
                 name = "corr_coeff_og_top",
                 xticklabels = [f'OG-{i}' for i in range(12)],
                 yticklabels = [f'OG-{i}' for i in range(12)],
                 outdir = output,
                 vmin = -1, vmax = 1,
                 colmap	= 'Spectral',
                 cbar_label = "Correlation Coefficient (-1 to +1)")
    plot_heatmap(data = corr_top_hb,
                 title = "Pearson Corr-Coeff / HB (TopS)",
                 name = "corr_coeff_hb_top",
                 xticklabels = [f'OG{i}-hb{j}' for i in range(12) for j in [2*i, 1+2*i]],
                 yticklabels = [f'OG{i}-hb{j}' for i in range(12) for j in [2*i, 1+2*i]],
                 outdir = output,
                 vmin = -1, vmax = 1,
                 colmap	= 'Spectral',
                 cbar_label = "Correlation Coefficient (-1 to +1)")


    # ==>>> pearson correl for channels from bottom sensor
    corr_bot_og, corr_bot_hb = get_correlation(cls_bot)
    plot_heatmap(data = corr_bot_og,
                 title = "Pearson Corr-Coeff / OG (BotS)",
                 name = "corr_coeff_og_bot",
                 xticklabels = [f'OG-{i}' for i in range(12)],
                 yticklabels = [f'OG-{i}' for i in range(12)],
                 outdir = output,
                 vmin = -1, vmax = 1,
                 colmap	= 'Spectral',
                 cbar_label = "Correlation Coefficient (-1 to +1)")
    plot_heatmap(data = corr_bot_hb,
                 title = "Pearson Corr-Coeff / HB (BotS)",
                 name = "corr_coeff_hb_bot",
                 xticklabels = [f'OG{i}-hb{j}' for i in range(12) for j in [2*i, 1+2*i]],
                 yticklabels = [f'OG{i}-hb{j}' for i in range(12) for j in [2*i, 1+2*i]],
                 outdir = output,
                 vmin = -1, vmax = 1,
                 colmap	= 'Spectral',
                 cbar_label = "Correlation Coefficient (-1 to +1)")





    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotter')
    parser.add_argument('-c', '--config', type=str, required=True, help="yaml configs to be used")
    parser.add_argument('-t', '--tag', type=str, required=False, default="", help="<output_dir>_<tag>")
   
    args = parser.parse_args()
        
    main(args)
