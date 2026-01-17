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


def get_correlation(cls_hb0, cls_hb1):
    cls_hb0_chipId = cls_hb0.chipId
    cls_hb1_chipId = cls_hb1.chipId + 8
    cls_chipId = ak.concatenate([cls_hb0_chipId, cls_hb1_chipId], axis=1)

    #from IPython import embed; embed();

    bincount_per_cbc = ak.sum(cls_chipId == 0, axis=1)[:,None] # for cbcId = 0
    for icbc in range(1,16): # 1 to 15
        temp = ak.sum(cls_chipId == icbc, axis=1)[:,None]
        bincount_per_cbc = ak.concatenate([bincount_per_cbc, temp], axis=1) 

    #bincount_per_cbc = ak.to_numpy(bincount_per_cbc)

    cls_chipId_encoded = ak.to_numpy(bincount_per_cbc)
    
    #cls_chipId_encoded = np.vstack([
    #    np.bincount(x, minlength=16) for x in cls_chipId
    #])
    #from IPython import embed; embed(); exit()
    
    corr = ak.to_numpy(np.corrcoef(cls_chipId_encoded, rowvar=False))
    return corr


def clean_white_noise(noise):
    white_noise = np.median(noise)
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


    OG = int(np.sort(np.unique(ak.to_numpy(ak.flatten(events.cluster.opticalGroupId))))[-1])
    #from IPython import embed; embed(); exit()
    hb0_id = 2*OG
    hb1_id = 2*OG + 1
    
    # =================================== #
    # ==>>> nClusters at different levels #
    # =================================== #
    nClusters_per_cbc = {
        f"opt_{OG}": {
            f"hb{hb0_id}_bottom": [],
            f"hb{hb0_id}_top"   : [],
            f"hb{hb1_id}_bottom": [],
            f"hb{hb1_id}_top"   : [],
        },
    }
    for iopt in range(OG,OG+1):
        cls_opt = events.cluster[events.cluster.opticalGroupId == iopt]
        #hb = [2*iopt, 2*iopt+1]
        #hb = [0, 1]
        for ihb in [hb0_id, hb1_id]:
            cls_hb = cls_opt[cls_opt.hybridId == ihb]
            for isen in range(2):
                key_sen = "bottom" if isen == 0 else "top"
                cls_sen = cls_hb[cls_hb.fromWhichSensor == isen]
                for icbc in range(8):
                    cls_cbc = cls_sen[cls_sen.chipId == icbc]
                    ncls_cbc = ak.sum(ak.num(cls_cbc.opticalGroupId, axis=1))
                    #nClusters_per_cbc[f'opt_{iopt}'][f'hb_{ihb}'][f's{isen}'].append([float(ncls_cbc), 0.0])
                    nClusters_per_cbc[f'opt_{iopt}'][f'hb{ihb}_{key_sen}'].append([float(ncls_cbc), 0.0])

    #from IPython import embed; embed(); exit()
                    
    plot_basic(x         = [i for i in range(8)],
               data_dict = nClusters_per_cbc[f'opt_{OG}'],
               title     = f"nClusters_per_chip",
               name      = f"nClusters_per_chip",
               xlabel    = "",
               ylabel    = "nClusters",
               xticklabels = [f'CBC_{i}' for i in range(8)],
               outdir    = output,
               linewidth = 1.5,
               #ylim      = [0, 14000.0],
               markersize = 5.0,
               fit       = False,
               legs      = ['hb0-bottom', 'hb0-top', 'hb1-bottom', 'hb1-top'])
    plot_basic(x         = [i for i in range(8)],
               data_dict = nClusters_per_cbc[f'opt_{OG}'],
               title     = f"rateClusters_per_chip",
               name      = f"rateClusters_per_chip",
               xlabel    = "",
               ylabel    = "Clusters rate",
               xticklabels = [f'CBC_{i}' for i in range(8)],
               outdir    = output,
               linewidth = 1.5,
               #ylim      = [0, 0.5],
               markersize = 5.0,
               fit       = False,
               legs      = ['hb0-bottom', 'hb0-top', 'hb1-bottom', 'hb1-top'],
               density   = True)
    
    # Plotting nClusters
    nclusters = ak.num(events.cluster.width, axis=1)


    get_ncls = lambda mask: ak.to_numpy(ak.sum(mask, axis=1))
    
    ncls_dict_hbs = {
        "OG": {
            "hb0": get_ncls(events.cluster.hybridId == hb0_id),
            "hb1": get_ncls(events.cluster.hybridId == hb1_id),
        }
    }
    hist_basic(bins      = [0., 150.0, 150], # [begin, end, nbins]
               data_dict = ncls_dict_hbs['OG'],
               title     = f"nClusters_per_hb",
               name      = f"nClusters_per_hb",
               xlabel    = "number of clusters",
               ylabel    = "number of events",
               outdir    = output,
               linewidth = 1.5,
               #ylim      = [0, 0.5],
               density   = False,
               logy      = True)


    #from IPython import embed; embed()
    
    
    ncls_dict_sensors = {
        "OG": {
            "bottom": get_ncls(events.cluster.fromWhichSensor == 0),
            "top": get_ncls(events.cluster.fromWhichSensor == 1),
        }
    }
    
    hist_basic(bins      = [0.,256.,256], # [begin, end, nbins]
               data_dict = ncls_dict_sensors['OG'],
               title     = f"nClusters_per_sensor",
               name      = f"nClusters_per_sensor",
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
        f"opt_{OG}": {
            f"hb_{hb0_id}": {f'CBC_{i}' : [] for i in range(8)},
            f"hb_{hb1_id}": {f'CBC_{i}' : [] for i in range(8)}
        }
    }
    for pair in pairs:
        cls = events.cluster[pair[0]:pair[1], :]
        for iopt in range(OG,OG+1):
            cls_opt = cls[cls.opticalGroupId == iopt]
            #hb = [2*iopt, 2*iopt+1]
            for ihb in [hb0_id, hb1_id]:
                cls_hb = cls_opt[cls_opt.hybridId == ihb]
                for icbc in range(8):
                    cls_cbc = cls_hb[cls_hb.chipId == icbc]
                    nHits = ak.to_numpy(ak.sum(cls_cbc.width, axis=1))

                    cluster_per_cbc_with_time[f'opt_{iopt}'][f'hb_{ihb}'][f'CBC_{icbc}'].append(
                        [float(np.mean(nHits)),
                         float(np.std(nHits)/np.sqrt(pair[1]-pair[0]))]
                    )

    plot_basic(x         = [i for i in range(len(pairs))],
               data_dict = cluster_per_cbc_with_time[f'opt_{OG}'][f'hb_{hb0_id}'],
               title     = f"nHits_with_time_hb0",
               name      = f"nHits_with_time_hb0",
               xlabel    = f"time stamps (every {step} events)",
               ylabel    = "nHits",
               outdir    = output,
               linewidth = 1.2,
               #ylim      = [0, 0.0045],
               fit       = False,
               fitmodel  = 'poly2')
    plot_basic(x         = [i for i in range(len(pairs))],
               data_dict = cluster_per_cbc_with_time[f'opt_{OG}'][f'hb_{hb1_id}'],
               title     = f"nHits_with_time_hb1",
               name      = f"nHits_with_time_hb1",
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
    nclusters_m1p1 = ak.to_numpy(ak.where(nclusters > 0, 1, -1))

    fs = config.get("FS") #244000  # Hz
    logger.warning(f"Check carefully ==> fs is {fs} Hz")
    f, t, Sxx = spectrogram(
        nclusters_m1p1,
        fs=fs,
        window='hann',
        nperseg=300000,
        noverlap=30000,
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
    Sxx_sum = np.concat((Sxx_sum_time[:,None], Sxx_err_time[:,None]), axis=1)

    #Sxx_sum_smooth_incl = np.concat((Sxx_sum_smooth[:,None], Sxx_err_time[:,None]), axis=1)


    # get white noise
    #Sxx_white_noise = np.percentile(Sxx_sum_time, 95)
    Sxx_clean = clean_white_noise(Sxx_sum_time)
    """
    Sxx_white_noise = np.median(Sxx_sum_time)
    Sxx_clean = Sxx_sum_time - Sxx_white_noise
    Sxx_clean[Sxx_clean < 0] = 0
    """
    Sxx_clean_incl = np.concat((Sxx_clean[:,None], Sxx_err_time[:,None]), axis=1)
    
    plot_basic(x         = f, #[i for i in range(Sxx_sum_time.shape[0])],
               data_dict = {"raw": Sxx_sum, "subtract_white_noise": Sxx_clean_incl},
               title     = f"hidden_noise_freq_incl_time",
               name      = f"hidden_noise_freq_incl_time",
               xlabel    = f"freq (Hz)",
               ylabel    = "amplitude (arbitrary)",
               outdir    = output,
               linewidth = 1.5,
               xlim      = [0, 14000],
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

    #from IPython import embed; embed(); exit()
    
    """
    #peaks_dict = {"hb0"}
    peaks_2d = []
    for i in range(8,10):
        print(f"hybrid: {i}")
        cls_hb = events.cluster[events.cluster.hybridId == i]
        for j in range(8):
            print(f'CBC: {j}')
            cls = cls_hb[cls_hb.chipId == j]
            nclusters = ak.num(cls.opticalGroupId, axis=1)
            print(f"#clusters : {ak.sum(nclusters, axis=0)}")
            nclusters_m1p1 = ak.to_numpy(ak.where(nclusters > 0, 1, -1))

            f, t, Sxx = spectrogram(
                nclusters_m1p1,
                fs=fs,
                window='hann',
                nperseg=300000,
                noverlap=30000,
                scaling='density',
                mode='magnitude'
            )
        
            plot_colormesh(x = t,
                           y = f,
                           z = Sxx,
                           shading='gouraud',
                           xlabel = "time (s)",
                           ylabel = "frquency (Hz)",
                           title  = f"hidden_noise_freq_hb{i-8}_CBC{j}",
                           name   = f"hidden_noise_freq_hb{i-8}_CBC{j}_with_time",
                           ylim   = [0,300],
                           outdir = output)

            # ==>>> Sum over time
            Sxx_sum_time = np.sum(Sxx, axis=1)
            Sxx_err_time = np.zeros_like(Sxx_sum_time)
            Sxx_sum = np.concat((Sxx_sum_time[:,None], Sxx_err_time[:,None]), axis=1)

            Sxx_clean = clean_white_noise(Sxx_sum_time)
            Sxx_clean_incl = np.concat((Sxx_clean[:,None], Sxx_err_time[:,None]), axis=1)
            
            plot_basic(x         = f,
                       data_dict = {"raw": Sxx_sum, "subtract_white_noise": Sxx_clean_incl},
                       title     = f"hidden_noise_freq_hb{i-8}_CBC{j}_incl_time",
                       name      = f"hidden_noise_freq_hb{i-8}_CBC{j}_incl_time",
                       xlabel    = f"freq (Hz)",
                       ylabel    = "amplitude (arbitrary)",
                       outdir    = output,
                       linewidth = 1.5,
                       xlim      = [0, 14000],
                       fit       = False)
            plot_basic(x         = f,
                       data_dict = {"raw": Sxx_sum, "subtract_white_noise": Sxx_clean_incl},
                       title     = f"hidden_noise_freq_hb{i-8}_CBC{j}_incl_time",
                       name      = f"hidden_noise_freq_hb{i-8}_CBC{j}_incl_time_zoom",
                       xlabel    = f"freq (Hz)",
                       ylabel    = "amplitude (arbitrary)",
                       outdir    = output,
                       linewidth = 1.5,
                       xlim      = [0, 300],
                       fit       = False)

            #peaks, properties = find_peaks(Sxx_sum_time, prominence=0.2 * Sxx_sum_time.max())
            peaks_pos, properties = find_peaks(Sxx_clean, prominence=0.25 * Sxx_clean.max())
            peaks = f[peaks_pos]
            print(f"peaks : {peaks}")
            #from IPython import embed; embed(); exit()
            temp = []
            for ipeak in peaks_pos_all:
                #temp.append(float(Sxx_sum_time[ipeak]))
                temp.append(float(Sxx_clean[ipeak]))

            peaks_2d.append(temp)


    peaks_2d  = np.array(peaks_2d)
    #from IPython import embed; embed(); exit()
    plot_heatmap(data = peaks_2d.T,
                 title = "amp_hidden_freq",
                 name = "amp_hidden_freq",
                 xticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                 yticklabels = [f'{f} Hz' for f in peaks_all],
                 outdir = output,
                 #vmin = -1, vmax = 1,
                 cbar_label = "aplitude")
    peaks_2d_hb0 = peaks_2d[0:8,:].tolist()
    peaks_2d_hb1 = peaks_2d[8:16,:].tolist()

    peaks_2d_hb0_dict = {f"CBC_{i}":item for i,item in enumerate(peaks_2d_hb0)}
    peaks_2d_hb1_dict = {f"CBC_{i}":item for i,item in enumerate(peaks_2d_hb1)}    
    
    plot_basic(x         = peaks_all,
               data_dict = peaks_2d_hb0_dict,
               title     = f"hidden_noise_amp_hb0",
               name      = f"hidden_noise_amp_hb0",
               xlabel    = f"freq (Hz)",
               ylabel    = "amplitude (arbitrary)",
               outdir    = output,
               linewidth = 1.2,
               #xlim      = [0, 1200],
               fit       = False)
    plot_basic(x         = peaks_all,
               data_dict = peaks_2d_hb1_dict,
               title     = f"hidden_noise_amp_hb1",
               name      = f"hidden_noise_amp_hb1",
               xlabel    = f"freq (Hz)",
               ylabel    = "amplitude (arbitrary)",
               outdir    = output,
               linewidth = 1.2,
               #xlim      = [0, 1200],
               fit       = False)


    
    """
    
    # ====================================================== #
    # ==>>> Keep only those events with at least one cluster #
    # ====================================================== #
    no_cluster_mask = ak.num(events.cluster.opticalGroupId, axis=1) == 0
    events = events[~no_cluster_mask]

    # clusters from hb0
    cls_hb0 = events.cluster[events.cluster.hybridId == hb0_id]
    # clusters from hb1
    cls_hb1 = events.cluster[events.cluster.hybridId == hb1_id]

    # ==>>> pearson correl coeff among all channels
    corr = get_correlation(cls_hb0, cls_hb1)    
    plot_heatmap(data = corr,
                 title = "Pearson Corr-Coeff / Module",
                 name = "corr_coeff",
                 xticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                 yticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                 outdir = output,
                 vmin = -1, vmax = 1,
                 colmap = 'Spectral',
                 cbar_label = "Correlation Coefficient (-1 to +1)")

    # cluster hb0 from top/bot sensor
    cls_hb0_top = cls_hb0[cls_hb0.fromWhichSensor == 1]
    cls_hb0_bot = cls_hb0[cls_hb0.fromWhichSensor == 0]
    # cluster hb1 from top/bot sensor
    cls_hb1_top = cls_hb1[cls_hb1.fromWhichSensor == 1]
    cls_hb1_bot = cls_hb1[cls_hb1.fromWhichSensor == 0]

    # ==>>> pearson correl for channels from top sensor
    corr_top = get_correlation(cls_hb0_top, cls_hb1_top)
    plot_heatmap(data = corr_top,
                 title = "Pearson Corr-Coeff / Module (TopS)",
                 name = "corr_coeff_top",
                 xticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                 yticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                 outdir = output,
                 vmin = -1, vmax = 1,
                 colmap	= 'Spectral',
                 cbar_label = "Correlation Coefficient (-1 to +1)")
    # ==>>> pearson correl for channels from bottom sensor
    corr_bot = get_correlation(cls_hb0_bot, cls_hb1_bot)
    plot_heatmap(data = corr_bot,
                 title = "Pearson Corr-Coeff / Module (BotS)",
                 name = "corr_coeff_bot",
                 xticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                 yticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
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
