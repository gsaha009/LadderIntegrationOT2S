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
from otutil import plot_basic, hist_basic, plot_heatmap, plot_colormesh, hist_2D, hist_cbc_group


dttag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger()
logger.info(f"date-time: {dttag}")




def get_correlation_chip(cls_hb0, cls_hb1):
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
    
    corr = ak.to_numpy(np.corrcoef(cls_chipId_encoded, rowvar=False))
    return corr



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
    white_noise = np.percentile(noise, 90)
    noise_clean = noise - white_noise
    noise_clean[noise_clean < 0] = 0
    return noise_clean


def get_hidden_freq(data = None, output = None, **kwargs):

    OG = kwargs.get("OG", None)
    fs = kwargs.get("FS", None)
    tag = kwargs.get("tag", "")
    
    if not fs:
        raise RuntimeError("no fs")
    
    nclusters = ak.num(data.opticalGroupId, axis=1)
    nclusters_m1p1 = ak.to_numpy(2*nclusters/np.max(nclusters) - 1)
    
    logger.warning(f"  Check carefully ==> fs is {fs} Hz")
    f, t, Sxx = spectrogram(
        nclusters_m1p1,
        fs=fs,
        window='hann',
        nperseg=10240,
        noverlap=int(0.85*10240),
        scaling='density',
        mode='magnitude'
    )

    plot_colormesh(x = t,
                   y = f,
                   z = Sxx,
                   shading='gouraud',
                   xlabel = "time (s)",
                   ylabel = "frquency (Hz)",
                   title  = f"hidden_noise_freq_OG_{OG}_{tag}" if OG is not None else "hidden_noise_freq_incl_{tag}",
                   name   = f"hidden_noise_freq_with_time_OG_{OG}_{tag}"if OG is not None else  "hidden_noise_freq_with_time_incl_{tag}",
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
               title     = f"hidden_noise_freq_incl_time_OG_{OG}_{tag}" if OG is not None else "hidden_noise_freq_incl_time_{tag}",
               name      = f"hidden_noise_freq_incl_time_OG_{OG}_{tag}" if OG is not None else "hidden_noise_freq_incl_time_{tag}",
               xlabel    = f"freq (Hz)",
               ylabel    = "amplitude (arbitrary)",
               outdir    = output,
               linewidth = 1.5,
               xlim      = [0, 14000],
               fit       = False)
    plot_basic(x         = f, #[i for i in range(Sxx_sum_time.shape[0])],
               data_dict = {"raw": Sxx_sum, "subtract_white_noise": Sxx_clean_incl},
               title     = f"hidden_noise_freq_incl_time_OG_{OG}_{tag}" if OG is not None else "hidden_noise_freq_incl_time_{tag}",
               name      = f"hidden_noise_freq_incl_time_zoom_OG_{OG}_{tag}" if OG is not None else "hidden_noise_freq_incl_time_{tag}",
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
    



def main(args):
    real_start = time.perf_counter()
    cpu_start  = time.process_time()
    
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
    str_ = None if args.stop is None else 0

    logger.warning(f"Ev. Start: {str_}")
    logger.warning(f"Ev. Stop : {args.stop}")
    
    events = processor(infile, "Events", start=str_, stop=args.stop)


    

    is_ladder = config.get('IS_LADDER')
    logger.info(f"Data from Ladder? : {is_ladder}")
    
    OGs = None

    if is_ladder:
        OGs = list(range(12))
    else:
        OGs = [int(np.sort(np.unique(ak.to_numpy(ak.flatten(events.cluster.opticalGroupId))))[-1])]

    logger.info(f"Optical Groups : {OGs}")
        
    #hb0_id = 2*OG
    #hb1_id = 2*OG + 1
    
    # =================================== #
    # ==>>> nClusters at different levels #
    # =================================== #

    total_nClusters = float(ak.sum(ak.num(events.cluster.opticalGroupId, axis=1)))
    get_ncls = lambda mask: ak.to_numpy(ak.sum(mask, axis=1))



    
    nClusters_per_OG = {
        "All_OptGroups": [],
    }
    nClusters_per_OG_HB = {
        "HB0": [],
        "HB1": [],
    }
    nClusters_per_OG_Sensor = {
        "BottomS":[], "TopS":[]
    }


    
    # ===>> nClusters per OG per Hybrid per Sensor
    for OG in OGs:
        logger.info(f"OG : {OG}")
        
        hb0_id = 2*OG
        hb1_id = 1 + 2*OG

        logger.info(f"  Hybrid_IDs : {hb0_id} & {hb1_id}")

        logger.info("  Plotting number of clusters for each CBCs per hybrid and sensor")

        cls_opt = events.cluster[events.cluster.opticalGroupId == OG]


        nClusters_per_OG["All_OptGroups"].append(ak.sum(cls_opt.opticalGroupId == OG))
        nClusters_per_OG_HB["HB0"].append(ak.sum(cls_opt.hybridId == hb0_id))
        nClusters_per_OG_HB["HB1"].append(ak.sum(cls_opt.hybridId == hb1_id))
        nClusters_per_OG_Sensor["BottomS"].append(ak.sum(cls_opt.fromWhichSensor == 0))
        nClusters_per_OG_Sensor["TopS"].append(ak.sum(cls_opt.fromWhichSensor == 1))


        
        nClusters_per_cbc = {
            f"OG_{OG}": {
                f"hybrid{hb0_id}_bottom": [],
                f"hybrid{hb0_id}_top"   : [],
                f"hybrid{hb1_id}_bottom": [],
                f"hybrid{hb1_id}_top"   : [],
            },
        }

        for ihb in [hb0_id, hb1_id]:
            cls_hb = cls_opt[cls_opt.hybridId == ihb]
            for isen_key, isen_val in {'bottom': 0, 'top': 1}.items():
                cls_sen = cls_hb[cls_hb.fromWhichSensor == isen_val]
                for icbc in range(8):
                    cls_cbc = cls_sen[cls_sen.chipId == icbc]
                    ncls_cbc = ak.sum(ak.num(cls_cbc.opticalGroupId, axis=1))
                    nClusters_per_cbc[f'OG_{OG}'][f'hybrid{ihb}_{isen_key}'].append([float(ncls_cbc), 0.0])

        plot_basic(x         = [i for i in range(8)],
                   data_dict = nClusters_per_cbc[f'OG_{OG}'],
                   title     = f"nClusters_per_CBC_OG_{OG}",
                   name      = f"nClusters_per_CBC_OG_{OG}",
                   xlabel    = "",
                   ylabel    = "nClusters",
                   xticklabels = [f'CBC_{i}' for i in range(8)],
                   outdir    = output,
                   linewidth = 1.5,
                   #ylim      = [0, 14000.0],
                   markersize = 5.0,
                   fit       = False,
                   legs      = [f'hb{hb0_id}-bottom', f'hb{hb0_id}-top',
                                f'hb{hb1_id}-bottom', f'hb{hb1_id}-top'])

        logger.info("  Plotting rate of clusters for each CBCs per hybrid and sensor")        
        
        plot_basic(x         = [i for i in range(8)],
                   data_dict = nClusters_per_cbc[f'OG_{OG}'],
                   title     = f"rateClusters_per_CBC_OG_{OG}",
                   name      = f"rateClusters_per_CBC_OG_{OG}",
                   xlabel    = "",
                   ylabel    = "Clusters rate",
                   xticklabels = [f'CBC_{i}' for i in range(8)],
                   outdir    = output,
                   linewidth = 1.5,
                   #ylim      = [0, 0.5],
                   markersize = 5.0,
                   fit       = False,
                   legs      = [f'hb{hb0_id}-bottom', f'hb{hb0_id}-top',
                                f'hb{hb1_id}-bottom', f'hb{hb1_id}-top'],
                   density   = True,
                   normfactor= total_nClusters)



        logger.info("  Ploting 1D hist of number of clusters per hybrid")
    
        ncls_dict_hbs = {
            f"OG_{OG}": {
                f"hybrid{hb0_id}": get_ncls(cls_opt.hybridId == hb0_id),
                f"hybrid{hb1_id}": get_ncls(cls_opt.hybridId == hb1_id),
            }
        }
        hist_basic(bins      = [0., 1000.0, 1001], # [begin, end, nbins]
                   data_dict = ncls_dict_hbs[f'OG_{OG}'],
                   title     = f"nClusters_per_hb_OG_{OG}",
                   name      = f"nClusters_per_hb_OG_{OG}",
                   xlabel    = "number of clusters",
                   ylabel    = "number of events",
                   outdir    = output,
                   linewidth = 1.5,
                   #ylim      = [0, 0.5],
                   density   = False,
                   logy      = True)

        logger.info("  Ploting 1D hist of number of clusters per sensor")
    
        ncls_dict_sensors = {
            f"OG_{OG}": {
                "bottom": get_ncls(cls_opt.fromWhichSensor == 0),
                "top": get_ncls(cls_opt.fromWhichSensor == 1),
            }
        }
        hist_basic(bins      = [0.,1000.,1001], # [begin, end, nbins]
                   data_dict = ncls_dict_sensors[f'OG_{OG}'],
                   title     = f"nClusters_per_sensor_OG_{OG}",
                   name      = f"nClusters_per_sensor_OG_{OG}",
                   xlabel    = "number of clusters",
                   ylabel    = "number of events",
                   outdir    = output,
                   linewidth = 1.5,
                   #ylim      = [0, 0.5],
                   density   = False,
                   logy      = True)
    

        logger.info("  Now Hits --> Plotting time evolution of number of hits per CBC")
        
        # ========================================================= #
        # ==>>>           Get time evolution wrt nHits              #
        # ========================================================= #
        hits_opt = events.hit[events.hit.opticalGroupId == OG]
        
        n_events = len(events)
        step = int(0.04*n_events)
        boundaries = [min(i, n_events) for i in range(0, n_events + step, step)]
        pairs = list(zip(boundaries[:-1], boundaries[1:]))
        hits_per_cbc_with_time = {
            f"OG_{OG}": {
                f"hb_{hb0_id}": {f'CBC_{i}' : [] for i in range(8)},
                f"hb_{hb1_id}": {f'CBC_{i}' : [] for i in range(8)}
            }
        }
        for pair in pairs:
            hits_pair = hits_opt[pair[0]:pair[1], :]
            for ihb in [hb0_id, hb1_id]:
                hits_hb = hits_pair[hits_pair.hybridId == ihb]
                for icbc in range(8):
                    hits_cbc = hits_hb[hits_hb.chipId == icbc]
                    nHits = ak.to_numpy(ak.num(hits_cbc.opticalGroupId, axis=1))

                    hits_per_cbc_with_time[f'OG_{OG}'][f'hb_{ihb}'][f'CBC_{icbc}'].append(
                        [float(np.mean(nHits)),
                         float(np.std(nHits)/np.sqrt(pair[1]-pair[0]))]
                    )

        logger.info("    For hybrid-{hb0_id}")

        plot_basic(x         = [i for i in range(len(pairs))],
                   data_dict = hits_per_cbc_with_time[f'OG_{OG}'][f'hb_{hb0_id}'],
                   title     = f"nHits_with_time_hb0_OG_{OG}",
                   name      = f"nHits_with_time_hb0_OG_{OG}",
                   xlabel    = f"time stamps (every {step} events)",
                   ylabel    = "nHits",
                   outdir    = output,
                   linewidth = 1.2,
                   #ylim      = [0, 0.0045],
                   fit       = False,
                   fitmodel  = 'poly2')
    
        logger.info("    For hybrid-{hb1_id}")
        
        plot_basic(x         = [i for i in range(len(pairs))],
                   data_dict = hits_per_cbc_with_time[f'OG_{OG}'][f'hb_{hb1_id}'],
                   title     = f"nHits_with_time_hb1_OG_{OG}",
                   name      = f"nHits_with_time_hb1_OG_{OG}",
                   xlabel    = f"time stamps (every {step} events)",
                   ylabel    = "nHits",
                   outdir    = output,
                   linewidth = 1.2,
                   #ylim      = [0, 0.0045],
                   fit       = False,
                   fitmodel  = 'poly2')


        #ncls_med = np.median(ak.to_numpy(ak.num(cls_opt.hybridId, axis=1)))
        #cls_per_event_wt = ak.ones_like(cls_opt.hybridId) #/float(ncls)
        
        flipX = False
        if (OG == 0) or (OG == 11):
            print("Flip X")
            flipX = True
        

        logger.info("  Plotting cluster-position vs. hybridId : Bottom Sensor")

        mask_bottom   = cls_opt.fromWhichSensor == 0
        cls_bottom    = cls_opt[mask_bottom]
        _pos_bottom   = ak.to_numpy(ak.flatten(cls_bottom.address)).astype(np.int64)
        hbid_bottom   = ak.to_numpy(ak.flatten(cls_bottom.hybridId % hb0_id))
        chipid_bottom = ak.to_numpy(ak.flatten(cls_bottom.chipId)).astype(np.int64)
        pos_bottom    = _pos_bottom + chipid_bottom*254

        #wt = ak.to_numpy(ak.flatten(cls_per_event_wt[mask_bottom])).astype(np.float64)
                
        hist_2D(xbins = [-0.5, 0.5, 1.5],
                ybins = np.arange(-0.5, 2032.5),
                xdata = hbid_bottom,
                ydata = pos_bottom,
                title = f"clspos_hbid_map_bottom_OG_{OG}",
                name  = f"clspos_hbid_map_bottom_OG_{OG}",
                xlabel = 'hybrids',
                ylabel = 'cluster position',
                outdir    = output,
                setYlabels='Strip',
                flipX = flipX,
                #vmin = 0, vmax = 800,
                #wt=wt
                )

        logger.info("  Plotting chipId vs. hybridId : Bottom Sensor")

        hist_2D(xbins = [-0.5, 0.5, 1.5],
                ybins = np.arange(-0.5, 8.5),
                xdata = hbid_bottom,
                ydata = chipid_bottom,
                title = f"chipid_hbid_map_bottom_OG_{OG}",
                name = f"chipid_hbid_map_bottom_OG_{OG}",
                xlabel = 'hybrids',
                ylabel = 'chip-id',
                outdir    = output,
                setYlabels='CBC',
                flipX = flipX,
                #vmin = 0, vmax = 80000,
                #wt=wt
                )
       


        mask_top = events.cluster.fromWhichSensor == 1
        cls_top = events.cluster[mask_top]
        _pos_top = ak.to_numpy(ak.flatten(cls_top.address)).astype(np.int64)
        hbid_top = ak.to_numpy(ak.flatten(cls_top.hybridId % hb0_id))
        chipid_top = ak.to_numpy(ak.flatten(cls_top.chipId)).astype(np.int64)
        pos_top = _pos_top + chipid_top*254

        #wt = ak.to_numpy(ak.flatten(cls_per_event_wt[mask_top])).astype(np.float64)
        
        logger.info("  Plotting cluster-position vs. hybridId : Top Sensor")
        
        hist_2D(xbins = [-0.5, 0.5, 1.5],
                ybins = np.arange(-0.5, 2032.5),
                xdata = hbid_top,
                ydata = pos_top,
                title = f"clspos_hbid_map_top_OG_{OG}",
                name = f"clspos_hbid_map_top_OG_{OG}",
                xlabel = 'hybrid-id',
                ylabel = 'cluster position',
                outdir    = output,
                setYlabels='Strip',
                flipX = flipX,
                #vmin = 0, vmax = 800,
                #wt=wt
                )

        logger.info("  Plotting chipId vs. hybridId : Top Sensor")

        hist_2D(xbins = [-0.5, 0.5, 1.5],
                ybins = np.arange(-0.5, 8.5),
                xdata = hbid_top,
                ydata = chipid_top,
                title = f"chipid_hbid_map_top_OG_{OG}",
                name = f"chipid_hbid_map_top_OG_{OG}",
                xlabel = 'hybrid-id',
                ylabel = 'chip-id',
                outdir    = output,
                setYlabels='CBC',
                flipX = flipX,
                #vmin = 0, vmax = 80000,
                #wt=wt
                )



        logger.info("  Plottong cluster position per CBC per hybrid")

        
        get_cls_pos = lambda cls: ak.to_numpy(ak.flatten(cls.address))
        cls_pos_og = {
            f'Hybrid_0': {f'CBC_{icbc}':{} for icbc in range(8)},
            f'Hybrid_1': {f'CBC_{icbc}':{} for icbc in range(8)},
        }
        for hb_key, hb_val in {'Hybrid_0': hb0_id, 'Hybrid_1': hb1_id}.items():
            cls_hb = cls_opt[cls_opt.hybridId == hb_val]
            for icbc in range(8):
                cls_cbc = cls_hb[cls_hb.chipId == icbc]
                cls_pos = {
                    "bottom": get_cls_pos(cls_cbc[(cls_cbc.fromWhichSensor == 0)]),
                    "top": get_cls_pos(cls_cbc[(cls_cbc.fromWhichSensor == 1)]),
                }
                cls_pos_og[hb_key][f'CBC_{icbc}'] = cls_pos

        #from IPython import embed; embed()
        hist_cbc_group(cls_pos_og['Hybrid_0'],
                       outdir = output,
                       name = f'hist_clspos_per_cbc_hb0_OG_{OG}',
                       title = f'cluster position (Hb0) OG_{OG}',
                       xlabel = "cluster position (Strip)",
                       ylabel = "no. of clusters",
                       forpos = True)
        hist_cbc_group(cls_pos_og['Hybrid_1'],
                       outdir = output,
                       name = f'hist_clspos_per_cbc_hb1_OG_{OG}',
                       title = f'cluster position (Hb1) OG_{OG}',
                       xlabel = "cluster position (Strip)",
                       ylabel = "no. of clusters",
                       forpos = True)

        
        logger.info("  Plottong hit position per CBC per hybrid")

        hits_pos_og = {
            f'Hybrid_0': {f'CBC_{icbc}':{} for icbc in range(8)},
            f'Hybrid_1': {f'CBC_{icbc}':{} for icbc in range(8)},
        }
        for hb_key, hb_val in {'Hybrid_0': hb0_id, 'Hybrid_1': hb1_id}.items():
            hits_hb = hits_opt[hits_opt.hybridId == hb_val]
            for icbc in range(8):
                hits_cbc = hits_hb[hits_hb.chipId == icbc]
                hits_pos = {
                    "bottom": get_cls_pos(hits_cbc[(hits_cbc.fromWhichSensor == 0)]),
                    "top": get_cls_pos(hits_cbc[(hits_cbc.fromWhichSensor == 1)]),
                }
                hits_pos_og[hb_key][f'CBC_{icbc}'] = hits_pos

        #from IPython import embed; embed()
        hist_cbc_group(hits_pos_og['Hybrid_0'],
                       outdir = output,
                       name = f'hist_hitpos_per_cbc_hb0_OG_{OG}',
                       title = f'hit position (Hb0) OG_{OG}',
                       xlabel = "hit position (Strip)",
                       ylabel = "no. of hits",
                       forpos=True)
        hist_cbc_group(hits_pos_og['Hybrid_1'],
                       outdir = output,
                       name = f'hist_hitpos_per_cbc_hb1_OG_{OG}',
                       title = f'hit position (Hb1) OG_{OG}',
                       xlabel = "hit position (Strip)",
                       ylabel = "no. of hits",
                       forpos=True)


        nhits_og = {
            f'Hybrid_0': {f'CBC_{icbc}':{} for icbc in range(8)},
            f'Hybrid_1': {f'CBC_{icbc}':{} for icbc in range(8)},
        }
        for hb_key, hb_val in {'Hybrid_0': hb0_id, 'Hybrid_1': hb1_id}.items():
            hits_hb = hits_opt[hits_opt.hybridId == hb_val]
            for icbc in range(8):
                hits_cbc = hits_hb[hits_hb.chipId == icbc]
                hits_pos = {
                    "bottom": get_ncls(hits_cbc.fromWhichSensor == 0),
                    "top": get_ncls(hits_cbc.fromWhichSensor == 1),
                }
                nhits_og[hb_key][f'CBC_{icbc}'] = hits_pos

        #from IPython import embed; embed()
        hist_cbc_group(nhits_og['Hybrid_0'],
                       outdir = output,
                       name = f'hist_nHits_per_cbc_hb0_OG_{OG}',
                       title = f'nHits (Hb0) OG_{OG}',
                       xlabel = "no. of hits (Strip)",
                       ylabel = "no. of events",
                       linewidth = 1.2,
                       forpos=True,
                       logy = True,
                       gap=127)
        hist_cbc_group(nhits_og['Hybrid_1'],
                       outdir = output,
                       name = f'hist_nHits_per_cbc_hb1_OG_{OG}',
                       title = f'nHits (Hb1) OG_{OG}',
                       xlabel = "no. of hits (Strip)",
                       ylabel = "no. of events",
                       linewidth = 1.2,
                       forpos=True,
                       logy = True,
                       gap=127)


        ncls_og = {
            f'Hybrid_0': {f'CBC_{icbc}':{} for icbc in range(8)},
            f'Hybrid_1': {f'CBC_{icbc}':{} for icbc in range(8)},
        }
        for hb_key, hb_val in {'Hybrid_0': hb0_id, 'Hybrid_1': hb1_id}.items():
            cls_hb = cls_opt[cls_opt.hybridId == hb_val]
            for icbc in range(8):
                cls_cbc = cls_hb[cls_hb.chipId == icbc]
                cls_pos = {
                    "bottom": get_ncls(cls_cbc.fromWhichSensor == 0),
                    "top": get_ncls(cls_cbc.fromWhichSensor == 1),
                }
                ncls_og[hb_key][f'CBC_{icbc}'] = cls_pos

        #from IPython import embed; embed()
        hist_cbc_group(ncls_og['Hybrid_0'],
                       outdir = output,
                       name = f'hist_nClusters_per_cbc_hb0_OG_{OG}',
                       title = f'nClusters (Hb0) OG_{OG}',
                       xlabel = "no. of clusters (Strip)",
                       ylabel = "no. of events",
                       linewidth = 1.2,
                       forpos=True,
                       logy = True,
                       gap=64)
        hist_cbc_group(ncls_og['Hybrid_1'],
                       outdir = output,
                       name = f'hist_nClusters_per_cbc_hb1_OG_{OG}',
                       title = f'nClusters (Hb1) OG_{OG}',
                       xlabel = "no. of clusters (Strip)",
                       ylabel = "no. of events",
                       linewidth = 1.2,
                       forpos=True,
                       logy = True,
                       gap=64)
        
        

        # ========================================================= #
        # ==>>> Extract hidden frequencies from Clusters            #
        # ========================================================= #
        get_hidden_freq(data = cls_opt,  output = output, OG = OG, FS = config.get("FS"), tag = "with_clusters")
        get_hidden_freq(data = hits_opt,  output = output, OG = OG, FS = config.get("FS"), tag = "with_hits")
        

        logger.info("  Plotting correlation among different CBCs per module")

        
        # ====================================================== #
        # ==>>> Keep only those events with at least one cluster #
        # ====================================================== #
        no_hit_mask = ak.num(hits_opt.opticalGroupId, axis=1) == 0
        sel_hits_opt = hits_opt[~no_hit_mask]

        # hitss from hb0
        hits_hb0 = sel_hits_opt[sel_hits_opt.hybridId == hb0_id]
        # hits from hb1
        hits_hb1 = sel_hits_opt[sel_hits_opt.hybridId == hb1_id]

        # ==>>> pearson correl coeff among all channels
        corr = get_correlation_chip(hits_hb0, hits_hb1)
        plot_heatmap(data = corr,
                     title = f"Pearson Corr-Coeff / Module (OG_{OG})",
                     name = f"corr_coeff_OG_{OG}",
                     xticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                     yticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                     outdir = output,
                     vmin = -1, vmax = 1,
                     colmap = 'Spectral',
                     cbar_label = "Correlation Coefficient (-1 to +1)")

        # hits hb0 from top/bot sensor
        hits_hb0_top = hits_hb0[hits_hb0.fromWhichSensor == 1]
        hits_hb0_bot = hits_hb0[hits_hb0.fromWhichSensor == 0]
        # hits hb1 from top/bot sensor
        hits_hb1_top = hits_hb1[hits_hb1.fromWhichSensor == 1]
        hits_hb1_bot = hits_hb1[hits_hb1.fromWhichSensor == 0]

        # ==>>> pearson correl for channels from top sensor
        corr_top = get_correlation_chip(hits_hb0_top, hits_hb1_top)
        plot_heatmap(data = corr_top,
                     title = f"Pearson Corr-Coeff / Module (TopS) (OG_{OG})",
                     name = f"corr_coeff_top_OG_{OG}",
                     xticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                     yticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                     outdir = output,
                     vmin = -1, vmax = 1,
                     colmap	= 'Spectral',
                     cbar_label = "Correlation Coefficient (-1 to +1)")
        # ==>>> pearson correl for channels from bottom sensor
        corr_bot = get_correlation_chip(hits_hb0_bot, hits_hb1_bot)
        plot_heatmap(data = corr_bot,
                     title = f"Pearson Corr-Coeff / Module (BotS) OG_{OG}",
                     name = f"corr_coeff_bot_OG_{OG}",
                     xticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                     yticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                     outdir = output,
                     vmin = -1, vmax = 1,
                     colmap	= 'Spectral',
                     cbar_label = "Correlation Coefficient (-1 to +1)")
        

        # =========>>> Loop over OGs ends here

    if is_ladder:

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

        
        
        get_hidden_freq(data = events.hit,  output = output, FS = config.get("FS"))
        
        
        # Module level correlation
        no_hit_mask = ak.num(events.hit.opticalGroupId, axis=1) == 0
        hits = events.hit[~no_hit_mask]
        
        
        # ==>>> pearson correl coeff among all channels
        corr_og, corr_hb = get_correlation(hits)
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
        hits_top = hits[hits.fromWhichSensor == 1]
        hits_bot = hits[hits.fromWhichSensor == 0]
        
        # ==>>> pearson correl for channels from top sensor
        corr_top_og, corr_top_hb = get_correlation(hits_top)
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
        corr_bot_og, corr_bot_hb = get_correlation(hits_bot)
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

    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=True,
                        help="yaml configs to be used")

    parser.add_argument('-t',
                        '--tag',
                        type=str,
                        required=False,
                        default="",
                        help="<output_dir>_<tag>")

    parser.add_argument('-s',
                        '--stop',
                        type=int,
                        required=False,
                        default=None,
                        help="<output_dir>_<tag>")

    
    
    args = parser.parse_args()
        
    main(args)
