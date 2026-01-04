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

from otutil import setup_logger, processor
from otutil import plot_basic, plot_heatmap


def get_correlation(cls_hb0, cls_hb1):
    cls_hb0_chipId = cls_hb0.chipId
    cls_hb1_chipId = cls_hb1.chipId + 8
    cls_chipId = ak.concatenate([cls_hb0_chipId, cls_hb1_chipId], axis=1)
    cls_chipId_encoded = np.vstack([
        np.bincount(x, minlength=16) for x in cls_chipId
    ])
    corr = ak.to_numpy(np.corrcoef(cls_chipId_encoded, rowvar=False))
    return corr


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

    output = f"{config.get('OUTPUT')}__{args.tag}"
    if os.path.exists(output):
        logger.info(f"Output dir {output} found")
    else:
        logger.info(f"Creating output dir {output}")
        os.mkdir(output)

    
    # ==>>> Process Events
    events = processor(infile, "Events")

    # ==>>> nClusters at different levels
    nClusters_per_cbc = {
        "opt_4": {
            "hb_8": {
                "s0": [],
                "s1": []
            },
            "hb_9": {
                "s0": [],
                "s1": []
            },
        }
    }
    for iopt in range(4,5):
        cls_opt = events.cluster[events.cluster.opticalGroupId == iopt]
        hb = [2*iopt, 2*iopt+1]
        for ihb in range(hb[0],1+hb[1]):
            cls_hb = cls_opt[cls_opt.hybridId == ihb]
            for isen in range(2):
                cls_sen = cls_hb[cls_hb.fromWhichSensor == isen]
                for icbc in range(8):
                    cls_cbc = cls_sen[cls_sen.chipId == icbc]
                    ncls_cbc = ak.sum(ak.num(cls_cbc.opticalGroupId, axis=1))
                    nClusters_per_cbc[f'opt_{iopt}'][f'hb_{ihb}'][f's{isen}'].append([float(ncls_cbc), 0.0])

    plot_basic(x         = [i for i in range(8)],
               data_dict = nClusters_per_cbc['opt_4']['hb_8'],
               title     = f"nClusters_hb0",
               name      = f"nClusters_hb0",
               xlabel    = "",
               ylabel    = "nClusters",
               xticklabels = [f'CBC_{i}' for i in range(8)],
               outdir    = output,
               linewidth = 1.5,
               ylim      = [0, 8000.0],
               markersize = 5.0,
               fit       = False,
               legs      = ['bottom sensor', 'top sensor'])
    plot_basic(x         = [i for i in range(8)],
               data_dict = nClusters_per_cbc['opt_4']['hb_9'],
               title     = f"nClusters_hb1",
               name      = f"nClusters_hb1",
               xlabel    = "",
               ylabel    = "nClusters",
               xticklabels = [f'CBC_{i}' for i in range(8)],
               outdir    = output,
               linewidth = 1.5,
               ylim      = [0, 14000.0],
               markersize = 5.0,
               fit       = False,
               legs      = ['bottom sensor', 'top sensor'])
    
    
    
    
    # ==>>> Get time evolution wrt nHits i.e. sum(clusterWidth)
    n_events = len(events)
    step = 200000
    boundaries = [min(i, n_events) for i in range(0, n_events + step, step)]
    pairs = list(zip(boundaries[:-1], boundaries[1:]))
    cluster_per_cbc_with_time = {
        "opt_4": {
            "hb_8": {f'CBC_{i}' : [] for i in range(8)},
            "hb_9": {f'CBC_{i}' : [] for i in range(8)}
        }
    }
    for pair in pairs:
        cls = events.cluster[pair[0]:pair[1], :]
        for iopt in range(4,5):
            cls_opt = cls[cls.opticalGroupId == iopt]
            hb = [2*iopt, 2*iopt+1]
            for ihb in range(hb[0],1+hb[1]):
                cls_hb = cls_opt[cls_opt.hybridId == ihb]
                for icbc in range(8):
                    cls_cbc = cls_hb[cls_hb.chipId == icbc]
                    nHits = ak.to_numpy(ak.sum(cls_cbc.width, axis=1))

                    cluster_per_cbc_with_time[f'opt_{iopt}'][f'hb_{ihb}'][f'CBC_{icbc}'].append(
                        [float(np.mean(nHits)),
                         float(np.std(nHits)/np.sqrt(pair[1]-pair[0]))]
                    )

    plot_basic(x         = [i for i in range(len(pairs))],
               data_dict = cluster_per_cbc_with_time['opt_4']['hb_8'],
               title     = f"nHits_with_time_hb0",
               name      = f"nHits_with_time_hb0",
               xlabel    = f"time stamps (every {step} events)",
               ylabel    = "nHits",
               outdir    = output,
               linewidth = 0.0,
               ylim      = [0, 0.0045],
               fit       = True,
               fitmodel  = 'poly2')
    plot_basic(x         = [i for i in range(len(pairs))],
               data_dict = cluster_per_cbc_with_time['opt_4']['hb_9'],
               title     = f"nHits_with_time_hb1",
               name      = f"nHits_with_time_hb1",
               xlabel    = f"time stamps (every {step} events)",
               ylabel    = "nHits",
               outdir    = output,
               linewidth = 0.0,
               ylim      = [0, 0.0045],
               fit       = True,
               fitmodel  = 'poly2')
    



    
    # ==>>> Keep only those events with at least one cluster
    no_cluster_mask = ak.num(events.cluster.opticalGroupId, axis=1) == 0
    events = events[~no_cluster_mask]

    # clusters from hb0
    cls_hb0 = events.cluster[events.cluster.hybridId == 8]
    # clusters from hb1
    cls_hb1 = events.cluster[events.cluster.hybridId == 9]

    # ==>>> pearson correl coeff among all channels
    corr = get_correlation(cls_hb0, cls_hb1)    
    plot_heatmap(data = corr,
                 title = "Pearson Corr-Coeff / Module",
                 name = "corr_coeff",
                 xticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                 yticklabels = [f'hb0-cbc{i}' for i in range(8)] + [f'hb1-cbc{i}' for i in range(8)],
                 outdir = output,
                 vmin = -1, vmax = 1,
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
                 cbar_label = "Correlation Coefficient (-1 to +1)")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotter')
    parser.add_argument('-c', '--config', type=str, required=True, help="yaml configs to be used")
    parser.add_argument('-t', '--tag', type=str, required=False, default="v1", help="<output_dir>_<tag>")
   
    args = parser.parse_args()
        
    main(args)
