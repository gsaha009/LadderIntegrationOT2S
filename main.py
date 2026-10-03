import os
import re
import sys
import time
import copy
import yaml
import json
import logging
import argparse
import datetime
import numpy as np
import pandas as pd
from glob import glob
from argparse import Namespace

from DataLoader import DataLoader
from PlotterUser import PlotterUser
from QualityCheck import QualityCheck
import sparseLadderRootFile


def extractModIDfromIV(fname):
    pattern = re.compile(
        r"^(?P<moduleid>.+)_HV#(?P<channelno>\d+)\.csv$"
    )
    match = pattern.fullmatch(fname)
    if match is None:
        raise RuntimeWarning(f'{fname} not parsed')
    module = match.group("moduleid")
    channel = int(match.group("channelno"))
    return module,channel
    

def deep_merge(a, b):
    for key, value in b.items():
        if key in a and isinstance(a[key], dict) and isinstance(value, dict):
            deep_merge(a[key], value)
        else:
            a[key] = value
    return a


class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[1;32m',    # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.msg = f"{color}{record.msg}{self.RESET}"

        fmt = "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
        formatter = logging.Formatter(fmt, '%Y-%m-%d:%H:%M:%S')
        return formatter.format(record)

def setup_logger(logf=""):
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Reset handlers to avoid duplicates on repeated calls
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Print in the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    # Save in a file
    if logf:
        file_handler = logging.FileHandler(
            logf, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
        ))
        logger.addHandler(file_handler)

    return logger
    

def looksLikePotatoConvertedFile(fname):
    pattern = re.compile(
        r"^(?P<module>.+?)_"
        r"(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}h\d{2}m\d{2}s)_"
        r"(?P<temperature>[+-]?\d+(?:\.\d+)?)C_"
        r"(?P<test>.+?)_"
        r"(?P<version>v[^.]+)"
        r"\.root$"
    )
    match = pattern.match(fname)
    if not match:
        return False
    else:
        module = match.group("module")
        timestamp = match.group("timestamp")
        temperature = match.group("temperature")
        test = match.group("test")
        version = match.group("version")
        #print(f'ModuleID: {module}, Temp: {temperature}C, Calibration: {test}, Ph2ACF version: {version}, timestamp: {timestamp}')

        return True

    
def main():
    
    real_start = time.perf_counter()
    cpu_start  = time.process_time()    
    dttag = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    parser = argparse.ArgumentParser(description='Plotter')


    # arguments for analysis / splitting root file
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        default="",
                        help="Ph2ACF Results Folder, please maintain the naming convention like this : <>")
    parser.add_argument('-ivtrx',
                        '--inputvtrx',
                        type=str,
                        required=True,
                        default="",
                        help="Ph2ACF Results Folder with PS monitor/IV, please maintain the naming convention like this : <>")
    #parser.add_argument('-b',
    #                    '--basedir',
    #                    type=str,
    #                    required=False,
    #                    default="/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs",
    #                    help="to find the Ph2ACF output dir")
    parser.add_argument("-nopreint",
                        "--nocomparisonwithpreint",
                        action = "store_true",
                        default = False,
                        help="By default, always compare same modules, between pre and post int")    
    parser.add_argument('-pft',
                        '--preintfiletype',
                        type=str,
                        required=False,
                        default="mod_final",
                        help="It could be mod_final (default) / burnin_cold / burin_warm")
    parser.add_argument('-d',
                        '--dump',
                        action='store_true',
                        default=False,
                        required=False,
                        help="save data in yaml before plotting?")
    parser.add_argument('-pext',
                        '--plotextension',
                        type=str,
                        default="png",
                        required=False,
                        help="plots in png or pdf?")
    parser.add_argument('-pdpi',
                        '--plotdpi',
                        type=int,
                        default=300,
                        required=False,
                        help="Plot DPI")
    parser.add_argument("-comparediffmods",
                        "--comparediffmods",
                        action      = "store_true",
                        default     = False,
                        help="By default, always compare same modules, between pre and post int")
    parser.add_argument("-fcmn",
                        "--fitcmnoise",
                        action      = "store_true",
                        default     = False,
                        help="Fit nHits with pure gaussian")
    parser.add_argument("-fsimcmn",
                        "--fitsimulcmnoise",
                        action      = "store_true",
                        default     = False,
                        help="Fit nHits simulatenously")
    parser.add_argument("-s",
                        "--split",
                        action      = "store_true",
                        default     = False,
                        help="Split the Merged ROOT file to be uploaded DCA")
    parser.add_argument("-og",
                        "--opticalgroup",
                        type=int,
                        nargs="+",
                        default=[-1],
                        required=False,
                        help="list of OGs; use -1 to select all (0–11) for making ROOT files Potato like")    
    parser.add_argument('-qccfg',
                        '--qcconfig',
                        type=str,
                        required=False,
                        default="qc_config.yaml",
                        help="Helper config for quality check from Ladder.root")
    parser.add_argument('-t',
                        '--tag',
                        type=str,
                        required=False,
                        default="",
                        help="extra tag to be added at the end of output folder name, if none, datetime will be added")

    
    
    args = parser.parse_args()


    # New
    # Create two separate configs from the name of the input path i.e. Ph2ACF Output Directory
    # TB2SLadder__<ladderID>__Pos_<ladder_position>__<cooling_temp>__<calibration_type>__<date_time>
    #   ladderID         : e.g. minus_6CP_9
    #   ladder_position  : 1 or 2
    #   cooling_temp     : -30C, 0C or +15C
    #   calibration_type : 2SquickTest / 2SfullTest / 2SvtrxOff
    #   date_time        : 2026-05-01_11h11m34s
    #
    # e.g. the folder name must be
    # TB2SLadder__minus_6CP_9__Pos_1__-30C__2SfullTest__2026-05-01_11h11m34s
    # TB2SLadders__pos_1__minus_6CP_9__pos_2__minus_6CP_8__+15C__2Sfulltest__2026-05-01_11h11m34s

    calibration_output_full_path = args.input
    if not os.path.isdir(calibration_output_full_path):
        raise FileNotFoundError(f"{calibration_output_full_path} not found")
    calibration_output_basedir = os.path.dirname(calibration_output_full_path)
    calibration_output = os.path.basename(calibration_output_full_path)

        
    pattern = re.compile(
        r"^TB2SLadders__"
        r"(?:pos_(?P<pos1>\d+)__"
        r"TB2S_Ladder_(?P<lad1>(?:minus|plus)_\d+CP_\d+)__)?"
        r"(?:pos_(?P<pos2>\d+)__"
        r"TB2S_Ladder_(?P<lad2>(?:minus|plus)_\d+CP_\d+)__)?"
        r"(?P<cooltemp>[+-]?\d+(?:\.\d+)?C)__"
        r"(?P<calibname>.*?)__"
        r"(?P<datetime>.+)"
        r"$"
    )
    
    m = pattern.match(calibration_output)

    ladder_1_pos = m['pos1']
    ladder_1_id  = m['lad1']
    ladder_2_pos = m['pos2']
    ladder_2_id  = m['lad2']
    cooltemp     = m['cooltemp']
    calibname    = m['calibname']
    calibdt      = m['datetime']

    outdir_name = f"TB2SLadders__pos_{ladder_1_pos}__TB2S_Ladder_{ladder_1_id}__pos_{ladder_2_pos}__TB2S_Ladder_{ladder_2_id}__{cooltemp}__{calibname}Analysis__{dttag}"
    outdir_main = os.path.join(calibration_output_basedir, outdir_name)
    if not os.path.isdir(outdir_main):
        os.mkdir(outdir_main)

    logger = setup_logger(logf=f'{outdir_main}/{outdir_name}.log')
    logger.info(f"Analysis date-time: {dttag}")

        
    #config_file_1 = "Configs/tests.yaml"
    #config_1 = None
    #with open(config_file_1,'r') as conf1:
    #    config_1 = yaml.safe_load(conf1)

    # to control switches
    config_1 = {
        'check_pedestal': True,
        'check_common_noise': True if 'full' in calibname else False,
        'fit_common_noise': args.fitcmnoise,
        'fit_simultaneous_common_noise': args.fitsimulcmnoise,
        'check_extra': True,
        'check_sensor_temperature': True,
        'compare_two_setup': True,
        'are_same_modules': not args.comparediffmods,
        'plot_extn': args.plotextension,
        'plot_dpi': args.plotdpi
    }

    
    config_file_hnames = "hist_names.yaml"
    config_hnames = None
    with open(config_file_hnames,'r') as confh:
        config_hnames = yaml.safe_load(confh)


    # fetch vtrxoff output
    vtrxoff_path = args.inputvtrx
    if not os.path.isdir(vtrxoff_path):
        raise FileNotFoundError(f"{vtrxoff_path} not found !")

    # monitor power-supply
    monitorPS = os.path.join(vtrxoff_path, 'MonitorPS.csv')
    if not os.path.exists(monitorPS):
        raise FileNotFoundError(f"MonitorPS.csv not found in {vtrxoff_path}")

    # check IVs
    IVs = os.path.join(vtrxoff_path, 'IV/iv_curves')
    if not os.path.isdir(IVs):
        raise FileNotFoundError(f"IV dir not found in {vtrxoff_path}")
    nIVfiles = len(os.listdir(IVs))
    if nIVfiles % 12 != 0:
        raise Exception(f"12 or 24 csv files must be inside IV")
    else:
        logger.info(f'{nIVfiles} csv files are inside IV')

    # Build 2 on-the-fly configs here
    # Use the info above
    # also, the moduleIDs and OG_index from JSON
    # and, for pre-int config, use PreIntResults

    ladder_dirs = []

    print(ladder_1_id, ladder_2_id)
    if ladder_1_pos:
        ladder_dirs.append(f'Pos_{ladder_1_pos}__TB2S_Ladder_{ladder_1_id}')
    else:
        logger.warning("No ladder found on ColdBox position 1")

    if ladder_2_pos:
        ladder_dirs.append(f'Pos_{ladder_2_pos}__TB2S_Ladder_{ladder_2_id}')
    else:
        logger.warning("No ladder found on ColdBox position 2")
        

    if len(ladder_dirs) == 0:
        raise RuntimeError("Both Ladders are missing ... have a bad day !")
    
    temp_name_ext = {"+15C": "ambient", "-30C": "cold"}[cooltemp]
    
    for ladder_dir in ladder_dirs:

        logger.info(f"Ladder : {ladder_dir}")
        ladder_path = os.path.join(calibration_output_full_path, ladder_dir)
        out_ladder_path = os.path.join(outdir_main, ladder_dir)
        if not os.path.isdir(out_ladder_path):
            logger.info(f"Creating {out_ladder_path} ...")
            os.mkdir(out_ladder_path)

        summary_path = os.path.join(out_ladder_path, 'Summary')
        if not os.path.isdir(summary_path):
            logger.info(f"Creating {summary_path} ...")
            os.mkdir(summary_path)
        
        # Create the analysis directory first
        analysis_path = f'{out_ladder_path}/AnalysisResults'
        logger.info(f'Creating AnalysisResults inside {out_ladder_path}')
        if os.path.isdir(analysis_path):
            logger.warning(f'Ah Oh ! AnalysisResults already exists, expected in the 2nd iteration and so on !')
            continue
        else:
            os.mkdir(analysis_path)
        
        lad_info = ladder_dir.split('__')
        ladpos = int(lad_info[0].split('_')[-1])
        ladid  = lad_info[-1].replace('TB2S_Ladder_','')

        # get OpticalGroups
        dataOG = {}
        OGs = list(range(12))
        HVchs = [(i+1) + 12*(ladpos-1) for i in OGs]

        for iOG in OGs:
            HVch = HVchs[iOG]
            for IV in os.listdir(IVs):
                IV_fname = os.path.basename(IV)
                modid,chno = extractModIDfromIV(IV_fname)
                if HVch == chno:
                    dataOG[f'OpticalGroup_{iOG}'] = modid

        logger.info(dataOG)
                    
        vtrxoff_tfiles_dqm = glob(f'{vtrxoff_path}/{ladder_dir}/{temp_name_ext}/MonitorResults/MonitorDQM*.root')
        if len(vtrxoff_tfiles_dqm) == 0:
            raise RuntimeError(f'MonitorDQM.root not found in {vtrxoff_path}/{ladder_dir}/{temp_name_ext}/MonitorResults')
        tfiles_main = glob(f'{ladder_path}/{temp_name_ext}/Results/Run_0/Results*.root')
        if len(tfiles_main) == 0:
            raise RuntimeError(f"no Ph2ACF root file found inside {ladder_path}/{temp_name_ext}/Results")
        tfiles_dqm = glob(f'{ladder_path}/{temp_name_ext}/MonitorResults/MonitorDQM*.root')
        if len(tfiles_dqm) == 0:
            raise RuntimeError(f"no Ph2ACF monitorDQM root file found inside {ladder_path}/{temp_name_ext}/MonitorResults")
        
        config_dict_postInt = {
            'maintag' : f'PostInt__Pos_{ladpos}__{ladid}',
            'ladder'  : True,
            'cooling' : cooltemp,
            'n_boards': 1,
            'opticalgroups': [0,1,2,3,4,5,6,7,8,9,10,11],
            'moduleinfo': {f'board_0_optical_{i}':val for i,(key, val) in enumerate(dataOG.items())},
            'files': {
                'tfile_main': tfiles_main[0],
                'tfile_dqm': tfiles_dqm[0],
            },
            'ps_monitor_file': monitorPS,
            'output': f'{analysis_path}/Results'
        }

        # prepare moduleID and filename dict
        modID_PreIntFileDict = {}
        PreIntFilePath = f'{ladder_path}/PreIntResults'
        if not os.path.isdir(PreIntFilePath):
            raise FileNotFoundError(f"{PreIntFilePath} not found !")
        
        for OG,modID in dataOG.items():
            calib_file_ = ""
            monitor_file_ = ""
            PreIntFilePathMod_ = f'{PreIntFilePath}/{modID}'
            if not os.path.isdir(PreIntFilePathMod_):
                raise FileNotFoundError(f"{modID} folder is found missing in {PreIntFilePath}... Check if the PreInt ROOT files downloaded from DCA !")
            files_ = os.listdir(PreIntFilePathMod_)
            if len(files_) > 1:
                calib_file_ = [file_ for file_ in files_ if file_.startswith('Results')][0]
                monitor_file_ = [file_ for file_ in files_ if file_.startswith('MonitorDQM')][0]
            else:
                preint_file = files_[0]
                if not looksLikePotatoConvertedFile(preint_file):
                    logger.warning(f'{preint_file} does not look like a potato converted file ! this preint comparison can be made optional (tbd)')
                calib_file_ = preint_file
                monitor_file_ = preint_file
            modID_PreIntFileDict[modID] = {
                'og': 0,
                'tfile_main': f'{PreIntFilePathMod_}/{calib_file_}',
                'tfile_dqm': f'{PreIntFilePathMod_}/{monitor_file_}'
            }
            
            
        
        config_dict_preInt = {
            'maintag' : f'PreInt__Pos_{ladpos}__{ladid}',
            'ladder'  : False,
            'cooling' : cooltemp,
            'n_boards': 1,
            'opticalgroups': [0,1,2,3,4,5,6,7,8,9,10,11],
            'files': modID_PreIntFileDict,
            'output': f'{analysis_path}/Results'
        }

        
        # dump these two dicts in yaml format inside analysis_path
        # create a dir configs first

        path_config = f'{analysis_path}/Configs'
        if not os.path.isdir(path_config):
            os.mkdir(path_config)
        else:
            logger.warning(f'configs dir found in {analysis_path}')

        logger.info(f"Writing Pre-int config in {path_config}")
        with open(f'{path_config}/configPreInt.yaml', 'w') as f1:
            yaml.dump(config_dict_preInt, f1, default_flow_style=False, sort_keys=False)
        logger.info(f"Writing Post-int config in {path_config}")
        with open(f'{path_config}/configPostInt.yaml', 'w') as f2:
            yaml.dump(config_dict_postInt, f2, default_flow_style=False, sort_keys=False)


        # Loading ROOT files 
        # Creating yaml files and later load those files to launch plotter
        logger.info("Hitting DataLoader to create Yaml files in same format from ROOT files with different format ...")
        outdir = None
        data = {}
        for config in [f'{path_config}/configPreInt.yaml', f'{path_config}/configPostInt.yaml']:
            
            with open(config,'r') as c:
                config = yaml.safe_load(c)
            if not config:
                raise RuntimeError("Must provide a yaml config for data loader")

            main_key = config.get("maintag")

            outdir = config.get("output")
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            
            outdir_tag = f"version_{dttag}" if args.tag == "" else f"version_{args.tag}"
            outdir = f"{outdir}/{outdir_tag}"
            logger.warning(f"Results dir : {outdir}")
            if os.path.isdir(outdir):
                logger.warning(f"{outdir} found")
            else:
                os.mkdir(outdir)
        

            config = deep_merge(copy.deepcopy(config), copy.deepcopy(config_hnames))
            
            dataobj = DataLoader(config_1, config, target=outdir)
            data_dict = dataobj.getData()
            data[main_key] = data_dict

            if args.dump:
                with open(f'{outdir}/data.yaml', 'w') as file:
                    yaml.dump(data, file, sort_keys=False, default_flow_style=False)
            logger.info("Data loading done ...")

            #from IPython import embed; embed(); exit()
    
        outdirP = f"{outdir}/Plots"
        if not os.path.exists(outdirP):
            logger.info(f"creating plot dir : {outdirP}")
            os.mkdir(outdirP)
        else:
            logger.warning(f"{outdirP} found")
            
        outdirF = f"{outdir}/Files"
        if not os.path.exists(outdirF):
            logger.info(f"creating file dir : {outdirF}")
            os.mkdir(outdirF)
        else:
            logger.warning(f"{outdirF} found")
            
        plotobj = PlotterUser(config_1,
                              data,
                              outdirP,
                              outdirF,
                              ladpos,
                              ladid)
        plotobj.plotEverything()
        
        logger.info("Plotting done ...")

        logger.info("Quality test begin ... fingers crossed !!! ")
        ladder_root_file = f'{outdirF}/TB2S_Ladder__pos_{ladpos}__{ladid}.root'
        if not os.path.exists(ladder_root_file):
            raise FileNotFoundError(f'{ladder_root_file} file does not exist :( take a coffee break, and then check analysis ... all the best !')

        qc_config_file = args.qcconfig
        if not os.path.exists(qc_config_file):
            raise FileNotFoundError(f'{qc_config_file} not found :( check the main analysis repo, should be there, dazzling like a star !')

        qcobj = QualityCheck(ladder_root_file, qc_config_file, dataOG)
        grade_dict,prepostcomp_dict = qcobj.analyze()
        df_grade, df_prepost = qcobj.get_df(grade_dict, prepostcomp_dict)

        logger.info(f"Grades \n{df_grade.head(100)}")
        logger.info(f"Pre-PostInt \n{df_prepost.head(100)}")
        
        #from IPython import embed; embed()
        df_grade.to_csv(f'{outdirF}/TB2S_Ladder__pos_{ladpos}__{ladid}__grade.csv')
        df_prepost.to_csv(f'{outdirF}/TB2S_Ladder__pos_{ladpos}__{ladid}__preint_postint_diff.csv')

        qcobj.plot_qc(data = df_grade,
                      outdir = summary_path)
        qcobj.plot_prepost(data = df_prepost,
                           outdir = summary_path)

        
        logger.info(f"QC done !")
        
        # Splitting root file
        logger.info("Preparing ROOT files for DCA")
        # Create the split directory first
        #split_path = f'{ladder_path}/ROOTfilesForDCA'
        split_path = f'{out_ladder_path}/ROOTfilesForDCA'
        logger.info(f'Creating ROOTfilesForDCA inside {ladder_path}')
        if os.path.isdir(split_path):
            logger.warning(f'Ah Oh ! ROOTfilesForDCA already exists, expected in the 2nd iteration and so on !')
        else:
            os.mkdir(split_path)

        # create config for split root file
        split_output_path = f'{split_path}/Results'
        if not os.path.isdir(split_output_path):
            os.mkdir(split_output_path)
        split_output_path = f'{split_output_path}'
        split_tag = f"v_{dttag}" if args.tag == "" else f"v_{args.tag}"
        split_output_path = f"{split_output_path}/{split_tag}"
        logger.warning(f"Results dir : {split_output_path}")
        if os.path.isdir(split_output_path):
            logger.warning(f"{split_output_path} found")
        else:
            os.mkdir(split_output_path)

        ph2acf_logs = glob(f'{ladder_path}/TB2S_Ladder_{ladid}_{temp_name_ext}_{calibname}_{ladpos}.log')
        if len(ph2acf_logs) == 0:
            raise FileNotFoundError('No Ph2ACF logfile found')
        
        split_config = {
            'LADDER': ladid,
            'LADDER_POS': int(ladpos),
            'COOLING_TEMP': int(cooltemp.split('C')[0]),
            'INPUT_FILE': tfiles_main[0],
            'DQM_FILE': tfiles_dqm[0],
            'DQM_VTRX_OFF_FILE': vtrxoff_tfiles_dqm[0],
            'PH2ACF_LOG_FILE': ph2acf_logs[0],
            'IV_FILE_DIR': IVs,
            'PS_MONITOR_FILE': monitorPS,
            'LADDER_ANALYSIS_ROOT_FILE': f'{outdirF}/TB2S_Ladder__pos_{ladpos}__{ladid}.root',
            'MODULE_POS': dataOG,
            'EXTRA_INFO': {
                'Setup': 'Ph2-ACF',
                'RunNo': 0,
                'ResultFolder': 'Results',
                'Location': 'Strasbourg',
                'Timezone': 'CET/GMT+2',
                'Operator': 'IPHC',
                'RunType': 'mod_int',
                'StationName': 'IPHC_Coldbox',
                'LadderSlot': ladpos,
                'Comment': 'OK'
            },
            'OUTPUT_PATH': split_output_path,
        }

        split_config_path = f'{split_path}/Configs'
        if not os.path.isdir(split_config_path):
            os.mkdir(split_config_path)
        
        with open(f'{split_config_path}/config_{ladid}.yaml', 'w') as sf:
            yaml.dump(split_config, sf, default_flow_style=False, sort_keys=False)

        logger.info('')
        split_args = Namespace(
            config=f'{split_config_path}/config_{ladid}.yaml',
            split=args.split,
            opticalgroup=args.opticalgroup,
        )
        sparseLadderRootFile.main(split_args)
            
        
    real_stop = time.perf_counter()
    cpu_stop  = time.process_time()
        
    logger.info(f"Real time : {real_stop - real_start:.3f} seconds")
    logger.info(f"CPU time  : {cpu_stop - cpu_start:.3f} seconds")


        
if __name__ == "__main__":
    main()
