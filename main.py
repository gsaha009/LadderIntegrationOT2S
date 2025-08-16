import os
import sys
import time
import numpy as np
import argparse
import datetime
import logging
import yaml

from DataLoader import DataLoader
from Plotter import Plotter


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


def setup_logger():
    # Create a logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Use the custom ColorFormatter
    formatter = ColorFormatter()
    console_handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger

    
def main():
    real_start = time.perf_counter()
    cpu_start  = time.process_time()
    
    dttag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger()
    logger.info(f"date-time: {dttag}")

    parser = argparse.ArgumentParser(description='Plotter')
    parser.add_argument('-cs', '--configs', nargs='+', type=str, required=False, help="yaml configs to be used")
    parser.add_argument('-d', '--dump', action='store_true', default=False, required=False, help="save data in yaml before plotting?")
    parser.add_argument('-t', '--tag', type=str, required=False, default="", help="config name")
   
    args = parser.parse_args()

    config_file_1 = "Configs/tests.yml"
    config_1 = None
    with open(config_file_1,'r') as conf1:
        config_1 = yaml.safe_load(conf1)

    # Loading ROOT files 
    # Creating yaml files and later load those files to launch plotter
    logger.info("Hitting DataLoader to create Yaml files in same format from ROOT files with different format ...")
    outdir = None
    data = {}
    for config in args.configs:
        with open(config,'r') as c:
            config = yaml.safe_load(c)
        main_key = config.get("maintag")
        if not config:
            raise RuntimeError("Must provide a yaml config for data loader")
        outdir_base = config.get("output", f"../Output/OutputDefault")
        if os.path.exists(outdir_base):
            logger.warning(f"{outdir_base} found")
        else:
            os.mkdir(outdir_base)
        
        outdir_name = f"{dttag}" if args.tag == "" else f"{args.tag}"
        outdir = f"{outdir_base}/{outdir_name}"
        if os.path.exists(outdir):
            logger.warning(f"{outdir} found")
        else:
            os.mkdir(outdir)

        dataobj = DataLoader(config_1, config, target=outdir)
        data_dict = dataobj.getData()
        data[main_key] = data_dict

    if args.dump:
        with open(f'{outdir}/data.yaml', 'w') as file:
            yaml.dump(data, file, sort_keys=False, default_flow_style=False)
    logger.info("Data loading done ...")

    #from IPython import embed; embed(); exit()
    
    outdir = f"{outdir}/Plots"
    if not os.path.exists(outdir):
        logger.info(f"creating plot dir : {outdir}")
        os.mkdir(outdir)
    else:
        logger.warning(f"{outdir} found")


    plotobj = Plotter(config_1,
                      data,
                      outdir)
    # comparing same modules from 2 different setup
    # or different modules from same setup?
    # diff_mods
    # same_mods
    #plotobj.plotEverything(diff_mods = diff_mods,
    #                       same_mods = same_mods)
    plotobj.plotEverything()
    
    real_stop = time.perf_counter()
    cpu_stop  = time.process_time()
    
    logger.info("Plotting done ...")
    logger.info(f"Real time : {real_stop - real_start:.3f} seconds")
    logger.info(f"CPU time  : {cpu_stop - cpu_start:.3f} seconds")
    
        
if __name__ == "__main__":
    main()
