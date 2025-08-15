import os
import sys
import csv
import yaml
import glob

input_paths = [
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTest_2S_18_5_NCP_10001_Apr28_RoomTemp",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTest_2S_18_5_NCP_10001_Apr29_ZeroTemp",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTest_2S_18_5_NCP_10001_Apr29_M30Temp"
]
#csv_file_name = f"{input_path}/run_calibration_log_Apr23.csv"


yaml_config = "quick_test_Apr23_2S_18_5_NCP_10001_v3.yml"


maindict = {}
maindict["n_boards"] = 1
maindict["n_opticals"] = 1
maindict["files"] = {}


for input_path in input_paths:
    temptag = input_path.split('_')[-1]
    maindict["files"][temptag] = {}
    csv_file_name = glob.glob(f"{input_path}/*.csv")[0]
    print(f"csv file : {csv_file_name}")
    with open(csv_file_name, mode='r') as csvf:
        csv_file = csv.reader(csvf)
        for line in csv_file:
            print(line)

            index = line[0]
            start = line[1]
            end   = line[2]
            main_root = line[3]
            dqm_root  = line[4]

            if start == "START": continue
        
            maindict["files"][temptag][f"{main_root}"] = {
                #"tstamp_start": start,
                #"tstamp_end"  : end,
                "tfile_main"  : f"{input_path}/Results_{main_root}.root",
                "tfile_dqm"   : f"{input_path}/{dqm_root}_{main_root}.root",
            }

with open(yaml_config, 'w') as outf:
    yaml.dump(maindict, outf, default_flow_style=False, sort_keys=False)
