import os
import sys
import csv
import yaml
import glob

input_paths = [
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10007",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10008",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10009",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10010",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10011",
    #"/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10012",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10013",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10014",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10015",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10016",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10017",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10018",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10019",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10020",
    "/Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/Results_QuickTests_KIT_May15/Results_2S_18_6_KIT-10021",
]
#csv_file_name = f"{input_path}/run_calibration_log_Apr23.csv"


yaml_config = "quick_test_KIT_May15_New.yml"


maindict = {}
maindict["n_boards"] = 1
maindict["n_opticals"] = 1
maindict["files"] = {}
maindict["files"]["RoomTemp"] = {}

for input_path in input_paths:
    mainkey = os.path.basename(input_path).replace("Results_","")
    rootf = glob.glob(f"{input_path}/Result*.root")[0]
    #dqmf  = glob.glob(f"{input_path}/Monitor*.root")[0]
    maindict["files"]["RoomTemp"][f"{mainkey}"] = {
        "tfile_main"  : rootf,
        #"tfile_dqm"   : dqmf,
    }

with open(yaml_config, 'w') as outf:
    yaml.dump(maindict, outf, default_flow_style=False, sort_keys=False)
