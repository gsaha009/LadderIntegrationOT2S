# Splitting Ladder ROOT file to separate Optical Groups
# Author : Gourab Saha, IPHC

# How to Run:
# python main.py Configs_split_ladder_root_file/file.yaml


import os
import sys
import yaml
import ROOT
import time
import argparse

def load_cfg(cfg):
    cfgdict = None
    with open(cfg, 'r') as f:
        cfgdict = yaml.safe_load(f)
    return cfgdict


def copy_dir(src_dir, dest_dir, skipdir = False):
    for key in src_dir.GetListOfKeys():
        name = key.GetName()
        obj = key.ReadObj()

        if isinstance(obj, ROOT.TDirectory):
            if skipdir == True:
                continue
            else:
                dest_dir.mkdir(name)
                subdir = dest_dir.GetDirectory(name)
                copy_dir(obj, subdir)
        else:
            dest_dir.cd()
            obj.Write(name)

def add_modID(TDir, modID, OG):
    ID_obj = ROOT.TObjString(modID)
    TDir.Delete(f"D_B(0)_NameId_OpticalGroup({OG});*")
    TDir.WriteObject(ID_obj, f"D_B(0)_NameId_OpticalGroup({OG})")
    return TDir

            
def main(args):

    REAL_START = time.perf_counter()
    CPU_START  = time.time()
    
    print(f"\nLadder --> Module level splitting ===>> {args.split}\n")
    if args.split == False:
        print("Just to add ModuleIDs per OpticalGroup")
        print("Use -s in the cmdline to enable splitting\n")
    
    CFG_FILE = args.config
    CFG = load_cfg(CFG_FILE)

    OUTDIR = CFG.get('OUTPUT_PATH')
    if OUTDIR is not None:
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)

    MODULE_POS = CFG.get("MODULE_POS")
    
    ladID    = CFG.get('LADDER')
    coolTemp = CFG.get('COOLING_TEMP')


    # Open input file in Read mode
    infilename = CFG.get("INPUT_FILE")
    infile = ROOT.TFile(infilename, "READ")
    print(f"Ph2ACF Ladder Output file as the input here : {infilename}\n")

    test_type = infile.Get("Detector").GetKey("CalibrationName_Detector").ReadObj()
    
    # Creating an output file in Write mode and then copy the contents
    outfile_ladder = None
    outfile_ladder_name = f"Results__{test_type}__Ladder_{ladID}__CO2_{coolTemp}.root"

    print(f"💾 ==> Copying file content from infile to an another file : {outfile_ladder_name}")
    outfile_ladder = ROOT.TFile.Open(f"{OUTDIR}/{outfile_ladder_name}", "RECREATE")
    det_out_ladder = outfile_ladder.mkdir("Detector")
    copy_dir(infile.Get("Detector"), det_out_ladder)
    outfile_ladder.Write() # Output file is written here
    infile.Close()         # Input file is closed 

    # Getting the Detector and Board_0 TDirectories from the output file
    TDir_Detector_Aux = outfile_ladder.Get("Detector")
    TDir_Board_Aux    = outfile_ladder.Get("Detector/Board_0")

    #test_type = TDir_Detector_Aux.GetKey("CalibrationName_Detector").ReadObj()
    print(f"Calibration     : {test_type}")    
    print(f"Ph2-ACF version : {TDir_Detector_Aux.GetKey('GitTag_Detector').ReadObj()}")
    print(f"Ph2-ACF commit  : {TDir_Detector_Aux.GetKey('GitCommitHash_Detector').ReadObj()}")
    print(f"Host PC         : {TDir_Detector_Aux.GetKey('Username_Detector').ReadObj()}@{TDir_Detector_Aux.GetKey('HostName_Detector').ReadObj()}")
    print(f"START           : {TDir_Detector_Aux.GetKey('CalibrationStartTimestamp_Detector').ReadObj()}")
    print(f"STOP            : {TDir_Detector_Aux.GetKey('CalibrationStopTimestamp_Detector').ReadObj()}\n")
    print(f"Board IP        : {TDir_Board_Aux.GetKey('D_NameId_Board_(0)').ReadObj()}")
    print(f"Ladder ID       : {ladID}")
    print(f"Cooling Temp    : {coolTemp}\n")


    # Loop over keys under Detector/Board_0 to find the OpticalGroup TDirectories
    for key in outfile_ladder.Get("Detector/Board_0").GetListOfKeys():
        name = key.GetName()
        obj  = key.ReadObj()

        OG = name.split('_')[-1]
        
        if not isinstance(obj, ROOT.TDirectory):
            continue
        if not "OpticalGroup" in name:
            continue

        
        modID = MODULE_POS[name]
        print(f"📝 ==> Adding ModuleID : {modID} to D_B(0)_NameId_OpticalGroup({OG})")
        add_modID(obj, modID, OG)

        
        if args.split == True:        
            # make outfile name
            print(f"⚙️  ==> Assigning Name to the ROOT file for {name}")
            lpgbt_id = obj.GetKey(f"D_B(0)_LpGBTFuseId_OpticalGroup({OG})").ReadObj()
            vtrx_id  = obj.GetKey(f"D_B(0)_VTRxFuseId_OpticalGroup({OG})").ReadObj()
            print(f" ... LpGBTID    : {lpgbt_id}")
            print(f" ... VTRxID     : {vtrx_id}")
            
            outfilename = f"Results__{test_type}__{modID}__{name}__Ladder_{ladID}__CO2_{coolTemp}.root"
            print(f" >>> File Name  : {outfilename}")

            outfilename = f"{OUTDIR}/{outfilename}"
            outfile = ROOT.TFile(outfilename, "RECREATE")
            
            det_out   = outfile.mkdir("Detector")        
            board_out = det_out.mkdir("Board_0")
            group_out = board_out.mkdir(name)

            print(f"💾 ==> Copying everything from OpticalGroup TDirectory to {name} ROOT file")
            copy_dir(obj, group_out)
            
            print(f"💾 ==> Copying TObjStrings from Board_0 TDirectory to {name} ROOT file")
            copy_dir(TDir_Board_Aux, board_out, skipdir=True)
            
            print(f"💾 ==> Copying TObjStrings from Detector TDirectory to {name} ROOT file")
            copy_dir(TDir_Detector_Aux, det_out, skipdir=True)

            outfile.Write()
            outfile.Close()
            print(f"✅ Wrote {outfilename}\n")

        
    print(f"✅ Wrote {outfile_ladder_name} with ModuleIDs as NameId_OpticalGroup \n")
    outfile_ladder.Close()
    #infile.Close()

    REAL_STOP = time.perf_counter()
    CPU_STOP  = time.time()

    print(f"Real time : {round(REAL_STOP - REAL_START, 2)} seconds")
    print(f"CPU time  : {round(CPU_STOP - CPU_START, 2)} seconds")



    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sparsing Ph2ACF ROOT file with more than one Optical Group')
    parser.add_argument("-c",
                        "--config", 
                        type        = str,
                        required    = True,
                        help="Config file") 

    parser.add_argument("-s",
                        "--split",
                        action      = "store_true",
                        default     = False,
                        help="Split the ROOT file?")

    args= parser.parse_args()
    
    main(args)
