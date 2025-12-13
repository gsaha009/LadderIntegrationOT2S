# Splitting Ladder ROOT file to separate Optical Groups
# Author : Gourab Saha, IPHC

# How to Run:
# python main.py Configs_split_ladder_root_file/file.yaml


import os
import sys
import yaml
import ROOT
import time

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


            
def main():

    REAL_START = time.perf_counter()
    CPU_START  = time.time()
    
    print("\nLadder --> Module level splitting ===>> \n")
    
    args = sys.argv
    if len(args) < 2:
        raise RuntimeError("Config file is missing")

    CFG_FILE = args[1]
    CFG = load_cfg(CFG_FILE)
    
    infilename = CFG.get("INPUT_FILE")
    test_type  = CFG.get("TEST_TYPE")
    print(f"File to split : {infilename}\n")

    OUTDIR = CFG.get('OUTPUT_PATH')
    MODULE_POS = CFG.get("MODULE_POS")
    
    infile = ROOT.TFile(infilename, "READ")
    
    TDir_Detector_Aux = infile.Get("Detector")
    TDir_Board_Aux    = infile.Get("Detector/Board_0")

    test_type = TDir_Detector_Aux.GetKey("CalibrationName_Detector").ReadObj()
    print(f"Calibration     : {test_type}")
    
    print(f"Ph2-ACF version : {TDir_Detector_Aux.GetKey('GitTag_Detector').ReadObj()}")

    print(f"Ph2-ACF commit  : {TDir_Detector_Aux.GetKey('GitCommitHash_Detector').ReadObj()}")
    print(f"Host PC         : {TDir_Detector_Aux.GetKey('Username_Detector').ReadObj()}@{TDir_Detector_Aux.GetKey('HostName_Detector').ReadObj()}")
    print(f"START           : {TDir_Detector_Aux.GetKey('CalibrationStartTimestamp_Detector').ReadObj()}")
    print(f"STOP            : {TDir_Detector_Aux.GetKey('CalibrationStopTimestamp_Detector').ReadObj()}\n")
    print(f"Board IP        : {TDir_Board_Aux.GetKey('D_NameId_Board_(0)').ReadObj()}")

    
    ladID    = CFG.get('LADDER')
    coolTemp = CFG.get('COOLING_TEMP')
    print(f"Ladder ID       : {ladID}")
    print(f"Cooling Temp    : {coolTemp}\n")
    
    for key in infile.Get("Detector/Board_0").GetListOfKeys():
        name = key.GetName()
        obj  = key.ReadObj()

        OG = name.split('_')[-1]
        
        if not isinstance(obj, ROOT.TDirectory):
            continue
        if not "OpticalGroup" in name:
            continue

        
        # make outfile name
        print(f"⚙️  ==> Assigning Name to the ROOT file for {name}")
        modID    = MODULE_POS[name]

        print(f" ... OGroup     : {name}")
        print(f" ... ModID      : {modID}")
        lpgbt_id = obj.GetKey(f"D_B(0)_LpGBTFuseId_OpticalGroup({OG})").ReadObj()
        vtrx_id  = obj.GetKey(f"D_B(0)_VTRxFuseId_OpticalGroup({OG})").ReadObj()
        print(f" ... LpGBTID    : {lpgbt_id}")
        print(f" ... VTRxID     : {vtrx_id}")
        
        outfilename = f"Results__{test_type}__{modID}__{name}__Ladder_{ladID}__CO2_{coolTemp}.root"
        print(f" >>> File Name  : {outfilename}")


        
        if OUTDIR is not None:
            if not os.path.exists(OUTDIR):
                os.mkdir(OUTDIR)
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

        
        
        print(f"📝 ==> Adding ModuleID : {modID} to D_B(0)_NameId_OpticalGroup({OG})")
        ID_obj = ROOT.TObjString(modID)
        group_out.Delete(f"D_B(0)_NameId_OpticalGroup({OG});*")

        group_out.WriteObject(ID_obj, f"D_B(0)_NameId_OpticalGroup({OG})")

        
        outfile.Write()
        outfile.Close()
        print(f"✅ Wrote {outfilename}\n")

    infile.Close()

    REAL_STOP = time.perf_counter()
    CPU_STOP  = time.time()

    print(f"Real time taken : {round(REAL_STOP - REAL_START, 2)} seconds")
    print(f"CPU time taken  : {round(CPU_STOP - CPU_START, 2)} seconds")



    
if __name__ == "__main__":
    main()
