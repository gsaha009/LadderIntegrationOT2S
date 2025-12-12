# Splitting Ladder ROOT file to separate Optical Groups
# Author : Gourab Saha, IPHC

# How to Run:
# python main.py <cfg_file.yaml>


import os
import sys
import yaml
import ROOT

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

    print("\nLadder --> Module level splitting ===>> \n")
    
    args = sys.argv
    if len(args) < 2:
        raise RuntimeError("Config file is missing")

    CFG_FILE = args[1]
    CFG = load_cfg(CFG_FILE)
    
    infilename = CFG.get("INPUT_FILE")
    test_type  = CFG.get("TEST_TYPE")
    print(f"File to split : {infilename}")
    print(f"Type of test  : {test_type}\n")

    OUTDIR = CFG.get('OUTPUT_PATH')
    MODULE_POS = CFG.get("MODULE_POS")
    
    infile = ROOT.TFile(infilename, "READ")
    
    TDir_Detector_Aux = infile.Get("Detector")
    TDir_Board_Aux = infile.Get("Detector/Board_0")
    
    for key in infile.Get("Detector/Board_0").GetListOfKeys():
        name = key.GetName()
        obj = key.ReadObj()

        if not isinstance(obj, ROOT.TDirectory):
            continue
        if not "OpticalGroup" in name:
            continue

        # make outfile name
        print(f"⚙️  ==> Assigning Name to the ROOT file for {name}")
        modID    = MODULE_POS[name]
        ladID    = CFG.get('LADDER')
        coolTemp = CFG.get('COOLING_TEMP')

        print(f" ... OGroup    : {name}")
        print(f" ... ModID     : {modID}")
        print(f" ... LadID     : {ladID}")
        print(f" ... Cool Temp : {coolTemp}")
        
        outfilename = f"Results__{test_type}__{modID}__{name}__Ladder_{ladID}__CO2_{coolTemp}.root"
        print(f" ===>>> File Name: {outfilename}")

        if OUTDIR is not None:
            if not os.path.exists(OUTDIR):
                os.mkdir(OUTDIR)
            outfilename = f"{OUTDIR}/{outfilename}"
                
        outfile = ROOT.TFile(outfilename, "RECREATE")

        det_out = outfile.mkdir("Detector")
        print(f"💾 ==> Copying TObjStrings from Detector TDirectory to the Detector correspnding to {name} ROOT file")
        copy_dir(TDir_Detector_Aux, det_out, skipdir=True)
        
        board_out = det_out.mkdir("Board_0")
        print(f"💾 ==> Copying TObjStrings from Board_0 TDirectory to the Board correspnding to {name} ROOT file")
        copy_dir(TDir_Board_Aux, board_out, skipdir=True)
        
        group_out = board_out.mkdir(name)

        print(f"💾 ==> Copying everything from OG TDirectory to the {name} ROOT file")
        copy_dir(obj, group_out)


        
        OG = name.split('_')[-1]        
        # modify the ModuleID
        print(f"📝 ==> Adding ModuleID : {MODULE_POS[name]} to D_B(0)_NameId_OpticalGroup({OG})")
        ID_obj = ROOT.TObjString(MODULE_POS[name])
        #ID_obj = ROOT.TObjString(f"D_B(0)_NameId_OpticalGroup({OG})")
        #ID_obj.SetName(MODULE_POS[name])
        group_out.Delete(f"D_B(0)_NameId_OpticalGroup({OG});*")
        group_out.WriteObject(ID_obj, f"D_B(0)_NameId_OpticalGroup({OG})")

        #from IPython import embed; embed()
        
        outfile.Write()
        outfile.Close()
        print(f"✅ Wrote {outfilename}\n")

    infile.Close()


if __name__ == "__main__":
    main()
