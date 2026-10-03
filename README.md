# LadderIntegrationOT2S

# 📊 Ph2_ACF 2S Module Quick Plotting Toolkit

Generate quick plots from **Ph2_ACF** quick/full test results for **2S modules** — whether tested individually in a **single-module box** (pre-integration) or integrated to a **ladder** and tested inside IPHC **cold-box** setup (post-integration).

## ✨ Features
- **Fast plotting** of Ph2_ACF quick/full test data
- Supports both **single module** and **ladder cold-box** configurations
- Flexible analysis levels:
  - **CBC-level** inspection
  - **Module-level** summary
  - **Cross-setup comparison** (single-module vs. ladder)

## 🛠 Requirements

Two options :
  - Use `cvmfs` : The easiest way
  - Or, use `Anaconda` to setup the environment : Easy, but messy

if `cvmfs` is not accessible, a bit longer process to be followed
```bash
# conda installation
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use the `yml` and build the environment
conda env create -f environment.yml
```

Or, if `cvmfs` is there, then life would be easier as mentioned above
```bash
# first clone the repo
git clone https://github.com/gsaha009/LadderIntegrationOT2S.git
source setup.sh
```

## 🤖 Now, lets start ...

  - Clone the main repository [for now, checkout `dev` branch]
  ```bash
  git clone --recurse-submodules https://github.com/gsaha009/LadderIntegrationOT2S.git
  git checkout dev
  ```
  - How to run?
    - The main script is `main.py`
    - two mandatory inputs are `Ph2ACF Calibration Results directory` and `Ph2ACF VTRXoff Results directory`
      ```bash
      python3 main.py -i /Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/FinalTest/TB2SLadders__pos_1__TB2S_Ladder_minus_6CP_9__pos_2__TB2S_Ladder_plus_5CP_4__+15C__2SfullTest__2026-10-02_10_58_18 -ivtrx /Users/gsaha/Work/IPHC/TrackerUpgrade/Inputs/FinalTest/TB2SLadders__pos_1__TB2S_Ladder_minus_6CP_9__pos_2__TB2S_Ladder_plus_5CP_4__+15C__vtrxoff__2026-10-02_10_50_02 -s [#to split the merged potato compatible ROOT file] -qccfg "bla.yaml" [#default is "qc_config.yaml"]
      ```
    - The two `Ph2ACF` Results-dir must have the folliwng structure:
        -
	```bash
	TB2SLadders__pos_1__TB2S_Ladder_<lad-1-id>__pos_2__TB2S_Ladder_<lad-2-id>__<cooling>__<calibname>__<datetime>
	# e.g.
	# lad-1-id  : minus_6CP_9
	# lad-2-id  : plus_5CP_4
	# cooling   : +15C or -30C
	# calibname : 2SfullTest / 2SquickTest / vtrxoff
	# datetime  : 2026-10-02_10_50_02
	```
	-
	```bash
TB2SLadders__pos_1__TB2S_Ladder_minus_6CP_9__pos_2__TB2S_Ladder_plus_5CP_4__+15C__vtrxoff__2026-10-02_10_50_02
├── IV
│   └── iv_curves
│       ├── 2S_18_5_BRN-01001_HV#1.csv
│       ├── 2S_18_6_KIT-10022_HV#13.csv
│       ├── 2S_18_6_KIT-10023_HV#20.csv
│       ├── 2S_18_6_KIT-10025_HV#24.csv
│       ├── 2S_18_6_KIT-10026_HV#21.csv
│       ├── 2S_18_6_KIT-10027_HV#22.csv
│       ├── 2S_18_6_KIT-10028_HV#19.csv
│       ├── 2S_18_6_KIT-10029_HV#18.csv
│       ├── 2S_18_6_KIT-10030_HV#17.csv
│       ├── 2S_18_6_KIT-10031_HV#16.csv
│       ├── 2S_18_6_KIT-10033_HV#14.csv
│       ├── 2S_18_6_NCP-10003_HV#5.csv
│       ├── 2S_18_6_NCP-10006_HV#2.csv
│       ├── 2S_18_6_NCP-10007_HV#3.csv
│       ├── 2S_18_6_NCP-10009_HV#23.csv
│       ├── 2S_18_6_NCP-10010_HV#15.csv
│       ├── 2S_18_6_NCP-10011_HV#10.csv
│       ├── 2S_18_6_NCP-10012_HV#4.csv
│       ├── 2S_18_6_NCP-10014_HV#6.csv
│       ├── 2S_18_6_NCP-10015_HV#7.csv
│       ├── 2S_18_6_NCP-10019_HV#11.csv
│       ├── 2S_18_6_NCP-10021_HV#12.csv
│       ├── 2S_18_6_NCP-10022_HV#9.csv
│       └── 2S_18_6_NCP-10023_HV#8.csv
├── LOG_MAIN_STEPS.csv
├── MonitorMarta.csv
├── MonitorPS.csv
├── Pos_1__TB2S_Ladder_minus_6CP_9
│   ├── TB2S_Ladder_minus_6CP_9_ambient_vtrxoff_1.log
│   └── ambient
│       ├── MonitorResults
│       │   └── MonitorDQM_2026-10-02_10-50-04.root
│       ├── Results
│       │   └── Run_0
│       │       └── Results.root
│       ├── RunNumbers.dat
│       ├── logs
│       │   ├── Ph2_ACF.log
│       │   ├── Ph2_ACF_debug.log
│       │   ├── Ph2_ACF_err.log
│       │   ├── Ph2_ACF_fatal.log
│       │   └── Ph2_ACF_warn.log
│       └── myeasylog.log
└── Pos_2__TB2S_Ladder_plus_5CP_4
    ├── TB2S_Ladder_plus_5CP_4_ambient_vtrxoff_2.log
    └── ambient
        ├── MonitorResults
        │   └── MonitorDQM_2026-10-02_10-50-06.root
        ├── Results
        │   └── Run_0
        │       └── Results.root
        ├── RunNumbers.dat
        ├── logs
        │   ├── Ph2_ACF.log
        │   ├── Ph2_ACF_debug.log
        │   ├── Ph2_ACF_err.log
        │   ├── Ph2_ACF_fatal.log
        │   └── Ph2_ACF_warn.log
        └── myeasylog.log
	```
	-
	```bash
TB2SLadders__pos_1__TB2S_Ladder_minus_6CP_9__pos_2__TB2S_Ladder_plus_5CP_4__+15C__2SfullTest__2026-10-02_10_58_18
├── Pos_1__TB2S_Ladder_minus_6CP_9
│   ├── PreIntResults
│   │   ├── 2S_18_5_BRN-01001
│   │   │   ├── MonitorDQM_2025-12-15_11-27-48.root
│   │   │   └── Results.root
│   │   ├── 2S_18_6_NCP-10003
│   │   │   ├── MonitorDQM_2026-01-12_12-39-01.root
│   │   │   └── Results.root
│   │   ├── 2S_18_6_NCP-10006
│   │   │   ├── MonitorDQM_2026-01-12_14-57-44.root
│   │   │   └── Results.root
│   │   ├── 2S_18_6_NCP-10007
│   │   │   ├── MonitorDQM_2026-01-12_14-28-29.root
│   │   │   └── Results.root
│   │   ├── 2S_18_6_NCP-10011
│   │   │   ├── MonitorDQM_2026-01-13_11-00-25.root
│   │   │   └── Results.root
│   │   ├── 2S_18_6_NCP-10012
│   │   │   ├── MonitorDQM_2026-01-12_14-01-11.root
│   │   │   └── Results.root
│   │   ├── 2S_18_6_NCP-10014
│   │   │   ├── MonitorDQM_2026-01-12_10-55-58.root
│   │   │   └── Results.root
│   │   ├── 2S_18_6_NCP-10015
│   │   │   ├── MonitorDQM_2026-01-12_10-10-43.root
│   │   │   └── Results.root
│   │   ├── 2S_18_6_NCP-10019
│   │   │   ├── MonitorDQM_2026-01-13_12-21-16.root
│   │   │   └── Results.root
│   │   ├── 2S_18_6_NCP-10021
│   │   │   ├── MonitorDQM_2026-01-13_13-32-22.root
│   │   │   └── Results.root
│   │   ├── 2S_18_6_NCP-10022
│   │   │   ├── MonitorDQM_2026-01-13_14-50-57.root
│   │   │   └── Results.root
│   │   └── 2S_18_6_NCP-10023
│   │       ├── MonitorDQM_2026-01-13_15-19-14.root
│   │       └── Results.root
│   ├── TB2S_Ladder_minus_6CP_9_ambient_2SfullTest_1.log
│   └── ambient
│       ├── MonitorResults
│       │   └── MonitorDQM_2026-10-02_10-58-20.root
│       ├── Results
│       │   └── Run_0
│       │       └── Results.root
│       ├── RunNumbers.dat
│       ├── logs
│       │   ├── Ph2_ACF.log
│       │   ├── Ph2_ACF_debug.log
│       │   ├── Ph2_ACF_err.log
│       │   ├── Ph2_ACF_fatal.log
│       │   └── Ph2_ACF_warn.log
│       └── myeasylog.log
└── Pos_2__TB2S_Ladder_plus_5CP_4
    ├── PreIntResults
    │   ├── 2S_18_6_KIT-10022
    │   │   └── 2S_18_6_KIT-10022_2025-08-05_15h05m03s_+23C_2SfullTest_v6-16.root
    │   ├── 2S_18_6_KIT-10023
    │   │   └── 2S_18_6_KIT-10023_2025-08-01_14h01m05s_+24C_2SfullTest_v6-16.root
    │   ├── 2S_18_6_KIT-10025
    │   │   └── 2S_18_6_KIT-10025_2025-08-01_11h38m21s_+24C_2SfullTest_v6-16.root
    │   ├── 2S_18_6_KIT-10026
    │   │   └── 2S_18_6_KIT-10026_2025-07-29_14h07m38s_+24C_2SfullTest_v6-16.root
    │   ├── 2S_18_6_KIT-10027
    │   │   └── 2S_18_6_KIT-10027_2025-07-29_13h33m00s_+23C_2SfullTest_v6-16.root
    │   ├── 2S_18_6_KIT-10028
    │   │   └── 2S_18_6_KIT-10028_2025-07-29_11h17m23s_+25C_2SfullTest_v6-16.root
    │   ├── 2S_18_6_KIT-10029
    │   │   └── 2S_18_6_KIT-10029_2025-07-29_10h39m28s_+24C_2SfullTest_v6-16.root
    │   ├── 2S_18_6_KIT-10030
    │   │   └── 2S_18_6_KIT-10030_2025-07-29_10h09m29s_+24C_2SfullTest_v6-16.root
    │   ├── 2S_18_6_KIT-10031
    │   │   └── 2S_18_6_KIT-10031_2025-07-29_09h32m00s_+24C_2SfullTest_v6-16.root
    │   ├── 2S_18_6_KIT-10033
    │   │   └── 2S_18_6_KIT-10033_2025-07-29_08h12m31s_+24C_2SfullTest_v6-16.root
    │   ├── 2S_18_6_NCP-10009
    │   │   ├── MonitorDQM_2026-01-14_09-11-27.root
    │   │   └── Results.root
    │   └── 2S_18_6_NCP-10010
    │       ├── MonitorDQM_2026-03-20_11-26-10.root
    │       └── Results.root
    ├── TB2S_Ladder_plus_5CP_4_ambient_2SfullTest_2.log
    └── ambient
        ├── MonitorResults
        │   └── MonitorDQM_2026-10-02_10-58-22.root
        ├── Results
        │   └── Run_0
        │       └── Results.root
        ├── RunNumbers.dat
        ├── logs
        │   ├── Ph2_ACF.log
        │   ├── Ph2_ACF_debug.log
        │   ├── Ph2_ACF_err.log
        │   ├── Ph2_ACF_fatal.log
        │   └── Ph2_ACF_warn.log
        └── myeasylog.log
	```
    - IMPORTANT:
      - The `PreInt Results` should be downloaded before hand, and the best strategy would be to check if the files are correct.
      - Perhaps, we should not add the `downloading Preint results` task to the maste GUI controller.
      - The files should come from `2SfullTest` and `Potato-converted` from `DCA`.
      - If one/two files are found missing from `DCA`, it would be better to test the modules in the single module box.
      - We need to coume-up with some smart idea, however, if we finally decide to overlook the pre-int results, the `preint check` can be made optional.


## The final output structure

```bash
TB2SLadders__pos_1__TB2S_Ladder_minus_6CP_9__pos_2__TB2S_Ladder_plus_5CP_4__+15C__2SfullTestAnalysis__2026-10-03_23_01_28
├── Pos_1__TB2S_Ladder_minus_6CP_9
│   ├── AnalysisResults
│   │   ├── Configs
│   │   │   ├── configPostInt.yaml
│   │   │   └── configPreInt.yaml
│   │   └── Results
│   │       └── version_2026-10-03_23_01_28
│   │           ├── Files
│   │           │   ├── TB2S_Ladder__pos_1__minus_6CP_9.root
│   │           │   ├── TB2S_Ladder__pos_1__minus_6CP_9__grade.csv
│   │           │   └── TB2S_Ladder__pos_1__minus_6CP_9__preint_postint_diff.csv
│   │           └── Plots
│   │               ├── Plot_BrokenChannels_hybrid0_allModules_compare.png
│   │               ├── Plot_BrokenChannels_hybrid1_allModules_compare.png
│   │               ├── Plot_CMN_frac_module_compare.png
│   │               ├── ...
│   │               ├── PostInt__Pos_1__minus_6CP_9
│   │               │   ├── 2S_18_5_BRN-01001
│   │               │   │   ├── CBCLevel
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_5_BRN-01001_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid1_2S_18_5_BRN-01001_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── ...
│   │               │   │   │   └── SCurve_2S_18_5_BRN-01001.png
│   │               │   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_5_BRN-01001_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   ├── LpGBT_EyeOpening.png
│   │               │   │   ├── ...
│   │               │   │   ├── Plot_StripNoise_bothHybrids_TopSensor_2S_18_5_BRN-01001_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   └── VTRx_LightYeildScan.png
│   │               │   ├── 2S_18_6_NCP-10003
│   │               │   │   ├── CBCLevel
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_6_NCP-10003_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── ...
│   │               │   │   │   └── SCurve_2S_18_6_NCP-10003.png
│   │               │   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_6_NCP-10003_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   ├── LpGBT_EyeOpening.png
│   │               │   │   ├── ...
│   │               │   │   └── VTRx_LightYeildScan.png
│   │               │   ├── 2S_18_6_NCP-10006
│   │               │   │   ├── CBCLevel
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_6_NCP-10006_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid1_2S_18_6_NCP-10006_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── ...
│   │               │   │   │   └── SCurve_2S_18_6_NCP-10006.png
│   │               │   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_6_NCP-10006_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   ├── LpGBT_EyeOpening.png
│   │               │   │   ├── ...
│   │               │   │   └── VTRx_LightYeildScan.png
│   │               │   ├── ...
│   │               │   ├── 2S_18_6_NCP-10023
│   │               │   │   ├── CBCLevel
│   │               │   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_6_NCP-10023_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   │   ├── ...
│   │               │   │   │   └── SCurve_2S_18_6_NCP-10023.png
│   │               │   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_6_NCP-10023_PostInt__Pos_1__minus_6CP_9.png
│   │               │   │   ├── LpGBT_EyeOpening.png
│   │               │   │   ├── ...
│   │               │   │   └── VTRx_LightYeildScan.png
│   │               │   ├── Eye_Opening_Scan_Pow_0p33.png
│   │               │   ├── Eye_Opening_Scan_Pow_0p67.png
│   │               │   ├── ...
│   │               │   ├── Sensor_Temp_OGs.png
│   │               │   └── VTRx_LightYield_Ladder.png
│   │               └── PreInt__Pos_1__minus_6CP_9
│   │                   ├── 2S_18_5_BRN-01001
│   │                   │   ├── CBCLevel
│   │                   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_5_BRN-01001_PreInt__Pos_1__minus_6CP_9.png
│   │                   │   │   ├── ...
│   │                   │   │   └── SCurve_2S_18_5_BRN-01001.png
│   │                   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_5_BRN-01001_PreInt__Pos_1__minus_6CP_9.png
│   │                   │   ├── ...
│   │                   │   └── VTRx_LightYeildScan.png
│   │                   ├── 2S_18_6_NCP-10023
│   │                   │   ├── CBCLevel
│   │                   │   │   ├── Hist_DelayVsThr_Hybrid0_2S_18_6_NCP-10023_PreInt__Pos_1__minus_6CP_9.png
│   │                   │   │   ├── ...
│   │                   │   │   └── SCurve_2S_18_6_NCP-10023.png
│   │                   │   ├── Hist_StripNoise_bothHybrids_bothSensors_2S_18_6_NCP-10023_PreInt__Pos_1__minus_6CP_9.png
│   │                   │   ├── ...
│   │                   │   └── VTRx_LightYeildScan.png
│   │                   ├── Eye_Opening_Scan_Pow_0p33.png
│   │                   ├── ...
│   │                   ├── Sensor_Temp_OGs.png
│   │                   └── VTRx_LightYield_Ladder.png
│   ├── ROOTfilesForDCA
│   │   ├── Configs
│   │   │   └── config_minus_6CP_9.yaml
│   │   └── Results
│   │       └── v_2026-10-03_23_01_28
│   │           ├── 2S_18_5_BRN-01001_2026-10-02_10h58m20s_+15C_2SfullTest_v6-29.root
│   │           ├── 2S_18_6_NCP-10003_2026-10-02_10h58m20s_+15C_2SfullTest_v6-29.root
│   │           ├── ...
│   │           ├── 2S_18_6_NCP-10023_2026-10-02_10h58m20s_+15C_2SfullTest_v6-29.root
│   │           └── TB2S_Ladder_minus_6CP_9_2026-10-02_10h58m20s_+15C_2SfullTest_v6-29.root
│   └── Summary
│       ├── quality_grade.pdf
│       └── quality_grade.png
├── Pos_2__TB2S_Ladder_plus_5CP_4
│   ├── AnalysisResults
│   │   ├── Configs
│   │   │   ├── configPostInt.yaml
│   │   │   └── configPreInt.yaml
│   │   └── Results
│   │       └── version_2026-10-03_23_01_28
│   │           ├── Files
│   │           │   ├── TB2S_Ladder__pos_2__plus_5CP_4.root
│   │           │   ├── TB2S_Ladder__pos_2__plus_5CP_4__grade.csv
│   │           │   └── TB2S_Ladder__pos_2__plus_5CP_4__preint_postint_diff.csv
│   │           └── Plots
│   │               ├── Plot_BrokenChannels_hybrid0_allModules_compare.png
│   │               ...
└── TB2SLadders__pos_1__TB2S_Ladder_minus_6CP_9__pos_2__TB2S_Ladder_plus_5CP_4__+15C__2SfullTestAnalysis__2026-10-03_23_01_28.log

```

## SubModules

| Path | Repository | Purpose |
|------|------------|---------|
| `modules/2SLadderCMNAna` | [GitHub Repo](https://github.com/gsaha009/2SLadderCMNAna) | CMNoise fitter |
| `modules/EventContainer` | [GitLab Repo](https://gitlab.cern.ch/gsaha/eventcontainer.git) | Read OTPhysics Root file |