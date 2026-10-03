# Plotter function
# Ladder Integration at IPHC
# Author: G.Saha

from Plotter import *
from util import *

from Fitter import Fitter
CMNmod = importlib.import_module("modules.2SLadderCMNAna.CMNFitter")

import logging
logger = logging.getLogger('main')


def clean_outliers(sensor_temps):
    sensor_temp_median = np.median(sensor_temps)
    sensor_temps_sel = sensor_temps[np.abs(sensor_temps-sensor_temp_median) < 1.0]

    sensor_temps_sel_grd = np.gradient(sensor_temps_sel)
    sensor_temps_sel = sensor_temps_sel[np.abs(sensor_temps_sel_grd) < 0.5]

    return sensor_temps_sel


class PlotterUser(Plotter):
    def __init__(self, testinfo, data, outdir, outdirF, ladpos, ladid, **kwargs):
        super().__init__(testinfo, data, outdir, outdirF, ladpos, ladid, **kwargs)

    def plot_sensor_temp(self,
                         sensor_temp_dict,
                         outdir = None,
                         **kwargs):
        
        pname     = kwargs.get('pname', 'default')
        #ylim      = kwargs.get("ylim", [-50.0, 50.0])
        
        #print(outdir)
        
        self.plot_basic_from_dict(data       = sensor_temp_dict,
                                  title      = f"Sensor_Temperature",
                                  name       = f"{pname}",
                                  outdir     = outdir,
                                  xlabel     = 'time stamps',
                                  ylabel     = 'sensor temperature (deg C)',
                                  linewidth  = 0.5,
                                  #ylim       = ylim
                                  )
        
        
    def simfit_and_analyse(self, noise_0_dict, noise_3_dict,
                           labels: list, title: str, name: str, outdir: str):
        fracs = []
        # loop over CBCs
        for i in range(8):
            nHits_0sigma = np.array(noise_0_dict[f'CBC_{i}'])[:,0]
            sigmaHits_0sigma = np.array(noise_0_dict[f'CBC_{i}'])[:,1]
            nHits_3sigma = np.array(noise_3_dict[f'CBC_{i}'])[:,0]
            sigmaHits_3sigma = np.array(noise_3_dict[f'CBC_{i}'])[:,1]
            
            logger.info(f"fitting nHits for CBC_{i}")
            sigma_fit, k_probs_fit, res = CMNmod.fit_k_and_sigma_from_hists(nHits_0sigma,
                                                                            nHits_3sigma,
                                                                            sigmaHits_0sigma,
                                                                            sigmaHits_3sigma)
            
            P_A, P_B, R_A, R_B = CMNmod.predict_distributions(sigma_fit, k_probs_fit)
            expA = 10000 * P_A
            expB = 10000 * P_B
            #from IPython import embed; embed(); exit()
            
            N_BINS_K = 20
            K_MIN, K_MAX = -2.0, 2.0
            BIN_EDGES = np.linspace(K_MIN, K_MAX, N_BINS_K + 1)
            BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
            BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]
            
            
            x = [ [np.arange(nHits_0sigma.shape[0]), np.arange(len(expA))] ,
                  [np.arange(nHits_3sigma.shape[0]), np.arange(len(expB))] ,
                  [BIN_CENTERS] ]
            y = [ [nHits_0sigma/10000, P_A] ,
                  [nHits_3sigma/10000, P_B] ,
                  [k_probs_fit] ]
            h3_width = BIN_WIDTH*0.9
            
            self.plot_fitted_result(x = x,
                                    y = y,
                                    w = h3_width,
                                    labels=labels,
                                    title = f"{title}_CBC_{i}",
                                    name = f"{name}_CBC_{i}",
                                    outdir = outdir)
            
            #fracs.append(float(R_A['frac_common']))
        
            mu_k = np.sum(BIN_CENTERS * k_probs_fit)
            var_k = np.sum((BIN_CENTERS - mu_k)**2 * k_probs_fit)
            sigma_k = np.sqrt(var_k)
            
            logger.info(f"sigma : {float(sigma_k)}")
            frac_sigma = float((sigma_k**2)/(sigma_k**2 + sigma_fit**2))
            logger.info(f"sigma_k / sigma_g: {frac_sigma}")
            #fracs.append(float(sigma_k))
            fracs.append(frac_sigma)
            
            #from IPython import embed; embed(); exit()
            
        return fracs

        
        
    def plotEverything(self):
        sensor_temps_setup = {}

        # pedestal
        pede_hb0_setup = {}
        pede_hb1_setup = {}

        # channel noise
        strip_noise_hb0_setup = {}
        strip_noise_hb0_bot_setup = {}
        strip_noise_hb0_top_setup = {}
        strip_noise_hb1_setup = {}
        strip_noise_hb1_bot_setup = {}
        strip_noise_hb1_top_setup = {}


        num_noisy_channels_hb0_setup = {}
        num_noisy_channels_hb1_setup = {}
        num_dead_channels_hb0_setup = {}
        num_dead_channels_hb1_setup = {}

        num_noisy_channels_hb0_cbc_setup  = {}
        num_noisy_channels_hb1_cbc_setup  = {}
        num_dead_channels_hb0_cbc_setup  = {}
        num_dead_channels_hb1_cbc_setup  = {}

        # common noise
        common_noise_setup = {}
        common_noise_bot_setup = {}
        common_noise_top_setup = {}
        common_noise_hb0_setup = {}
        common_noise_hb0_bot_setup = {}
        common_noise_hb0_top_setup = {}
        common_noise_hb1_setup = {}
        common_noise_hb1_bot_setup = {}
        common_noise_hb1_top_setup = {}

        common_noise_fit_hb0_cbc_setup = {}
        common_noise_fit_hb0_cbc_top_sensor_setup = {}
        common_noise_fit_hb0_cbc_bot_sensor_setup = {}

        common_noise_fit_hb1_cbc_setup = {}
        common_noise_fit_hb1_cbc_top_sensor_setup = {}
        common_noise_fit_hb1_cbc_bot_sensor_setup = {}

        common_noise_giovanni_hb0_cbc_setup = {}
        common_noise_giovanni_hb0_cbc_top_sensor_setup = {}
        common_noise_giovanni_hb0_cbc_bot_sensor_setup = {}

        common_noise_giovanni_hb1_cbc_setup = {}
        common_noise_giovanni_hb1_cbc_top_sensor_setup = {}
        common_noise_giovanni_hb1_cbc_bot_sensor_setup = {}
        
        common_noise_iphc_hb0_cbc_setup = {}
        common_noise_iphc_hb0_cbc_top_sensor_setup = {}
        common_noise_iphc_hb0_cbc_bot_sensor_setup = {}

        common_noise_iphc_hb1_cbc_setup = {}
        common_noise_iphc_hb1_cbc_top_sensor_setup = {}
        common_noise_iphc_hb1_cbc_bot_sensor_setup = {}

        common_noise_crude_hb0_cbc_setup = {}
        common_noise_crude_hb1_cbc_setup = {}

        common_noise_frac_potato_hb0_cbc_setup = {}
        common_noise_frac_potato_hb1_cbc_setup = {}        

        common_noise_frac_simfit_hb0_cbc_setup = {}
        common_noise_frac_simfit_hb1_cbc_setup = {}

        eye_cross_frac_offset_0p3_setup = {}
        eye_cross_frac_offset_0p7_setup = {}
        eye_cross_frac_offset_1p0_setup = {}
        # crossingWidthAtyMax
        eye_cross_wd_ymax_0p3_setup = {}
        eye_cross_wd_ymax_0p7_setup = {}
        eye_cross_wd_ymax_1p0_setup = {}
        # eyeOpeningArea
        eye_open_area_0p3_setup = {}
        eye_open_area_0p7_setup = {}
        eye_open_area_1p0_setup = {}
        
        # VTRx
        LightYield_total_setup = {}
        LightYield_avg_diff_setup = {}
        LightYield_mod_slope_setup = {}
        LightYield_bias_slope_setup = {}
        
        # Create ROOT file
        rootfile = f"{self.outdirF}/TB2S_Ladder__pos_{self.ladpos}__{self.ladid}.root"
        rootptr  = ROOT.TFile(rootfile, "RECREATE")
        laddir   = rootptr.mkdir("Ladder")
        #self.store_modIDs(modIDdict = ,
        #                  tdir = laddir)
        
        
        allModuleIDs = []

        for datakey, dataval in self.data.items():
            logger.info(f"Setup : {datakey}")
            """
            data:
            
            IPHC_coldBox:
              2S_18_6_KIT-10019:
                strip_noise_hb0:
                ... 
            """
            
            # Create different output dirs for different setup
            _outdir = f"{self.outdir}/{datakey}"
            if not os.path.exists(_outdir): os.mkdir(_outdir)
            #_outdirCBC = f"{_outdir}/CBCLevel"
            #if not os.path.exists(_outdirCBC): os.mkdir(_outdirCBC)

            
            
            sensor_temps_mod = []

            pede_hb0_mod = []
            pede_hb1_mod = []
            
            # initialize the dictionaries to prepare module level data
            strip_noise_hb0_mod = []
            strip_noise_hb0_bot_mod = []
            strip_noise_hb0_top_mod = []
            strip_noise_hb1_mod = []
            strip_noise_hb1_bot_mod = []
            strip_noise_hb1_top_mod = []

            num_noisy_channels_hb0_mod = []
            num_noisy_channels_hb1_mod = []
            num_dead_channels_hb0_mod = []
            num_dead_channels_hb1_mod = []
            
            num_noisy_channels_hb0_cbc_mod  = []
            num_noisy_channels_hb1_cbc_mod  = []
            num_dead_channels_hb0_cbc_mod  = []
            num_dead_channels_hb1_cbc_mod  = []
            
            # common noise
            common_noise_mod = []
            common_noise_bot_mod = []
            common_noise_top_mod = []
            common_noise_hb0_mod = []
            common_noise_hb0_bot_mod = []
            common_noise_hb0_top_mod = []
            common_noise_hb1_mod = []
            common_noise_hb1_bot_mod = []
            common_noise_hb1_top_mod = []

            common_noise_fit_hb0_cbc_mod = []
            common_noise_fit_hb0_cbc_top_mod = []
            common_noise_fit_hb0_cbc_bot_mod = []
            
            common_noise_fit_hb1_cbc_mod = []
            common_noise_fit_hb1_cbc_top_mod = []
            common_noise_fit_hb1_cbc_bot_mod = []


            common_noise_giovanni_hb0_cbc_mod = []
            common_noise_giovanni_hb0_cbc_top_mod = []
            common_noise_giovanni_hb0_cbc_bot_mod = []
            
            common_noise_giovanni_hb1_cbc_mod = []
            common_noise_giovanni_hb1_cbc_top_mod = []
            common_noise_giovanni_hb1_cbc_bot_mod = []

            common_noise_iphc_hb0_cbc_mod = []
            common_noise_iphc_hb0_cbc_top_mod = []
            common_noise_iphc_hb0_cbc_bot_mod = []
            
            common_noise_iphc_hb1_cbc_mod = []
            common_noise_iphc_hb1_cbc_top_mod = []
            common_noise_iphc_hb1_cbc_bot_mod = []

            common_noise_crude_hb0_cbc_mod = []
            common_noise_crude_hb1_cbc_mod = []

            common_noise_frac_potato_hb0_cbc_mod = []
            common_noise_frac_potato_hb1_cbc_mod = []

            common_noise_frac_simfit_hb0_cbc_mod = []
            common_noise_frac_simfit_hb1_cbc_mod = []

            # CICtoLpGBT_PhaseAlignmentEfficiency
            CICtoLpGBT_PhaseAlignmentEfficiency_mod = []
            CICtoLpGBT_BestPhase_mod = []
            CICtoLpGBT_PatternMatchingErrorRate_mod = []
            
            BERTerrorRate_mod = []
            RegisterMatchingEfficiency_mod = []

            bestDelay_mod = []
            bestTh_mod = []

            # BERT Error
            BERT_Err = []
            BERT_nbits = []
            FEC_Err = []

            # eyeCrossingFractionalOffset
            eye_cross_frac_offset_0p3_mod = []
            eye_cross_frac_offset_0p7_mod = []
            eye_cross_frac_offset_1p0_mod = []
            # crossingWidthAtyMax
            eye_cross_wd_ymax_0p3_mod = []
            eye_cross_wd_ymax_0p7_mod = []
            eye_cross_wd_ymax_1p0_mod = []
            # eyeOpeningArea
            eye_open_area_0p3_mod = []
            eye_open_area_0p7_mod = []
            eye_open_area_1p0_mod = []

            # VTRx
            LightYield_total_mod = []
            LightYield_avg_diff_mod = []
            LightYield_mod_slope_mod = []
            LightYield_bias_slope_mod = []
            
            vtrx_lightYieldScan_mod = {}
            eye_open_pow_0p3_mod = {}
            eye_open_pow_0p7_mod = {}
            eye_open_pow_1p0_mod = {}

            sen_temp_dict = {}
            
            
            # define dict to save info per module level
            moduleIDs = []
            for iOG, (moduleID, moduleDict) in enumerate(dataval.items()):
                logger.info(f"Module ID : {moduleID}")
                moduleIDs.append(moduleID)

                _outdirMod = f"{_outdir}/{moduleID}"
                if not os.path.exists(_outdirMod): os.mkdir(_outdirMod)
                
                _outdirModCBC = f"{_outdirMod}/CBCLevel"
                if not os.path.exists(_outdirModCBC): os.mkdir(_outdirModCBC)

                
                # Now we have the access to the noise dict
                """
                strip_noise_hb0:
                allCBC:
                - [6.766048431396484, 0.05587056279182434]
                - [5.9948577880859375, 0.04917684197425842]
                - ...
                CBC_0:
                - ...
                ... 
                
                """


                # ---------------------------------------------------- #
                #                         SCurve                       #
                #         EyeOpeningPower, BERT Err, FEC Err           #
                # ---------------------------------------------------- #
                scurve_dict = moduleDict['SCurve']
                self.plot_ROOT_2Dhist_AllCBCs(hist_dict=scurve_dict,
                                              name=f'SCurve_{moduleID}',
                                              title=f'SCurve_{moduleID}',
                                              outdir=_outdirModCBC)
                
                eye_open_dict = {
                    #"VTRx_LightYieldScan": moduleDict["VTRx_LightYieldScan"],
                    "LpGBT_EyeOpeningScan_Power_0.33": moduleDict["LpGBT_EyeOpeningScan_Power_0.33"],
                    "LpGBT_EyeOpeningScan_Power_0.67": moduleDict["LpGBT_EyeOpeningScan_Power_0.67"],
                    "LpGBT_EyeOpeningScan_Power_1.00": moduleDict["LpGBT_EyeOpeningScan_Power_1.00"],
                }
                self.plot_ROOT_2Dhist(hist_dict = eye_open_dict,
                                      name = f"LpGBT_EyeOpening",
                                      title = f"LpGBT_EyeOpening",
                                      outdir = _outdirMod,
                                      nRows = 1,
                                      nCols = 3)

                eye_open_pow_0p3_mod[moduleID] = moduleDict["LpGBT_EyeOpeningScan_Power_0.33"]
                eye_open_pow_0p7_mod[moduleID] = moduleDict["LpGBT_EyeOpeningScan_Power_0.67"]
                eye_open_pow_1p0_mod[moduleID] = moduleDict["LpGBT_EyeOpeningScan_Power_1.00"]
                    

                eyeCrossingFractionalOffset,crossingWidthAtyMax,eyeOpeningArea = get_eye_opening_potato_like(moduleDict["LpGBT_EyeOpeningScan_Power_0.33"])
                eye_cross_frac_offset_0p3_mod.append(eyeCrossingFractionalOffset)
                eye_cross_wd_ymax_0p3_mod.append(crossingWidthAtyMax)
                eye_open_area_0p3_mod.append(eyeOpeningArea)
                    
                eyeCrossingFractionalOffset,crossingWidthAtyMax,eyeOpeningArea = get_eye_opening_potato_like(moduleDict["LpGBT_EyeOpeningScan_Power_0.67"])
                eye_cross_frac_offset_0p7_mod.append(eyeCrossingFractionalOffset)
                eye_cross_wd_ymax_0p7_mod.append(crossingWidthAtyMax)
                eye_open_area_0p7_mod.append(eyeOpeningArea)
                
                eyeCrossingFractionalOffset,crossingWidthAtyMax,eyeOpeningArea = get_eye_opening_potato_like(moduleDict["LpGBT_EyeOpeningScan_Power_1.00"])
                eye_cross_frac_offset_1p0_mod.append(eyeCrossingFractionalOffset)
                eye_cross_wd_ymax_1p0_mod.append(crossingWidthAtyMax)
                eye_open_area_1p0_mod.append(eyeOpeningArea)
                
                
                totalBERTErrorBits,totalBitsTested,totalFECBits = bert_error_histos(moduleDict["BERTerrorRate"],
                                                                                    moduleDict["BERTtestedBitCounter"],
                                                                                    moduleDict["FECerrorCounter"])
                
                #print(totalBERTErrorBits, totalBitsTested, totalFECBits)
                BERT_Err.append(totalBERTErrorBits)
                BERT_nbits.append(totalBitsTested)
                FEC_Err.append(totalFECBits)
                
                
                # ---------------------------------------------------- #
                #                     VTRxLightYield                   #
                #                PhAlignEff, Best Delay                #
                # ---------------------------------------------------- #
                    
                self.plot_ROOT_2Dhist(hist_dict = {'VTRx_LightYieldScan': moduleDict["VTRx_LightYieldScan"]},
                                      name = f"VTRx_LightYeildScan",
                                      title = f"VTRx_LightYeildScan",
                                      outdir = _outdirMod,
                                      nRows = 1,
                                      nCols = 1)
                
                vtrx_lightYieldScan_mod[moduleID] = moduleDict["VTRx_LightYieldScan"]

                    

                totLightYield,averagePercentageDifference, modulationSlope, biasSlope = fill_vtrx_light_yield(
                    moduleDict["VTRx_LightYieldScan"]
                )
                #print(averagePercentageDifference, modulationSlope, biasSlope)
                LightYield_total_mod.append(totLightYield)
                LightYield_avg_diff_mod.append(averagePercentageDifference)
                LightYield_mod_slope_mod.append(modulationSlope)
                LightYield_bias_slope_mod.append(biasSlope)
                
                    

                    
                CICtoLpGBT_PhaseAlignmentEfficiency_mod.append(
                    list(moduleDict['CICtoLpGBT_PhaseAlignmentEfficiency']['Hybrid_0'].values()))
                CICtoLpGBT_PhaseAlignmentEfficiency_mod.append(
                    list(moduleDict['CICtoLpGBT_PhaseAlignmentEfficiency']['Hybrid_1'].values()))
                
                
                CICtoLpGBT_BestPhase_mod.append(
                    list(moduleDict['CICtoLpGBT_BestPhase']['Hybrid_0'].values()))
                CICtoLpGBT_BestPhase_mod.append(
                    list(moduleDict['CICtoLpGBT_BestPhase']['Hybrid_1'].values()))
                
                
                BERTerrorRate_mod.append(
                    list(moduleDict['BERTerrorRateData']['Hybrid_0'].values()))
                BERTerrorRate_mod.append(
                    list(moduleDict['BERTerrorRateData']['Hybrid_1'].values()))
                
                
                RegisterMatchingEfficiency_mod.append(
                    list(moduleDict['RegisterMatchingEfficiency']['Hybrid_0'].values()))
                RegisterMatchingEfficiency_mod.append(
                    list(moduleDict['RegisterMatchingEfficiency']['Hybrid_1'].values()))
                
                
                CICtoLpGBT_PatternMatchingErrorRate_mod.append(
                    list(moduleDict['CICtoLpGBT_PatternMatchingErrorRate']['Hybrid_0'].values())
                )
                CICtoLpGBT_PatternMatchingErrorRate_mod.append(
                    list(moduleDict['CICtoLpGBT_PatternMatchingErrorRate']['Hybrid_1'].values())
                )
                
                
                thVsDelayDict = moduleDict['ThresholdVsDelay']
                for ihb in range(2):
                    delay = None
                    threshold = []
                    thVsDelayDict_hb = thVsDelayDict[f'Hybrid_{ihb}']
                    for icbc in range(8):
                        temp = thVsDelayDict_hb[f'CBC_{icbc}']
                        x = temp['delay']
                        if icbc == 0:
                            delay = np.array(x)
                        y = np.array(temp['th'])
                        yerr = np.array(temp['thErr'])
                        threshold.append(np.concatenate((y[:,None], yerr[:,None]), axis=1))
                            
                    self.plot_basic(x          = delay,
                                    data_list  = threshold,
                                    legends    = ["Chip 0", "Chip 1", "Chip 2", "Chip 3",
                                                  "Chip 4", "Chip 5", "Chip 6", "Chip 7"],
                                    title      = f"DelayVsThr_Hb{ihb}_{moduleID}",
                                    name       = f"Hist_DelayVsThr_Hybrid{ihb}_{moduleID}_{datakey}",
                                    xlabel     = "Delay [ns]",
                                    ylabel     = "50% threshold [VcTh]",
                                    marker     = "o",
                                    linewidth  = 0.5,
                                    elinewidth = 0.0,
                                    capsize    = 0.0,
                                    markersize = 1.0,
                                    #ylim       = [20.0,27.0],
                                    outdir     = _outdirModCBC)
                    

                bestThDelayDict = moduleDict['BestThresholdAndDelay']
                #for ihb in range(2):
                hb0dict = bestThDelayDict[f'Hybrid_0']
                delay_hb0 = []
                th_hb0 = []
                for icbc in range(8):
                    delay_hb0.append(hb0dict[f'CBC_{icbc}']['bestDelay'])
                    th_hb0.append(hb0dict[f'CBC_{icbc}']['bestThreshold'])
                hb1dict = bestThDelayDict[f'Hybrid_1']
                delay_hb1 = []
                th_hb1 = []
                for icbc in range(8):
                    delay_hb1.append(hb0dict[f'CBC_{icbc}']['bestDelay'])
                    th_hb1.append(hb0dict[f'CBC_{icbc}']['bestThreshold'])                            
                delay_cbc = delay_hb0 + [0] + delay_hb1
                th_cbc = th_hb0 + [0] + th_hb1
                
                bestDelay_mod.append(delay_cbc)
                bestTh_mod.append(th_cbc)


                # ---------------------------------------------------- #
                #                  Sensor temperature                  #
                # ---------------------------------------------------- #
                
                time_stamps = np.array(moduleDict['time_stamps'])
                sensor_temps = np.array(moduleDict['sensor_temps'])
                sensor_temps_clean = clean_outliers(sensor_temps)
                #sensor_temps_clean = sensor_temps[sensor_temps_mask]
                #time_stamps_clean = time_stamps[sensor_temps_mask]
                    
                #sensor_temps_mod.append(float(sensor_temps[-1]))
                sensor_temps_mod.append([float(np.mean(sensor_temps_clean)), float(np.std(sensor_temps_clean))])
                #from IPython import embed; embed(); exit()
                
                sen_temp_dict[f'OpticalGroup_{iOG}'] = sensor_temps_clean
                #from IPython import embed;embed()
                        
                self.plot_basic(x          = np.arange(time_stamps.shape[0]),
                                data_list  = [np.concatenate((sensor_temps[:,None], np.zeros_like(sensor_temps)[:,None]), axis=1)],
                                legends    = ["sensor temp"],
                                title      = f"Sensor Temperature: {moduleID}",
                                name       = f"Plot_SensorTemp_{moduleID}_{datakey}",
                                #xlabel     = "Time Stamps",
                                ylabel     = "Sensor Temperature (deg C)",
                                xticklabels=time_stamps.tolist(),
                                marker     = "o",
                                linewidth  = 1.2,
                                markersize = 2.5,
                                #ylim       = [20.0,27.0],
                                outdir     = _outdirMod,
                                nticks     = 40 if time_stamps.shape[0] > 40 else None)
                    
                # ---------------------------------------------------- #
                #                     Channel Noise                    #
                # ---------------------------------------------------- #
                
                strip_noise_hb0_dict = moduleDict['strip_noise_hb0']
                strip_noise_hb0 = strip_noise_hb0_dict['allCBC']
                strip_noise_hb0_mod.append(strip_noise_hb0)
                
                strip_noise_hb0_bot_dict = moduleDict['strip_noise_hb0_bot']
                strip_noise_hb0_bot = strip_noise_hb0_bot_dict['allCBC']
                strip_noise_hb0_bot_mod.append(strip_noise_hb0_bot)
                
                strip_noise_hb0_top_dict = moduleDict['strip_noise_hb0_top']
                strip_noise_hb0_top = strip_noise_hb0_top_dict['allCBC']
                strip_noise_hb0_top_mod.append(strip_noise_hb0_top)
                
                strip_noise_hb1_dict = moduleDict['strip_noise_hb1']
                strip_noise_hb1 = strip_noise_hb1_dict['allCBC']
                strip_noise_hb1_mod.append(strip_noise_hb1)                    
                
                strip_noise_hb1_bot_dict = moduleDict['strip_noise_hb1_bot']
                strip_noise_hb1_bot = strip_noise_hb1_bot_dict['allCBC']
                strip_noise_hb1_bot_mod.append(strip_noise_hb1_bot)
                
                strip_noise_hb1_top_dict = moduleDict['strip_noise_hb1_top']
                strip_noise_hb1_top = strip_noise_hb1_top_dict['allCBC']
                strip_noise_hb1_top_mod.append(strip_noise_hb0_top)                    
                
                
                self.plot_basic(x          = np.arange(len(strip_noise_hb0)),
                                data_list  = [strip_noise_hb0, strip_noise_hb1],
                                legends    = ["hybrid 0", "hybrid 1"],
                                title      = f"StripNoise_{moduleID}",
                                name       = f"Plot_StripNoise_bothHybrids_{moduleID}_{datakey}",
                                xlabel     = "Channel",
                                ylabel     = "Noise [VcTh]",
                                ylim       = [0.0,12.0],
                                outdir     = _outdirMod)

                    
                # Plotting strip noise channel wise : [hb0_bot, hb1_bot]
                self.plot_basic(x          = np.arange(len(strip_noise_hb0_top)),
                                data_list  = [strip_noise_hb0_bot, strip_noise_hb1_bot],
                                legends    = ["hybrid 0", "hybrid 1"],
                                title      = f"StripNoise_Bottom_{moduleID}",
                                name       = f"Plot_StripNoise_bothHybrids_BottomSensor_{moduleID}_{datakey}",
                                xlabel     = "Channel",
                                ylabel     = "Noise [VcTh]",
                                ylim       = [0.0,12.0],
                                outdir     = _outdirMod)
                
                # Plotting strip noise channel wise : [hb0_top, hb1_top]
                self.plot_basic(x          = np.arange(len(strip_noise_hb0_top)),
                                data_list  = [strip_noise_hb0_top, strip_noise_hb1_top],
                                legends    = ["hybrid 0", "hybrid 1"],
                                title      = f"StripNoise_Top_{moduleID}",
                                name       = f"Plot_StripNoise_bothHybrids_TopSensor_{moduleID}_{datakey}",
                                xlabel     = "Channel",
                                ylabel     = "Noise [VcTh]",
                                ylim       = [0.0,12.0],
                                outdir     = _outdirMod)
                
                
                # Plotting hist for stripNoise : [hb0_bot, hb0_top, hb1_bot, hb1_top]
                self.hist_basic(bins       = np.linspace(2,10,80),
                                data_list  = [strip_noise_hb0_bot, strip_noise_hb0_top, strip_noise_hb1_bot, strip_noise_hb1_top],
                                legends    = ["hybrid 0 bottom", "hybrid 0 top", "hybrid 1 bottom", "hybrid 1 top"],
                                title      = f"StripNoise_{moduleID}",
                                name       = f"Hist_StripNoise_bothHybrids_bothSensors_{moduleID}_{datakey}",
                                xlabel     = "Noise [VcTh]",
                                ylabel     = "Entries",
                                linewidth  = 2,
                                outdir     = _outdirMod,
                                colors     = ["#165a86","#cc660b","#165a86","#cc660b"],
                                linestyles = ["-","-","--","--"])
                
                
                # Plotting hist for stripNoise per CBC : [hb0 cbc level]
                self.hist_basic(bins       = np.linspace(2,10,80),
                                data_list  = [strip_noise_hb0_dict['CBC_0'],
                                              strip_noise_hb0_dict['CBC_1'],
                                              strip_noise_hb0_dict['CBC_2'],
                                              strip_noise_hb0_dict['CBC_3'],
                                              strip_noise_hb0_dict['CBC_4'],
                                              strip_noise_hb0_dict['CBC_5'],
                                              strip_noise_hb0_dict['CBC_6'],
                                              strip_noise_hb0_dict['CBC_7']],
                                legends    = ["Chip 0", "Chip 1", "Chip 2", "Chip 3", "Chip 4", "Chip 5", "Chip 6", "Chip 7"],
                                title      = f"StripNoise_Hb0_{moduleID}",
                                name       = f"Hist_StripNoiseCBC_Hybrid0_bothSensors_{moduleID}_{datakey}",
                                xlabel     = "Noise [VcTh]",
                                ylabel     = "Entries",
                                linewidth  = 1.2,
                                outdir     = _outdirModCBC,
                                colors     = ["#0B1A2F", "#152C4D", "#1F3E6C", "#29508A", "#3362A9", "#3D74C7", "#4796E6", "#61B4FF"],
                                linestyles = 8*["-"])
                
                
                
                # ------------------------------------------------------------------ #
                #                Get Noisy & Dead channel fractions                  #
                # ------------------------------------------------------------------ #
                
                # per hybrid
                hb0_noisy_channels, hb0_dead_channels = get_noisy_and_dead_channels(strip_noise_hb0)
                num_noisy_channels_hb0_mod.append(len(hb0_noisy_channels))
                num_dead_channels_hb0_mod.append(len(hb0_dead_channels))
                
                hb1_noisy_channels, hb1_dead_channels = get_noisy_and_dead_channels(strip_noise_hb1)
                num_noisy_channels_hb1_mod.append(len(hb1_noisy_channels))
                num_dead_channels_hb1_mod.append(len(hb1_dead_channels))
                
                # per CBC
                noisy_ch_list_hb0, dead_ch_list_hb0 = get_noisy_and_dead_channels_cbc(strip_noise_hb0_dict)
                noisy_ch_list_hb1, dead_ch_list_hb1 = get_noisy_and_dead_channels_cbc(strip_noise_hb1_dict)
                
                num_noisy_channels_hb0_cbc_mod.append(noisy_ch_list_hb0)
                num_dead_channels_hb0_cbc_mod.append(dead_ch_list_hb0)
                num_noisy_channels_hb1_cbc_mod.append(noisy_ch_list_hb1)
                num_dead_channels_hb1_cbc_mod.append(dead_ch_list_hb1)
                
                #from IPython import embed; embed()
                
                
                # Plotting hist for stripNoise per CBC : [hb1 cbc level]
                self.hist_basic(bins       = np.linspace(2,10,80),
                                data_list  = [strip_noise_hb1_dict['CBC_0'],
                                              strip_noise_hb1_dict['CBC_1'],
                                              strip_noise_hb1_dict['CBC_2'],
                                              strip_noise_hb1_dict['CBC_3'],
                                              strip_noise_hb1_dict['CBC_4'],
                                              strip_noise_hb1_dict['CBC_5'],
                                              strip_noise_hb1_dict['CBC_6'],
                                              strip_noise_hb1_dict['CBC_7']],
                                legends    = ["Chip 0", "Chip 1", "Chip 2", "Chip 3", "Chip 4", "Chip 5", "Chip 6", "Chip 7"],
                                title      = f"StripNoise_Hb1_{moduleID}",
                                name       = f"Hist_StripNoiseCBC_Hybrid1_bothSensors_{moduleID}_{datakey}",
                                xlabel     = "Noise [VcTh]",
                                ylabel     = "Entries",
                                linewidth  = 1.2,
                                outdir     = _outdirModCBC,
                                colors     = ["#2E0000", "#4A0A0A", "#661515", "#821F1F", "#9E2A2A", "#BA3535", "#D65050", "#F26B6B"],
                                linestyles = 8*["-"])
                
                
                # Get the mean and std of noise per CBC
                strip_noise_hb0_bot_per_cbc_mean_std_list = get_mean_std_per_cbc(strip_noise_hb0_bot_dict)
                strip_noise_hb0_top_per_cbc_mean_std_list = get_mean_std_per_cbc(strip_noise_hb0_top_dict)
                #
                self.plot_group(x           = np.arange(8),
                                data_list   = [[strip_noise_hb0_bot_per_cbc_mean_std_list,
                                                strip_noise_hb0_top_per_cbc_mean_std_list]],
                                legends     = [["Bottom Sensor",
                                                "Top Sensor"]],
                                title       = f"StripNoise_Hb0_{moduleID}",
                                name        = f"Plot_StripNoise_perCBC_Hybrid0_bothSensors_{moduleID}_{datakey}",
                                xticklabels = [f"CBC_{i}" for i in range(8)],
                                ylim        = [2.0,10.0],
                                ylabel      = "Noise [VcTh]",
                                outdir      = _outdirModCBC,
                                marker      = "o",
                                markersize  = 2.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0)
                
                strip_noise_hb1_bot_per_cbc_mean_std_list = get_mean_std_per_cbc(strip_noise_hb1_bot_dict)
                strip_noise_hb1_top_per_cbc_mean_std_list = get_mean_std_per_cbc(strip_noise_hb1_top_dict)
                self.plot_group(x           = np.arange(8),
                                data_list   = [[strip_noise_hb1_bot_per_cbc_mean_std_list,
                                                strip_noise_hb1_top_per_cbc_mean_std_list]],
                                legends     = [["Bottom Sensor",
                                                "Top Sensor"]],
                                title       = f"StripNoise_Hb1_{moduleID}",
                                name        = f"Plot_StripNoise_perCBC_Hybrid1_bothSensors_{moduleID}_{datakey}",
                                xticklabels = [f"CBC_{i}" for i in range(8)],
                                ylim        = [2.0,10.0],
                                ylabel      = "Noise [VcTh]",
                                outdir      = _outdirModCBC,
                                marker      = "o",
                                markersize  = 2.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0)
                
                
                strip_noise_hb0_per_cbc_mean_std_list = get_mean_std_per_cbc(strip_noise_hb0_dict)
                strip_noise_hb1_per_cbc_mean_std_list = get_mean_std_per_cbc(strip_noise_hb1_dict)
                self.plot_group(x           = np.arange(8),
                                data_list   = [[strip_noise_hb0_per_cbc_mean_std_list,
                                                strip_noise_hb1_per_cbc_mean_std_list]],
                                legends     = [["Hybrid 0",
                                                "Hybrid 1"]],
                                title       = f"StripNoise_{moduleID}",
                                name        = f"Plot_StripNoise_perCBC_bothHybrids_bothSensors_{moduleID}_{datakey}",
                                xticklabels = [f"CBC_{i}" for i in range(8)],
                                ylim        = [2.0,10.0],
                                ylabel      = "Noise [VcTh]",
                                outdir      = _outdirModCBC,
                                marker      = "o",
                                markersize  = 2.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0)
                



                # ---------------------------------------------------- #
                #                 Common Mode Noise                    #
                # ---------------------------------------------------- #
                    
                if self.testinfo.get("check_common_noise") == True:
                    # module level
                    common_noise_module = moduleDict['common_noise_module']
                    common_noise_mod.append(common_noise_module)
                    # module (bottom sensor)
                    common_noise_module_bot = moduleDict['common_noise_module_bot']
                    common_noise_bot_mod.append(common_noise_module_bot)
                    # module (top sensor)
                    common_noise_module_top = moduleDict['common_noise_module_top']
                    common_noise_top_mod.append(common_noise_module_top)
                    
                    # hb-0
                    common_noise_hb0_dict = moduleDict['common_noise_hb0']
                    common_noise_hb0 = common_noise_hb0_dict['allCBC']
                    common_noise_hb0_mod.append(common_noise_hb0)
                    # hb-0 (bottom)
                    common_noise_hb0_bot_dict = moduleDict['common_noise_hb0_bot']
                    common_noise_hb0_bot = common_noise_hb0_bot_dict['allCBC']
                    common_noise_hb0_bot_mod.append(common_noise_hb0_bot)
                    # hb-0 (top)
                    common_noise_hb0_top_dict = moduleDict['common_noise_hb0_top']
                    common_noise_hb0_top = common_noise_hb0_top_dict['allCBC']
                    common_noise_hb0_top_mod.append(common_noise_hb0_top)
                    # hb-1
                    common_noise_hb1_dict = moduleDict['common_noise_hb1']
                    common_noise_hb1 = common_noise_hb1_dict['allCBC']
                    common_noise_hb1_mod.append(common_noise_hb1)
                    # hb-1 (bottom)
                    common_noise_hb1_bot_dict = moduleDict['common_noise_hb1_bot']
                    common_noise_hb1_bot = common_noise_hb1_bot_dict['allCBC']
                    common_noise_hb1_bot_mod.append(common_noise_hb1_bot)
                    # hb-1 (top)
                    common_noise_hb1_top_dict = moduleDict['common_noise_hb1_top']
                    common_noise_hb1_top = common_noise_hb1_top_dict['allCBC']
                    common_noise_hb1_top_mod.append(common_noise_hb1_top)
                    
                    
                    # Plotting common noise (for whole module)
                    self.plot_basic(x          = np.arange(len(common_noise_module)),
                                    data_list  = [common_noise_module],
                                    legends    = ["Module"],
                                    title      = f"CMNoise_{moduleID}",
                                    name       = f"Plot_CommonModeNoise_module_{moduleID}_{datakey}",
                                    xlabel     = "Number of hits",
                                    ylabel     = "Number of events",
                                    #ylim       = [0.0,12.0],
                                    outdir     = _outdirMod,
                                    linewidth  = 0.0,
                                    markersize = 0.5,
                                    capsize    = 0.0,
                                    elinewidth = 0.05,
                                    fitlinewidth = 2.0,
                                    fit        = self.testinfo.get("fit_common_noise"),
                                    #fitmodel   = self.__gauss_model,
                                    #fitfunc    = self.__gauss_fit,
                                    #mean_init  = 1500.0,
                                    #sigma_init = 40.0
                                    )
                    # Plotting common noise (top and bot sensors at module level)
                    self.plot_basic(x          = np.arange(len(common_noise_module_bot)),
                                    data_list  = [common_noise_module_bot, common_noise_module_top],
                                    legends    = ["Bottom Sensor", "Top Sensor"],
                                    title      = f"CMNoise_{moduleID}",
                                    name       = f"Plot_CommonModeNoise_module_bothSensors_{moduleID}_{datakey}",
                                    xlabel     = "Number of hits",
                                    ylabel     = "Number of events",
                                    #ylim       = [0.0,12.0],
                                    outdir     = _outdirMod,
                                    linewidth  = 0.0,
                                    markersize = 0.5,
                                    capsize    = 0.0,
                                    elinewidth = 0.05,
                                    fitlinewidth = 2.0,
                                    fit        = self.testinfo.get("fit_common_noise"),
                                    #fitmodel   = self.__gauss_model,
                                    #fitfunc    = self.__gauss_fit,
                                    #mean_init  = 800.0,
                                    #sigma_init = 30.0
                                    )
                    # Plotting common noise (hybrids at module level)
                    self.plot_basic(x          = np.arange(len(common_noise_hb0)),
                                    data_list  = [common_noise_hb0, common_noise_hb1],
                                    legends    = ["Hybrid 0", "Hybrid 1"],
                                    title      = f"CMNoise_{moduleID}",
                                    name       = f"Plot_CommonModeNoise_module_bothHybrids_{moduleID}_{datakey}",
                                    xlabel     = "Number of hits",
                                    ylabel     = "Number of events",
                                    #ylim       = [0.0,12.0],
                                    outdir     = _outdirMod,
                                    linewidth  = 0.0,
                                    markersize = 0.5,
                                    capsize    = 0.0,
                                    elinewidth = 0.05,
                                    fitlinewidth = 2.0,
                                    fit        = self.testinfo.get("fit_common_noise"),
                                    #fitmodel   = self.__gauss_model,
                                    #fitfunc    = self.__gauss_fit,
                                    #mean_init  = 800.0,
                                    #sigma_init = 100.0
                                    )
                    # Plotting common noise (Top sensor)
                    self.plot_basic(x          = np.arange(len(common_noise_hb0_top)),
                                    data_list  = [common_noise_hb0_top, common_noise_hb1_top],
                                    legends    = ["Hybrid 0", "Hybrid 1"],
                                    title      = f"CMNoise_Top_{moduleID}",
                                    name       = f"Plot_CommonModeNoise_bothHybrids_topSensor_{moduleID}_{datakey}",
                                    xlabel     = "Number of hits",
                                    ylabel     = "Number of events",
                                    #ylim       = [0.0,12.0],
                                    linewidth  = 0.0,
                                    markersize = 0.5,
                                    capsize    = 0.0,
                                    elinewidth = 0.05,
                                    fitlinewidth = 2.0,
                                    outdir     = _outdirMod,
                                    fit        = self.testinfo.get("fit_common_noise"),
                                    #fitmodel   = self.__gauss_model,
                                    #fitfunc    = self.__gauss_fit
                                    )
                    # Plotting common noise (Bottom sensor)
                    self.plot_basic(x          = np.arange(len(common_noise_hb0_bot)),
                                    data_list  = [common_noise_hb0_bot, common_noise_hb1_bot],
                                    legends    = ["Hybrid 0", "Hybrid 1"],
                                    title      = f"CMNoise_Bot_{moduleID}",
                                    name       = f"Plot_CommonModeNoise_bothHybrids_bottomSensor_{moduleID}_{datakey}",
                                    xlabel     = "Number of hits",
                                    ylabel     = "Number of events",
                                    #ylim       = [0.0,12.0],
                                    outdir     = _outdirMod,
                                    linewidth  = 0.0,
                                    markersize = 0.5,
                                    capsize    = 0.0,
                                    elinewidth = 0.05,
                                    fitlinewidth = 2.0,
                                    fit        = self.testinfo.get("fit_common_noise"),
                                    #fitmodel   = self.__gauss_model,
                                    #fitfunc    = self.__gauss_fit
                                    )
                    

                    # Plotting hist for stripNoise per CBC : [hb0 cbc level]
                    self.plot_basic(x          = np.arange(len(common_noise_hb0_dict["CBC_0"])),
                                    data_list  = [common_noise_hb0_dict['CBC_0'],
                                                  common_noise_hb0_dict['CBC_1'],
                                                  common_noise_hb0_dict['CBC_2'],
                                                  common_noise_hb0_dict['CBC_3'],
                                                  common_noise_hb0_dict['CBC_4'],
                                                  common_noise_hb0_dict['CBC_5'],
                                                  common_noise_hb0_dict['CBC_6'],
                                                  common_noise_hb0_dict['CBC_7']],
                                    legends    = [f"CBC{i}" for i in range(8)],
                                    title      = f"CMNoiseCBC_hb0_{moduleID}",
                                    name       = f"Plot_CommonNoiseDistributionCBC_Hybrid0_bothSensors_{moduleID}_{datakey}",
                                    xlabel     = "Number of hits",
                                    ylabel     = "Number of events",
                                    outdir     = _outdirModCBC,
                                    colors     = ["#0B1A2F", "#152C4D", "#1F3E6C", "#29508A", "#3362A9", "#3D74C7", "#4796E6", "#61B4FF"],
                                    linestyles = 8*["-"],
                                    linewidth  = 1.0,
                                    markersize = 0.3,
                                    capsize    = 0.0,
                                    elinewidth = 0.2,
                                    fit        = self.testinfo.get("fit_common_noise"),
                                    #fitmodel   = self.__gauss_model,
                                    #fitfunc    = self.__gauss_fit,
                                    fitlinewidth = 0.5,
                                    #mean_init  = 100.0,
                                    #sigma_init = 20.0
                                    )
                    
                    self.plot_basic(x          = np.arange(len(common_noise_hb1_dict["CBC_0"])),
                                    data_list  = [common_noise_hb1_dict['CBC_0'],
                                                  common_noise_hb1_dict['CBC_1'],
                                                  common_noise_hb1_dict['CBC_2'],
                                                  common_noise_hb1_dict['CBC_3'],
                                                  common_noise_hb1_dict['CBC_4'],
                                                  common_noise_hb1_dict['CBC_5'],
                                                  common_noise_hb1_dict['CBC_6'],
                                                  common_noise_hb1_dict['CBC_7']],
                                    legends    = [f"CBC{i}" for i in range(8)],
                                    title      = f"CMNoiseCBC_hb1_{moduleID}",
                                    name       = f"Plot_CommonNoiseDistributionCBC_Hybrid1_bothSensors_{moduleID}_{datakey}",
                                    xlabel     = "Number of hits",
                                    ylabel     = "Number of events",
                                    outdir     = _outdirModCBC,
                                    colors     = ["#2E0000", "#4A0A0A", "#661515", "#821F1F", "#9E2A2A", "#BA3535", "#D65050", "#F26B6B"],
                                    linestyles = 8*["-"],
                                    linewidth  = 1.0,
                                    markersize = 0.3,
                                    capsize    = 0.0,
                                    elinewidth = 0.2,
                                    fit        = self.testinfo.get("fit_common_noise"),
                                    #fitmodel   = self.__gauss_model,
                                    #fitfunc    = self.__gauss_fit,
                                    fitlinewidth = 0.5,
                                    #mean_init  = 100.0,
                                    #sigma_init = 20.0
                                    )
                    

                    
                    # Get the mean and std of noise per CBC
                    common_noise_hb0_per_cbc_mean_std_list = get_cmn_mean_std_per_cbc(common_noise_hb0_dict)
                    common_noise_hb1_per_cbc_mean_std_list = get_cmn_mean_std_per_cbc(common_noise_hb1_dict)
                    self.plot_group(x           = np.arange(8),
                                    data_list   = [[common_noise_hb0_per_cbc_mean_std_list,
                                                    common_noise_hb1_per_cbc_mean_std_list]],
                                    legends     = [["hb0", "hb1"]],
                                    title       = f"CMNoise_{moduleID}",
                                    name        = f"Plot_CommonNoiseCBC_bothHybrids_{moduleID}_{datakey}",
                                    xticklabels = [f"CBC_{i}" for i in range(8)],
                                    ylim        = [50, 200],
                                    ylabel      = "Common Noise",
                                    outdir      = _outdirModCBC,
                                    marker      = "o",
                                    markersize  = 2.5,
                                    capsize     = 1.5,
                                    elinewidth  = 1.0)
                    common_noise_hb0_bot_per_cbc_mean_std_list = get_cmn_mean_std_per_cbc(common_noise_hb0_bot_dict)
                    common_noise_hb0_top_per_cbc_mean_std_list = get_cmn_mean_std_per_cbc(common_noise_hb0_top_dict)                    
                    common_noise_hb1_bot_per_cbc_mean_std_list = get_cmn_mean_std_per_cbc(common_noise_hb1_bot_dict)
                    common_noise_hb1_top_per_cbc_mean_std_list = get_cmn_mean_std_per_cbc(common_noise_hb1_top_dict)
                    self.plot_group(x           = np.arange(8),
                                    data_list   = [[common_noise_hb0_bot_per_cbc_mean_std_list, common_noise_hb1_bot_per_cbc_mean_std_list,
                                                    common_noise_hb0_top_per_cbc_mean_std_list, common_noise_hb1_top_per_cbc_mean_std_list]],
                                    legends     = [["hybrid 0 bottom", "hybrid 1 bottom",
                                                    "hybrid 0 top", "hybrid 1 top"]],
                                    title       = f"CMNoise_{moduleID}",
                                    name        = f"Plot_CommonNoiseCBC_bothHybrids_bothSensors_{moduleID}_{datakey}",
                                    xticklabels = [f"CBC_{i}" for i in range(8)],
                                    ylim        = [20, 100],
                                    ylabel      = "Common Noise",
                                    outdir      = _outdirModCBC,
                                    marker      = "o",
                                    markersize  = 2.5,
                                    capsize     = 1.5,
                                    elinewidth  = 1.0)

                    
                    
                    common_noise_giovanni_hb0_cbc_mod.append(extractCMN_giovanni(nchannels=254,
                                                                                 mean=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,0],
                                                                                 std=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,1]).tolist())
                    common_noise_giovanni_hb1_cbc_mod.append(extractCMN_giovanni(nchannels=254,
                                                                                 mean=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,0],
                                                                                 std=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,1]).tolist())
                    
                    common_noise_iphc_hb0_cbc_mod.append(extractCMN_iphc(nchannels=254,
                                                                         mean=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,0],
                                                                         std=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,1]).tolist())
                    common_noise_iphc_hb1_cbc_mod.append(extractCMN_iphc(nchannels=254,
                                                                         mean=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,0],
                                                                         std=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,1]).tolist())
                    
                    common_noise_crude_hb0_cbc_mod.append(extractCMN_crude(nchannels=254,
                                                                           mean=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,0],
                                                                           std=np.array(common_noise_hb0_per_cbc_mean_std_list)[:,1]).tolist())
                    common_noise_crude_hb1_cbc_mod.append(extractCMN_crude(nchannels=254,
                                                                           mean=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,0],
                                                                           std=np.array(common_noise_hb1_per_cbc_mean_std_list)[:,1]).tolist())
                    
                    

                    #from IPython import embed; embed(); exit()
                    
                    
                    append_fit_result = lambda target, source: target.append(
                        [
                            source[f'CBC_{i}_fit_params']['cmnFraction']
                            for i in range(8)
                            if f'CBC_{i}_fit_params' in source.keys()
                        ]
                    )
                    # saving the cmn from fit per CBC
                    #common_noise_fit_hb0_cbc_mod.append([common_noise_hb0_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                    append_fit_result(common_noise_fit_hb0_cbc_mod, common_noise_hb0_dict)
                    append_fit_result(common_noise_fit_hb0_cbc_bot_mod, common_noise_hb0_bot_dict)
                    append_fit_result(common_noise_fit_hb0_cbc_top_mod, common_noise_hb0_top_dict)
                    append_fit_result(common_noise_fit_hb1_cbc_mod, common_noise_hb1_dict)
                    append_fit_result(common_noise_fit_hb1_cbc_bot_mod, common_noise_hb1_bot_dict)
                    append_fit_result(common_noise_fit_hb1_cbc_top_mod, common_noise_hb1_top_dict)
                    
                    append_cmn_potato_result = lambda target, source: target.append(
                        [
                            extractCMN_potato(source[f'CBC_{i}']) for i in range(8)
                        ]
                    )
                    append_cmn_potato_result(common_noise_frac_potato_hb0_cbc_mod, common_noise_hb0_dict)
                    append_cmn_potato_result(common_noise_frac_potato_hb1_cbc_mod, common_noise_hb1_dict)
                    
                    #common_noise_fit_hb0_cbc_bot_mod[martaTemp].append([common_noise_hb0_bot_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                    #common_noise_fit_hb0_cbc_bot_mod[martaTemp].append(
                    #    [
                    #        common_noise_hb0_bot_dict[f'CBC_{i}_fit_params']['cmnFraction']
                    #        for i in range(8)
                    #        
                    #common_noise_fit_hb0_cbc_top_mod[martaTemp].append([common_noise_hb0_top_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                    #common_noise_fit_hb1_cbc_mod[martaTemp].append([common_noise_hb1_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                    #common_noise_fit_hb1_cbc_bot_mod[martaTemp].append([common_noise_hb1_bot_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                    #common_noise_fit_hb1_cbc_top_mod[martaTemp].append([common_noise_hb1_top_dict[f'CBC_{i}_fit_params']['cmnFraction'] for i in range(8)])
                    
                    
                    if self.testinfo.get("fit_simultaneous_common_noise") == True:
                        common_noise_3sigma_hb0_dict = moduleDict['common_3sigma_noise_hb0']
                        common_noise_3sigma_hb1_dict = moduleDict['common_3sigma_noise_hb1']
                        
                        cmn_res_hb0 = self.simfit_and_analyse(common_noise_hb0_dict,
                                                              common_noise_3sigma_hb0_dict,
                                                              labels=["nHits : 0 sigma", "nHits : 3 sigma", "CMNoise"],
                                                              title = f"CMN_fitted_hb0_{moduleID}",
                                                              name = f"Plot_CommonNoiseCBC_hb0_bothSensors_{moduleID}_{datakey}",
                                                              outdir = _outdirModCBC)
                        common_noise_frac_simfit_hb0_cbc_mod.append(cmn_res_hb0)
                        
                        cmn_res_hb1 = self.simfit_and_analyse(common_noise_hb1_dict,
                                                              common_noise_3sigma_hb1_dict,
                                                              labels=["nHits : 0 sigma", "nHits : 3 sigma", "CMNoise"],
                                                              title = f"CMN_fitted_hb1_{moduleID}",
                                                              name = f"Plot_CommonNoiseCBC_hb1_bothSensors_{moduleID}_{datakey}",
                                                              outdir = _outdirModCBC)
                        
                        common_noise_frac_simfit_hb1_cbc_mod.append(cmn_res_hb1)
                        
                        #from IPython import embed; embed()
                            
                        """
                        # loop over CBCs
                        for i in range(8):
                        fracs = []
                        nHits_0sigma = np.array(common_noise_hb0_dict[f'CBC_{i}'])[:,0]
                        sigmaHits_0sigma = np.array(common_noise_hb0_dict[f'CBC_{i}'])[:,1]
                        nHits_3sigma = np.array(common_noise_3sigma_hb0_dict[f'CBC_{i}'])[:,0]
                        sigmaHits_3sigma = np.array(common_noise_3sigma_hb0_dict[f'CBC_{i}'])[:,1]
                        
                        logger.info(f"fitting nHits for CBC_{i}")
                        sigma_fit, k_probs_fit, res = CMNmod.fit_k_and_sigma_from_hists(nHits_0sigma,
                        nHits_3sigma,
                        sigmaHits_0sigma,
                        sigmaHits_3sigma)
                        
                        P_A, P_B, R_A, R_B = CMNmod.predict_distributions(sigma_fit, k_probs_fit)
                        expA = 10000 * P_A
                        expB = 10000 * P_B
                        #from IPython import embed; embed(); exit()
                        
                        N_BINS_K = 20
                        K_MIN, K_MAX = -2.0, 2.0
                        BIN_EDGES = np.linspace(K_MIN, K_MAX, N_BINS_K + 1)
                        BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
                        BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]
                        
                        
                        x = [ [np.arange(nHits_0sigma.shape[0]), np.arange(len(expA))] ,
                        [np.arange(nHits_3sigma.shape[0]), np.arange(len(expB))] ,
                        [BIN_CENTERS] ]
                        y = [ [nHits_0sigma/10000, P_A] ,
                        [nHits_3sigma/10000, P_B] ,
                        [k_probs_fit] ]
                        h3_width = BIN_WIDTH*0.9
                        
                        self.plot_fitted_result(x = x,
                        y = y,
                        w = h3_width,
                        labels=["nHits : 0 sigma", "nHits : 3 sigma", "CMNoise"],
                        title = f"CMN_fitted_hb0_{moduleID}_CBC{i}",
                        name = f"Plot_CommonNoiseCBC_hb0_bothSensors_{moduleID}_{datakey}_CBC_{i}",
                        outdir = _outdirCBC)
                        
                        from IPython import embed; embed()
                        fracs.append(R_A['frac_common'])
                        
                        common_noise_frac_simfit_hb0_cbc_mod[martaTemp].append(fracs)
                        """
                    else:
                        logger.warning("Skip common noise extraction by fitting nHits simultaneously for 0 & 3 sigma noise")

                            
                else:
                    logger.warning("skip plotting common mode noise")



                pede_hb0_dict = moduleDict['pedestal_hb0']
                pede_hb1_dict = moduleDict['pedestal_hb1']
                # Total pede
                pede_hb0 = pede_hb0_dict['CBC_0'] + pede_hb0_dict['CBC_1'] + pede_hb0_dict['CBC_2'] + pede_hb0_dict['CBC_3'] + pede_hb0_dict['CBC_4'] + pede_hb0_dict['CBC_5'] + pede_hb0_dict['CBC_6'] + pede_hb0_dict['CBC_7']
                pede_hb0_mod.append(pede_hb0)
                pede_hb1 = pede_hb1_dict['CBC_0'] + pede_hb1_dict['CBC_1'] + pede_hb1_dict['CBC_2'] + pede_hb1_dict['CBC_3'] + pede_hb1_dict['CBC_4'] + pede_hb1_dict['CBC_5'] + pede_hb1_dict['CBC_6'] + pede_hb1_dict['CBC_7']
                pede_hb1_mod.append(pede_hb1)
                
                self.plot_basic(x          = np.arange(len(pede_hb0)),
                                data_list  = [pede_hb0, pede_hb1],
                                legends    = ["hybrid 0", "hybrid 1"],
                                title      = f"Pedestal_{moduleID}",
                                name       = f"Plot_Pedestal_bothHybrids_{moduleID}_{datakey}",
                                xlabel     = "Channel",
                                ylabel     = "Pedestal [VcTh]",
                                ylim       = [595.0,605.0],
                                outdir     = _outdirMod)
                
                self.hist_basic(bins       = np.linspace(590,610,80),
                                data_list  = [pede_hb0_dict['CBC_0'],
                                              pede_hb0_dict['CBC_1'],
                                              pede_hb0_dict['CBC_2'],
                                              pede_hb0_dict['CBC_3'],
                                              pede_hb0_dict['CBC_4'],
                                              pede_hb0_dict['CBC_5'],
                                              pede_hb0_dict['CBC_6'],
                                              pede_hb0_dict['CBC_7']],
                                legends    = ["Chip 0", "Chip 1", "Chip 2", "Chip 3", "Chip 4", "Chip 5", "Chip 6", "Chip 7"],
                                title      = f"PedestalCBC_Hb0_{moduleID}",
                                name       = f"Hist_PedestalCBC_Hybrid0_bothSensors_{moduleID}_{datakey}",
                                xlabel     = "Pedestal [VcTh]",
                                ylabel     = "Entries",
                                linewidth  = 1.2,
                                outdir     = _outdirModCBC,
                                colors     = ["#0B1A2F", "#152C4D", "#1F3E6C", "#29508A", "#3362A9", "#3D74C7", "#4796E6", "#61B4FF"],
                                linestyles = 8*["-"])
                self.hist_basic(bins       = np.linspace(590,610,80),
                                data_list  = [pede_hb1_dict['CBC_0'],
                                              pede_hb1_dict['CBC_1'],
                                              pede_hb1_dict['CBC_2'],
                                              pede_hb1_dict['CBC_3'],
                                              pede_hb1_dict['CBC_4'],
                                              pede_hb1_dict['CBC_5'],
                                              pede_hb1_dict['CBC_6'],
                                              pede_hb1_dict['CBC_7']],
                                legends    = ["Chip 0", "Chip 1", "Chip 2", "Chip 3", "Chip 4", "Chip 5", "Chip 6", "Chip 7"],
                                title      = f"PedestalCBC_Hb1_{moduleID}",
                                name       = f"Hist_PedestalCBC_Hybrid1_bothSensors_{moduleID}_{datakey}",
                                xlabel     = "Pedestal [VcTh]",
                                ylabel     = "Entries",
                                linewidth  = 1.2,
                                outdir     = _outdirModCBC,
                                colors     = ["#2E0000", "#4A0A0A", "#661515", "#821F1F", "#9E2A2A", "#BA3535", "#D65050", "#F26B6B"],
                                linestyles = 8*["-"])
                    
                        
                        
                # Get the mean and std of noise per CBC
                pede_hb0_per_cbc_mean_std_list = get_mean_std_per_cbc(pede_hb0_dict)
                pede_hb1_per_cbc_mean_std_list = get_mean_std_per_cbc(pede_hb1_dict)


                #from IPython import embed; embed()  
                
                self.plot_group(x           = np.arange(8),
                                data_list   = [[pede_hb0_per_cbc_mean_std_list,
                                                pede_hb1_per_cbc_mean_std_list]],
                                legends     = [["hybrid 0",
                                                "hybrid 1"]],
                                title       = f"Pedestal_{moduleID}",
                                name        = f"Plot_PedestalCBC_bothHybrids_bothSensors_{moduleID}_{datakey}",
                                xticklabels = [f"CBC_{i}" for i in range(8)],
                                ylim        = [595, 605],
                                ylabel      = "Pedestal [VcTh]",
                                outdir      = _outdirModCBC,
                                marker      = "o",
                                markersize  = 2.5,
                                capsize     = 1.5,
                                elinewidth  = 1.0)
                    

            # Loop over modules end here
            allModuleIDs += moduleIDs
            #from IPython import embed; embed()
            
            self.plot_ROOT_2Dhist(hist_dict = vtrx_lightYieldScan_mod,
                                  name = f"VTRx_LightYield_Ladder",
                                  title = f"VTRx_LightYield_Ladder",
                                  outdir = _outdir,
                                  nRows  = 2,
                                  nCols  = 6)
            self.plot_ROOT_2Dhist(hist_dict = eye_open_pow_0p3_mod,
                                  name = f"Eye_Opening_Scan_Pow_0p33",
                                  title = f"EyeOpenScan_Pow0p33",
                                  outdir = _outdir,
                                  nRows  = 2,
                                  nCols  = 6,
                                  pType = "COLZ")
            self.plot_ROOT_2Dhist(hist_dict = eye_open_pow_0p7_mod,
                                  name = f"Eye_Opening_Scan_Pow_0p67",
                                  title = f"EyeOpenScan_Pow0p67",
                                  outdir = _outdir,
                                  nRows  = 2,
                                  nCols  = 6,
                                  pType = "COLZ")
            self.plot_ROOT_2Dhist(hist_dict = eye_open_pow_1p0_mod,
                                  name = f"Eye_Opening_Scan_Pow_1p00",
                                  title = f"EyeOpenScan_Pow1p00",
                                  outdir = _outdir,
                                  nRows  = 2,
                                  nCols  = 6,
                                  pType = "COLZ")

            
            logger.info("Plotting Noise per Module ...")

            if self.testinfo.get("check_extra") == True:

                self.plot_heatmap(CICtoLpGBT_PhaseAlignmentEfficiency_mod,
                                  title       = f"CICtoLpGBT-PhaseAlignEff",
                                  name        = f"Plot_CICtoLpGBT_PhaseAlignmentEfficiency_{datakey}",
                                  xticklabels = [f'{id}-hb{j}' for id in moduleIDs for j in range(2)],
                                  yticklabels = ['L1', 'Stub0', 'Stub1', 'Stub2', 'Stub3', 'Stub4'],
                                  colmap      = "RdYlGn",
                                  cb_label    = "Efficiency",
                                  vmin        = 0.0,
                                  vmax        = 1.0,
                                  outdir      = _outdir)

                self.plot_heatmap(CICtoLpGBT_BestPhase_mod,
                                  title       = f"CICtoLpGBT-BestPhase",
                                  name        = f"Plot_CICtoLpGBT_BestPhase_{datakey}",
                                  xticklabels = [f'{id}-hb{j}' for id in moduleIDs for j in range(2)],
                                  yticklabels = ['L1', 'Stub0', 'Stub1', 'Stub2', 'Stub3', 'Stub4'],
                                  colmap      = mcolors.ListedColormap(["#A8D5BA", "#D9534F"]),
                                  norm        = mcolors.BoundaryNorm([0, 15, 16], 2),
                                  cb_label    = "BestPhase",
                                  #vmin        = 0,
                                  #vmax        = 15,
                                  outdir      = _outdir,
                                  annotint    = True)

                self.plot_heatmap(BERTerrorRate_mod,
                                  title       = f"LpGBTtoFPGA-BERTErrorRate",
                                  name        = f"Plot_LpGBTtoFPGA_BERTErrorRate_{datakey}",
                                  xticklabels = [f'{id}-hb{j}' for id in moduleIDs for j in range(2)],
                                  yticklabels = ['L1', 'Stub0', 'Stub1', 'Stub2', 'Stub3', 'Stub4'],
                                  colmap      = "RdYlGn_r",
                                  cb_label    = "Eroor Rate",
                                  vmin        = 0.0,
                                  vmax        = 1.0,
                                  outdir      = _outdir)

                self.plot_heatmap(RegisterMatchingEfficiency_mod,
                                  title       = f"CBCCIC-RegMatchEff",
                                  name        = f"Plot_CBCCIC_RegisterMatchingEfficiency_{datakey}",
                                  xticklabels = [f'{id}-hb{j}' for id in moduleIDs for j in range(2)],
                                  yticklabels = [f'CBC{i}' for i in range(8)]+['CIC'],
                                  colmap      = "RdYlGn",
                                  cb_label    = "Efficiency",
                                  vmin        = 0.0,
                                  vmax        = 1.0,
                                  outdir      = _outdir)
                
                self.plot_heatmap(bestDelay_mod,
                                  title       = f"Best Delay [ns]",
                                  name        = f"Plot_best_delay_{datakey}",
                                  xticklabels = moduleIDs,
                                  yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                  colmap      = "GnBu",
                                  cb_label    = "delay [ns]",
                                  vmin        = 50.0,
                                  vmax        = 70.0,
                                  outdir      = _outdir)

                self.plot_heatmap(bestTh_mod,
                                  title       = f"Best Threshold [VcTh]",
                                  name        = f"Plot_best_threshold_{datakey}",
                                  xticklabels = moduleIDs,
                                  yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                  colmap      = "GnBu",
                                  cb_label    = "best threshold [VcTh]",
                                  vmin        = 550.0,
                                  vmax        = 580.0,
                                  outdir      = _outdir)

                self.plot_heatmap(CICtoLpGBT_PatternMatchingErrorRate_mod,
                                  title       = f"CICtoLpGBT-PatternMatchingErrorRate",
                                  name        = f"Plot_CICtoLpGBT_PatternMatchingErrorRate_{datakey}",
                                  xticklabels = [f'{id}-hb{j}' for id in moduleIDs for j in range(2)],
                                  yticklabels = ['L1', 'Stub0', 'Stub1', 'Stub2', 'Stub3', 'Stub4'],
                                  colmap      = 'RdYlGn_r',
                                  vmin        = 0.0,
                                  vmax        = 1.0,
                                  cb_label    = "ErrorRate",
                                  #vmin        = 0,
                                  #vmax        = 15,
                                  outdir      = _outdir,
                                  annotint    = False)
                

                eye_cross_frac_offset_0p3_mod = np.array(eye_cross_frac_offset_0p3_mod)
                eye_cross_frac_offset_0p7_mod = np.array(eye_cross_frac_offset_0p7_mod)
                eye_cross_frac_offset_1p0_mod = np.array(eye_cross_frac_offset_1p0_mod)
                self.plot_basic(x          = np.arange(eye_cross_frac_offset_0p3_mod.shape[0]),
                                data_list  = [np.concatenate((eye_cross_frac_offset_0p3_mod[:,None], np.zeros_like(eye_cross_frac_offset_0p3_mod)[:,None]), axis=1),
                                              np.concatenate((eye_cross_frac_offset_0p7_mod[:,None], np.zeros_like(eye_cross_frac_offset_0p7_mod)[:,None]), axis=1),
                                              np.concatenate((eye_cross_frac_offset_1p0_mod[:,None], np.zeros_like(eye_cross_frac_offset_1p0_mod)[:,None]), axis=1)],
                                legends    = ["EyeOpPower-0.33", "EyeOpPower-0.67", "EyeOpPower-1.00"],
                                title      = f"EyeCrossingFractionalOffset",
                                name       = f"Plot_EyeCrossingFractionalOffset_allMods_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "Voltage [DAC units]",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 6.5,
                                ylim       = [0.0,30.0],
                                outdir     = _outdir,
                                ncols      = 3)

                eye_cross_wd_ymax_0p3_mod = np.array(eye_cross_wd_ymax_0p3_mod)
                eye_cross_wd_ymax_0p7_mod = np.array(eye_cross_wd_ymax_0p7_mod)
                eye_cross_wd_ymax_1p0_mod = np.array(eye_cross_wd_ymax_1p0_mod)                
                self.plot_basic(x          = np.arange(eye_cross_wd_ymax_0p3_mod.shape[0]),
                                data_list  = [np.concatenate((eye_cross_wd_ymax_0p3_mod[:,None], np.zeros_like(eye_cross_wd_ymax_0p3_mod)[:,None]), axis=1),
                                              np.concatenate((eye_cross_wd_ymax_0p7_mod[:,None], np.zeros_like(eye_cross_wd_ymax_0p7_mod)[:,None]), axis=1),
                                              np.concatenate((eye_cross_wd_ymax_1p0_mod[:,None], np.zeros_like(eye_cross_wd_ymax_1p0_mod)[:,None]), axis=1)],
                                legends    = ["EyeOpPower-0.33", "EyeOpPower-0.67", "EyeOpPower-1.00"],
                                title      = f"EyeCrossingWidthAtMaxVoltage",
                                name       = f"Plot_EyeCrossingWidthAtMaxVoltage_allMods_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "Time [ps]",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 6.5,
                                ylim       = [0.0,50.0],
                                outdir     = _outdir,
                                ncols      = 3)

                eye_open_area_0p3_mod = np.array(eye_open_area_0p3_mod)
                eye_open_area_0p7_mod = np.array(eye_open_area_0p7_mod)
                eye_open_area_1p0_mod = np.array(eye_open_area_1p0_mod)
                self.plot_basic(x          = np.arange(eye_open_area_0p3_mod.shape[0]),
                                data_list  = [np.concatenate((eye_open_area_0p3_mod[:,None], np.zeros_like(eye_open_area_0p3_mod)[:,None]), axis=1),
                                              np.concatenate((eye_open_area_0p7_mod[:,None], np.zeros_like(eye_open_area_0p7_mod)[:,None]), axis=1),
                                              np.concatenate((eye_open_area_1p0_mod[:,None], np.zeros_like(eye_open_area_1p0_mod)[:,None]), axis=1)],
                                legends    = ["EyeOpPower-0.33", "EyeOpPower-0.67", "EyeOpPower-1.00"],
                                title      = f"EyeOpeningArea",
                                name       = f"Plot_EyeOpeningArea_allMods_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "Voltage [DAC unit] x Time [ps]",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 6.5,
                                ylim       = [50.0,90.0],
                                outdir     = _outdir,
                                ncols      = 3)

                BERT_Err = np.array(BERT_Err)
                FEC_Err  = np.array(FEC_Err)

                self.plot_basic(x          = np.arange(BERT_Err.shape[0]),
                                data_list  = [np.concatenate((BERT_Err[:,None], np.zeros_like(BERT_Err)[:,None]), axis=1),
                                              np.concatenate((FEC_Err[:,None], np.zeros_like(FEC_Err)[:,None]), axis=1)],
                                legends    = ["BERTError", "FECError"],
                                title      = f"BERT-FEC-Error",
                                name       = f"Plot_BertFecError_allMods_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "bits",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 6.5,
                                #ylim       = [50.0,90.0],
                                outdir     = _outdir,
                                ncols      = 1)                                

                BERT_nbits = np.array(BERT_nbits)
                self.plot_basic(x          = np.arange(BERT_nbits.shape[0]),
                                data_list  = [np.concatenate((BERT_nbits[:,None], np.zeros_like(BERT_nbits)[:,None]), axis=1)],
                                legends    = ["BERT-nBits"],
                                title      = f"BERT-nBits",
                                name       = f"Plot_nbits_allMods_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "bits",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 6.5,
                                #ylim       = [50.0,90.0],
                                outdir     = _outdir,
                                ncols      = 1)          


                # VTRx
                LightYield_total_mod = np.array(LightYield_total_mod)
                LightYield_avg_diff_mod = np.array(LightYield_avg_diff_mod)
                LightYield_mod_slope_mod = np.array(LightYield_mod_slope_mod)
                LightYield_bias_slope_mod = np.array(LightYield_bias_slope_mod)

                self.plot_basic(x          = np.arange(LightYield_total_mod.shape[0]),
                                data_list  = [np.concatenate((LightYield_total_mod[:,None],
                                                              np.zeros_like(LightYield_total_mod)[:,None]),
                                                             axis=1)],
                                legends    = ["VTRX-LightYield-Total"],
                                title      = f"VTRX_Yield_Total",
                                name       = f"Plot_VTRX_Yield_Total_allMods_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "Total Yield",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 6.5,
                                #ylim       = [50.0,90.0],
                                outdir     = _outdir,
                                ncols      = 1)          
                self.plot_basic(x          = np.arange(LightYield_avg_diff_mod.shape[0]),
                                data_list  = [np.concatenate((LightYield_avg_diff_mod[:,None],
                                                              np.zeros_like(LightYield_avg_diff_mod)[:,None]),
                                                             axis=1)],
                                legends    = ["VTRX-LightYield-AvgDiff"],
                                title      = f"VTRX_Yield_Avg",
                                name       = f"Plot_VTRX_Yield_Avg_allMods_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "Avg. Difference",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 6.5,
                                #ylim       = [50.0,90.0],
                                outdir     = _outdir,
                                ncols      = 1)          
                self.plot_basic(x          = np.arange(LightYield_mod_slope_mod.shape[0]),
                                data_list  = [np.concatenate((LightYield_mod_slope_mod[:,None],
                                                              np.zeros_like(LightYield_mod_slope_mod)[:,None]),
                                                             axis=1)],
                                legends    = ["VTRX-LightYield-ModSlope"],
                                title      = f"VTRX_Yield_Modulation_Slope",
                                name       = f"Plot_VTRX_Yield_ModSlope_allMods_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "light yield / modulation",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 6.5,
                                #ylim       = [50.0,90.0],
                                outdir     = _outdir,
                                ncols      = 1)          
                self.plot_basic(x          = np.arange(LightYield_bias_slope_mod.shape[0]),
                                data_list  = [np.concatenate((LightYield_bias_slope_mod[:,None],
                                                              np.zeros_like(LightYield_bias_slope_mod)[:,None]),
                                                             axis=1)],
                                legends    = ["VTRX-LightYield-BiasSlope"],
                                title      = f"VTRX_Yield_Bias_Slope",
                                name       = f"Plot_VTRX_Yield_BiasSlope_allMods_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "light yield / bias",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 6.5,
                                #ylim       = [50.0,90.0],
                                outdir     = _outdir,
                                ncols      = 1)

                
                
            if self.testinfo.get("check_sensor_temperature") == True:
                #from IPython import embed; embed()
                sensor_temps = np.array(sensor_temps_mod)
                self.plot_basic(x          = np.arange(sensor_temps.shape[0]),
                                data_list  = [sensor_temps],
                                #data_list  = [np.concatenate((sensor_temps[:,None],
                                #np.zeros_like(sensor_temps)[:,None]), axis=1)],
                                legends    = ["sensor temp"],
                                title      = f"Sensor Temperature",
                                name       = f"Plot_SensorTemp_allMods_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "Sensor Temperature (deg C)",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 4.5,
                                #ylim       = [18.0,29.0],
                                outdir     = _outdir,
                                capsize    = 1.0)
                #from IPython import embed; embed()

                self.plot_sensor_temp(sen_temp_dict,
                                      outdir = _outdir,
                                      pname = "Sensor_Temp_OGs",
                                      #ylim = [17.0, 33.0]
                                      )
                
            else:
                logger.warning("skip plotting sensor temperature")

                
            self.plot_box(data_list_1 = strip_noise_hb0_mod,
                          data_list_2 = strip_noise_hb1_mod,
                          legends     = ["hybrid 0", "hybrid 1"],
                          title       = f"StripNoise",
                          name        = f"Plot_StripNoiseBox_allModules_bothHybrids_{datakey}",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          outdir      = _outdir)
                
            self.plot_box(data_list_1 = strip_noise_hb0_bot_mod,
                          data_list_2 = strip_noise_hb1_bot_mod,
                          legends     = ["hybrid 0", "hybrid 1"],
                          title       = f"StripNoise_bottomSensor",
                          name        = f"Plot_StripNoiseBox_allModules_bothHybrids_bottomSensor_{datakey}",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          outdir      = _outdir)
            
            self.plot_box(data_list_1 = strip_noise_hb0_top_mod,
                          data_list_2 = strip_noise_hb1_top_mod,
                          legends     = ["hybrid 0", "hybrid 1"],
                          title       = f"StripNoise_topSensor",
                          name        = f"Plot_StripNoiseBox_allModules_bothHybrids_topSensor_{datakey}",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          outdir      = _outdir)

            # Preparing to plot the mean and std
            bot_noise_hb0 = np.array(strip_noise_hb0_bot_mod)[:,:,0]
            bot_noise_hb0_mean_std = np.concatenate((np.mean(strip_noise_hb0_bot_mod, axis=1)[:,None],
                                                     np.std(strip_noise_hb0_bot_mod, axis=1)[:,None]), axis=1)
            bot_noise_hb1 = np.array(strip_noise_hb1_bot_mod)[:,:,0]
            bot_noise_hb1_mean_std = np.concatenate((np.mean(strip_noise_hb1_bot_mod, axis=1)[:,None],
                                                     np.std(strip_noise_hb1_bot_mod, axis=1)[:,None]), axis=1)
            top_noise_hb0 = np.array(strip_noise_hb0_top_mod)[:,:,0]
            top_noise_hb0_mean_std = np.concatenate((np.mean(strip_noise_hb0_top_mod, axis=1)[:,None],
                                                     np.std(strip_noise_hb0_top_mod, axis=1)[:,None]), axis=1)
            top_noise_hb1 = np.array(strip_noise_hb1_top_mod)[:,:,0]
            top_noise_hb1_mean_std = np.concatenate((np.mean(strip_noise_hb1_top_mod, axis=1)[:,None],
                                                     np.std(strip_noise_hb1_top_mod, axis=1)[:,None]), axis=1)

            #from IPython import embed; embed()
            """
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[bot_noise_hb0_mean_std,
                                            bot_noise_hb1_mean_std],
                                           [top_noise_hb0_mean_std,
                                            top_noise_hb1_mean_std]],
                            legends     = [["hb0 (bottom sensor)", "hb1 (bottom sensor)"],
                                           ["hb0 (top sensor)",    "hb1 (top sensor)"]],
                            title       = f"StripNoise",
                            name        = f"Plot_StripNoise_allModules_bothHybrids_bothSensors_{datakey}",
                            xticklabels = moduleIDs,
                            ylim        = [3.5,8.5],
                            ylabel      = "Noise [VcTh]",
                            outdir      = _outdir,
                            marker      = "o",
                            markersize  = 2.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0)
            """

            noisy_channels_hb0_cbc = np.array(num_noisy_channels_hb0_cbc_mod)
            noisy_channels_hb1_cbc = np.array(num_noisy_channels_hb1_cbc_mod)
            noisy_channels_cbc = np.concatenate((noisy_channels_hb0_cbc,
                                                 np.zeros_like(noisy_channels_hb0_cbc[:,:1]),
                                                 noisy_channels_hb1_cbc), axis=1)

            #from IPython import embed; embed()                
            self.plot_heatmap(noisy_channels_cbc.astype(int),
                              title       = f"nNoisyCh",
                              name        = f"Plot_NumNoisyCh_bothHybrids_{datakey}",
                              xticklabels = moduleIDs,
                              yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                              colmap      = "coolwarm",
                              cb_label    = "nChannels",
                              #vmin        = 0.5,
                              #vmax        = 1.5,
                              outdir      = _outdir)
            
            dead_channels_hb0_cbc = np.array(num_dead_channels_hb0_cbc_mod)
            dead_channels_hb1_cbc = np.array(num_dead_channels_hb1_cbc_mod)
            dead_channels_cbc = np.concatenate((dead_channels_hb0_cbc,
                                                np.zeros_like(dead_channels_hb0_cbc[:,:1]),
                                                dead_channels_hb1_cbc), axis=1)
            
            self.plot_heatmap(dead_channels_cbc.astype(int),
                              title       = f"nDeadCh",
                              name        = f"Plot_NumDeadCh_bothHybrids_{datakey}",
                              xticklabels = moduleIDs,
                              yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                              colmap      = "coolwarm",
                              cb_label    = "nChannels",
                              #vmin        = 0.5,
                              #vmax        = 1.5,
                              outdir      = _outdir)
                
                

            # same for common mode noise
            if self.testinfo.get("check_common_noise") == True:

                if self.testinfo.get("fit_simultaneous_common_noise") == True:
                    cmn_frac_hb0_cbc = np.array(common_noise_frac_simfit_hb0_cbc_mod)
                    cmn_frac_hb1_cbc = np.array(common_noise_frac_simfit_hb1_cbc_mod)
                    cmn_frac_cbc = np.concatenate((cmn_frac_hb0_cbc,
                                                   np.zeros_like(cmn_frac_hb0_cbc[:,:1]),
                                                   cmn_frac_hb1_cbc), axis=1)
                        
                    self.plot_heatmap(cmn_frac_cbc,
                                      title       = f"CMNFrac_SimFit",
                                      name        = f"Plot_CMNFrac_SimFit_bothHybrids_{datakey}",
                                      xticklabels = moduleIDs,
                                      yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                      colmap      = "coolwarm",
                                      cb_label    = "nChannels",
                                      vmin        = 0.0,
                                      vmax        = 1.0,
                                      outdir      = _outdir)
                    
                    
                cmn_noise_hb0_mean_std, cmn_noise_hb0_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_mod)[:,:,0])
                cmn_noise_hb1_mean_std, cmn_noise_hb1_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_mod)[:,:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_mean_std, cmn_noise_hb1_mean_std],
                                legends    = ["Hybrid 0", "Hybrid 1"],
                                title      = f"#Hits (50% Occ) (µ)",
                                name       = f"Plot_nHitsMean_bothHybrids_{datakey}",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (µ)",
                                markersize = 10,
                                ylim       = [500.0,1200.0],
                                outdir     = _outdir,
                                fit        = False)
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_sigma_std, cmn_noise_hb1_sigma_std],
                                legends    = ["Hybrid 0", "Hybrid 1"],
                                title      = f"#Hits (50% Occ) (σ)",
                                name       = f"Plot_nHitsStd_bothHybrids_{datakey}",
                                xticklabels = moduleIDs,
                                ylabel     = "CM Noise",
                                markersize = 10,
                                ylim       = [50,250],
                                outdir     = _outdir,
                                fit        = False)
                    

                cmn_noise_hb0_bot_mean_std, cmn_noise_hb0_bot_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_bot_mod)[:,:,0])
                cmn_noise_hb1_bot_mean_std, cmn_noise_hb1_bot_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_bot_mod)[:,:,0])
                
                cmn_noise_hb0_top_mean_std, cmn_noise_hb0_top_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_top_mod)[:,:,0])                
                cmn_noise_hb1_top_mean_std, cmn_noise_hb1_top_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_top_mod)[:,:,0])
                    
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_bot_mean_std, cmn_noise_hb1_bot_mean_std,
                                              cmn_noise_hb0_top_mean_std, cmn_noise_hb1_top_mean_std],
                                legends    = ["Hybrid 0 bottom", "Hybrid 1 bottom",
                                              "Hybrid 0 top", "Hybrid 1 top"],
                                title      = f"#Hits (50% Occ) (µ)",
                                name       = f"Plot_nHitsMean_bothHybrids_bothSensors_{datakey}",
                                xticklabels= moduleIDs,
                                ylabel     = "#hits (µ)",
                                markersize = 10,
                                ylim       = [100,700],
                                outdir     = _outdir,
                                fit        = False)
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_bot_sigma_std, cmn_noise_hb1_bot_sigma_std,
                                              cmn_noise_hb0_top_sigma_std, cmn_noise_hb1_top_sigma_std],
                                legends    = ["Hybrid 0 bottom", "Hybrid 1 bottom",
                                              "Hybrid 0 top", "Hybrid 1 top"],
                                title      = f"#Hits (50% Occ) (σ)",
                                name       = f"Plot_nHitsStd_bothHybrids_bothSensors_{datakey}",
                                xticklabels = moduleIDs,
                                ylabel     = "CM Noise",
                                markersize = 10,
                                ylim       = [20,160],
                                outdir     = _outdir,
                                fit        = False)



                common_noise_giovanni_hb0_cbc = np.array(common_noise_giovanni_hb0_cbc_mod)
                common_noise_giovanni_hb1_cbc = np.array(common_noise_giovanni_hb1_cbc_mod)
                common_noise_giovanni_cbc = np.concatenate((common_noise_giovanni_hb0_cbc,
                                                            np.zeros_like(common_noise_giovanni_hb0_cbc[:,:1]),
                                                            common_noise_giovanni_hb1_cbc), axis=1)
                    
                self.plot_heatmap(common_noise_giovanni_cbc,
                                  title       = f"CMN frac",
                                  name        = f"Plot_CMN_Fraction_Giovanni_bothHybrids_{datakey}",
                                  xticklabels = moduleIDs,
                                  yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                  colmap      = "coolwarm",
                                  #vmin        = 0.5,
                                  #vmax        = 1.5,
                                  outdir      = _outdir) 
                
                common_noise_iphc_hb0_cbc = np.array(common_noise_iphc_hb0_cbc_mod)
                common_noise_iphc_hb1_cbc = np.array(common_noise_iphc_hb1_cbc_mod)
                common_noise_iphc_cbc = np.concatenate((common_noise_iphc_hb0_cbc,
                                                        np.zeros_like(common_noise_iphc_hb0_cbc[:,:1]),
                                                        common_noise_iphc_hb1_cbc), axis=1)
                
                self.plot_heatmap(common_noise_iphc_cbc,
                                  title       = f"CMN frac",
                                  name        = f"Plot_CMN_Fraction_IPHC_bothHybrids_{datakey}",
                                  xticklabels = moduleIDs,
                                  yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                  colmap      = "coolwarm",
                                  vmin        = 0.0,
                                  vmax        = 0.5,
                                  outdir      = _outdir)
                    
                    
                common_noise_potato_hb0_cbc = np.array(common_noise_frac_potato_hb0_cbc_mod)
                common_noise_potato_hb1_cbc = np.array(common_noise_frac_potato_hb1_cbc_mod)
                common_noise_potato_cbc = np.concatenate((common_noise_potato_hb0_cbc,
                                                          np.zeros_like(common_noise_potato_hb0_cbc[:,:1]),
                                                          common_noise_potato_hb1_cbc), axis=1)

                self.plot_heatmap(common_noise_potato_cbc,
                                  title       = f"CMN frac",
                                  name        = f"Plot_CMN_Fraction_Potato_bothHybrids_{datakey}",
                                  xticklabels = moduleIDs,
                                  yticklabels = [f"Hb0_CBC{i}" for i in range(8)] + ["SEH"] + [f"Hb1_CBC{i}" for i in range(8)],
                                  colmap      = "coolwarm",
                                  #vmin        = 0.5,
                                  #vmax        = 1.5,
                                  outdir      = _outdir)



                    
                #from IPython import embed; embed()
                    


                
    
                #from IPython import embed; embed(); exit()
                is_empty = all(not x for x in common_noise_fit_hb0_cbc_mod)
                if not is_empty:
                    self.plot_heatmap(common_noise_fit_hb0_cbc_mod,
                                      title       = f"CMN-hb0",
                                      name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb0_{datakey}",
                                      xticklabels = moduleIDs,
                                      colmap      = "coolwarm",
                                      outdir      = _outdir) 
                    self.plot_heatmap(common_noise_fit_hb0_cbc_top_mod,
                                      title       = f"CMN-hb0-top",
                                      name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb0_top_{datakey}",
                                      xticklabels = moduleIDs,
                                      colmap      = "coolwarm",
                                      outdir      = _outdir)
                    self.plot_heatmap(common_noise_fit_hb0_cbc_bot_mod,
                                      title       = f"CMN-hb0-bot",
                                      name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb0_bot_{datakey}",
                                      xticklabels = moduleIDs,
                                      colmap      = "coolwarm",
                                      outdir      = _outdir)

                    self.plot_heatmap(common_noise_fit_hb1_cbc_mod,
                                      title       = f"CMN-hb1",
                                      name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb1_{datakey}",
                                      xticklabels = moduleIDs,
                                      colmap      = "coolwarm",
                                      outdir      = _outdir) 
                    self.plot_heatmap(common_noise_fit_hb1_cbc_top_mod,
                                      title       = f"CMN-hb1-top",
                                      name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb1_top_{datakey}",
                                      xticklabels = moduleIDs,
                                      colmap      = "coolwarm",
                                      outdir      = _outdir)
                    self.plot_heatmap(common_noise_fit_hb1_cbc_bot_mod,
                                      title       = f"CMN-hb1-bot",
                                      name        = f"Plot_CMN_Fraction_Ph2ACF_fit_hb1_bot_{datakey}",
                                      xticklabels = moduleIDs,
                                      colmap      = "coolwarm",
                                      outdir      = _outdir)




                    self.plot_heatmap((np.array(common_noise_giovanni_hb0_cbc)/np.array(common_noise_fit_hb0_cbc)).tolist(),
                                      title       = f"CMN-hb0-ratio",
                                      name        = f"Plot_CMN_Fraction_Giovanni_by_fit_hb0_{datakey}",
                                      xticklabels = moduleIDs,
                                      colmap      = "coolwarm",
                                      vmin        = 0.5, vmax = 1.5,
                                      cb_label    = "Giovanni_Form/Ph2ACF_Fit",
                                      outdir      = _outdir) 
                    self.plot_heatmap((np.array(common_noise_giovanni_hb1_cbc)/np.array(common_noise_fit_hb1_cbc)).tolist(),
                                      title       = f"CMN-hb1-ratio",
                                      name        = f"Plot_CMN_Fraction_Giovanni_by_fit_hb1_{datakey}",
                                      xticklabels = moduleIDs,
                                      colmap      = "coolwarm",
                                      vmin        = 0.5, vmax = 1.5,
                                      cb_label    = "Giovanni_Form/Ph2ACF_Fit",
                                      outdir      = _outdir) 
                    self.plot_heatmap((np.array(common_noise_iphc_hb0_cbc)/np.array(common_noise_fit_hb0_cbc)).tolist(),
                                      title       = f"CMN-hb0-ratio",
                                      name        = f"Plot_CMN_Fraction_Iphc_by_fit_hb0_{datakey}",
                                      xticklabels = moduleIDs,
                                      colmap      = "coolwarm",
                                      vmin        = 0.5, vmax = 1.5,
                                      cb_label    = "IPHC_Form/Ph2ACF_Fit",
                                      outdir      = _outdir) 
                    self.plot_heatmap((np.array(common_noise_iphc_hb1_cbc)/np.array(common_noise_fit_hb1_cbc)).tolist(),
                                      title       = f"CMN-hb1-ratio",
                                      name        = f"Plot_CMN_Fraction_Iphc_by_fit_hb1_{datakey}",
                                      xticklabels = moduleIDs,
                                      colmap      = "coolwarm",
                                      vmin        = 0.5, vmax = 1.5,
                                      cb_label	  = "IPHC_Form/Ph2ACF_Fit",
                                      outdir      = _outdir) 
                    
                    #common_noise_crude_hb0_cbc = common_noise_crude_hb0_cbc_mod[temp]
                    #self.plot_heatmap(common_noise_crude_hb0_cbc,
                    #                  title       = f"CMN-hb0: {temp}",
                    #                  name        = f"Plot_CMN_Fraction_Crude_hb0_{temp}_{datakey}",
                    #                  xticklabels = moduleIDs,
                    #                  colmap      = "coolwarm",
                    #                  vmin        = 0.5, vmax = +1.0,
                    #                  cb_label	  = "(σ - 0.5x√µ)/σ",
                    #                  outdir      = _outdirCBC) 
                    #common_noise_crude_hb1_cbc = common_noise_crude_hb1_cbc_mod[temp]
                    #self.plot_heatmap(common_noise_crude_hb1_cbc,
                    #                  title       = f"CMN-hb1: {temp}",
                    #                  name        = f"Plot_CMN_Fraction_Crude_hb1_{temp}_{datakey}",
                    #                  xticklabels = moduleIDs,
                    #                  colmap      = "coolwarm",
                    #                  vmin        = 0.5, vmax = +1.0,
                    #                  cb_label    = "(σ - 0.5x√µ)/σ",
                    #                  outdir      = _outdirCBC) 


                    
                    
            else:
                logger.warning("skipping module wise common mode noise comparison")




            modIDs = [f"OpticalGroup_{i}:{ModID}" for i,ModID in enumerate(moduleIDs)]
            ladsubdir_key = datakey.split('_')[0]
            ladsubdir = laddir.mkdir(ladsubdir_key)
            self.store_modIDs(modIDlist = modIDs,
                              tdir = ladsubdir)
            # ------------ Plot add in ROOT file ------------- #

            #from IPython import embed; embed()
            
            strip_noise_hb0_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_mod)[:,:,0])
            strip_noise_hb1_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_mod)[:,:,0])
            strip_noise_hb0_bot_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_bot_mod)[:,:,0])
            strip_noise_hb0_top_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_top_mod)[:,:,0])
            strip_noise_hb1_bot_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_bot_mod)[:,:,0])
            strip_noise_hb1_top_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_top_mod)[:,:,0])

            
            cmn_noise_mean, cmn_noise_sigma = get_cmn_mean_sigma_for_plotting(np.array(common_noise_mod)[:,:,0])
            CMN_mod_frac = extractCMN(
                nchannels = np.array(common_noise_mod).shape[1],
                mean      = cmn_noise_mean[:,0],
                std       = cmn_noise_sigma[:,0]
            )[:,0]/100.0
            cmn_noise_hb0_mean, cmn_noise_hb0_sigma = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_mod)[:,:,0])
            CMN_hb0_frac = extractCMN(
                nchannels = np.array(common_noise_hb0_mod).shape[1],
                mean      = cmn_noise_hb0_mean[:,0],
                std       = cmn_noise_hb0_sigma[:,0]
            )[:,0]/100.0
            cmn_noise_hb1_mean, cmn_noise_hb1_sigma = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_mod)[:,:,0])
            CMN_hb1_frac = extractCMN(
                nchannels = np.array(common_noise_hb1_mod).shape[1],
                mean      = cmn_noise_hb1_mean[:,0],
                std       = cmn_noise_hb1_sigma[:,0]
            )[:,0]/100.0
            cmn_noise_hb0_bot_mean, cmn_noise_hb0_bot_sigma = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_bot_mod)[:,:,0])
            CMN_hb0_bot_frac = extractCMN(
                nchannels = np.array(common_noise_hb0_bot_mod).shape[1],
                mean      = cmn_noise_hb0_bot_mean[:,0],
                std       = cmn_noise_hb0_bot_sigma[:,0]
            )[:,0]/100.0
            cmn_noise_hb0_top_mean, cmn_noise_hb0_top_sigma = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_top_mod)[:,:,0])
            CMN_hb0_top_frac = extractCMN(
                nchannels = np.array(common_noise_hb0_top_mod).shape[1],
                mean      = cmn_noise_hb0_top_mean[:,0],
                std       = cmn_noise_hb0_top_sigma[:,0]
            )[:,0]/100.0
            cmn_noise_hb1_bot_mean, cmn_noise_hb1_bot_sigma = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_bot_mod)[:,:,0])
            CMN_hb1_bot_frac = extractCMN(
                nchannels = np.array(common_noise_hb1_bot_mod).shape[1],
                mean      = cmn_noise_hb1_bot_mean[:,0],
                std       = cmn_noise_hb1_bot_sigma[:,0]
            )[:,0]/100.0
            cmn_noise_hb1_top_mean, cmn_noise_hb1_top_sigma = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_top_mod)[:,:,0])
            CMN_hb1_top_frac = extractCMN(
                nchannels = np.array(common_noise_hb1_top_mod).shape[1],
                mean      = cmn_noise_hb1_top_mean[:,0],
                std       = cmn_noise_hb1_top_sigma[:,0]
            )[:,0]/100.0
            
            #from IPython import embed; embed()
            lincol = ROOT.kBlue if datakey.startswith('PreInt') else ROOT.kRed+1

            #from IPython import embed; embed()
            
            pede_hb0_mean_std = get_noise_mean_sigma_for_plotting(np.array(pede_hb0_mod)[:,:,0])
            pede_hb1_mean_std = get_noise_mean_sigma_for_plotting(np.array(pede_hb1_mod)[:,:,0])

            self.to_root_temp(x      = np.arange(pede_hb0_mean_std.shape[0]),
                              y      = pede_hb0_mean_std[:,0],
                              yerr   = pede_hb0_mean_std[:,1],
                              title  = 'Pedestal - Hybrid 0',
                              name   = 'pedestal_hb0',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Pedestal [VcTh]",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(pede_hb1_mean_std.shape[0]),
                              y      = pede_hb1_mean_std[:,0],
                              yerr   = pede_hb1_mean_std[:,1],
                              title  = 'Pedestal - Hybrid 1',
                              name   = 'pedestal_hb1',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Pedestal [VcTh]",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(pede_hb0_mean_std.shape[0]),
                              y      = pede_hb0_mean_std[:,1],
                              yerr   = None,
                              title  = 'Pedestal - Hybrid 0',
                              name   = 'pedestal_std_hb0',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Pedestal std [VcTh]",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(pede_hb1_mean_std.shape[0]),
                              y      = pede_hb1_mean_std[:,1],
                              yerr   = None,
                              title  = 'Pedestal - Hybrid 1',
                              name   = 'pedestal_std_hb1',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Pedestal std [VcTh]",
                              lincol = lincol)
            

            self.to_root_temp(x      = np.arange(strip_noise_hb0_mean_std.shape[0]),
                              y      = strip_noise_hb0_mean_std[:,0],
                              yerr   = strip_noise_hb0_mean_std[:,1],
                              title  = 'Strip Noise - Hybrid 0',
                              name   = 'strip_noise_hb0',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Noise [VcTh]",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(strip_noise_hb0_bot_mean_std.shape[0]),
                              y      = strip_noise_hb0_bot_mean_std[:,0],
                              yerr   = strip_noise_hb0_bot_mean_std[:,1],
                              title  = 'Strip Noise - Hybrid 0 - Bottom',
                              name   = 'strip_noise_hb0_bottom',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Noise [VcTh]",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(strip_noise_hb0_top_mean_std.shape[0]),
                              y      = strip_noise_hb0_top_mean_std[:,0],
                              yerr   = strip_noise_hb0_top_mean_std[:,1],
                              title  = 'Strip Noise - Hybrid 0 - Top',
                              name   = 'strip_noise_hb0_top',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Noise [VcTh]",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(strip_noise_hb1_mean_std.shape[0]),
                              y      = strip_noise_hb1_mean_std[:,0],
                              yerr   = strip_noise_hb1_mean_std[:,1],
                              title  = 'Strip Noise - Hybrid 1',
                              name   = 'strip_noise_hb1',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Noise [VcTh]",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(strip_noise_hb1_bot_mean_std.shape[0]),
                              y      = strip_noise_hb1_bot_mean_std[:,0],
                              yerr   = strip_noise_hb1_bot_mean_std[:,1],
                              title  = 'Strip Noise - Hybrid 1 - Bottom',
                              name   = 'strip_noise_hb1_bottom',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Noise [VcTh]",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(strip_noise_hb1_top_mean_std.shape[0]),
                              y      = strip_noise_hb1_top_mean_std[:,0],
                              yerr   = strip_noise_hb1_top_mean_std[:,1],
                              title  = 'Strip Noise - Hybrid 1 - Top',
                              name   = 'strip_noise_hb1_top',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Noise [VcTh]",
                              lincol = lincol)


            self.to_root_temp(x      = np.arange(CMN_hb0_frac.shape[0]),
                              y      = CMN_hb0_frac,
                              yerr   = None,
                              title  = 'Common Mode Noise Frac - Hybrid 0',
                              name   = 'common_noise_frac_hb0',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Common Mode Noise fraction",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(CMN_hb1_frac.shape[0]),
                              y      = CMN_hb1_frac,
                              yerr   = None,
                              title  = 'Common Mode Noise Frac - Hybrid 1',
                              name   = 'common_noise_frac_hb1',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Common Mode Noise fraction",
                              lincol = lincol)

            
            CMN_hb0 = strip_noise_hb0_mean_std[:,0] * CMN_hb0_frac
            CMN_hb0_err = strip_noise_hb0_mean_std[:,1] * CMN_hb0_frac
            CMN_hb1 = strip_noise_hb1_mean_std[:,0] * CMN_hb1_frac
            CMN_hb1_err = strip_noise_hb1_mean_std[:,1] * CMN_hb1_frac

            self.to_root_temp(x      = np.arange(CMN_hb0.shape[0]),
                              y      = CMN_hb0,
                              yerr   = CMN_hb0_err,
                              title  = 'Common Mode Noise - Hybrid 0',
                              name   = 'common_noise_hb0',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Common Mode Noise [VcTh]",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(CMN_hb1.shape[0]),
                              y      = CMN_hb1,
                              yerr   = CMN_hb1_err,
                              title  = 'Common Mode Noise - Hybrid 1',
                              name   = 'common_noise_hb1',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Common Mode Noise [VcTh]",
                              lincol = lincol)

            sensor_temps_mod = np.array(sensor_temps_mod)
            self.to_root_temp(x      = np.arange(sensor_temps_mod.shape[0]),
                              y      = sensor_temps_mod[:,0],
                              yerr   = sensor_temps_mod[:,1],
                              title  = 'Sensor Temperature',
                              name   = 'sensor_temperature',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "Temperature (C)",
                              lincol = lincol)

            
            self.to_root_temp(x      = np.arange(len(num_noisy_channels_hb0_mod)),
                              y      = num_noisy_channels_hb0_mod,
                              title  = 'Noisy Channels - Hybrid 0',
                              name   = 'n_noisy_ch_hb0',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "# Noisy Channels",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(len(num_noisy_channels_hb1_mod)),
                              y      = num_noisy_channels_hb1_mod,
                              title  = 'Noisy Channels - Hybrid 1',
                              name   = 'n_noisy_ch_hb1',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "# Noisy Channels",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(len(num_dead_channels_hb0_mod)),
                              y      = num_dead_channels_hb0_mod,
                              title  = 'Broken Channels - Hybrid 0',
                              name   = 'n_broken_ch_hb0',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "# Broken Channels",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(len(num_dead_channels_hb1_mod)),
                              y      = num_dead_channels_hb1_mod,
                              title  = 'Broken Channels - Hybrid 1',
                              name   = 'n_broken_ch_hb1',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "# Broken Channels",
                              lincol = lincol)
            
            self.to_root_temp(x      = np.arange(LightYield_total_mod.shape[0]),
                              y      = LightYield_total_mod,
                              title  = 'VTRX Total LightYield',
                              name   = 'vtrx_lightyield_total',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "VTRX LightYield Total",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(LightYield_avg_diff_mod.shape[0]),
                              y      = LightYield_avg_diff_mod,
                              title  = 'VTRX LightYield AvgDiff',
                              name   = 'vtrx_lightyield_avgdiff',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "VTRX LightYield AvgDiff",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(LightYield_mod_slope_mod.shape[0]),
                              y      = LightYield_mod_slope_mod,
                              title  = 'VTRX LightYield per Modulation',
                              name   = 'vtrx_lightyield_permod',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "VTRX LightYield PerModulation",
                              lincol = lincol)
            self.to_root_temp(x      = np.arange(LightYield_bias_slope_mod.shape[0]),
                              y      = LightYield_bias_slope_mod,
                              title  = 'VTRX LightYield per Bias',
                              name   = 'vtrx_lightyield_perbias',
                              tdir   = ladsubdir,
                              xlabel = "OpticalGroup",
                              ylabel = "VTRX LightYield PerBias",
                              lincol = lincol)




            
            # for comparison
            sensor_temps_setup[datakey] = sensor_temps_mod
            strip_noise_hb0_setup[datakey] = strip_noise_hb0_mod
            strip_noise_hb0_bot_setup[datakey] = strip_noise_hb0_bot_mod
            strip_noise_hb0_top_setup[datakey] = strip_noise_hb0_top_mod
            strip_noise_hb1_setup[datakey] = strip_noise_hb1_mod
            strip_noise_hb1_bot_setup[datakey] = strip_noise_hb1_bot_mod
            strip_noise_hb1_top_setup[datakey] = strip_noise_hb1_top_mod

            num_noisy_channels_hb0_setup[datakey] = num_noisy_channels_hb0_mod
            num_noisy_channels_hb1_setup[datakey] = num_noisy_channels_hb1_mod
            num_dead_channels_hb0_setup[datakey] = num_dead_channels_hb0_mod
            num_dead_channels_hb1_setup[datakey] = num_dead_channels_hb1_mod
            
            common_noise_setup[datakey] = common_noise_mod
            common_noise_bot_setup[datakey] = common_noise_bot_mod
            common_noise_top_setup[datakey] = common_noise_top_mod
            common_noise_hb0_setup[datakey] = common_noise_hb0_mod
            common_noise_hb0_bot_setup[datakey] = common_noise_hb0_bot_mod
            common_noise_hb0_top_setup[datakey] = common_noise_hb0_top_mod
            common_noise_hb1_setup[datakey] = common_noise_hb1_mod
            common_noise_hb1_bot_setup[datakey] = common_noise_hb1_bot_mod
            common_noise_hb1_top_setup[datakey] = common_noise_hb1_top_mod

            common_noise_fit_hb0_cbc_setup[datakey] = common_noise_fit_hb0_cbc_mod
            common_noise_fit_hb0_cbc_top_sensor_setup[datakey] = common_noise_fit_hb0_cbc_top_mod
            common_noise_fit_hb0_cbc_bot_sensor_setup[datakey] = common_noise_fit_hb0_cbc_bot_mod
            common_noise_fit_hb1_cbc_setup[datakey] = common_noise_fit_hb1_cbc_mod
            common_noise_fit_hb1_cbc_top_sensor_setup[datakey] = common_noise_fit_hb1_cbc_top_mod
            common_noise_fit_hb1_cbc_bot_sensor_setup[datakey] = common_noise_fit_hb1_cbc_bot_mod

            common_noise_giovanni_hb0_cbc_setup[datakey] = common_noise_giovanni_hb0_cbc_mod
            common_noise_giovanni_hb1_cbc_setup[datakey] = common_noise_giovanni_hb1_cbc_mod

            common_noise_iphc_hb0_cbc_setup[datakey] = common_noise_iphc_hb0_cbc_mod
            common_noise_iphc_hb1_cbc_setup[datakey] = common_noise_iphc_hb1_cbc_mod

            common_noise_crude_hb0_cbc_setup[datakey] = common_noise_crude_hb0_cbc_mod
            common_noise_crude_hb1_cbc_setup[datakey] = common_noise_crude_hb1_cbc_mod

            common_noise_frac_potato_hb0_cbc_setup[datakey] = common_noise_frac_potato_hb0_cbc_mod
            common_noise_frac_potato_hb1_cbc_setup[datakey] = common_noise_frac_potato_hb1_cbc_mod

            eye_cross_frac_offset_0p3_setup[datakey] = eye_cross_frac_offset_0p3_mod
            eye_cross_frac_offset_0p7_setup[datakey] = eye_cross_frac_offset_0p7_mod
            eye_cross_frac_offset_1p0_setup[datakey] = eye_cross_frac_offset_1p0_mod
            
            eye_cross_wd_ymax_0p3_setup[datakey] = eye_cross_wd_ymax_0p3_mod
            eye_cross_wd_ymax_0p7_setup[datakey] = eye_cross_wd_ymax_0p7_mod
            eye_cross_wd_ymax_1p0_setup[datakey] = eye_cross_wd_ymax_1p0_mod

            eye_open_area_0p3_setup[datakey] = eye_open_area_0p3_mod
            eye_open_area_0p7_setup[datakey] = eye_open_area_0p7_mod
            eye_open_area_1p0_setup[datakey] = eye_open_area_1p0_mod

            LightYield_total_setup[datakey] = LightYield_total_mod
            LightYield_avg_diff_setup[datakey] = LightYield_avg_diff_mod
            LightYield_mod_slope_setup[datakey] = LightYield_mod_slope_mod
            LightYield_bias_slope_setup[datakey] = LightYield_bias_slope_mod

            
        # Loop over setup ends here    
        #from IPython import embed; embed(); exit()

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
        #                                          Comparing two setup                                               #
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
        if self.testinfo.get("compare_two_setup") == True:
            logger.info(" ===> Comparing two setup ...")
            setup_keys = list(strip_noise_hb0_bot_setup.keys())
            setup_1 = setup_keys[0]
            setup_2 = setup_keys[1]
        
            logger.info(f"Setup-1 : {setup_1}, Setup-2 : {setup_2}")
        
            tick_offset = 0.1
            box_offset = 0.1
            if self.testinfo.get("are_same_modules") == False:
                moduleIDs = np.array(allModuleIDs).reshape(2,-1).T.tolist()
                moduleIDs = [f"{mid[0]}\n{mid[1]}" for mid in moduleIDs]
                tick_offset = 0.22
                box_offset = 0.52

            #cool_temps_setup_1 = list(strip_noise_hb0_bot_setup[setup_1])
            #cool_temps_setup_2 = list(strip_noise_hb0_bot_setup[setup_2])

            #cool_temps = [temp for temp in cool_temps_setup_1 if temp in cool_temps_setup_2]

            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
            #                                     WARNING: FEW THIGS ARE HARDCODED                                         #
            #                        Here, one can use hardcoding to compare different setup                               #
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
            #from IPython import embed; embed(); exit()

            #from IPython import embed; embed()
            n_noisy_channels_hb0_set_1 = np.array(num_noisy_channels_hb0_setup[setup_1])
            n_noisy_channels_hb1_set_1 = np.array(num_noisy_channels_hb1_setup[setup_1])
            n_noisy_channels_hb0_set_2 = np.array(num_noisy_channels_hb0_setup[setup_2])
            n_noisy_channels_hb1_set_2 = np.array(num_noisy_channels_hb1_setup[setup_2])
                
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[np.concatenate((n_noisy_channels_hb0_set_1[:,None], np.zeros_like(n_noisy_channels_hb0_set_1)[:,None]), axis=1).tolist(),
                                            np.concatenate((n_noisy_channels_hb0_set_2[:,None], np.zeros_like(n_noisy_channels_hb0_set_2)[:,None]), axis=1).tolist()]],
                            legends     = [[f"hb0_{setup_1}", f"hb0_{setup_2}"]],
                            title       = f"n_noisy_ch_hb0",
                            name        = f"Plot_NoisyChannels_hybrid0_allModules_compare",
                            xticklabels = moduleIDs,
                            #ylim        = [4.0,8.0],
                            ylabel      = "nNoisyChannels",
                            outdir      = self.outdir,
                            marker      = "o",
                            markerfacecolor=None,
                            markersize  = 7.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0,
                            tick_offset = tick_offset)
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[np.concatenate((n_noisy_channels_hb1_set_1[:,None], np.zeros_like(n_noisy_channels_hb1_set_1)[:,None]), axis=1).tolist(),
                                            np.concatenate((n_noisy_channels_hb1_set_2[:,None], np.zeros_like(n_noisy_channels_hb1_set_2)[:,None]), axis=1).tolist()]],
                            legends     = [[f"hb1_{setup_1}", f"hb1_{setup_2}"]],
                            title       = f"n_noisy_ch_hb1",
                            name        = f"Plot_NoisyChannels_hybrid1_allModules_compare",
                            xticklabels = moduleIDs,
                            #ylim        = [4.0,8.0],
                            ylabel      = "nNoisyChannels",
                            outdir      = self.outdir,
                            marker      = "o",
                            markerfacecolor=None,
                            markersize  = 7.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0,
                            tick_offset = tick_offset)


            n_dead_channels_hb0_set_1 = np.array(num_dead_channels_hb0_setup[setup_1])
            n_dead_channels_hb1_set_1 = np.array(num_dead_channels_hb1_setup[setup_1])
            n_dead_channels_hb0_set_2 = np.array(num_dead_channels_hb0_setup[setup_2])
            n_dead_channels_hb1_set_2 = np.array(num_dead_channels_hb1_setup[setup_2])
                
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[np.concatenate((n_dead_channels_hb0_set_1[:,None], np.zeros_like(n_dead_channels_hb0_set_1)[:,None]), axis=1).tolist(),
                                            np.concatenate((n_dead_channels_hb0_set_2[:,None], np.zeros_like(n_dead_channels_hb0_set_2)[:,None]), axis=1).tolist()]],
                            legends     = [[f"hb0_{setup_1}", f"hb0_{setup_2}"]],
                            title       = f"n_broken_ch_hb0",
                            name        = f"Plot_BrokenChannels_hybrid0_allModules_compare",
                            xticklabels = moduleIDs,
                            #ylim        = [4.0,8.0],
                            ylabel      = "nBrokenChannels",
                            outdir      = self.outdir,
                            marker      = "o",
                            markerfacecolor=None,
                            markersize  = 7.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0,
                            tick_offset = tick_offset)
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[np.concatenate((n_dead_channels_hb1_set_1[:,None], np.zeros_like(n_dead_channels_hb1_set_1)[:,None]), axis=1).tolist(),
                                            np.concatenate((n_dead_channels_hb1_set_2[:,None], np.zeros_like(n_dead_channels_hb1_set_2)[:,None]), axis=1).tolist()]],
                            legends     = [[f"hb1_{setup_1}", f"hb1_{setup_2}"]],
                            title       = f"n_broken_ch_hb1",
                            name        = f"Plot_BrokenChannels_hybrid1_allModules_compare",
                            xticklabels = moduleIDs,
                            #ylim        = [4.0,8.0],
                            ylabel      = "nBrokenChannels",
                            outdir      = self.outdir,
                            marker      = "o",
                            markerfacecolor=None,
                            markersize  = 7.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0,
                            tick_offset = tick_offset)

                
                
            if self.testinfo.get("check_sensor_temperature") == True:
                # ===>> Sensor temperature
                sensor_temps_setup_1 = sensor_temps_setup[setup_1]
                sensor_temps_setup_2 = sensor_temps_setup[setup_2]
                    
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [sensor_temps_setup_1, sensor_temps_setup_2],
                                legends    = [f"{setup_1}", f"{setup_2}"],
                                title      = f"Sensor Temperature",
                                name       = f"Plot_SensorTemp_allModules_compare",
                                xticklabels= moduleIDs,
                                ylabel     = "Sensor Temperature (deg C)",
                                marker     = "o",
                                linewidth  = 1.5,
                                markersize = 4.5,
                                #ylim       = [18.0,29.0],
                                outdir     = self.outdir,
                                tick_offset = tick_offset)
            else:
                logger.warning("skip comparing sensor temperature")

            
            # ===>> strip noise hybrid 0
            strip_noise_hb0_setup_1 = strip_noise_hb0_setup[setup_1]
            strip_noise_hb0_setup_2 = strip_noise_hb0_setup[setup_2]
            # box plot
            self.plot_box(data_list_1 = strip_noise_hb0_setup_1,
                          data_list_2 = strip_noise_hb0_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hb0",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid0_compare",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          offset      = box_offset,
                          outdir      = self.outdir,
                          box_offset  = box_offset)
            # group plot
            strip_noise_hb0_setup_1_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_setup_1)[:,:,0])
            strip_noise_hb0_setup_2_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_setup_2)[:,:,0])
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[strip_noise_hb0_setup_1_mean_std,
                                            strip_noise_hb0_setup_2_mean_std]],
                            legends     = [[f"{setup_1}", f"{setup_2}"]],
                            title       = f"StripNoise_hb0",
                            name        = f"Plot_StripNoise_allModules_hybrid0_compare",
                            xticklabels = moduleIDs,
                            ylim        = [4.0,8.0],
                            ylabel      = "Noise [VcTh]",
                            outdir      = self.outdir,
                            marker      = "o",
                            markerfacecolor=None,
                            markersize  = 5.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0,
                            tick_offset = tick_offset)
            

            # ===>> strip noise hybrid 1
            # box plot
            strip_noise_hb1_setup_1 = strip_noise_hb1_setup[setup_1]
            strip_noise_hb1_setup_2 = strip_noise_hb1_setup[setup_2]
            self.plot_box(data_list_1 = strip_noise_hb1_setup_1,
                          data_list_2 = strip_noise_hb1_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hb1",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid1_compare",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          offset      = box_offset,
                          outdir      = self.outdir,
                          box_offset  = box_offset)
            # group plot
            strip_noise_hb1_setup_1_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_setup_1)[:,:,0])
            strip_noise_hb1_setup_2_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_setup_2)[:,:,0])
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[strip_noise_hb1_setup_1_mean_std,
                                            strip_noise_hb1_setup_2_mean_std]],
                            legends     = [[f"{setup_1}", f"{setup_2}"]],
                            title       = f"StripNoise_hb1",
                            name        = f"Plot_StripNoise_allModules_hybrid1_compare",
                            xticklabels = moduleIDs,
                            ylim        = [4.0,8.0],
                            ylabel      = "Noise [VcTh]",
                            outdir      = self.outdir,
                            marker      = "o",
                            markerfacecolor=None,
                            markersize  = 5.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0,
                            tick_offset = tick_offset)
                
                
            # ===>> strip noise hybrid 0 (bottom)
            # box plot
            strip_noise_hb0_bot_setup_1 = strip_noise_hb0_bot_setup[setup_1]
            strip_noise_hb0_bot_setup_2 = strip_noise_hb0_bot_setup[setup_2]
            self.plot_box(data_list_1 = strip_noise_hb0_bot_setup_1,
                          data_list_2 = strip_noise_hb0_bot_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hb0_bottom",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid0_bottomSensor_compare",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          offset      =	box_offset,
                          outdir      = self.outdir,
                          box_offset  = box_offset)
            # group plot
            strip_noise_hb0_bot_setup_1_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_bot_setup_1)[:,:,0])
            strip_noise_hb0_bot_setup_2_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_bot_setup_2)[:,:,0])
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[strip_noise_hb0_bot_setup_1_mean_std,
                                            strip_noise_hb0_bot_setup_2_mean_std]],
                            legends     = [[f"{setup_1}", f"{setup_2}"]],
                            title       = f"StripNoise_hb0_bottom",
                            name        = f"Plot_StripNoise_allModules_hybrid0_bottomSensor_compare",
                            xticklabels = moduleIDs,
                            ylim        = [4.0,8.0],
                            ylabel      = "Noise [VcTh]",
                            outdir      = self.outdir,
                            marker      = "o",
                            markerfacecolor=None,
                            markersize  = 5.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0,
                            tick_offset = tick_offset)
            
            # ===>> strip noise hybrid 1 (bottom)
            # box plot        
            strip_noise_hb1_bot_setup_1 = strip_noise_hb1_bot_setup[setup_1]
            strip_noise_hb1_bot_setup_2 = strip_noise_hb1_bot_setup[setup_2]
            self.plot_box(data_list_1 = strip_noise_hb1_bot_setup_1,
                          data_list_2 = strip_noise_hb1_bot_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hb1_bottom",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid1_bottomSensor_compare",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          offset      = box_offset,
                          outdir      = self.outdir,
                          box_offset  = box_offset)
            # group plot
            strip_noise_hb1_bot_setup_1_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_bot_setup_1)[:,:,0])
            strip_noise_hb1_bot_setup_2_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_bot_setup_2)[:,:,0])
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[strip_noise_hb1_bot_setup_1_mean_std, strip_noise_hb1_bot_setup_2_mean_std]],
                            legends     = [[f"{setup_1}", f"{setup_2}"]],
                            title       = f"StripNoise_hb1_bottom",
                            name        = f"Plot_StripNoise_allModules_hybrid1_bottomSensor_compare",
                            xticklabels = moduleIDs,
                            ylim        = [4.0,8.0],
                            ylabel      = "Noise [VcTh]",
                            outdir      = self.outdir,
                            marker      = "o",
                            markerfacecolor=None,
                            markersize  = 5.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0,
                            tick_offset = tick_offset)
            

            # ===>> strip noise hybrid 0 (top)
            # box plot
            strip_noise_hb0_top_setup_1 = strip_noise_hb0_top_setup[setup_1]
            strip_noise_hb0_top_setup_2 = strip_noise_hb0_top_setup[setup_2]
            self.plot_box(data_list_1 = strip_noise_hb0_top_setup_1,
                          data_list_2 = strip_noise_hb0_top_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hb0_top",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid0_topSensor_compare",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          offset      = box_offset,
                          outdir      = self.outdir,
                          box_offset  = box_offset)
            # group plot
            strip_noise_hb0_top_setup_1_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_top_setup_1)[:,:,0])
            strip_noise_hb0_top_setup_2_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb0_top_setup_2)[:,:,0])
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[strip_noise_hb0_top_setup_1_mean_std, strip_noise_hb0_top_setup_2_mean_std]],
                            legends     = [[f"{setup_1}", f"{setup_2}"]],
                            title       = f"StripNoise_hb0_top",
                            name        = f"Plot_StripNoise_allModules_hybrid0_topSensor_compare",
                            xticklabels = moduleIDs,
                            ylim        = [4.0,8.0],
                            ylabel      = "Noise [VcTh]",
                            outdir      = self.outdir,
                            marker      = "o",
                            markerfacecolor=None,
                            markersize  = 5.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0,
                            tick_offset = tick_offset)
            
        
            # ===>> strip noise hybrid 1 (top)
            # box plot
            strip_noise_hb1_top_setup_1 = strip_noise_hb1_top_setup[setup_1]
            strip_noise_hb1_top_setup_2 = strip_noise_hb1_top_setup[setup_2]
            self.plot_box(data_list_1 = strip_noise_hb1_top_setup_1,
                          data_list_2 = strip_noise_hb1_top_setup_2,
                          legends     = [f"{setup_1}", f"{setup_2}"],
                          title       = f"StripNoise_hb1_top",
                          name        = f"Plot_StripNoiseBox_allModules_Hybrid1_topSensor_compare",
                          xticklabels = moduleIDs,
                          ylabel      = "Noise [VcTh]",
                          offset      = box_offset,
                          outdir      = self.outdir,
                          box_offset  = box_offset)
            # group plot
            strip_noise_hb1_top_setup_1_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_top_setup_1)[:,:,0])
            strip_noise_hb1_top_setup_2_mean_std = get_noise_mean_sigma_for_plotting(np.array(strip_noise_hb1_top_setup_2)[:,:,0])
            self.plot_group(x           = np.arange(len(moduleIDs)),
                            data_list   = [[strip_noise_hb1_top_setup_1_mean_std,
                                            strip_noise_hb1_top_setup_2_mean_std]],
                            legends     = [[f"{setup_1}", f"{setup_2}"]],
                            title       = f"StripNoise_hb1_top",
                            name        = f"Plot_StripNoise_allModules_hybrid1_topSensor_compare",
                            xticklabels = moduleIDs,
                            ylim        = [4.0,8.0],
                            ylabel      = "Noise [VcTh]",
                            outdir      = self.outdir,
                            marker      = "o",
                            markerfacecolor=None,
                            markersize  = 5.5,
                            capsize     = 1.5,
                            elinewidth  = 1.0,
                            tick_offset = tick_offset)

        
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
            #        Comparing common mode noise             #
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
            if self.testinfo.get("check_common_noise") == True:
                # ===>> nHits mean and std
                cmn_noise_setup_1_mean_std, cmn_noise_setup_1_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_setup[setup_1])[:,:,0])
                cmn_noise_setup_2_mean_std, cmn_noise_setup_2_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_setup[setup_2])[:,:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_setup_1_mean_std, cmn_noise_setup_2_mean_std],
                                legends    = [f"{setup_1}", f"{setup_2}"],
                                title      = f"#hits (50% Occ) (µ)",
                                name       = f"Plot_nHitsMean_module_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (µ)",
                                markersize = 10,
                                ylim       = [1200.0,2200.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_setup_1_sigma_std, cmn_noise_setup_2_sigma_std],
                                legends    = [f"{setup_1}", f"{setup_2}"],
                                title      = f"#hits (50% Occ) (σ)",
                                name       = f"Plot_nHitsStd_module_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (σ)",
                                markersize = 10,
                                ylim       = [50.0,250.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset=tick_offset)
                # ===>> Extract CMN
                CMN_setup_1 = extractCMN(nchannels = np.array(common_noise_setup[setup_1]).shape[1],
                                         mean = cmn_noise_setup_1_mean_std[:,0],
                                         std = cmn_noise_setup_1_sigma_std[:,0])
                CMN_setup_2 = extractCMN(nchannels = np.array(common_noise_setup[setup_2]).shape[1],
                                         mean = cmn_noise_setup_2_mean_std[:,0],
                                         std = cmn_noise_setup_2_sigma_std[:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [CMN_setup_1, CMN_setup_2],
                                legends    = [f"{setup_1}", f"{setup_2}"],
                                title      = f"CMNoise_fraction",
                                name       = f"Plot_CMN_frac_module_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "CMN (%)",
                                markersize = 10,
                                ylim       = [0.0,10.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
            

                # ===>>> nHits top and bottom
                cmn_noise_bot_setup_1_mean_std, cmn_noise_bot_setup_1_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_bot_setup[setup_1])[:,:,0])
                cmn_noise_bot_setup_2_mean_std, cmn_noise_bot_setup_2_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_bot_setup[setup_2])[:,:,0])
                cmn_noise_top_setup_1_mean_std, cmn_noise_top_setup_1_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_top_setup[setup_1])[:,:,0]) 
                cmn_noise_top_setup_2_mean_std, cmn_noise_top_setup_2_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_top_setup[setup_2])[:,:,0])
                
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_bot_setup_1_mean_std, cmn_noise_bot_setup_2_mean_std,
                                              cmn_noise_top_setup_1_mean_std, cmn_noise_top_setup_2_mean_std],
                                legends    = [f"bot: {setup_1}", f"bot: {setup_2}",
                                              f"top: {setup_1}", f"top: {setup_2}"],
                                title      = f"#hits (50%Occ) (µ) : sensors",
                                name       = f"Plot_nHitsMean_module_bothSensors_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (µ)",
                                markersize = 10,
                                ylim       = [500.0,1200.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_bot_setup_1_sigma_std, cmn_noise_bot_setup_2_sigma_std,
                                              cmn_noise_top_setup_1_sigma_std, cmn_noise_top_setup_2_sigma_std],
                                legends    = [f"bot: {setup_1}", f"bot: {setup_2}",
                                              f"top: {setup_1}", f"top: {setup_2}"],
                                title      = f"#hits (50% Occ) (σ) : sensors",
                                name       = f"Plot_nHitsStd_module_bothSensors_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (σ)",
                                markersize = 10,
                                ylim       = [0.0,200.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                # ===>> Extract CMN
                CMN_bot_setup_1 = extractCMN(nchannels = np.array(common_noise_bot_setup[setup_1]).shape[1],
                                                    mean = cmn_noise_bot_setup_1_mean_std[:,0],
                                                    std = cmn_noise_bot_setup_1_sigma_std[:,0])
                CMN_bot_setup_2 = extractCMN(nchannels = np.array(common_noise_bot_setup[setup_2]).shape[1],
                                                    mean = cmn_noise_bot_setup_2_mean_std[:,0],
                                                    std = cmn_noise_bot_setup_2_sigma_std[:,0])
                CMN_top_setup_1 = extractCMN(nchannels = np.array(common_noise_top_setup[setup_1]).shape[1],
                                                    mean = cmn_noise_top_setup_1_mean_std[:,0],
                                                    std = cmn_noise_top_setup_1_sigma_std[:,0])
                CMN_top_setup_2 = extractCMN(nchannels = np.array(common_noise_top_setup[setup_2]).shape[1],
                                                    mean = cmn_noise_top_setup_2_mean_std[:,0],
                                                    std = cmn_noise_top_setup_2_sigma_std[:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [CMN_bot_setup_1, CMN_bot_setup_2,
                                              CMN_top_setup_1, CMN_top_setup_2],
                                legends    = [f"bot: {setup_1}", f"bot: {setup_2}",
                                              f"top: {setup_1}", f"top: {setup_2}"],
                                title      = f"CMNoise (sensors)",
                                name       = f"Plot_CMN_module_bothSensors_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "CMN Fraction",
                                markersize = 10,
                                ylim       = [0.0,40.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                

                # nHits both hybrids
                cmn_noise_hb0_setup_1_mean_std, cmn_noise_hb0_setup_1_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_setup[setup_1])[:,:,0])
                cmn_noise_hb0_setup_2_mean_std, cmn_noise_hb0_setup_2_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb0_setup[setup_2])[:,:,0])
                cmn_noise_hb1_setup_1_mean_std, cmn_noise_hb1_setup_1_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_setup[setup_1])[:,:,0]) 
                cmn_noise_hb1_setup_2_mean_std, cmn_noise_hb1_setup_2_sigma_std = get_cmn_mean_sigma_for_plotting(np.array(common_noise_hb1_setup[setup_2])[:,:,0])
                
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_setup_1_mean_std, cmn_noise_hb0_setup_2_mean_std,
                                              cmn_noise_hb1_setup_1_mean_std, cmn_noise_hb1_setup_2_mean_std],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"#hits (50% Occ) (µ) : hybrids",
                                name       = f"Plot_nHitsMean_module_bothHybrids_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (µ)",
                                markersize = 10,
                                ylim       = [500.0,1200.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_setup_1_sigma_std, cmn_noise_hb0_setup_2_sigma_std,
                                              cmn_noise_hb1_setup_1_sigma_std, cmn_noise_hb1_setup_2_sigma_std],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"#hits (50% Occ) (σ) : hybrids",
                                name       = f"Plot_nHitsStd_module_bothHybrids_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (σ)",
                                markersize = 10,
                                ylim       = [50.0,300.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                    
                CMN_hb0_setup_1 = extractCMN(nchannels = np.array(common_noise_hb0_setup[setup_1]).shape[1],
                                             mean = cmn_noise_hb0_setup_1_mean_std[:,0],
                                             std = cmn_noise_hb0_setup_1_sigma_std[:,0])
                CMN_hb0_setup_2 = extractCMN(nchannels = np.array(common_noise_hb0_setup[setup_2]).shape[1],
                                             mean = cmn_noise_hb0_setup_2_mean_std[:,0],
                                             std = cmn_noise_hb0_setup_2_sigma_std[:,0])
                CMN_hb1_setup_1 = extractCMN(nchannels = np.array(common_noise_hb1_setup[setup_1]).shape[1],
                                             mean = cmn_noise_hb1_setup_1_mean_std[:,0],
                                             std = cmn_noise_hb1_setup_1_sigma_std[:,0])
                CMN_hb1_setup_2 = extractCMN(nchannels = np.array(common_noise_hb1_setup[setup_2]).shape[1],
                                             mean = cmn_noise_hb1_setup_2_mean_std[:,0],
                                             std = cmn_noise_hb1_setup_2_sigma_std[:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [CMN_hb0_setup_1, CMN_hb0_setup_2,
                                              CMN_hb1_setup_1, CMN_hb1_setup_2],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"CMNoise frac (hybrids)",
                                name       = f"Plot_CMN_fraction_module_bothHybrids_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "CMN Fraction",
                                markersize = 10,
                                ylim       = [0.0,40.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                    
                
                
                cmn_noise_hb0_bot_setup_1_mean_std, cmn_noise_hb0_bot_setup_1_sigma_std = get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb0_bot_setup[setup_1])[:,:,0]
                )
                cmn_noise_hb0_bot_setup_2_mean_std, cmn_noise_hb0_bot_setup_2_sigma_std = get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb0_bot_setup[setup_2])[:,:,0]
                )
                    
                cmn_noise_hb1_bot_setup_1_mean_std, cmn_noise_hb1_bot_setup_1_sigma_std = get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb1_bot_setup[setup_1])[:,:,0]
                )
                cmn_noise_hb1_bot_setup_2_mean_std, cmn_noise_hb1_bot_setup_2_sigma_std = get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb1_bot_setup[setup_2])[:,:,0]
                )
                
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_bot_setup_1_mean_std, cmn_noise_hb0_bot_setup_2_mean_std,
                                              cmn_noise_hb1_bot_setup_1_mean_std, cmn_noise_hb1_bot_setup_2_mean_std],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"#hits (50% Occ) botSensor (µ)",
                                name       = f"Plot_nHitsMean_module_bothHybrids_bottomSensor_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (µ)",
                                markersize = 10,
                                ylim       = [200.0,600.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_bot_setup_1_sigma_std, cmn_noise_hb0_bot_setup_2_sigma_std,
                                              cmn_noise_hb1_bot_setup_1_sigma_std, cmn_noise_hb1_bot_setup_2_sigma_std],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"#hits (50% Occ) botSensor (σ)",
                                name       = f"Plot_nHitsStd_module_bothHybrids_bottomSensor_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (σ)",
                                markersize = 10,
                                ylim       = [10.0,160.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset=tick_offset)
                    
                CMN_hb0_bot_setup_1 = extractCMN(nchannels = np.array(common_noise_hb0_bot_setup[setup_1]).shape[1],
                                                 mean = cmn_noise_hb0_bot_setup_1_mean_std[:,0],
                                                 std = cmn_noise_hb0_bot_setup_1_sigma_std[:,0])
                CMN_hb0_bot_setup_2 = extractCMN(nchannels = np.array(common_noise_hb0_bot_setup[setup_2]).shape[1],
                                                 mean = cmn_noise_hb0_bot_setup_2_mean_std[:,0],
                                                 std = cmn_noise_hb0_bot_setup_2_sigma_std[:,0])
                CMN_hb1_bot_setup_1 = extractCMN(nchannels = np.array(common_noise_hb1_bot_setup[setup_1]).shape[1],
                                                 mean = cmn_noise_hb1_bot_setup_1_mean_std[:,0],
                                                 std = cmn_noise_hb1_bot_setup_1_sigma_std[:,0])
                CMN_hb1_bot_setup_2 = extractCMN(nchannels = np.array(common_noise_hb1_bot_setup[setup_2]).shape[1],
                                                 mean = cmn_noise_hb1_bot_setup_2_mean_std[:,0],
                                                 std = cmn_noise_hb1_bot_setup_2_sigma_std[:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [CMN_hb0_bot_setup_1, CMN_hb0_bot_setup_2,
                                              CMN_hb1_bot_setup_1, CMN_hb1_bot_setup_2],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"CMNoise (bottom sensor)",
                                name       = f"Plot_CMN_module_bothHybrids_bottomSensor_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "CMN Fraction",
                                markersize = 10,
                                ylim       = [0.0,40.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                
                cmn_noise_hb0_top_setup_1_mean_std, cmn_noise_hb0_top_setup_1_sigma_std = get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb0_top_setup[setup_1])[:,:,0]
                )
                cmn_noise_hb0_top_setup_2_mean_std, cmn_noise_hb0_top_setup_2_sigma_std = get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb0_top_setup[setup_2])[:,:,0]
                )
                
                cmn_noise_hb1_top_setup_1_mean_std, cmn_noise_hb1_top_setup_1_sigma_std = get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb1_top_setup[setup_1])[:,:,0]
                )
                cmn_noise_hb1_top_setup_2_mean_std, cmn_noise_hb1_top_setup_2_sigma_std = get_cmn_mean_sigma_for_plotting(
                    np.array(common_noise_hb1_top_setup[setup_2])[:,:,0]
                )
                    
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_top_setup_1_mean_std, cmn_noise_hb0_top_setup_2_mean_std,
                                              cmn_noise_hb1_top_setup_1_mean_std, cmn_noise_hb1_top_setup_2_mean_std],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"#hits (50% Occ) topSensor (µ)",
                                name       = f"Plot_nHitsMean_module_bothHybrids_topSensor_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (µ)",
                                markersize = 10,
                                ylim       = [200.0,600.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset = tick_offset)
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [cmn_noise_hb0_top_setup_1_sigma_std, cmn_noise_hb0_top_setup_2_sigma_std,
                                              cmn_noise_hb1_top_setup_1_sigma_std, cmn_noise_hb1_top_setup_2_sigma_std],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"#hits (50% Occ) topSensor (σ)",
                                name       = f"Plot_nHitsStd_module_bothHybrids_topSensor_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "#hits (σ)",
                                markersize = 10,
                                ylim       = [10.0,160.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset = tick_offset)
                
                CMN_hb0_top_setup_1 = extractCMN(nchannels = np.array(common_noise_hb0_top_setup[setup_1]).shape[1],
                                                 mean = cmn_noise_hb0_top_setup_1_mean_std[:,0],
                                                 std = cmn_noise_hb0_top_setup_1_sigma_std[:,0])
                CMN_hb0_top_setup_2 = extractCMN(nchannels = np.array(common_noise_hb0_top_setup[setup_2]).shape[1],
                                                 mean = cmn_noise_hb0_top_setup_2_mean_std[:,0],
                                                 std = cmn_noise_hb0_top_setup_2_sigma_std[:,0])
                CMN_hb1_top_setup_1 = extractCMN(nchannels = np.array(common_noise_hb1_top_setup[setup_1]).shape[1],
                                                 mean = cmn_noise_hb1_top_setup_1_mean_std[:,0],
                                                 std = cmn_noise_hb1_top_setup_1_sigma_std[:,0])
                CMN_hb1_top_setup_2 = extractCMN(nchannels = np.array(common_noise_hb1_top_setup[setup_2]).shape[1],
                                                 mean = cmn_noise_hb1_top_setup_2_mean_std[:,0],
                                                 std = cmn_noise_hb1_top_setup_2_sigma_std[:,0])
                self.plot_basic(x          = np.arange(len(moduleIDs)),
                                data_list  = [CMN_hb0_top_setup_1, CMN_hb0_top_setup_2,
                                              CMN_hb1_top_setup_1, CMN_hb1_top_setup_2],
                                legends    = [f"hb0: {setup_1}", f"hb0: {setup_2}",
                                              f"hb1: {setup_1}", f"hb1: {setup_2}"],
                                title      = f"CMNoise (top sensor)",
                                name       = f"Plot_CMN_module_bothHybrids_topSensor_compare",
                                xticklabels = moduleIDs,
                                ylabel     = "CMN Fraction",
                                markersize = 10,
                                ylim       = [0.0,40.0],
                                outdir     = self.outdir,
                                fit        = False,
                                tick_offset= tick_offset)
                
            else:
                logger.warning(f"skip comparing common mode noise between {setup_1} and {setup_2}")

 
            LightYield_total_setup_1 = LightYield_total_setup[setup_1]
            LightYield_total_setup_2 = LightYield_total_setup[setup_2]
            
            self.plot_basic(x          = np.arange(len(moduleIDs)),
                            data_list  = [np.concatenate((LightYield_total_setup_1[:,None],
                                                          np.zeros_like(LightYield_total_setup_1)[:,None]),
                                                         axis=1).tolist(),
                                          np.concatenate((LightYield_total_setup_2[:,None],
                                                          np.zeros_like(LightYield_total_setup_2)[:,None]),
                                                         axis=1).tolist()],
                            legends    = [f"{setup_1}", f"{setup_2}"],
                            title      = f"LightYield_total",
                            name       = f"Plot_LightYield_total_compare",
                            xticklabels = moduleIDs,
                            ylabel     = "VTRX Total Light Yield",
                            linewidth  = 1.5,
                            markersize = 10,
                            ylim       = [20000.0,40000.0],
                            outdir     = self.outdir,
                            fit        = False,
                            tick_offset= tick_offset)
            
            LightYield_avg_diff_setup_1 = LightYield_avg_diff_setup[setup_1]
            LightYield_avg_diff_setup_2 = LightYield_avg_diff_setup[setup_2]
            
            self.plot_basic(x          = np.arange(len(moduleIDs)),
                            data_list  = [np.concatenate((LightYield_avg_diff_setup_1[:,None],
                                                          np.zeros_like(LightYield_avg_diff_setup_1)[:,None]),
                                                         axis=1).tolist(),
                                          np.concatenate((LightYield_avg_diff_setup_2[:,None],
                                                          np.zeros_like(LightYield_avg_diff_setup_2)[:,None]),
                                                         axis=1).tolist()],
                            legends    = [f"{setup_1}", f"{setup_2}"],
                            title      = f"LightYield_avg_diff",
                            name       = f"Plot_LightYield_avg_diff_compare",
                            xticklabels = moduleIDs,
                            ylabel     = "VTRX LightYield avg diff",
                            linewidth  = 1.5,
                            markersize = 10,
                            ylim       = [0.7,1.3],
                            outdir     = self.outdir,
                            fit        = False,
                            tick_offset= tick_offset)


            LightYield_mod_slope_setup_1 = LightYield_mod_slope_setup[setup_1]
            LightYield_mod_slope_setup_2 = LightYield_mod_slope_setup[setup_2]
            
            self.plot_basic(x          = np.arange(len(moduleIDs)),
                            data_list  = [np.concatenate((LightYield_mod_slope_setup_1[:,None],
                                                          np.zeros_like(LightYield_mod_slope_setup_1)[:,None]),
                                                         axis=1).tolist(),
                                          np.concatenate((LightYield_mod_slope_setup_2[:,None],
                                                          np.zeros_like(LightYield_mod_slope_setup_2)[:,None]),
                                                         axis=1).tolist()],
                            legends    = [f"{setup_1}", f"{setup_2}"],
                            title      = f"LightYield_mod_slope",
                            name       = f"Plot_LightYield_mod_slope_compare",
                            xticklabels = moduleIDs,
                            ylabel     = "VTRX LightYield Mod Slope",
                            linewidth  = 1.5,
                            markersize = 10,
                            ylim       = [-50.0,50.0],
                            outdir     = self.outdir,
                            fit        = False,
                            tick_offset= tick_offset)

            LightYield_bias_slope_setup_1 = LightYield_bias_slope_setup[setup_1]
            LightYield_bias_slope_setup_2 = LightYield_bias_slope_setup[setup_2]
            
            self.plot_basic(x          = np.arange(len(moduleIDs)),
                            data_list  = [np.concatenate((LightYield_bias_slope_setup_1[:,None],
                                                          np.zeros_like(LightYield_bias_slope_setup_1)[:,None]),
                                                         axis=1).tolist(),
                                          np.concatenate((LightYield_bias_slope_setup_2[:,None],
                                                          np.zeros_like(LightYield_bias_slope_setup_2)[:,None]),
                                                         axis=1).tolist()],
                            legends    = [f"{setup_1}", f"{setup_2}"],
                            title      = f"LightYield_bias_slope",
                            name       = f"Plot_LightYield_bias_slope_compare",
                            xticklabels = moduleIDs,
                            ylabel     = "VTRX LightYield Bias Slope",
                            linewidth  = 1.5,
                            markersize = 10,
                            ylim       = [-50.0,50.0],
                            outdir     = self.outdir,
                            fit        = False,
                            tick_offset= tick_offset)


            eye_cross_wd_ymax_0p3_setup_1 = eye_cross_frac_offset_0p3_setup[setup_1]
            eye_cross_wd_ymax_0p3_setup_2 = eye_cross_frac_offset_0p3_setup[setup_2]
            self.plot_basic(x          = np.arange(len(moduleIDs)),
                            data_list  = [np.concatenate((eye_cross_wd_ymax_0p3_setup_1[:,None],
                                                          np.zeros_like(eye_cross_wd_ymax_0p3_setup_1)[:,None]),
                                                         axis=1).tolist(),
                                          np.concatenate((eye_cross_wd_ymax_0p3_setup_2[:,None],
                                                          np.zeros_like(eye_cross_wd_ymax_0p3_setup_2)[:,None]),
                                                         axis=1).tolist()],
                            legends    = [f"{setup_1}", f"{setup_2}"],
                            title      = f"EyeOpn_p0p3_Cross_Wd_MaxV",
                            name       = f"Plot_EyeOpn_p0p3_Cross_Wd_MaxV_compare",
                            xticklabels = moduleIDs,
                            ylabel     = "EyeOpnening Cross Width MaxV",
                            linewidth  = 1.5,
                            markersize = 10,
                            ylim       = [0.0,50.0],
                            outdir     = self.outdir,
                            fit        = False,
                            tick_offset= tick_offset,
                            ncols      = 2)
            eye_cross_wd_ymax_0p7_setup_1 = eye_cross_frac_offset_0p7_setup[setup_1]
            eye_cross_wd_ymax_0p7_setup_2 = eye_cross_frac_offset_0p7_setup[setup_2]
            self.plot_basic(x          = np.arange(len(moduleIDs)),
                            data_list  = [np.concatenate((eye_cross_wd_ymax_0p7_setup_1[:,None],
                                                          np.zeros_like(eye_cross_wd_ymax_0p7_setup_1)[:,None]),
                                                         axis=1).tolist(),
                                          np.concatenate((eye_cross_wd_ymax_0p7_setup_2[:,None],
                                                          np.zeros_like(eye_cross_wd_ymax_0p7_setup_2)[:,None]),
                                                         axis=1).tolist()],
                            legends    = [f"{setup_1}", f"{setup_2}"],
                            title      = f"EyeOpn_p0p7_Cross_Wd_MaxV",
                            name       = f"Plot_EyeOpn_p0p7_Cross_Wd_MaxV_compare",
                            xticklabels = moduleIDs,
                            ylabel     = "EyeOpnening Cross Width MaxV",
                            linewidth  = 1.5,
                            markersize = 10,
                            ylim       = [0.0,50.0],
                            outdir     = self.outdir,
                            fit        = False,
                            tick_offset= tick_offset,
                            ncols      = 2)
            eye_cross_wd_ymax_1p0_setup_1 = eye_cross_frac_offset_1p0_setup[setup_1]
            eye_cross_wd_ymax_1p0_setup_2 = eye_cross_frac_offset_1p0_setup[setup_2]
            self.plot_basic(x          = np.arange(len(moduleIDs)),
                            data_list  = [np.concatenate((eye_cross_wd_ymax_1p0_setup_1[:,None],
                                                          np.zeros_like(eye_cross_wd_ymax_1p0_setup_1)[:,None]),
                                                         axis=1).tolist(),
                                          np.concatenate((eye_cross_wd_ymax_1p0_setup_2[:,None],
                                                          np.zeros_like(eye_cross_wd_ymax_1p0_setup_2)[:,None]),
                                                         axis=1).tolist()],
                            legends    = [f"{setup_1}", f"{setup_2}"],
                            title      = f"EyeOpn_p1p0_Cross_Wd_MaxV",
                            name       = f"Plot_EyeOpn_p1p0_Cross_Wd_MaxV_compare",
                            xticklabels = moduleIDs,
                            ylabel     = "EyeOpnening Cross Width MaxV",
                            linewidth  = 1.5,
                            markersize = 10,
                            ylim       = [0.0,50.0],
                            outdir     = self.outdir,
                            fit        = False,
                            tick_offset= tick_offset,
                            ncols      = 2)

            
        else:
            logger.warning("skip comparing two setup")
        
