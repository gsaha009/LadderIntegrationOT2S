import os
import sys
from datetime import datetime
import numpy as np
import ROOT
from glob import glob
#from util import *

import logging
logger = logging.getLogger('main')

class DataLoader:
    def __init__(self,
                 testinfo: dict,
                 fileinfo: dict,
                 **kwargs):
        """
          Description
        """
        self.testinfo = testinfo
        self.fileinfo = fileinfo
        self.outdir = kwargs.get("target", "../Output")
        self.dttag = kwargs.get("dttag", "")
        self.mask_noisy = kwargs.get("mask_noisy", False)
        self.mask_noise_level = kwargs.get("mask_noise_level", 15)

        self.in_kira = self.fileinfo['single_module_box']
        self.in_ladder = self.fileinfo['ladder']

        

    def __mask_noisy(self,arr):
        arr = np.array(arr)
        if self.mask_noisy:
            mask = arr > self.mask_noise_level
            return arr[~mask]
        else:
            return arr
        
    def __get_hist(self, root_ptr, **kwargs):
        hist = None
        namelist = kwargs.get("hnamelist", [])
        assert len(namelist) > 0, "list of histnames must not be empty"
        for name in namelist:
            hist  = root_ptr.Get(name)
            if isinstance(hist, ROOT.TH1F):
                break
        return hist


    def __get_hist_array(self, hist):
        hlist = []
        nbins = hist.GetNbinsX()
        for i in range(nbins):
            hlist.append([hist.GetBinContent(i+1),
                          hist.GetBinError(i+1)])
        #hist.delete()
        return hlist

    def __format_datetime(self, timestamps):
        readable_stamps = [datetime.fromtimestamp(ts) for ts in timestamps]
        formatted_stamps = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in readable_stamps]
        return formatted_stamps
    
    
    def __prepare_noise_data(self, root_ptr, ibrd, iopt, **kwargs):
        noise_dict = {}
        root_dir  = f"Detector/Board_{ibrd}/OpticalGroup_{iopt}"

        # Module level hists if any
        hname_mod = kwargs.get("hname_mod", [])
        hname_mod_bot = kwargs.get("hname_mod_bot", [])
        hname_mod_top = kwargs.get("hname_mod_top", [])

        
        hname_hb     = kwargs.get("hname_hb", [])
        hname_hb_bot = kwargs.get("hname_hb_bot", [])
        hname_hb_top = kwargs.get("hname_hb_top", [])

        hname_cbc     = kwargs.get("hname_cbc", [])
        hname_cbc_bot = kwargs.get("hname_cbc_bot", [])
        hname_cbc_top = kwargs.get("hname_cbc_top", [])
        
        cbc_level  = kwargs.get("cbc_level", True)
        noise_type = kwargs.get("noise_type", "") # strip or common
        


        if ("common" in noise_type) and (self.testinfo.get("check_common_noise") == True):
            #print(f"{root_dir}/D_B({ibrd})_{hname_mod}_OpticalGroup({iopt})")
            #print(f"{root_dir}/D_B({ibrd})_{hname_mod_bot}_OpticalGroup({iopt})")
            #print(f"{root_dir}/D_B({ibrd})_{hname_mod_top}_OpticalGroup({iopt})")

            hnames = [f"{root_dir}/D_B({ibrd})_{name}_OpticalGroup({iopt})" for name in hname_mod]
            noise_hist_mod = self.__get_hist(root_ptr, hnamelist=hnames)
            noise_hist_mod_list = self.__get_hist_array(noise_hist_mod)
            noise_dict[f"{noise_type}_noise_module"] = noise_hist_mod_list

            hnames = [f"{root_dir}/D_B({ibrd})_{name}_OpticalGroup({iopt})" for name in hname_mod_bot]
            noise_hist_mod_bot = self.__get_hist(root_ptr, hnamelist=hnames)
            noise_hist_mod_bot_list = self.__get_hist_array(noise_hist_mod_bot)
            noise_dict[f"{noise_type}_noise_module_bot"] = noise_hist_mod_bot_list

            hnames = [f"{root_dir}/D_B({ibrd})_{name}_OpticalGroup({iopt})" for name in hname_mod_top]
            noise_hist_mod_top = self.__get_hist(root_ptr, hnamelist=hnames)
            noise_hist_mod_top_list = self.__get_hist_array(noise_hist_mod_top)
            noise_dict[f"{noise_type}_noise_module_top"] = noise_hist_mod_top_list
        
        #hb1 = 2*iopt
        #hb2 = 2*iopt+1

        for idx in range(2):
            hb = 2*iopt + idx

            hnames = [f"{root_dir}/Hybrid_{hb}/D_B({ibrd})_O({iopt})_{name}_Hybrid({hb})" for name in hname_hb]
            ch_noise_hist_hb = self.__get_hist(root_ptr, hnamelist=hnames)
            ch_noise_hist_hb_list = self.__get_hist_array(ch_noise_hist_hb)
            noise_dict[f"{noise_type}_noise_hb{idx}"] = {'allCBC': ch_noise_hist_hb_list}

            hnames = [f"{root_dir}/Hybrid_{hb}/D_B({ibrd})_O({iopt})_{name}_Hybrid({hb})" for name in hname_hb_bot]
            ch_noise_hist_hb_bot = self.__get_hist(root_ptr, hnamelist=hnames)
            ch_noise_hist_hb_bot_list = self.__get_hist_array(ch_noise_hist_hb_bot)
            noise_dict[f"{noise_type}_noise_hb{idx}_bot"] = {'allCBC': ch_noise_hist_hb_bot_list}

            hnames = [f"{root_dir}/Hybrid_{hb}/D_B({ibrd})_O({iopt})_{name}_Hybrid({hb})" for name in hname_hb_top]
            ch_noise_hist_hb_top = self.__get_hist(root_ptr, hnamelist=hnames)
            ch_noise_hist_hb_top_list = self.__get_hist_array(ch_noise_hist_hb_top)
            noise_dict[f"{noise_type}_noise_hb{idx}_top"] = {'allCBC': ch_noise_hist_hb_top_list}


            if cbc_level == True:
                for icbc in range(8):
                    hnames = [f"{root_dir}/Hybrid_{hb}/CBC_{icbc}/D_B({ibrd})_O({iopt})_H({hb})_{name}_Chip({icbc})" for name in hname_cbc]
                    hnames = hnames + [f"{root_dir}/Hybrid_{hb}/Chip_{icbc}/D_B({ibrd})_O({iopt})_H({hb})_{name}_Chip({icbc})" for name in hname_cbc]
                    ch_noise_distr_hb_cbc = self.__get_hist(root_ptr, hnamelist=hnames)
                    ch_noise_distr_hb_cbc_list = self.__get_hist_array(ch_noise_distr_hb_cbc)
                    noise_dict[f"{noise_type}_noise_hb{idx}"][f"CBC_{icbc}"] = ch_noise_distr_hb_cbc_list

                    hnames = [f"{root_dir}/Hybrid_{hb}/CBC_{icbc}/D_B({ibrd})_O({iopt})_H({hb})_{name}_Chip({icbc})" for name in hname_cbc_bot]
                    hnames = hnames + [f"{root_dir}/Hybrid_{hb}/Chip_{icbc}/D_B({ibrd})_O({iopt})_H({hb})_{name}_Chip({icbc})" for name in hname_cbc_bot]
                    ch_noise_distr_hb_bot_cbc = self.__get_hist(root_ptr, hnamelist=hnames)
                    ch_noise_distr_hb_bot_cbc_list = self.__get_hist_array(ch_noise_distr_hb_bot_cbc)
                    noise_dict[f"{noise_type}_noise_hb{idx}_bot"][f"CBC_{icbc}"] = ch_noise_distr_hb_bot_cbc_list

                    hnames = [f"{root_dir}/Hybrid_{hb}/CBC_{icbc}/D_B({ibrd})_O({iopt})_H({hb})_{name}_Chip({icbc})" for name in hname_cbc_top]
                    hnames = hnames + [f"{root_dir}/Hybrid_{hb}/Chip_{icbc}/D_B({ibrd})_O({iopt})_H({hb})_{name}_Chip({icbc})" for name in hname_cbc_top]
                    ch_noise_distr_hb_top_cbc = self.__get_hist(root_ptr, hnamelist=hnames)
                    ch_noise_distr_hb_top_cbc_list = self.__get_hist_array(ch_noise_distr_hb_top_cbc)
                    noise_dict[f"{noise_type}_noise_hb{idx}_top"][f"CBC_{icbc}"] = ch_noise_distr_hb_top_cbc_list

                    if "common" in noise_type:
                        fit = ch_noise_distr_hb_cbc.GetFunction("chipFit")
                        if not fit:
                            logger.warning("... Cannot find fit function 'chipFit' in histogram ... skipping ...")
                            continue
                        params = {fit.GetParName(i):fit.GetParameters()[i] for i in range(fit.GetNpar())}

                        fit_bot = ch_noise_distr_hb_bot_cbc.GetFunction("chipFit")
                        params_bot = {fit_bot.GetParName(i):fit_bot.GetParameters()[i] for i in range(fit_bot.GetNpar())}

                        fit_top = ch_noise_distr_hb_top_cbc.GetFunction("chipFit")
                        params_top = {fit_top.GetParName(i):fit_top.GetParameters()[i] for i in range(fit_top.GetNpar())}

                        noise_dict[f"{noise_type}_noise_hb{idx}"][f"CBC_{icbc}_fit_params"] = params
                        noise_dict[f"{noise_type}_noise_hb{idx}_bot"][f"CBC_{icbc}_fit_params"] = params_bot
                        noise_dict[f"{noise_type}_noise_hb{idx}_top"][f"CBC_{icbc}_fit_params"] = params_top
                        #from IPython import embed; embed(); exit()
                        
        return noise_dict



    def __prepare_pede_data(self, root_ptr, ibrd, iopt, **kwargs):
        pede_dict = {}
        root_dir  = f"Detector/Board_{ibrd}/OpticalGroup_{iopt}"

        hname_cbc     = kwargs.get("hname_cbc", [])
                
        #hb1 = 2*iopt
        #hb2 = 2*iopt+1

        for idx in range(2):
            pede_dict[f"pedestal_hb{idx}"] = {}
            hb = 2*iopt + idx
            for icbc in range(8):
                hnames = [f"{root_dir}/Hybrid_{hb}/CBC_{icbc}/D_B({ibrd})_O({iopt})_H({hb})_{name}_Chip({icbc})" for name in hname_cbc]
                hnames = hnames + [f"{root_dir}/Hybrid_{hb}/Chip_{icbc}/D_B({ibrd})_O({iopt})_H({hb})_{name}_Chip({icbc})" for name in hname_cbc]
                ch_pede_distr_hb_cbc = self.__get_hist(root_ptr, hnamelist=hnames)
                ch_pede_distr_hb_cbc_list = self.__get_hist_array(ch_pede_distr_hb_cbc)
                pede_dict[f"pedestal_hb{idx}"][f"CBC_{icbc}"] = ch_pede_distr_hb_cbc_list
            
        return pede_dict

    


    def __get_noise_data_for_ladder(self, nboards, nopticals, root_file_info, module_info):
        main_noise_dict = {}

        for temperature_key, test_iter_dict in root_file_info.items():
            logger.info(f"Cooling temperature : {temperature_key}")
            for test_iter, file_dict in test_iter_dict.items():
                logger.info(f"Run : {test_iter}")

                root_file = file_dict["tfile_main"]
                logger.info(f"==> ROOT File : {test_iter} ==> {root_file}")
                dqm_file = file_dict["tfile_dqm"]
                logger.info(f"==> DQM ROOT File : {test_iter} ==> {dqm_file}")

                root_ptr = ROOT.TFile(root_file, "r")
                dqm_ptr  = ROOT.TFile(dqm_file, "r")
                
                # Iterating over nboards
                for ibrd in range(nboards):
                    logger.info(f"BeBoard : {ibrd}")
                    # Iterating over nOpticals
                    #for iopt in range(nopticals):
                    for iopt in range(nopticals[0], nopticals[1]):
                        logger.info(f"Optical Group : {iopt}")
                        module_tag = module_info[f"board_{ibrd}_optical_{iopt}"]

                        # $$$$$$$$$$$$$$$$$$$$$ Data Structure $$$$$$$$$$$$$$$$ #
                        if module_tag in main_noise_dict:
                            main_noise_dict[module_tag].update({
                                temperature_key: {
                                    test_iter: {}
                                }
                            })
                        else:
                            main_noise_dict[module_tag] = {
                                temperature_key: {
                                    test_iter: {}
                                }
                            }

                        if self.testinfo.get("check_sensor_temperature") == True:
                            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                            #              Sensor Temperature and timestamps from DQM ROOT file               #
                            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                            dqm_dir = f"Detector/Board_{ibrd}/OpticalGroup_{iopt}"
                            
                            graph_temp = dqm_ptr.Get(f"{dqm_dir}/D_B({ibrd})_LpGBT_DQM_SensorTemp_OpticalGroup({iopt})")
                            get_point_x = lambda graph_temp : [graph_temp.GetPointX(i) for i in range(graph_temp.GetN())]
                            get_point_y = lambda graph_temp : [graph_temp.GetPointY(i) for i in range(graph_temp.GetN())]
                            time_stamps  = get_point_x(graph_temp)
                            sensor_temps = get_point_y(graph_temp)
                            
                            # Save the timestamp & temp in the dict
                            main_noise_dict[module_tag][temperature_key][test_iter].update({"sensor_temps": sensor_temps})
                            main_noise_dict[module_tag][temperature_key][test_iter].update({"time_stamps" : self.__format_datetime(time_stamps)})
                        
                        else:
                            logger.warning("skip checking sensor temperature data")
                        

                        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                        #              Channel/STrip noise from the main results ROOT file                #
                        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                        strip_noise_dict = self.__prepare_noise_data(root_ptr,
                                                                     ibrd,
                                                                     iopt,
                                                                     hname_hb=self.fileinfo['strip_noise_hname_hybrid_level'],
                                                                     hname_hb_bot=self.fileinfo['strip_noise_hname_hybrid_level_bottom'],
                                                                     hname_hb_top=self.fileinfo['strip_noise_hname_hybrid_level_top'],
                                                                     hname_cbc=self.fileinfo['strip_noise_hname_chip_level'],
                                                                     hname_cbc_bot=self.fileinfo['strip_noise_hname_chip_level_bottom'],
                                                                     hname_cbc_top=self.fileinfo['strip_noise_hname_chip_level_top'],
                                                                     cbc_level=True,
                                                                     noise_type="strip")
                        
                        #main_noise_dict[module_tag][temperature_key][test_iter] = strip_noise_dict
                        main_noise_dict[module_tag][temperature_key][test_iter].update(strip_noise_dict)
                        
                        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                        #                Common mode noise from the main results ROOT file                #
                        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                        if self.testinfo.get("check_common_noise") == True:
                            common_noise_dict = self.__prepare_noise_data(root_ptr,
                                                                          ibrd,
                                                                          iopt,
                                                                          hname_mod=self.fileinfo['common_noise_hname_module_level'],
                                                                          hname_mod_bot=self.fileinfo['common_noise_hname_module_level_bottom'],
                                                                          hname_mod_top=self.fileinfo['common_noise_hname_module_level_top'],
                                                                          hname_hb=self.fileinfo['common_noise_hname_hybrid_level'],
                                                                          hname_hb_bot=self.fileinfo['common_noise_hname_hybrid_level_bottom'],
                                                                          hname_hb_top=self.fileinfo['common_noise_hname_hybrid_level_top'],
                                                                          hname_cbc=self.fileinfo['common_noise_hname_chip_level'],
                                                                          hname_cbc_bot=self.fileinfo['common_noise_hname_chip_level_bottom'],
                                                                          hname_cbc_top=self.fileinfo['common_noise_hname_chip_level_top'],
                                                                          cbc_level=True,
                                                                          noise_type="common")
                            
                            main_noise_dict[module_tag][temperature_key][test_iter].update(common_noise_dict)

                            if self.testinfo.get("fit_simultaneous_common_noise") == True:
                                common_3sigma_noise_dict = self.__prepare_noise_data(root_ptr,
                                                                                     ibrd,
                                                                                     iopt,
                                                                                     hname_mod=self.fileinfo['common_noise_3sigma_hname_module_level'],
                                                                                     hname_mod_bot=self.fileinfo['common_noise_3sigma_hname_module_level_bottom'],
                                                                                     hname_mod_top=self.fileinfo['common_noise_3sigma_hname_module_level_top'],
                                                                                     hname_hb=self.fileinfo['common_noise_3sigma_hname_hybrid_level'],
                                                                                     hname_hb_bot=self.fileinfo['common_noise_3sigma_hname_hybrid_level_bottom'],
                                                                                     hname_hb_top=self.fileinfo['common_noise_3sigma_hname_hybrid_level_top'],
                                                                                     hname_cbc=self.fileinfo['common_noise_3sigma_hname_chip_level'],
                                                                                     hname_cbc_bot=self.fileinfo['common_noise_3sigma_hname_chip_level_bottom'],
                                                                                     hname_cbc_top=self.fileinfo['common_noise_3sigma_hname_chip_level_top'],
                                                                                     cbc_level=True,
                                                                                     noise_type="common_3sigma")
                                
                                main_noise_dict[module_tag][temperature_key][test_iter].update(common_3sigma_noise_dict)

                            
                        else:
                            logger.warning("skip checking common mode noise")


                        if self.testinfo.get("check_pedestal") == True:
                            pede_dict = self.__prepare_pede_data(root_ptr,
                                                                 ibrd,
                                                                 iopt,
                                                                 hname_cbc=self.fileinfo['pede_hname_chip_level'])
                            main_noise_dict[module_tag][temperature_key][test_iter].update(pede_dict)
                        else:
                            logger.info("skip checking pedestal info")



        return main_noise_dict




    def __get_noise_data_for_kira(self, nboards, nopticals, root_file_info):
        main_noise_dict = {}

        from_db = False
        temperature_key = '+15 deg'

        for mod_key, test_iter_dict in root_file_info.items():
            logger.info(f"Module : {mod_key}")
            for test_iter, file_dict in test_iter_dict.items():
                logger.info(f"Run : {test_iter}")

                root_file = file_dict["tfile_main"]
                logger.info(f"==> ROOT File : {test_iter} ==> {root_file}")
                dqm_file = file_dict["tfile_dqm"]
                logger.info(f"==> DQM ROOT File : {test_iter} ==> {dqm_file}")
                root_ptr = ROOT.TFile(root_file, "r")
                if dqm_file == root_file:
                    logger.info(f"extracting environment temp : looks like files from DB")
                    from_db = True
                    
                root_ptr = ROOT.TFile(root_file, "r")
                dqm_ptr  = ROOT.TFile(dqm_file, "r")
                
                # Iterating over nboards
                for ibrd in range(nboards):
                    logger.info(f"BeBoard : {ibrd}")
                    # Iterating over nOpticals
                    for iopt in range(nopticals[0], nopticals[1]):
                        logger.info(f"Optical Group : {iopt}")
                        module_tag = mod_key

                        # $$$$$$$$$$$$$$$$$$$$$ Data Structure $$$$$$$$$$$$$$$$ #
                        if module_tag in main_noise_dict:
                            main_noise_dict[module_tag].update({
                                temperature_key: {
                                    test_iter: {}
                                }
                            })
                        else:
                            main_noise_dict[module_tag] = {
                                temperature_key: {
                                    test_iter: {}
                                }
                            }
                        
                        if self.testinfo.get("check_sensor_temperature") == True:
                            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                            #              Sensor Temperature and timestamps from DQM ROOT file               #
                            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                            dqm_dir = f"Detector/Board_{ibrd}/OpticalGroup_{iopt}"

                            graph_temp = None
                            if not from_db:
                                graph_temp = dqm_ptr.Get(f"{dqm_dir}/D_B({ibrd})_LpGBT_DQM_SensorTemp_OpticalGroup({iopt})")
                            else :
                                graph_temp = dqm_ptr.Get("Monitor/ENV_Temperature")
                                if not isinstance(graph_temp, ROOT.TGraph):
                                    graph_temp = dqm_ptr.Get("Monitor/ENV_temperature")
                            
                            get_point_x = lambda graph_temp : [graph_temp.GetPointX(i) for i in range(graph_temp.GetN())]
                            get_point_y = lambda graph_temp : [graph_temp.GetPointY(i) for i in range(graph_temp.GetN())]
                            time_stamps  = get_point_x(graph_temp)
                            sensor_temps = get_point_y(graph_temp)

                            # Save the timestamp & temp in the dict
                            main_noise_dict[module_tag][temperature_key][test_iter].update({"time_stamps" : self.__format_datetime(time_stamps)})
                            main_noise_dict[module_tag][temperature_key][test_iter].update({"sensor_temps": sensor_temps})
                        else:
                            logger.warning("skip checking sensor temperature data")
                            
                        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                        #              Channel/STrip noise from the main results ROOT file                #
                        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                        strip_noise_dict = self.__prepare_noise_data(root_ptr,
                                                                     ibrd,
                                                                     iopt,
                                                                     hname_hb=self.fileinfo['strip_noise_hname_hybrid_level'],
                                                                     hname_hb_bot=self.fileinfo['strip_noise_hname_hybrid_level_bottom'],
                                                                     hname_hb_top=self.fileinfo['strip_noise_hname_hybrid_level_top'],
                                                                     hname_cbc=self.fileinfo['strip_noise_hname_chip_level'],
                                                                     hname_cbc_bot=self.fileinfo['strip_noise_hname_chip_level_bottom'],
                                                                     hname_cbc_top=self.fileinfo['strip_noise_hname_chip_level_top'],
                                                                     cbc_level=True,
                                                                     noise_type="strip")
                        
                        main_noise_dict[module_tag][temperature_key][test_iter].update(strip_noise_dict)
                        
                        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                        #                Common mode noise from the main results ROOT file                #
                        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
                        if self.testinfo.get("check_common_noise") == True:
                            common_noise_dict = self.__prepare_noise_data(root_ptr,
                                                                          ibrd,
                                                                          iopt,
                                                                          hname_mod=self.fileinfo['common_noise_hname_module_level'],
                                                                          hname_mod_bot=self.fileinfo['common_noise_hname_module_level_bottom'],
                                                                          hname_mod_top=self.fileinfo['common_noise_hname_module_level_top'],
                                                                          hname_hb=self.fileinfo['common_noise_hname_hybrid_level'],
                                                                          hname_hb_bot=self.fileinfo['common_noise_hname_hybrid_level_bottom'],
                                                                          hname_hb_top=self.fileinfo['common_noise_hname_hybrid_level_top'],
                                                                          hname_cbc=self.fileinfo['common_noise_hname_chip_level'],
                                                                          hname_cbc_bot=self.fileinfo['common_noise_hname_chip_level_bottom'],
                                                                          hname_cbc_top=self.fileinfo['common_noise_hname_chip_level_top'],
                                                                          cbc_level=True,
                                                                          noise_type="common")
                            
                            main_noise_dict[module_tag][temperature_key][test_iter].update(common_noise_dict)

                            if self.testinfo.get("fit_simultaneous_common_noise") == True:
                                common_3sigma_noise_dict = self.__prepare_noise_data(root_ptr,
                                                                                     ibrd,
                                                                                     iopt,
                                                                                     hname_mod=self.fileinfo['common_noise_3sigma_hname_module_level'],
                                                                                     hname_mod_bot=self.fileinfo['common_noise_3sigma_hname_module_level_bottom'],
                                                                                     hname_mod_top=self.fileinfo['common_noise_3sigma_hname_module_level_top'],
                                                                                     hname_hb=self.fileinfo['common_noise_3sigma_hname_hybrid_level'],
                                                                                     hname_hb_bot=self.fileinfo['common_noise_3sigma_hname_hybrid_level_bottom'],
                                                                                     hname_hb_top=self.fileinfo['common_noise_3sigma_hname_hybrid_level_top'],
                                                                                     hname_cbc=self.fileinfo['common_noise_3sigma_hname_chip_level'],
                                                                                     hname_cbc_bot=self.fileinfo['common_noise_3sigma_hname_chip_level_bottom'],
                                                                                     hname_cbc_top=self.fileinfo['common_noise_3sigma_hname_chip_level_top'],
                                                                                     cbc_level=True,
                                                                                     noise_type="common_3sigma")
                                
                                main_noise_dict[module_tag][temperature_key][test_iter].update(common_3sigma_noise_dict)
                            

                        if self.testinfo.get("check_pedestal") == True:
                            pede_dict = self.__prepare_pede_data(root_ptr,
                                                                 ibrd,
                                                                 iopt,
                                                                 hname_cbc=self.fileinfo['pede_hname_chip_level'])
                            main_noise_dict[module_tag][temperature_key][test_iter].update(pede_dict)
                        else:
                            logger.info("skip checking pedestal info")

        return main_noise_dict


        

    def getData(self):
        """
          Read the dictionary contains file names and other conditions required for plotting
          
        """
        main_noise_dict = {}
        
        nboards        = self.fileinfo["n_boards"]
        #nopticals      = self.fileinfo["n_opticals"]
        nopticals      = [self.fileinfo["n_opticals_start"], self.fileinfo["n_opticals_stop"]]
        root_file_info = self.fileinfo["files"]

        if self.in_kira & self.in_ladder:
            raise RuntimeError("single_module_box & ladder: Both can not be yes")
        
        
        if self.in_ladder:
            module_info    = self.fileinfo["moduleinfo"]
            main_noise_dict = self.__get_noise_data_for_ladder(nboards,
                                                               nopticals,
                                                               root_file_info,
                                                               module_info)
        elif self.in_kira:
            main_noise_dict = self.__get_noise_data_for_kira(nboards,
                                                             nopticals,
                                                             root_file_info)
        
        else:
            raise RuntimeError("At least one of the single_module_box & ladder must be True")


        return main_noise_dict
