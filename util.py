import os
import ROOT
import numpy as np


def get_mean_std_per_cbc(noise_dict):
    out_list = []
    for i in range(8):
        key = f"CBC_{i}"
        temp = np.array(noise_dict[key])
        temp_mean = np.mean(temp[:,0])
        temp_std  = np.std(temp[:,0])
        temp_list = [temp_mean, temp_std]
        out_list.append(temp_list)
    
    return out_list


def get_mean_std_for_cmn(x, val, axis=0):
    mean = np.average(x, weights=val, axis=axis)
    sigma = np.sqrt(np.average((x - mean)**2, weights=val, axis=axis))
    return [round(mean,2), round(sigma,2)]
    
def get_mean_std_for_cmn_per_module(arr):
    x = np.arange(arr.shape[-1])
    arr = arr.tolist()
    outlist = []
    for item in arr:
        _arr = np.array(item)
        mean_sigma = get_mean_std_for_cmn(x, _arr)
        mean = mean_sigma[0]
        sigma = mean_sigma[1]
        outlist.append([float(mean), float(sigma)])
        
    return np.array(outlist)


def get_cmn_mean_std_per_cbc(noise_dict):
    out_list = []
    for i in range(8):
        key = f"CBC_{i}"
        temp = np.array(noise_dict[key])
        x = np.arange(temp.shape[0])
        mean_std = get_mean_std_for_cmn(x, temp[:,0])
        temp_mean = float(mean_std[0])
        temp_std  =	float(mean_std[1])
        temp_list =	[temp_mean, temp_std]
        out_list.append(temp_list)
            
    return out_list


def extractCMN_iphc(nchannels=None, mean=None, std=None):
    """
    Ref: https://indico.cern.ch/event/1465528/contributions/6170035/attachments/2949439/5184080/systemtest_1710_JT.pdf
    """
    alpha = 2*np.pi*(std**2-mean*(1-mean/nchannels))/(nchannels*(nchannels-1))
    cmn = np.sqrt(np.sin(alpha)/(1-np.sin(alpha)))
    return cmn


def extractCMN_giovanni(nchannels=None, mean=None, std=None):
    """
    Ref: CMNoiseFraction ( Giovanni's Note )
    """
    cmn = np.sqrt((1/(nchannels - 1)) * (std**2/(mean*(1 - (mean/nchannels))) - 1))
    return cmn
    
def extractCMN(nchannels=None, mean=None, std=None):
    cmn = extractCMN_iphc(nchannels=nchannels, mean=mean, std=std)*100
    return np.concatenate((cmn[:,None], np.zeros_like(cmn)[:,None]), axis=1)

def extractCMN_crude(nchannels=None, mean=None, std=None):
    std_expected = np.sqrt(mean)/2.0
    cmn = (std - std_expected)/std
    return cmn

def extractCMN_potato(hitsarr):
    val = np.array(hitsarr)[:,0]
    nch = float(val.shape[0])
    ch  = np.arange(nch)
    mask = ((ch < int(nch*0.2)) | (ch > int(nch*0.8)))
    val_pass = val[mask]
    cmn_frac = float(np.sum(val_pass)/np.sum(val))
    return cmn_frac




def get_cmn_mean_sigma_for_plotting(arr):
    arr_mean_sigma = get_mean_std_for_cmn_per_module(arr)
    arr_mean = np.concatenate((arr_mean_sigma[:,0:1],
                               np.zeros_like(arr_mean_sigma[:,0:1])), axis=1)
    arr_sigma = np.concatenate((arr_mean_sigma[:,1:2],
                                np.zeros_like(arr_mean_sigma[:,1:2])), axis=1)
    return arr_mean, arr_sigma


def get_noise_mean_sigma_for_plotting(arr):
    return np.concatenate((np.mean(arr, axis=1)[:,None], np.std(arr, axis=1)[:,None]), axis=1)


def rearrange_arrs(arr):
    arr = np.array(arr).reshape(2,-1).T.reshape(-1)
    return arr


def get_noisy_and_dead_channels(noise_array):
    noise = np.array(noise_array)[:,0]
    channels = np.arange(noise.shape[0])
    #median = np.median(noise)
    #mad = np.median(np.abs(noise - median))
    #modified_z_scores = 0.6745 * (noise - median) / mad
    #pos_mask = modified_z_scores > 3.5
    ##neg_mask = modified_z_scores < -2*3.5
    #neg_mask = noise < 3.0
    pos_mask = noise > 8.0
    neg_mask = noise < 4.0
    pos_outliers = channels[pos_mask]
    neg_outliers = channels[neg_mask]
    return pos_outliers.tolist(), neg_outliers.tolist()

def get_noisy_and_dead_channels_cbc(strip_noise_dict):
    noisy_ch_list = []
    dead_ch_list = []
    for i in range(8):
        hb0_noise_ = strip_noise_dict[f"CBC_{i}"]
        #from IPython import embed; embed(); exit()
        noisy_channels, dead_channels = get_noisy_and_dead_channels(hb0_noise_)
        noisy_ch_list.append(len(noisy_channels))
        dead_ch_list.append(len(dead_channels)+(254-len(hb0_noise_)))
    return noisy_ch_list, dead_ch_list



def get_eye_opening_potato_like(histoEyeOpeningScan,
                                eyeOpeningThresholdWrtMaxCut = 0.8):
    """
    References:
      https://gitlab.cern.ch/cms_tk_ph2/potato/-/blob/develop/inc/NamesDefinition.h
      https://gitlab.cern.ch/cms_tk_ph2/potato/-/blob/develop/src/Analyzer2S.cpp#L173
      https://gitlab.cern.ch/cms_tk_ph2/potato/-/blob/develop/src/Analyzer.cpp#L645
    """

    zMax = histoEyeOpeningScan.GetMaximum()
    yProjection = histoEyeOpeningScan.ProjectionY("", 0, -1, "");
    yBinMax     = yProjection.GetMaximumBin()

    eyeCrossingVoltage    = histoEyeOpeningScan.GetYaxis().GetBinCenter(yBinMax)
    eyeCrossingVoltageSum = yProjection.GetBinContent(yBinMax)
    eyeCrossingVoltageMinus1Sum = yProjection.GetBinContent(yBinMax-1)
    eyeCrossingVoltagePlus1Sum  = yProjection.GetBinContent(yBinMax+1)

    #Delta around max allows to find a little better the real peak
    delta = 0.5*(eyeCrossingVoltageMinus1Sum-eyeCrossingVoltagePlus1Sum)/(eyeCrossingVoltageMinus1Sum+eyeCrossingVoltagePlus1Sum-2*eyeCrossingVoltageSum)
    eyeCrossingFractionalOffset = eyeCrossingVoltage + delta

    #After the projected Y max has been found, the histo is sliced at that point to get the crossing width of the eye
    xProjectionAtMaxy = histoEyeOpeningScan.ProjectionX("", yBinMax, yBinMax, "")

    threshold = eyeOpeningThresholdWrtMaxCut*zMax
    crossingWidthAtyMax = 0

    for i in range(1, xProjectionAtMaxy.GetNbinsX() + 1):
        if xProjectionAtMaxy.GetBinContent(i) < threshold:
            crossingWidthAtyMax += xProjectionAtMaxy.GetBinWidth(i)

    eyeOpeningArea = 0.0

    for i in range(1, histoEyeOpeningScan.GetNbinsX() + 1):
        for j in range(1, histoEyeOpeningScan.GetNbinsY() + 1):
            binContent = histoEyeOpeningScan.GetBinContent(i, j)
            if binContent > threshold:
                eyeOpeningArea += 100.0

    eyeOpeningArea /= (
        histoEyeOpeningScan.GetNbinsX()
        * histoEyeOpeningScan.GetNbinsY()
    )

    #print(f"LPGBT_EYE_OFF        : {eyeCrossingFractionalOffset}")
    #print(f"LPGBT_EYE_TIME_WIDTH : {crossingWidthAtyMax}")
    #print(f"LPGBT_EYE_AREA       : {eyeOpeningArea}")

    return eyeCrossingFractionalOffset, crossingWidthAtyMax, eyeOpeningArea



def bert_error_histos(histoBERTerrorRate,
                      histoBERTtestedBitCounter,
                      histoFECerrorCounter):

    # Number of X bins
    lines = histoBERTtestedBitCounter.GetNbinsX()

    # Access raw histogram arrays
    errorRatePerLine = histoBERTerrorRate.GetArray()
    bitsTestedPerLine = histoBERTtestedBitCounter.GetArray()

    totalBERTErrorBits = 0.0
    totalBitsTested = 0.0

    for i in range(lines):
        totalBERTErrorBits += (
            bitsTestedPerLine[i] * errorRatePerLine[i]
        )

        totalBitsTested += bitsTestedPerLine[i]

    totalFECBits = histoFECerrorCounter.Integral()

    return totalBERTErrorBits,totalBitsTested,totalFECBits


def fill_vtrx_light_yield(histoVTRxLightYield):
    nBinsX = histoVTRxLightYield.GetNbinsX()
    nBinsY = histoVTRxLightYield.GetNbinsY()

    biasRange = (
        histoVTRxLightYield.GetXaxis().GetBinCenter(nBinsX)
        - histoVTRxLightYield.GetXaxis().GetBinCenter(1)
    )

    modulationRange = (
        histoVTRxLightYield.GetYaxis().GetBinCenter(nBinsY)
        - histoVTRxLightYield.GetYaxis().GetBinCenter(1)
    )

    # --------------------------------------------------
    # Modulation slope
    # Fit Y projection for each X bin
    # --------------------------------------------------

    modulationSlope = 0.0

    for xbin in range(1, nBinsX + 1):
        projection = histoVTRxLightYield.ProjectionY(
            f"projectionX_{xbin}",
            xbin,
            xbin
        )

        fit = ROOT.TF1(
            f"fit_modulation_{xbin}",
            "pol1",
            projection.GetXaxis().GetXmin(),
            projection.GetXaxis().GetXmax()
        )
        
        projection.Fit(fit, "Q")
        modulationSlope += fit.GetParameter(1)

    modulationSlope /= nBinsX

    # --------------------------------------------------
    # Bias slope
    # Fit X projection for each Y bin
    # --------------------------------------------------

    biasSlope = 0.0

    for ybin in range(1, nBinsY + 1):
        projection = histoVTRxLightYield.ProjectionX(
            f"projectionY_{ybin}",
            ybin,
            ybin
        )

        fit = ROOT.TF1(
            f"fit_bias_{ybin}",
            "pol1",
            projection.GetXaxis().GetXmin(),
            projection.GetXaxis().GetXmax()
        )

        projection.Fit(fit, "Q")
        biasSlope += fit.GetParameter(1)

    biasSlope /= nBinsY

    # --------------------------------------------------
    # Bin-by-bin percentage difference
    # --------------------------------------------------

    totalYield = 0.0
    totalPercentageDifference = 0.0

    bin_1_1 = 900.0
    bin_5_1 = 1200.0
    bin_1_5 = 700.0
    bin_5_5 = 1050.0
    sigma = 0.0

    for i in range(1, nBinsX + 1):
        for j in range(1, nBinsY + 1):

            fx = float(i - 1) / 4.0
            fy = float(j - 1) / 4.0

            # Bilinear interpolation between the 4 corners
            referenceValue = (
                (1.0 - fx) * (1.0 - fy) * bin_1_1
                + fx * (1.0 - fy) * bin_5_1
                + (1.0 - fx) * fy * bin_1_5
                + fx * fy * bin_5_5
            )

            referenceValue -= sigma

            yieldval = histoVTRxLightYield.GetBinContent(i, j)
            percent = 1.0 - (
                yieldval / referenceValue
            )
            totalYield += yieldval

            # Only add percentages if measured value is below reference
            if percent > 0:
                totalPercentageDifference += percent

    averagePercentageDifference = (
        1.0 - totalPercentageDifference / (nBinsX * nBinsY)
    )

    return totalYield,averagePercentageDifference, modulationSlope, biasSlope
