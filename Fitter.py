import os
import numpy as np

import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares

import logging
logger = logging.getLogger('main')


class Fitter:
    def __init__(self,
                 x: np.array,
                 y: np.array,
                 yerr: np.array,
                 **kwargs):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.modeltype = kwargs.get("modeltype", "pure_gauss")
        self.fitparams = kwargs.get("fitparams", {})
        self.fitparamsrange = kwargs.get("fitparamsrange", None)

        if self.modeltype == "poly2":
            self.model = self.__gauss_model
        else:
            logger.warning("No fitmodeltype, default is pure gaussian")
            self.model = self.__gauss_model
            self.params = {
                "amp": self.fitparams.get("amp", 0.1),
                "mean": self.fitparams.get("mean", 500.0),
                "sigma": self.fitparams.get("sigma", 100.0)
            }
            self.params_range = [(0.1, 500.1), (300, 3000), (10.0, 255.0)]
            self.result = self.__fit()
                    

    def __optimise(self):
        lsq = LeastSquares(self.x, self.y, self.yerr, self.model)
        m = Minuit(lsq, **self.params)
        m.limits = self.params_range
        m.fixed = False
        m.migrad() # finds minimum of least_squares function
        m.hesse()  # accurately computes uncertainties
        return m
        
        
    def __gauss_model(self, x, amp, mean, sigma):
        return amp * np.exp(-0.5 * ((x - mean) / sigma)**2)

    
    def __fit(self):
        fit_label_text = ""
        m_fit = self.__optimise()
        red_chi2 = m_fit.fmin.reduced_chi2
        params = m_fit.values.to_dict()
        fit_val = self.model(self.x, *m_fit.values)
        
        fit_valid = red_chi2 < 10.0

        if self.modeltype == 'pure_gauss':
            fit_mean = round(params['mean'],2)
            fit_sigma = round(params['sigma'],2)
            fit_label_text = f"$\\chi^2$/$n_\\mathrm{{dof}}$={m_fit.fval:.1f} / {m_fit.ndof:.0f} = {m_fit.fmin.reduced_chi2:.1f}: (µ = {round(fit_mean,1)}, σ = {round(fit_sigma,1)})"

            
        if fit_valid:
            return (fit_val, fit_label_text)
        else:
            return None
