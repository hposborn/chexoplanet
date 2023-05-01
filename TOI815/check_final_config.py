#!/usr/bin/env python

import sys
sys.path=sys.path+['/home/hosborn/home1/python/','/shares/home1/hosborn/python/']

import numpy as np
import pandas as pd
from astropy.io import ascii

# DACE import
from dace.cheops import Cheops
#filesystem path import
from pathlib import Path
#Description of the method:
from dace.cheops import Cheops
import os
import time

from MonoTools.MonoTools import lightcurve
from chexoplanet import tools, newfit
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

name="TOI-815"
#fks=["CH_PR110048_TG037801_V0200","CH_PR110045_TG005001_V0200","CH_PR110048_TG038101_V0200","CH_PR110048_TG041801_V0200","CH_PR110048_TG041901_V0200"]

#Getting the info:
df=pd.read_csv("TIC00102840239_2022-04-17_0_mcmc_output_short.csv",index_col=0)

ntrans=int(np.round((df.loc['t0_c','mean']-df.loc['t0_2_c','mean'])/35))

info = pd.concat([pd.Series({'TIC ID':102840239, 'Epoch (BJD)':df.loc['t0_b','mean']+2457000,
                             'Epoch (BJD) err':df.loc['t0_b','sd'],
                                'Depth (ppm)':1e6*df.loc['ror_b','mean']**2,
                             'Duration (hours)':df.loc['tdur_b[0]','mean']*24,
                                'Period (days)':df.loc['per_b','mean'],
                             'Period (days) err':df.loc['per_b','sd']},name='b'),
                 pd.Series({'TIC ID':102840239, 'Epoch (BJD)':df.loc['t0_c','mean']+2457000,
                             'Epoch (BJD) err':df.loc['t0_c','sd'],
                                'Depth (ppm)':1e6*df.loc['ror_c','mean']**2,
                             'Duration (hours)':df.loc['tdur_c','mean']*24,
                                'Period (days)':(df.loc['t0_c','mean']-df.loc['t0_2_c','mean'])/ntrans,
                             'Period (days) err':np.sqrt(2*df.loc['t0_c','sd']**2+2*df.loc['t0_2_c','sd']**2)},name='c')],axis=1).T
print(info.iloc[1])
#53.58133729513230
#35.72d
mod = newfit.chexo_model(name, radec=SkyCoord(155.871938*u.deg,-43.834925*u.deg), overwrite=True, 
                         comment="all_cheops",save_file_loc="/home/hosborn/home1/data/Cheops_data")

mod.get_tess(tic=102840239)
mod.get_cheops()

for nix in range(info.shape[0]):
    mod.add_planet("bcdef"[nix],
                   tcen=float(info.iloc[nix]['Epoch (BJD)']),tcen_err=float(info.iloc[nix]['Epoch (BJD) err']),
                   tdur=float(info.iloc[nix]['Duration (hours)'])/24,depth=float(info.iloc[nix]['Depth (ppm)'])/1e6,
                   period=float(info.iloc[nix]['Period (days)']),
                   period_err=2*float(info.iloc[nix]['Period (days) err']))
print(mod.planets)
#Initialising lightcurves:
mod.init_lc(fit_gp=False, fit_flat=True, cut_oot=True, bin_oot=False)
mod.init_cheops(use_bayes_fact=True, use_signif=False, overwrite=False)

#Initialising full model:
mod.init_model(use_mstar=False, use_logg=True,fit_phi_spline=True,fit_phi_gp=False,
               constrain_lds=True, fit_ttvs=False, assume_circ=True)
mod.plot_cheops(save=True,show_detrend=True)
mod.model_comparison_cheops()
mod.plot_phot(save=True)
mod.plot_transits_fold(save=True)
mod.sample_model()
#mod.plot_rollangle_gps(save=True)
mod.plot_rollangle_splines(save=True)
mod.plot_cheops(save=True,show_detrend=True)
mod.plot_transits_fold(save=True)
mod.plot_phot(save=True)
if mod.fit_ttvs:
    mod.plot_ttvs()

mod.plot_tess(save=True)
mod.save_model_to_file()

os.system("find . -name \"*.png\" -o -name \"*.csv\" | tar -cf "+name+"_all_visits.tar.gz -T -")
print("mkdir /Volumes/LUVOIR/Cheops_data/"+name+";scp hosborn@horus.unibe.ch:/home/hosborn/home1/python/chexoplanet/"+name+"/"+name+"_all_visits.tar.gz /Volumes/LUVOIR/Cheops_data/"+name)
