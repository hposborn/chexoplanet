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
import pymc3 as pm
import pymc3_ext as pmx
from patsy import dmatrix
import theano.tensor as tt
from matplotlib import pyplot
name="HIP41378"
mod = newfit.chexo_model(name, radec=SkyCoord(126.615828*u.deg,10.080371*u.deg), overwrite=False, save_file_loc="/home/hosborn/home1/data/Cheops_data")
mod.get_tess(tic=366443426)
mod.get_cheops()
mod.save_model_to_file()
#mod.load_model_from_file("/home/hosborn/home1/data/Cheops_data/HD114082/HD114082_20230330_348762_0.9_3.75_0.55_0.020_1.25_auto_2_3.0_0.33_logK_3.0_9.0_model.pkl")
model_params={}
mod.cheops_lc['phi']=mod.cheops_lc['phi'].values%360
mod.cheops_lc=mod.cheops_lc.sort_values("time")
#mask_allphi_sorting=np.argsort(mod.cheops_lc['phi'].values).astype(int)
mask_allphi_sorting=np.argsort(mod.cheops_lc['phi'].values).astype(int)
mask_alltime_sorting=np.argsort(mask_allphi_sorting).astype(int)
cheops_linear_decorrs=['bg','deltaT','centroidx','centroidy']
norm_cheops_dat={}
for linpar in cheops_linear_decorrs:
    combdat=mod.cheops_lc[linpar]
    norm_cheops_dat[linpar]=(combdat - np.nanmedian(combdat[mod.cheops_lc['mask'].values]))/np.nanstd(combdat[mod.cheops_lc['mask'].values])

with pm.Model() as abs_model:
    # -------------------------------------------
    #         Cheops detrending (linear)
    # -------------------------------------------
    model_params['logs_cheops'] = pm.Normal("logs_cheops", mu=np.log(np.nanmedian(abs(np.diff(mod.cheops_lc.loc[mod.cheops_lc['mask'],'raw_flux'].values)))), sd=3)
    #Initialising linear (and quadratic) parameters:
    model_params['linear_decorr_dict']={}#i:{} for i in mod.cheops_filekeys}
    for decorr in cheops_linear_decorrs:
        model_params['linear_decorr_dict'][decorr]=pm.Normal(decorr,mu=0,sd=np.nanmedian(abs(np.diff(mod.cheops_lc['raw_flux']))))
    #Creating the flux correction vectors:
    model_params['cheops_obs_mean']=pm.Normal("cheops_obs_mean",mu=np.nanmedian(mod.cheops_lc['raw_flux']),sd=0.01*np.nanmedian(mod.cheops_lc['raw_flux'].values))
    #Linear detrending only
    model_params['cheops_flux_cor'] = pm.Deterministic("cheops_flux_cor", tt.sum([model_params['linear_decorr_dict'][lvar]*norm_cheops_dat[lvar] for lvar in cheops_linear_decorrs], axis=0))
    # -------------------------------------------
    #      Cheops detrending (roll angle GP)
    # -------------------------------------------
    #Fit a single spline to all rollangle data
    minmax=(np.min(mod.cheops_lc.loc[mod.cheops_lc['mask'],'phi']),np.max(mod.cheops_lc.loc[mod.cheops_lc['mask'],'phi']))
    n_knots=int(np.round((minmax[1]-minmax[0])/mod.spline_bkpt_cad))
    knot_list = np.quantile(mod.cheops_lc.loc[mod.cheops_lc['mask'],'phi'],np.linspace(0,1,n_knots))
    B = dmatrix("bs(phi, knots=knots, degree="+str(int(mod.spline_order))+", include_intercept=True) - 1",{"phi": np.sort(mod.cheops_lc['phi'].values), "knots": knot_list[1:-1]})
    model_params['splines'] = pm.Normal("splines", mu=0, sd=3*np.nanmedian(abs(np.diff(mod.cheops_lc.loc[mod.cheops_lc['mask'].values,'raw_flux'].values))), shape=B.shape[1], testval=np.random.normal(0.0,1e-4,B.shape[1]))
    model_params['spline_model_allphi'] = pm.Deterministic("spline_model_allphi", tt.dot(np.asarray(B, order="F"), model_params['splines'].T))
    model_params['spline_model_alltime'] = pm.Deterministic("spline_model_alltime", model_params['spline_model_allphi'][mask_alltime_sorting])
    # -------------------------------------------
    #      Evaluating log likelihoods            
    # -------------------------------------------
    cheops_sigma2s = mod.cheops_lc.loc[mod.cheops_lc['mask'].values,'raw_flux_err'].values ** 2 + tt.exp(model_params['logs_cheops'])**2
    model_params['llk_cheops'] = pm.Normal("llk_cheops", 
                                           mu = model_params['cheops_obs_mean'] + model_params['cheops_flux_cor'][mod.cheops_lc['mask'].values] + model_params['spline_model_alltime'][mod.cheops_lc['mask'].values],
                                           sd = tt.sqrt(cheops_sigma2s), observed=mod.cheops_lc.loc[mod.cheops_lc['mask'],'raw_flux'].values)
    decorrvars = [model_params['linear_decorr_dict'][par] for par in model_params['linear_decorr_dict']] + [model_params['cheops_obs_mean'],model_params['splines']]
    comb_soln = pmx.optimize(vars=decorrvars)
    #More complex transit fit. Also RVs:
    #Doing everything:
    init_soln = pmx.optimize(start=comb_soln)
    #trace=pmx.sample(tune=500, draws=600, cores=6, start=init_soln)

plt.figure(1,figsize=(9,4))

#plt.plot(mod.cheops_lc['time'][mod.cheops_lc['mask'].values], mod.cheops_lc['raw_flux'][mod.cheops_lc['mask'].values], 
#         '.k',markersize=0.9,alpha=0.5)
med_rawflux=np.nanmedian(mod.cheops_lc['raw_flux'][mod.cheops_lc['mask'].values])

plt.plot(mod.cheops_lc['time'][mod.cheops_lc['mask'].values], 
         1000*((mod.cheops_lc['raw_flux'][mod.cheops_lc['mask'].values]-(init_soln['cheops_flux_cor'][mod.cheops_lc['mask'].values]+init_soln['spline_model_alltime'][mod.cheops_lc['mask'].values]))/med_rawflux-1), 
         '.k',markersize=1.4,alpha=0.75)
binlc=tools.bin_lc_segment(np.column_stack((mod.cheops_lc['time'][mod.cheops_lc['mask'].values], 
                                            1000*((mod.cheops_lc['raw_flux'][mod.cheops_lc['mask'].values]-(init_soln['cheops_flux_cor'][mod.cheops_lc['mask'].values]+init_soln['spline_model_alltime'][mod.cheops_lc['mask'].values]))/med_rawflux-1),
                                            1000*mod.cheops_lc['raw_flux_err'][mod.cheops_lc['mask'].values]/med_rawflux)),1/48)
plt.errorbar(binlc[:,0],binlc[:,1],yerr=binlc[:,2],fmt='.')
plt.ylim(-12,7)
plt.plot([np.min(mod.cheops_lc['time']),np.max(mod.cheops_lc['time'])],[-1.55,-1.55],':k',linewidth=2,alpha=0.3)
plt.text(np.min(mod.cheops_lc['time'])+0.5,-4.95,"Expected transit Depth",horizontalalignment="left",verticalalignment="bottom",fontsize=10)
plt.arrow(np.max(mod.cheops_lc['time'])-0.75,-4.8,dx=0.55/2,dy=0,head_width=0.07,head_length=0.06)
plt.arrow(np.max(mod.cheops_lc['time'])-0.75,-4.8,dx=-0.55/2,dy=0,head_width=0.07,head_length=0.06)
plt.text(np.max(mod.cheops_lc['time'])-0.75,-4.55,"Duration",fontsize=9.5,horizontalalignment="center")
plt.savefig(os.path.join(mod.save_file_loc,mod.name,mod.unq_name+"_absolute_photometry.png"))

plt.figure(2)
plt.plot(mod.cheops_lc['phi'],init_soln['spline_model_alltime'],'.',label='spline',alpha=0.4)
plt.plot(mod.cheops_lc['phi'],init_soln['cheops_flux_cor'],'.',label='flux_cor',alpha=0.4)
plt.plot(mod.cheops_lc['phi'],mod.cheops_lc['raw_flux']-init_soln['cheops_obs_mean'],'.k',
         markersize=0.75,label='flux',alpha=0.4)
plt.legend()
plt.ylim(-1e5,1e5)
plt.ylabel("Relative Flux [ppt]")
plt.xlabel("Time [HJD]")
plt.savefig(os.path.join(mod.save_file_loc,mod.name,mod.unq_name+"_decorrelations.png"))

# plt.figure(3)
# plt.plot(mod.cheops_lc['time'][mod.cheops_lc['mask'].values], 
#          mod.cheops_lc['raw_flux'][mod.cheops_lc['mask'].values], 
#          '.k',markersize=0.9,alpha=0.5)
# plt.plot(mod.cheops_lc['time'][mod.cheops_lc['mask'].values], 
#          (init_soln['cheops_obs_mean']+init_soln['cheops_flux_cor'][mod.cheops_lc['mask'].values]+init_soln['spline_model_alltime'][mod.cheops_lc['mask'].values]), 
#          ':r',markersize=0.9,alpha=0.5)
# plt.xlim(2.03e7,2.05e7)
# plt.savefig(os.path.join(mod.save_file_loc,mod.name,mod.unq_name+"_model.png"))
