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

from MonoTools.MonoTools import lightcurve
from chexoplanet import fit
import matplotlib.pyplot as plt

name="TOI2519"
fks=["PR130056_TG005301_V0200"]

pipe_dataloc="/home/hosborn/.astropy/cache/.pipe-cheops/"
out_dir="/home/hosborn/home1/data/Cheops_data/"+name

#Getting the info:

info = pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
info=info.loc[info['TOI'].values.astype(int)==int(name[3:])]
info

#Getting lightcurve

lc=lightcurve.multilc(info.iloc[0]['TIC ID'],'tess',load=False,do_search=True)

#Selecting PSF:
psfs=pd.read_csv(pipe_dataloc+"ref_lib_data/psf_lib/PSF_list.txt",
                 delim_whitespace=True,header=None,na_values="?")
psf_file=psfs[0].values[np.nanargmin(abs(float(lc.all_ids['tess']['data']['Teff'])-psfs[3].values.astype(float)))]

if not os.path.isdir(out_dir):
    #Creating the folder
    os.mkdir(out_dir)

if not os.path.exists(pipe_dataloc+"data_root/"+name):
    #Linking the folder to the PIPE location
    os.system("ln -s "+out_dir+" "+pipe_dataloc+"data_root/"+name)
for fk in fks:
    if not os.path.isdir(os.path.join(out_dir,fk)):
        #Downloading Cheops data:
        Cheops.download('all', {'file_key': {'contains': "CH_"+fk}},
                        output_directory=out_dir, output_filename=name+'_dace_download.tar.gz')
        #Extracting .tar.gz download:
        os.system("tar -xvzf "+out_dir+'/'+name+'_dace_download.tar.gz -C '+out_dir)
        #Deleting it
        os.system("rm "+out_dir+'/'+name+'_dace_download.tar.gz')


fitrad=np.clip(int(20+5*np.round(3*(12.5-lc.all_ids['tess']['data']['Tmag'])/5)),20,45)
#Running PIPE:
for fk in fks:
    if not os.path.exists(out_dir+"/"+fk+"/Outdata"):
        #Running PIPE:
        from pipe import PipeParam, PipeControl
        for visit in [fk]:
            pps = PipeParam(name, visit)
            pps.psflib = int(psf_file.split("_")[-1][:-4]) #HD189
            pps.bgstars = True 
            pps.limflux = 1e-5
            pps.darksub = False#True
            pps.dark_level = 2
            pps.remove_static = True
            pps.save_static = False
            pps.static_psf_rad = False
            pps.smear_fact = 5.5
            pps.smear_resid = False
            pps.non_lin_tweak = True
            pps.klip = 1
            pps.sigma_clip = 15
            pps.empiric_noise = True
            pps.empiric_sigma_clip = 4
            pps.save_noise_cubes = False
            pps.save_psfmodel = True
            pps.fitrad = fitrad
            pc = PipeControl(pps)
            pc.process_eigen()

mod = fit.chexo_model(name, radec=lc.radec, overwrite=True, comment="running_via_textfile")
mod.add_lc(lc.time+2457000,lc.flux,lc.flux_err)
for fk in fks:
    if os.path.exists(out_dir+"/"+fk+"/Outdata/00000/"+name+"_"+fk+"_im.fits"):
        mod.add_cheops_lc(filekey=fk,fileloc=out_dir+"/"+fk+"/Outdata/00000/"+name+"_"+fk+"_im.fits",PIPE=True,DRP=False)
    elif os.path.exists(out_dir+"/"+fk+"/Outdata/00000/"+name+"_"+fk+"_sa.fits"):
        mod.add_cheops_lc(filekey=fk,fileloc=out_dir+"/"+fk+"/Outdata/00000/"+name+"_"+fk+"_sa.fits",PIPE=True,DRP=False)
    else:
        print("ERROR with files in "+out_dir+"/"+fk+"/Outdata/00000")
        
#Radius info from lightcurve data (TIC)
if 'eneg_Rad' in lc.all_ids['tess']['data'] and lc.all_ids['tess']['data']['eneg_Rad'] is not None and lc.all_ids['tess']['data']['eneg_Rad']>0:
    Rstar=lc.all_ids['tess']['data'][['rad','eneg_Rad','epos_Rad']].values
else:
    Rstar=lc.all_ids['tess']['data'][['rad','e_rad','e_rad']].values
if 'eneg_Teff' in lc.all_ids['tess']['data'] and lc.all_ids['tess']['data']['eneg_Teff'] is not None and lc.all_ids['tess']['data']['eneg_Teff']>0:
    Teff=lc.all_ids['tess']['data'][['Teff','eneg_Teff','epos_Teff']].values
else:
    Teff=lc.all_ids['tess']['data'][['Teff','e_Teff','e_Teff']].values
if 'eneg_logg' in lc.all_ids['tess']['data'] and lc.all_ids['tess']['data']['eneg_logg'] is not None and lc.all_ids['tess']['data']['eneg_logg']>0:
    logg=lc.all_ids['tess']['data'][['logg','eneg_logg','epos_logg']].values
else:
    logg=lc.all_ids['tess']['data'][['logg','e_logg','e_logg']].values
print(Rstar,Teff,logg)
mod.init_starpars(Rstar=Rstar,Teff=Teff,logg=logg)

for nix in range(info.shape[0]):
    mod.add_planet("bcdef"[nix],
                   tcen=float(info.iloc[nix]['Epoch (BJD)']),tcen_err=float(info.iloc[nix]['Epoch (BJD) err']),
                   tdur=float(info.iloc[nix]['Duration (hours)'])/24,depth=float(info.iloc[nix]['Depth (ppm)'])/1e6,
                   period=float(info.iloc[nix]['Period (days)']),period_err=2*float(info.iloc[nix]['Period (days) err']))
print(mod.planets)
#Initialising lightcurves:
mod.init_lc(fit_gp=False, fit_flat=True, cut_oot=True, bin_oot=False)
mod.init_cheops(use_bayes_fact=True, use_signif=False, overwrite=False)

#Initialising full model:
mod.init_model(use_mstar=False, use_logg=True,fit_phi_spline=True,fit_phi_gp=False,
               constrain_lds=True, fit_ttvs=False, assume_circ=True)

#mod.plot_rollangle_gps(save=True)
mod.plot_rollangle_splines(save=True)
mod.plot_cheops(save=True,show_detrend=True)
mod.plot_tess(save=True)
mod.plot_transits_fold(save=True)

mod.model_comparison_cheops()

mod.save_model_to_file()