from chexoplanet import newfit, tools
from MonoTools.MonoTools import lightcurve
import numpy as np
import pandas as pd
import os
import glob
from dace.cheops import Cheops
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body
import time

all_cheops_obs=[]
while len(all_cheops_obs)==0:
    all_cheops_obs=pd.DataFrame(Cheops.query_database(filters={"file_key":{"contains":"PR130056"}}))
    if len(all_cheops_obs)==0:
        time.sleep(15)
        from dace.cheops import Cheops

#print(all_cheops_obs['obj_id_catname'].values)
#Getting RaDec from string:
obs_radecs=SkyCoord([rd.split(" / ")[0] for rd in all_cheops_obs['obj_pos_coordinates_hms_dms'].values], 
                [rd.split(" / ")[1] for rd in all_cheops_obs['obj_pos_coordinates_hms_dms'].values],
                unit=(u.hourangle,u.deg))
all_cheops_obs['ra']=obs_radecs.ra.deg
all_cheops_obs['dec']=obs_radecs.dec.deg

#Reading TOI data from web or file (updates every 10 days)
round_date=np.round(Time.now().jd,-1)
if not os.path.exists("TOI_tab_jd_"+str(int(round_date))+".csv"):
    info=pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
    info.to_csv("TOI_tab_jd_"+str(int(round_date))+".csv")
else:
    info=pd.read_csv("TOI_tab_jd_"+str(int(round_date))+".csv",index_col=0)
info['star_TOI']=info['TOI'].values.astype(int)

#Using Observed RA/Dec and TOI RA/Dec to match these:
toi_radecs=SkyCoord(info['RA'].values,info['Dec'].values,
                    unit=(u.hourangle,u.deg),
                    pm_ra_cosdec=np.array([pmra if np.isfinite(pmra) else 0.0 for pmra in info['PM RA (mas/yr)'].values])*u.mas/u.year,
                    pm_dec=np.array([pmdec if np.isfinite(pmdec) else 0.0 for pmdec in info['PM Dec (mas/yr)'].values])*u.mas/u.year,
                    obstime=Time(2015.5,format='jyear'))
toi_radecs_epoch2000 = toi_radecs.apply_space_motion(Time(2000.0,format='jyear'))
idx, d2d, _ = obs_radecs.match_to_catalog_sky(toi_radecs_epoch2000)
print(np.min(d2d.arcsec),np.max(d2d.arcsec))
#all_cheops_obs.iloc[np.argmin(d2d.arcsec)],all_cheops_obs.iloc[np.argmax(d2d.arcsec)])
assert np.all(d2d < 3*u.arcsec), "All the Observed TOIs should match RA/Dec to the TOI catalogue"
all_cheops_obs['star_TOI']=info["star_TOI"].values[idx]
all_cheops_obs['TIC ID']=info["TIC ID"].values[idx]

#Reading input data from PHT2
input_ors=pd.read_csv("/home/hosborn/home1/python/chexoplanet/Chateaux/Chateaux_ORs_apr23.csv")

#Changing non-obvious names to the TOI:
#input_ors['TOI_name'] = [change_dic[o] if o in change_dic else o.replace("-","").upper() for o in input_ors['Target Name'].values]
input_ors['file_key']=["CH_PR"+str(int(input_ors.iloc[i]['Programme Type']))+str(int(input_ors.iloc[i]['Programme Id'])).zfill(4)+"_TG"+str(int(input_ors.iloc[i]['Observation Request Id'])).zfill(4)+"01_V0200" for i in range(len(input_ors))]
#print(all_cheops_obs['TOI_name'].values,input_ors['TOI_name'].values)

for toi in np.unique(all_cheops_obs['star_TOI'].values):
    if len(glob.glob("/home/hosborn/home1/data/Cheops_data/TOI"+str(toi)+"/*refit_all_horus_model.pkl"))>0:
        print("Skipping TOI="+str(toi),"("+str(len(glob.glob("/home/hosborn/home1/data/Cheops_data/TOI"+str(toi)+"/*refit_all_horus_model.pkl")))+"files found)")
        continue
    #print("Modelling TOI="+str(toi),"("+str(len(glob.glob("/home/hosborn/home1/data/Cheops_data/TOI"+str(toi)+"/*refit_all_horus_model.pkl")))+"files found)")
    #Getting the data from TOI list, OR list and Observation list for this TOI:
    these_obs=all_cheops_obs.loc[all_cheops_obs['star_TOI']==toi]
    these_tois=info.loc[info['star_TOI']==toi].sort_values('Period (days)')
    these_ors=input_ors.loc[input_ors['Target Name']==these_obs.iloc[0]['obj_id_catname']]
    fks=[f.replace("CH_","") for f in these_obs.loc[:,'file_key'].values]
    print(toi,fks)

    #Figuring out which TOI links to which observation (if any)
    these_obs['TOI']=np.zeros(len(these_obs))
    print(len(pd.unique(these_ors['file_key'])),len(these_ors['file_key'].values))
    for i,iob in these_obs.iterrows():
        #print(fk,"CH_"+fk in these_obs['file_key'].values,"CH_"+fk in these_ors['file_key'].values)
        fk=iob['file_key']
        ior=these_ors.loc[these_ors['file_key']==fk]#.iloc[0]
        itoi=np.argmin((these_tois['Period (days)'].values.astype(float)-float(ior["Transit Period [day]"]))**2)
        these_obs.loc[i,'TOI']=these_tois['TOI'].values[itoi]
        these_ors.loc[these_ors['file_key']=="CH_"+fk,'TOI']=these_tois['TOI'].values[itoi]    
    
    #Loading lightcurve
    lc=lightcurve.multilc(these_tois.iloc[0]['TIC ID'],'tess',load=False)
                          #radec=SkyCoord(these_obs.iloc[0]['ra']*u.deg,these_obs.iloc[0]['dec']*u.deg),load=False)

    #Initialising model
    mod = newfit.chexo_model("TOI"+str(toi), radec=lc.radec, overwrite=True, 
                             comment="refit_all_horus",save_file_loc="/home/hosborn/home1/data/Cheops_data")
    Rstar, Teff, logg = tools.starpars_from_MonoTools_lc(lc)
    mod.init_starpars(Rstar=Rstar,Teff=Teff,logg=logg)
    mod.add_lc(lc.time+2457000,lc.flux,lc.flux_err)
    for fk in fks:
        mod.add_cheops_lc(filekey=fk,fileloc=None, download=True, PIPE=True, DRP=False,
                          mag=lc.all_ids['tess']['data']['Tmag'])
    
    #for i,itoi in these_tois.iterrows():
    #    toi_to_or_link[i]=np.argmin((itoi['Period (days)']-these_ors["Transit Period [day]"].values.astype(float))**2)
    #    toi_to_obs_link[i]=these_obs['filekey'].values.index(these_ors.loc[toi_to_or_link[i],'filekey'])
        
    for nix in range(these_tois.shape[0]):
        itoi=these_tois['TOI'].values[nix]
        if itoi in these_ors['TOI'].values:
            #We have scheduled this TOI - it has updated period from our fit
            mod.add_planet("bcdef"[nix],
                        tcen=float(these_ors.loc[these_ors['TOI']==itoi]['Transit Time  [BJD_TDB]']),
                        tcen_err=float(these_tois.iloc[nix]['Epoch (BJD) err']),
                        tdur=float(these_tois.iloc[nix]['Duration (hours)'])/24,
                        depth=float(these_tois.iloc[nix]['Depth (ppm)'])/1e6,
                        period=float(these_ors.loc[these_ors['TOI']==itoi]['Transit Period [day]']),
                        period_err=2*float(these_tois.iloc[nix]['Period (days) err']))
        else:
            #We don't have an observation/OR for this one
            mod.add_planet("bcdef"[nix],
                        tcen=float(these_tois.iloc[nix]['Epoch (BJD)']),
                        tcen_err=float(these_tois.iloc[nix]['Epoch (BJD) err']),
                        tdur=float(these_tois.iloc[nix]['Duration (hours)'])/24,
                        depth=float(these_tois.iloc[nix]['Depth (ppm)'])/1e6,
                        period=float(these_tois.iloc[nix]['Period (days)']),
                        period_err=2*float(these_tois.iloc[nix]['Period (days) err']))
    print(mod.planets)
    try:
        #Initialising lightcurves:
        mod.init_lc(fit_gp=False, fit_flat=True, cut_oot=True, bin_oot=False)
        mod.init_cheops(use_bayes_fact=True, use_signif=False, overwrite=False)

        #Initialising full model:
        mod.init_model(use_mstar=False, use_logg=True,fit_phi_spline=True,fit_phi_gp=False,
                    constrain_lds=True, fit_ttvs=False, assume_circ=True)

        mod.sample_model()
        mod.model_comparison_cheops()

        #mod.plot_rollangle_gps(save=True)
        mod.plot_rollangle_splines(save=True)
        mod.plot_cheops(save=True,show_detrend=True)
        mod.plot_tess(save=True)
        mod.plot_transits_fold(save=True)
        
        df=mod.save_trace_summary()
        mod.save_model_to_file()
        
        mod.MakeExoFopFiles(list(["TOI"+t for t in these_tois['TOI'].values.astype(str)]),
                            upload_loc="/home/hosborn/home1/data/Cheops_data/ChATeAUX/")
    except:
        print("MakeExoFopFiles failed for ",toi)
    #Bundling into a pickle:
    os.system("find . -name \""+mod.save_file_loc+"/*.png\" -o -name \""+mod.save_file_loc+"/*.csv\"  -o -name \""+mod.save_file_loc+"/*.pkl\" | tar -cf "+mod.save_file_loc+"/"+mod.name+"_all_visits.tar.gz -T -")
    #print("mkdir /Volumes/LUVOIR/Cheops_data/"+name+";scp hosborn@horus.unibe.ch:/home/hosborn/home1/python/chexoplanet/"+name+"/"+name+"_all_visits.tar.gz /Volumes/LUVOIR/Cheops_data/"+name)


