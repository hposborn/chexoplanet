import exoplanet as xo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter    
import pandas as pd

from astropy.io import fits
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body

import pickle
import os.path
from datetime import date
import os
import glob
import time
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("theano").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)

floattype=np.float64
tablepath = os.path.join(os.path.dirname(__file__),'tables')

try:
    import aesara_theano_fallback.tensor as tt
except:
    import theano.tensor as tt
import pymc3 as pm
import pymc3_ext as pmx
from celerite2.theano import terms as theano_terms
import celerite2.theano

from .tools import *

class chexo_model():
    """Fit Cheops with Exoplanet core model class"""

    def __init__(self, targetname, overwrite=True, radec=None, comment=None, **kwargs):
        """Initialising chexo_model.

        Args:
            targetname (str): Targetname as string
            overwrite (bool, optional): Whether to overwrite the past settings/model. Defaults to True.
            radec (astropy.SkyCoord object, optional): Coordinates of the star (used for Moon glint calculation). Defaults to None.
            comment (str, optional): Comment to be added to the unique filename. Defaults to None.
        """
        self.name=targetname
        self.overwrite=overwrite
        self.comment=comment
        self.defaults={'debug':False,           # debug - bool - print debug statements?
                       'load_from_file':False,  # load_from_file - bool - Load previous model?
                       'save_file_loc':None,    # save_file_loc - str - String file location to save data
                       'fit_gp':True,           # fit_gp - bool - co-fit a GP.
                       'fit_flat':False,        # fit_flat - bool - flatten the lightcurve before modelling
                       'flat_knotdist':0.9,     # flat_knotdist - float - Length of knotdistance for flattening spline (in days)
                       'train_gp':True,         # train_gp - bool - Train the GP hyperparameters on out-of-transit data
                       'cut_distance':3.75,     # cut_distance - float - cut out points further than cut_distance*Tdur. 0.0 means no cutting
                       'mask_distance': 0.55,   # Distance, in transit durations, from set transits, to "mask" as in-transit data when e.g. flattening.
                       'cut_oot':False,         # cut_oot - bool - Cut points outside the cut_distance when fitting
                       'bin_size':1/48,         # bin_size - float - Size of binned points (defaults to 30mins)
                       'bin_oot':True,          # bin_oot - bool - Bin points outside the cut_distance to 30mins
                       'pred_all':False,        # Do we predict all time array, or only a cut-down version?
                       'use_bayes_fact':True,   # Determine the detrending factors to use with a Bayes Factor
                       'use_signif':False,      # Determine the detrending factors to use by simply selecting those with significant non-zero coefficients
                       'signif_thresh':1.25,    # Threshold for detrending parameters in sigma
                       'use_multinest':False,   # use_multinest - bool - currently not supported
                       'use_pymc3':True,        # use_pymc3 - bool
                       'assume_circ':False,     # assume_circ - bool - Assume circular orbits (no ecc & omega)?
                       'timing_sd_durs':0.33,   # timing_sd_durs - float - The standard deviation to use (in units of transit duration) when setting out timing priors
                       'fit_ttvs':False,        # Fit a TTVorbit exoplanet model which searches for TTVs
                       'split_periods':None,    # Fit for multiple split periods. Input must be None or a dict matching mod.planets with grouped indexes for those transits to group
                       'ttv_prior':'Normal',    # What prior to have for individual transit times. Possibilities: "Normal","Uniform","BoundNormal"
                       'fit_phi_gp':False,      # fit_phi_gp - bool - co-fit a GP to the roll angle.
                       'fit_phi_spline':True,   # fit_phi_spline - bool - co-fit a spline model to the roll angle
                       'spline_bkpt_cad':9.,    # spline_bkpt_cad - float - The spline breakpoint cadence in degrees. Default is 9deg
                       'spline_order':3.,       # spline_order - int - Thespline order. Defaults to 3 (cubic)
                       'common_phi_model':True, # common_phi_model - bool - Fit the same roll angle GP trend common across all visits
                       'ecc_prior':'auto',      # ecc_prior - string - 'uniform', 'kipping' or 'vaneylen'. If 'auto' we decide based on multiplicity
                       'npoly_rv':2,            # npoly_rv - int - order of polynomial fit to RVs
                       'rv_mass_prior':'logK',  # rv_mass_prior - str - What mass prior to use. "logK" = normal prior on log amplitude of K, "popMp" = population-derived prior on logMp, "K" simple normal prior on K.
                       'use_mstar':True,        # use_mstar - bool - Whether to model using the stellar Mass (otherwise set use_logg)
                       'use_logg':False,        # use_logg - bool - Whether to model using the stellar logg (otherwise Mass)
                       'constrain_lds':True,    # constrain_lds - bool - Use constrained LDs from model or unconstrained?
                       'ld_mult':3.,            # ld_mult - float - How much to multiply theoretical LD param uncertainties
                       'fit_contam':False}      # fit_contam - bool - Fit for "second light" (i.e. a binary or planet+blend)

        self.descr_dict={}

        if radec is not None:
            self.radec=radec

        self.planets={}
        self.cheops_filekeys=[]

        for param in self.defaults:
            if not hasattr(self,param) or self.overwrite:
                setattr(self,param,self.defaults[param])
        self.update(**kwargs)                    

        #Initalising save locations
        if self.load_from_file and not self.overwrite:
            #Catching the case where the file doesnt exist:
            try:
                self.load_model_from_file(loadfile=self.save_file_loc)
            except:
                self.load_from_file = False

        self.percentiles={'-2sig':2.2750132, '-1sig':15.8655254, 'med':50., '+1sig':84.1344746, '+2sig':97.7249868}

    def get_tess(self,tic=None,**kwargs):
        """Automatically download archival photometric (i.e. TESS) data using MonoTools.lightcurve
        This also automatically initialises the stellar parameters from the TIC catalogue info.
        tic (int, optional): TIC ID. If not present, we can get it from the TOI list
        """
        try:
            from MonoTools.MonoTools import lightcurve
            from . import tools
        except:
            raise ImportError("Cannot import MonoTools. Check it is installed, or initialise the class with `get_tess=False` and add a lightcurve with `mod.add_lc`")
        #First we need the TIC ID
        if tic is not None:
            self.monotools_lc=lightcurve.multilc(int(tic),'tess',**kwargs)
        elif "TIC" in self.name:
            self.monotools_lc=lightcurve.multilc(int(self.name.replace('-','')[3:]),'tess',**kwargs)
        elif "TOI" in self.name:
            self.filter_TOI()
            self.monotools_lc=lightcurve.multilc(self.init_toi_data.iloc[0]['TIC ID'],'tess',**kwargs)
        Rstar, Teff, logg = starpars_from_MonoTools_lc(self.monotools_lc)
        self.init_starpars(Rstar=Rstar,Teff=Teff,logg=logg)
        
        #Getting cadences for each mission and adding them to the model:
        lc_dic_flipped={v: k for k, v in lightcurve.tools.lc_dic.items()}
        all_missions=np.unique([cad.split("_")[0] for cad in self.monotools_lc.cadence_list])
        misss=np.array([c[:2] for c in self.monotools_lc.cadence])
        for unq_miss_id in all_missions:
            print(unq_miss_id)
            cad_ix=(self.monotools_lc.mask)&(misss==unq_miss_id)
            self.add_lc(self.monotools_lc.time[cad_ix]+2457000,self.monotools_lc.flux[cad_ix],self.monotools_lc.flux_err[cad_ix],source=lc_dic_flipped[unq_miss_id])

    def get_cheops(self,catname=None, n_prog=None, distthresh=3,download=True,PIPE=True,fks=None,**kwargs):
        """Automatically download CHEOPS data using Dace
        """
        try:
            from dace_query.cheops import Cheops
        except:
            raise ImportError("Cannot import Dace. Check it is installed, or initialise the class with `get_cheops=False` and add a lightcurve with `mod.add_cheops_lc`")
        catname= self.name if catname is None else catname
        try:
            these_cheops_obs=pd.DataFrame(Cheops.query_database(filters={"obj_id_catname":{"contains":catname},"file_key":{"contains":"_V020"}}))
            if len(these_cheops_obs)==0:
                from dace_query.cheops import Cheops
                raise ValueError("No objects returned for name="+catname)
            elif hasattr(self,'radec'):
                obs_radecs=SkyCoord([rd.split(" / ")[0] for rd in these_cheops_obs['obj_pos_coordinates_hms_dms'].values], 
                                    [rd.split(" / ")[1] for rd in these_cheops_obs['obj_pos_coordinates_hms_dms'].values],
                                    unit=(u.hourangle,u.deg))
                assert np.all(self.radec.separation(obs_radecs).arcsec<distthresh), "The RA/DEC of CHEOPS visits does not match the Ra/DEC included above"
        except:
            assert hasattr(self,'radec'), "If indexing by name does not work, we must have an RA/Dec coordinate"
            #Could not get it using the name, trying with the programme and the coordinates
            if n_prog is not None:
                all_cheops_obs=pd.DataFrame(Cheops.query_database(filters={"prog_id":{"contains":str(int(n_prog))},"file_key":{"contains":"_V020"}}))
            else:
                #Getting all data/all programmes
                all_cheops_obs=pd.DataFrame(Cheops.query_database(filters={"prog_id":{"contains":"CHEOPS"},"file_key":{"contains":"_V020"}}))
            #Finding which target we have:
            print(all_cheops_obs)
            all_radecs=SkyCoord([rd.split(" / ")[0] for rd in all_cheops_obs['obj_pos_coordinates_hms_dms'].values], 
                                [rd.split(" / ")[1] for rd in all_cheops_obs['obj_pos_coordinates_hms_dms'].values],
                                unit=(u.hourangle,u.deg))
            these_cheops_obs=all_cheops_obs.loc[self.radec.separation(all_radecs).arcsec<distthresh]
        print(these_cheops_obs['file_key'].values)
        if fks is not None:
            fks=[f.replace("CH_","") for f in fks]
            these_cheops_obs=these_cheops_obs.loc[np.array([f.replace("CH_","") in fks for f in these_cheops_obs['file_key'].values])]
        print(these_cheops_obs['file_key'].values)

        assert these_cheops_obs.shape[0]>0, "No matches found in the CHEOPS database. Are you logged in via your .dacerc file?"
        for fk in these_cheops_obs['file_key'].values:
            ifk=fk[3:] if fk[:3]=="CH_" else fk
            if hasattr(self,'monotools_lc'):
                self.add_cheops_lc(filekey=ifk, fileloc=None, download=download, PIPE=PIPE, DRP=~PIPE,
                                   mag=self.monotools_lc.all_ids['tess']['data']['GAIAmag'],**kwargs)
            else:
                self.add_cheops_lc(filekey=ifk, fileloc=None, download=download, PIPE=PIPE, DRP=~PIPE,
                                   mag=these_cheops_obs.loc[these_cheops_obs['file_key']==fk,'obj_mag_v'],**kwargs)
                                   #self.monotools_lc.all_ids['tess']['data']['GAIAmag']


    def filter_TOI(self,threshdist=3):
        if not hasattr(self,'toi_cat'):
            self.get_TOI()
        if "TOI" in self.name and int(self.name.replace('-','')[3:]) in self.toi_cat['star_TOI'].values:
            #Adding this as planet array
            self.init_toi_data=self.toi_cat.loc[self.toi_cat['star_TOI']==int(self.name.replace('-','')[3:])]
        elif hasattr(self,'radec'):
            toi_radecs=SkyCoord(info['RA'].values,info['Dec'].values,unit=(u.hourangle,u.deg))
            seps = self.radec.separation(toi_radecs)
            assert np.min(seps.arcsec)<3, "Must have at least one TOI within "+str(threshdist)+"arcsec"
            self.init_toi_data=self.toi_cat.loc[self.toi_cat['star_TOI']==self.toi_cat.iloc[np.argmin(seps)]['star_TOI']]
        else:
            raise ValueError()

    def get_TOI(self):
        round_date=int(np.round(Time.now().jd,-1))
        if not os.path.exists(os.path.join(tablepath,"TOI_tab_jd_"+str(round_date)+".csv")):
            self.toi_cat=pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
            self.toi_cat['star_TOI']=self.toi_cat['TOI'].values.astype(int)
            self.toi_cat.to_csv(os.path.join(tablepath,"TOI_tab_jd_"+str(round_date)+".csv"))
        else:
            self.toi_cat=pd.read_csv(os.path.join(tablepath,"TOI_tab_jd_"+str(round_date)+".csv"),index_col=0)

    def update(self,**kwargs):
        #Updating settings
        for param in kwargs:
            if param in self.defaults:
                setattr(self,param,kwargs[param])
        if self.save_file_loc is None:
            self.save_file_loc=os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(self.save_file_loc,self.name)):
            os.mkdir(os.path.join(self.save_file_loc,self.name))
        bools=['debug','load_from_file','fit_gp','fit_flat','train_gp','cut_oot','bin_oot','pred_all','use_bayes_fact','use_signif',
               'use_multinest','use_pymc3','assume_circ','fit_ttvs','fit_phi_gp','fit_phi_spline','common_phi_model','use_mstar','use_logg',
               'constrain_lds','fit_contam']
        boolstr=''
        for i in bools:
            boolstr+=str(int(getattr(self,i)))
            
        nonbools=['flat_knotdist','cut_distance','mask_distance','bin_size','signif_thresh','ecc_prior','npoly_rv','ld_mult','timing_sd_durs','rv_mass_prior','spline_order','spline_bkpt_cad']
        nonboolstrs=[]
        for i in nonbools:
            nonboolstrs+=[str(getattr(self,i)) if len(str(getattr(self,i)))<5 else str(getattr(self,i))[:5]]
        comm="" if self.comment is None else "_"+self.comment
        self.unq_name=self.name+"_"+date.today().strftime("%Y%m%d")+"_"+str(int(boolstr, 2))+"_"+"_".join(nonboolstrs)+comm


    def add_lc(self, time, flux, flux_err, mask=None, source='tess'):
        if not hasattr(self,'lcs'):
            self.lcs={}
        self.lcs[source]=pd.DataFrame({'time':time,'flux':flux,'flux_err':flux_err})
        self.lcs[source]['mask']=np.tile(True,len(time)) if mask is None else mask

    def run_PIPE(self,out_dir,fk,mag=None,overwrite=False,make_psf=False,binary=False,**kwargs):
        """
         Run Pipe to extract PSFs. This is a wrapper around the : py : func : ` ~psf. pipeline. PipeParam ` class and it's subarray extraction function
         
         Args:
         	 out_dir: directory to save the results
         	 fk: Filenames of the PSF subarray
         	 mag: Magnitude of the subarray in Hz
         	 overwrite: Whether to overwrite the files or not. If True they will be overwritten
             make_psf: Use PIPE to make a specific PSF for this target (long)
             binary: Whether there is a nearby contaminant within the PSF that we want to model using the BG-star-fits
         Returns: 
         	 A list of : py : class : ` ~psf. pipeline. PipelineParam `
        """
        try:
            from pipe import PipeParam, PipeControl, config
        except:
            raise ImportError("PIPE not importable.")

        mag=10.5 if mag is None else mag

        out_dir=os.path.join(self.save_file_loc,self.name)
        pipe_refdataloc=config.get_conf_paths()[1]
        
        # folds = glob.glob(os.path.join(out_dir,fk,"Outdata","000??"))
        # folds = {int(pf[0].split('/')[-1]):pf for pf in folds[0] if len(glob.glob(os.path.join(pf,"*.fits")))>0}
        # n_max = list(folds.keys())[np.argmax(list(folds.keys()))]
        
        #Checking if we have an Outdata file but no PIPE outputs (in which case we delete)
        if (os.path.exists(os.path.join(out_dir,fk,"Outdata","00000")) and len(glob.glob(os.path.join(out_dir,fk,"Outdata","00000","*.fits")))==0) or (os.path.exists(os.path.join(out_dir,fk,"Outdata")) and len(glob.glob(os.path.join(out_dir,fk,"Outdata","*")))==0):
            os.system("rm -r "+os.path.join(out_dir,fk,"Outdata"))
        if not os.path.exists(os.path.join(out_dir,fk,"Outdata")) or overwrite:
            #os.system("mkdir "+os.path.join(out_dir,fk,"Outdata"))
            #os.system("mkdir "+os.path.join(out_dir,fk,"Outdata","00000"))
            #Running PIPE:
            from pipe import PipeParam, PipeControl
            pps = PipeParam(self.name, fk)#, outdir=os.path.join(out_dir,fk,"Outdata","00000"))
            #pps.bgstars = True 
            pps.fit_bgstars = False
            #pps.limflux = 1e-5
            pps.darksub = True
            #pps.dark_level = 2
            #pps.remove_static = True
            #pps.save_static = False
            #pps.static_psf_rad = False
            pps.sa_optimise = False
            #pps.smear_fact = 5.5
            pps.psf_score = None
            pps.psf_min_num = 12
            pps.cti_corr = True
            #pps.smear_resid = False
            #pps.smear_resid_sa = True
            #pps.non_lin_tweak = True
            pps.klip = 1
            #pps.sigma_clip = 15
            #pps.empiric_noise = True
            #pps.empiric_sigma_clip = 4
            #pps.save_noise_cubes = False
            pps.save_psfmodel = True
            #pps.fitrad = fitrad
            #pps.psflib = psf_locs2[best]
            #print(psf_locs2[best])
            #pc.make_psf_lib()
            #if make_psf:
            #    pc = PipeControl(pps)
            #    pc.make_psf_lib()
            #else:
            #    #pps.psflib = pipe_refdataloc+"/psf_lib/"+psf_locs[closest_subarr]
            #pc = PipeControl(pps)
            if binary:
                pps.fit_bgstars = False
                #pps.binary = True
                #pps.robust_centre_binary = True
                #starcat=fits.open(glob.glob(os.path.join(out_dir,fk,"*_StarCatalogue-*.fits"))[0])
                #pps.fix_flux2 = True
                #pps.secondary = np.argmin((starcat[1].data['MAG_GAIA'][0]-starcat[1].data['MAG_GAIA'][1:])+starcat[1].data['DISTANCE'][1:]/15)
                pc = PipeControl(pps)
                pc.process_eigen()
                #pc.process_binary()
                if make_psf:
                    pc.make_psf_lib()
            else:
                pc = PipeControl(pps)
                pc.process_eigen()
                if make_psf:
                    pc.make_psf_lib()
            # Returns the path to the output PSF file.
            if pc.pps.file_im is not None:
                return os.path.join(pc.pp.pps.outdir, f'{pc.pp.pps.name}_{pc.pp.pps.visit}_im.fits')
            else:
                return os.path.join(pc.pp.pps.outdir, f'{pc.pp.pps.name}_{pc.pp.pps.visit}_sa.fits')
        else:
            #Looping through "Outdata" folders - higher number = most recent = best
            # Return the latest folder in out_dir.
            for folder in np.sort(np.array(glob.glob(os.path.join(out_dir,fk,"Outdata","000*"))))[::-1]:
                # Returns the name of the file in the folder.
                if len(glob.glob(os.path.join(folder,self.name+"*_im.fits")))>0:
                    return glob.glob(os.path.join(folder,self.name+"*_im.fits"))[0]
                elif len(glob.glob(os.path.join(folder,self.name+"*_sa.fits")))>0:
                    return glob.glob(os.path.join(folder,self.name+"*_sa.fits"))[0]
                else:
                    continue
            raise ValueError("Unable to either run PIPE extractions or return PIPE fits file")


    def add_cheops_lc(self, filekey, fileloc=None, download=True, ylims=(-15,15),
                      DRP=True, PIPE=False, PIPE_bin_src=None, mag=None, **kwargs):
        """AI is creating summary for add_cheops_lc

        Args:
            filekey (str): Unique filekey for this Cheops lightcurve
            fileloc (str, optional): Location of lightcurve fits file
            download (bool, optional): Should we download from DACE?
            ylims (tuple, optional): Limits below/above which to cut (-15,15)
            DRP (bool, optional): Is this a DRP file? Defaults to True.
            PIPE (bool, optional): Is this a PIPE file? Defaults to False.
            PIPE_bin_src (int, optional): If this is a PIPE file with two stars (ie binary model) which should we model? Defaults to None.
            mag (float, optional): Magnitude needed by PIPE to guess fit radius? Defaults to 10.5
        """

        filekey=filekey[3:] if filekey[:3]=="CH_" else filekey

        assert filekey not in self.cheops_filekeys, "Duplicated CHEOPS filekeys"

        self.update(**kwargs)

        assert DRP^PIPE, "Must have either DRP or PIPE flagged"
        out_dir=os.path.join(self.save_file_loc,self.name)
        
        if fileloc is None and download:
            from dace_query.cheops import Cheops
            n_attempts=0
            while not os.path.isdir(os.path.join(out_dir,filekey)) and n_attempts<5:
                try:
                    if not os.path.isdir(os.path.join(out_dir,filekey)):
                        #Downloading Cheops data:
                        if PIPE:
                            print("Downloading "+filekey+" with Dace")
                            Cheops.download('all', {'file_key': {'contains': "CH_"+str(filekey)}},
                                            output_directory=out_dir, output_filename=self.name+'_'+filekey+'_dace_download.tar.gz')
                            print("Succeeded downloading "+filekey+" with Dace")
                        elif DRP:
                            Cheops.download('lightcurves', {'file_key': {'contains': "CH_"+str(filekey)}},
                                            output_directory=out_dir, output_filename=self.name+'_'+filekey+'_dace_download.tar.gz')
                        os.system("tar -xvzf "+out_dir+'/'+self.name+'_'+filekey+'_dace_download.tar.gz -C '+out_dir)
                        #Deleting it
                        os.system("rm "+out_dir+'/'+self.name+'_'+filekey+'_dace_download.tar.gz')
                except:
                    time.sleep(15)
                print("new dir exists?",os.path.isdir(os.path.join(out_dir,filekey)))
                if not os.path.isdir(os.path.join(out_dir,filekey)):
                    time.sleep(15)
                    n_attempts+=1
            assert os.path.isdir(os.path.join(out_dir,filekey)), "Unable to download filekey "+filekey+" using Dace."
        
        if fileloc is None and download and PIPE:
            fileloc = self.run_PIPE(out_dir,filekey,mag,**kwargs)

        if not hasattr(self,"cheops_lc"):
            self.cheops_lc = pd.DataFrame()
        
        if PIPE:
            binchar=str(int(PIPE_bin_src)) if PIPE_bin_src is not None else ''
            sources={'time':'BJD_TIME', 'flux':'FLUX'+binchar, 'flux_err':'FLUXERR'+binchar, 
                     'phi':'ROLL', 'bg':'BG', 'centroidx':'XC'+binchar,
                     'centroidy':'YC'+binchar, 'deltaT':'thermFront_2','smear':None}
        elif DRP:
            sources={'time':'BJD_TIME', 'flux':'FLUX', 'flux_err':'FLUXERR', 
                     'phi':'ROLL_ANGLE', 'bg':'BACKGROUND', 'centroidx':'CENTROID_X', 
                     'centroidy':'CENTROID_Y', 'deltaT':None, 'smear':'SMEARING_LC'}
        f=fits.open(fileloc)
        iche=pd.DataFrame()
        for s in sources:
            if sources[s] is not None:
                if s=='flux_err' and sources[s] not in f[1].data:
                    iche[s]=np.sqrt(f[1].data[sources['flux']])
                else:
                    iche[s]=f[1].data[sources[s]]
            if s=='flux':
                iche['raw_flux']=iche[s].values
                iche[s]=(iche[s].values/np.nanmedian(f[1].data[sources['flux']])-1)*1000
            if s=='flux_err':
                iche['raw_flux_err']=iche[s].values
                iche[s]=(iche[s].values/np.nanmedian(f[1].data[sources['flux']]))*1000
            if s=='phi':
                if hasattr(self,'radec') and self.radec is not None:
                    moon_coo = get_body('moon', Time(iche['time'],format='jd',scale='tdb'))
                    v_moon = np.arccos(
                            np.cos(moon_coo.ra.radian)*np.cos(moon_coo.dec.radian)*np.cos(self.radec.ra.radian)*np.cos(self.radec.dec.radian) +
                            np.sin(moon_coo.ra.radian)*np.cos(moon_coo.dec.radian)*np.sin(self.radec.ra.radian)*np.cos(self.radec.dec.radian) +
                            np.sin(moon_coo.dec.radian)*np.sin(self.radec.dec.radian))
                    dv_rot = np.degrees(np.arcsin(np.sin(moon_coo.ra.radian-self.radec.ra.radian)*np.cos(moon_coo.dec.radian)/np.sin(v_moon)))
                    iche['cheops_moon_angle']=(iche[s].values[:]-dv_rot)%360
                iche[s+"_orig"]=iche[s].values[:]
                iche[s]=roll_rollangles(iche[s].values)
        iche['xoff']=iche['centroidx']-np.nanmedian(iche['centroidx'])
        iche['yoff']=iche['centroidy']-np.nanmedian(iche['centroidy'])
        iche['phi_sorting']=np.argsort(iche['phi'].values)
        iche['time_sorting']=np.argsort(iche['time'].values[iche['phi_sorting'].values])

        #Getting moon-object angle:

        iche['filekey']=np.tile(filekey,len(f[1].data[sources['time']]))
        
        #Performing simple anomaly masking using background limit, nans, and flux outliers:
        bgthresh=np.percentile(iche['bg'].values,95)*1.5
        iche['mask']=(~np.isnan(iche['flux']))&(~np.isnan(iche['flux_err']))&cut_anom_diff(iche['flux'].values)&(iche['bg']<bgthresh)&(iche['flux']>ylims[0])&(iche['flux']<ylims[-1])
        iche.loc[iche['mask'],'mask']&=cut_anom_diff(iche['flux'].values[iche['mask']])
        
        iche['mask_phi_sorting']=np.tile(-1,len(iche['mask']))
        iche['mask_time_sorting']=np.tile(-1,len(iche['mask']))
        iche.loc[iche['mask'],'mask_phi_sorting']=np.argsort(iche.loc[iche['mask'],'phi'].values).astype(int)
        iche.loc[iche['mask'],'mask_time_sorting']=np.argsort(iche.loc[iche['mask'],'mask_phi_sorting'].values)

        self.cheops_lc=self.cheops_lc.append(iche)

        if not hasattr(self,'cheops_filekeys'):
            self.cheops_filekeys=[filekey]
        else:
            self.cheops_filekeys+=[filekey]
        
        if not hasattr(self,'cheops_fk_mask'):
            self.cheops_fk_mask={}
        for fk in self.cheops_filekeys:
            self.cheops_fk_mask[fk]=(self.cheops_lc['filekey'].values==fk)&(self.cheops_lc['mask'].values)
        
        self.descr_dict[filekey]={'src':["DRP","PIPE"][int(PIPE)],
                                  'len':(iche['time'].values[-1]-iche['time'].values[0]),
                                  'start':Time(iche['time'][0],format='jd').isot,
                                  'cad':np.nanmedian(np.diff(iche['time'])),
                                  'rms':1e3*np.nanstd(iche.loc[iche['mask'],'flux'])}

    def add_rvs(self,x,y,yerr,name, overwrite=False, **kwargs):
        """AI is creating summary for add_rvs

        Args:
            x ([type]): [description]
            y ([type]): [description]
            yerr ([type]): [description]
            name ([type]): [description]

        Optional args:
            npoly_rv (int): 
        """

        self.update(**kwargs)

        #initialising stored dicts:
        if not hasattr(self,"rvs") or overwrite:
            self.rvs = pd.DataFrame()
        if not hasattr(self,"rv_medians") or overwrite:
            self.rv_medians={}
        if not hasattr(self,"rv_stds") or overwrite:
            self.rv_stds={}

        irv=pd.DataFrame({'time':x,'y':y,'yerr':yerr,'scope':np.tile(name,len(x))})
        self.rvs=self.rvs.append(irv)
        
        #Adding median and std info to stored dicts:
        self.rv_medians[name]=np.nanmedian(y)
        self.rv_stds[name]=np.nanstd(y)

        #Making index array:
        self.rv_instr_ix=np.column_stack([self.rvs['scope'].values==scope for scope in self.rv_medians])
        
        self.rv_x_ref=np.average(self.rvs['time'])
        self.rv_t = np.arange(np.nanmin(self.rvs['time']-10), np.nanmax(self.rvs['time']+10), 0.0666*np.min([self.planets[pl]['period'] for pl in self.planets]))

    def init_starpars(self,Rstar=None,Teff=None,logg=None,FeH=0.0,rhostar=None,Mstar=None, **kwargs):
        """Adds stellar parameters to model

        Args:
            Rstar (list, optional): Stellar radius in Rsol in format [value, neg_err, pos_err]. Defaults to np.array([1.0,0.08,0.08]).
            Teff (list, optional): Stellar effective Temperature in K in format [value, neg_err, pos_err]. Defaults to np.array([5227,100,100]).
            logg (list, optional): Stellar logg in cgs in format [value, neg_err, pos_err]. Defaults to np.array([4.3,1.0,1.0]).
            FeH (float, optional): Stellar log Metallicity. Defaults to 0.0.
            rhostar (list, optional): Stellar density in rho_sol (1.411gcm^-3) in format [value, neg_err, pos_err]. Defaults to None.
            Mstar (float or list, optional): Stellar mass in Msol either as a float or in format [value, neg_err, pos_err]. Defaults to None.
            
        Optional kwargs:
            use_mstar (boolean, optional): Whether to model using the stellar Mass (otherwise set use_logg). Defaults to True
            use_logg (boolean, optional): Whether to model using the stellar logg (otherwise Mass). Defaults to False

        """
        self.update(**kwargs)

        if Rstar is None and hasattr(self.lc,'all_ids') and 'tess' in self.lc.all_ids and 'data' in self.lc.all_ids['tess'] and 'rad' in self.lc.all_ids['tess']['data']:
            #Radius info from lightcurve data (TIC)
            if 'eneg_Rad' in self.lc.all_ids['tess']['data'] and self.lc.all_ids['tess']['data']['eneg_Rad'] is not None and self.lc.all_ids['tess']['data']['eneg_Rad']>0:
                Rstar=self.lc.all_ids['tess']['data'][['rad','eneg_Rad','epos_Rad']].values
            else:
                Rstar=self.lc.all_ids['tess']['data'][['rad','e_rad','e_rad']].values
            if 'eneg_Teff' in self.lc.all_ids['tess']['data'] and self.lc.all_ids['tess']['data']['eneg_Teff'] is not None and self.lc.all_ids['tess']['data']['eneg_Teff']>0:
                Teff=self.lc.all_ids['tess']['data'][['Teff','eneg_Teff','epos_Teff']].values
            else:
                Teff=self.lc.all_ids['tess']['data'][['Teff','e_Teff','e_Teff']].values
            if 'eneg_logg' in self.lc.all_ids['tess']['data'] and self.lc.all_ids['tess']['data']['eneg_logg'] is not None and self.lc.all_ids['tess']['data']['eneg_logg']>0:
                logg=self.lc.all_ids['tess']['data'][['logg','eneg_logg','epos_logg']].values
            else:
                logg=self.lc.all_ids['tess']['data'][['logg','e_logg','e_logg']].values
        if Rstar is None:
            Rstar=np.array([1.0,0.08,0.08])
        if Teff is None:
            Teff=np.array([5227,100,100])
        if logg is None:
            logg=np.array([4.3,1.0,1.0])

        self.Rstar=np.array(Rstar).astype(float)
        self.Teff=np.array(Teff).astype(float)
        self.logg=np.array(logg).astype(float)
        self.FeH=FeH

        if Mstar is not None:
            self.Mstar = Mstar if type(Mstar)==float else float(Mstar[0])
        #Here we only have a mass, radius, logg- Calculating rho two ways (M/R^3 & logg/R), and doing weighted average
        if rhostar is None:
            rho_logg=[np.power(10,self.logg[0]-4.43)/self.Rstar[0]]
            rho_logg+=[np.power(10,self.logg[0]+self.logg[1]-4.43)/(self.Rstar[0]-self.Rstar[1])/rho_logg[0]-1.0,
                       1.0-np.power(10,self.logg[0]-self.logg[2]-4.43)/(self.Rstar[0]+self.Rstar[2])/rho_logg[0]]
            if Mstar is not None:
                rho_MR=[Mstar[0]/self.Rstar[0]**3]
                rho_MR+=[(Mstar[0]+Mstar[1])/(self.Rstar[0]-abs(self.Rstar[1]))**3/rho_MR[0]-1.0,
                         1.0-(Mstar[0]-abs(Mstar[2]))/(self.Rstar[0]+self.Rstar[2])**3/rho_MR[0]]
                #Weighted sums of two avenues to density:
                rhostar=[rho_logg[0]*(rho_MR[1]+rho_MR[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2])+
                         rho_MR[0]*(rho_logg[1]+rho_logg[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2])]
                rhostar+=[rhostar[0]*(rho_logg[1]*(rho_MR[1]+rho_MR[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2])+
                                      rho_MR[1]*(rho_logg[1]+rho_logg[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2])),
                          rhostar[0]*(rho_logg[2]*(rho_MR[1]+rho_MR[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2])+
                                      rho_MR[2]*(rho_logg[1]+rho_logg[2])/(rho_logg[1]+rho_logg[2]+rho_MR[1]+rho_MR[2]))]
                self.Mstar=Mstar
            else:
                rhostar=[rho_logg[0],rho_logg[0]*rho_logg[1],rho_logg[0]*rho_logg[2]]

            self.rhostar=np.array(rhostar).astype(float)
            if Mstar is None:
                self.Mstar=[rhostar[0]*self.Rstar[0]**3]
                self.Mstar+=[self.Mstar[0]-((rhostar[0]-rhostar[1])*(self.Rstar[0]-self.Rstar[1])**3),((rhostar[0]+rhostar[2])*(self.Rstar[0]+self.Rstar[2])**3)-self.Mstar[0]]
        else:
            self.rhostar=np.array(rhostar).astype(float)
            if Mstar is None:
                self.Mstar=rhostar[0]*self.Rstar[0]**3

    def add_planets_from_toi(self):
        """Add all TOIs to the model as planets
        """
        assert hasattr(self,'init_toi_data')

        self.init_toi_data=self.init_toi_data.sort_values('Period (days)')
        for i,row in enumerate(self.init_toi_data.iterrows()):
            assert "bcdefgh"[i] not in self.planets or overwrite, "Name is already stored as a planet"
            self.add_planet(name="bcdefgh"[i],
                            tcen=float(row[1]['Epoch (BJD)']),
                            tcen_err=float(row[1]['Epoch (BJD) err']),
                            tdur=float(row[1]['Duration (hours)'])/24,
                            depth=float(row[1]['Depth (ppm)'])/1e6,
                            period=float(row[1]['Period (days)']),
                            period_err=float(row[1]['Period (days) err']))

    def add_planet(self, name, tcen, period, tdur, depth, tcen_err=None,
                   period_err=None, b=None, rprs=None, K=None, overwrite=False):
        """Add planet to the model

        Args:
            name (str): Name associated with planet (e.g. b or c)
            tcen (float): transit epoch in same units as time array (i.e. TJD)
            period (float): transit period in same units as time array (i.e. days)
            tdur (float): transit duration in days
            depth (float): transit depth as ratio
            tcen_err (float,optional): transit epoch error (optional)
            period_err (float,optional): transit period error in same units as time array (i.e. days)
            b (float,optional): impact parameter
            rprs (float,optional): radius ratio
            K (float,optional): RV semi-amplitude in m/s
        """
        assert name not in self.planets or overwrite, "Name is already stored as a planet"
        
        if period_err is None:
            period_err = self.timing_sd_durs*tdur/period

        if rprs is None:
            assert depth<0.25 #Depth must be a ratio (not in mmags)
            rprs=np.sqrt(depth)

        if b is None:
            rho_S=self.rhostar[0] if hasattr(self,'rhostar') else 1.0
            #Estimating b from simple geometry:

            b=np.clip((1+rprs)**2 - (tdur*86400)**2 * \
                                ((3*period*86400) / (np.pi**2*6.67e-11*rho_S*1410))**(-2/3),
                                0.01,2.0)**0.5
        self.planets[name]={'tcen':tcen,'tcen_err':tcen_err if tcen_err is not None else 0.25*tdur,
                            'period':period,'period_err':period_err,'tdur':tdur,'depth':depth,
                            'b':b,'rprs':rprs,'K':K}

    def init_lc(self, xmask=None, **kwargs):
        """Initialise survey (i.e. TESS) lightcurve. 
        This will create a lightcurve as the lc_fit object.

        Optional:
            fit_gp (bool) - co-fit a GP.
            fit_flat (bool) - flatten the lightcurve before modelling
            cut_distance (float) - cut out points further than cut_distance*Tdur. 0.0 means no cutting
            mask_distance (float) - Distance, in transit durations, from set transits, to "mask" as in-transit data when e.g. flattening.
            cut_oot (bool) - Cut points outside the cut_distance when fitting
            bin_oot (bool) - Bin points outside the cut_distance to 
            bin_size (float) - Size of binned points (defaults to 30mins)
            flat_knotdist (float) - Length of knotdistance for flattening spline (in days)
        """

        self.update(**kwargs)

        assert hasattr(self,'planets'), "In order to run e.g. smoothing or GP fits, we need to flag planet transits. Please run `add_planet` before `init_lc`"
        assert ~(self.cut_oot&self.bin_oot), "Cannot both cut and bin out of transit data. Pick one."
        assert ~(self.fit_flat&self.fit_gp), "Cannot both flatten data and fit GP. Choose one"        
        
        
        #masking, binning, flattening light curve

        if not hasattr(self,'binlc'):
            self.binlc={}
        for src in self.lcs:
            self.lcs[src]['mask']=~np.isnan(self.lcs[src]['flux'].values)&~np.isnan(self.lcs[src]['flux_err'].values)
            self.lcs[src]['mask'][self.lcs[src]['mask']]=cut_anom_diff(self.lcs[src]['flux'].values[self.lcs[src]['mask']])
            self.lcs[src]['mask'][self.lcs[src]['mask']]=cut_anom_diff(self.lcs[src]['flux'].values[self.lcs[src]['mask']])
            self.lcs[src]['in_trans'] = np.tile(False,len(self.lcs[src]['mask']))
            self.lcs[src]['near_trans'] = np.tile(False,len(self.lcs[src]['mask']))
            for pl in self.planets:
                self.lcs[src]['in_trans']+=abs((self.lcs[src]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<(self.mask_distance*self.planets[pl]['tdur'])
                self.lcs[src]['near_trans']+=abs((self.lcs[src]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.cut_distance*self.planets[pl]['tdur']

            #FLATTENING
            if self.fit_flat:
                spline, newmask = kepler_spline(self.lcs[src]['time'].values[self.lcs[src]['mask']],
                                                self.lcs[src]['flux'].values[self.lcs[src]['mask']], 
                                                transit_mask=~self.lcs[src]['in_trans'][self.lcs[src]['mask']],bk_space=self.flat_knotdist)
                self.lcs[src]['spline']=np.tile(np.nan,len(self.lcs[src]['time']))
                self.lcs[src].loc[self.lcs[src]['mask'],'spline']=spline
                self.lcs[src]['flux_flat']=self.lcs[src]['flux'].values
                self.lcs[src]['flux_flat'][self.lcs[src]['mask']]-=self.lcs[src]['spline']


            #BINNING
            ibinlc=bin_lc_segment(np.column_stack((self.lcs[src]['time'].values[self.lcs[src]['mask']],
                                                    self.lcs[src]['flux'].values[self.lcs[src]['mask']],
                                                    self.lcs[src]['flux_err'].values[self.lcs[src]['mask']])),
                                  self.bin_size)
            print(ibinlc)
            bin_ix=~np.isnan(np.sum(ibinlc,axis=1))
            self.binlc[src]=pd.DataFrame({'time':ibinlc[bin_ix,0],'flux':ibinlc[bin_ix,1],'flux_err':ibinlc[bin_ix,2]})
            if self.fit_flat:
                ibinlc2=bin_lc_segment(np.column_stack((self.lcs[src]['time'].values[self.lcs[src]['mask']],
                                                        self.lcs[src]['flux_flat'].values[self.lcs[src]['mask']],
                                                        self.lcs[src]['flux_err'].values[self.lcs[src]['mask']])),
                                           self.bin_size)
                self.binlc[src]['flux_flat']=ibinlc2[bin_ix,1]
                splinebin=bin_lc_segment(np.column_stack((self.lcs[src]['time'].values[self.lcs[src]['mask']],
                                         self.lcs[src]['spline'].values[self.lcs[src]['mask']],
                                         self.lcs[src]['flux_err'].values[self.lcs[src]['mask']])),
                                         self.bin_size)
                self.binlc[src]['spline']=splinebin[:,1]
            self.binlc[src]['in_trans'] = np.tile(False,len(self.binlc[src]['time']))
            self.binlc[src]['near_trans'] = np.tile(False,len(self.binlc[src]['time']))
            for pl in self.planets:
                self.binlc[src]['in_trans']+=abs((self.binlc[src]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.mask_distance*self.planets[pl]['tdur']
                self.binlc[src]['near_trans']+=abs((self.binlc[src]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.cut_distance*self.planets[pl]['tdur']
        
        if not hasattr(self,'lc_fit') or self.overwrite:
            self.lc_fit={scope:pd.DataFrame() for scope in self.lcs}
        for src in self.lcs:
            vals=['time','flux','flux_err','in_trans','near_trans']
            if self.fit_flat: vals+=['spline']
            for val in vals:
                srcval='flux_flat' if val=='flux' and self.fit_flat else val
                if self.cut_oot:
                    #Cutting far-from-transit values:
                    self.lc_fit[src][val]=self.lcs[src].loc[self.lcs[src]['mask']&self.lcs[src]['near_trans'],srcval]
                    #newvals=self.lcs[src].loc[self.lcs[src]['mask']&self.lcs[src]['near_trans'],srcval]
                elif self.bin_oot:
                    #Merging the binned and raw timeseries so that near-transit data is raw and far-from-transit is binned:
                    #newvals=np.hstack((self.lcs[src].loc[self.lcs[src]['mask']&self.lcs[src]['near_trans'],srcval],self.binlc[src].loc[~self.binlc[src]['near_trans'],srcval]))
                    self.lc_fit[src][val]=np.hstack((self.lcs[src].loc[self.lcs[src]['mask']&self.lcs[src]['near_trans'],srcval],self.binlc[src].loc[~self.binlc[src]['near_trans'],srcval]))
                else:
                    #newvals=self.lcs[src].loc[self.lcs[src]['mask'],srcval]
                    self.lc_fit[src][val]=self.lcs[src].loc[self.lcs[src]['mask'],srcval]
                # if srcval not in self.lc_fit.columns:
                #     self.lc_fit[val]=newvals
                # else:
                #     self.lc_fit[val]=np.hstack((self.lc_fit[val],newvals))

        #     #Adding source to the array:
        #     if 'src' not in self.lc_fit.columns:
        #         self.lc_fit['src']=np.tile(src,len(newvals))
        #     else:
        #         self.lc_fit['src']=np.hstack((self.lc_fit['src'],np.tile(src,len(newvals))))
            self.lc_fit[src]=self.lc_fit[src].sort_values('time')          
        # #Making an index array for the fit lightcurve according to source:
        # self.lc_fit_src_index=np.zeros((len(self.lc_fit),len(self.lcs)))
        # for isrc,src in self.lcs:
        #     self.lc_fit_src_index[self.lc_fit['src']==src,isrc]=1

        if self.train_gp and self.fit_gp:
            self.init_gp(**kwargs)

        for scope in self.lcs:
            if not hasattr(self,'ld_dists'):
                self.ld_dists={}
            self.ld_dists[scope]=get_lds(1200,self.Teff[:2],self.logg[:2],how=scope)

    def init_gp(self, **kwargs):
        """Initiliasing photometry GP on e.g. TESS

        Optional
        """
        self.update(**kwargs)

        from celerite2.theano import terms as theano_terms
        import celerite2.theano

        with pm.Model() as ootmodel:
            logs={}
            for scope in self.lcs:
                logs[scope] = pm.Normal(scope+"_logs", mu=np.log(np.std(self.lc_fit[scope]['flux']))+2, sd=1)
            
            #Initialising the SHO frequency
            max_cad=self.bin_size
            lcrange=27
            av_dur = np.average([self.planets[key]['tdur'] for key in self.planets])
            success=False;target=0.05
            exps=np.array([np.log((2*np.pi)/(av_dur)),np.log((2*np.pi)/(0.25*lcrange))])
            log_w0 = pm.Normal("log_w0",mu=(exps[0]+exps[1])/2,sd=np.ptp(exps)/8)
            w0 = pm.Deterministic("w0",tt.exp(log_w0))
            # while not success and target<0.21:
            #     try:
            #         low=(2*np.pi)/(0.25*lcrange/(target/0.05))
            #         up=(2*np.pi)/(25*av_dur*(target/0.05))
            #         w0 = pm.InverseGamma("w0",testval=(2*np.pi)/10,**pmx.estimate_inverse_gamma_parameters(lower=low,upper=up))
            #         success=True
            #     except:
            #         low=(2*np.pi)/(10)
            #         up=(2*np.pi)/(6*max_cad)
            #         target*=1.15
            #         success=False
            #print("w0",success,target,low,up,(2*np.pi)/low,(2*np.pi)/up)
            
            #Initialising the power:
            success=False;target=0.01
            maxpower=1.0*np.max([np.nanstd(self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans'],'flux'].values) for scope in self.lcs])
            minpower=0.02*np.min([np.nanmedian(abs(np.diff(self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans'],'flux'].values))) for scope in self.lcs])
            
            while not success and target<0.25:
                try:
                    power = pm.InverseGamma("power",testval=minpower*5,
                                            **pmx.estimate_inverse_gamma_parameters(lower=minpower,
                                                                                    upper=maxpower/(target/0.01),
                                                                                    target=0.1))
                    success=True
                except:
                    target*=1.15
                    success=False
            #print("power",success,target)
            S0 = pm.Deterministic("S0", power/(w0**4))

            # GP model for the light curve
            kernel = theano_terms.SHOTerm(S0=S0, w0=w0, Q=1/np.sqrt(2))
            means={}
            for scope in self.lcs:
                print(scope, np.nanmedian(self.lcs[scope]['flux']))
                means[scope] = pm.Normal(scope+"_mean", mu=0.0, sd=10.0, testval=np.nanmedian(self.lcs[scope]['flux']))
            gp = celerite2.theano.GaussianProcess(kernel)
            allt=np.hstack([self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans'],'time'].values for scope in self.lcs])
            ally=np.hstack([self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans'],'flux'].values for scope in self.lcs])[np.argsort(allt)]
            allyerr=np.hstack([self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans'],'flux_err'].values for scope in self.lcs])[np.argsort(allt)]
            allsrcs=np.hstack([np.tile(iscope,len(self.lc_fit[list(self.lcs.keys())[iscope]]['time'])) for iscope in np.arange(len(self.lcs))])[np.argsort(allt)]
            allsrcs=np.column_stack((np.isin(allsrcs,i) for i in range(len(self.lcs))))
            allt=np.sort(allt)
            gp.compute(allt,diag=allyerr ** 2 + tt.dot(allsrcs,tt.exp([logs[scope] for scope in logs])**2), quiet=True)
            gp.marginal("obs", observed=ally-tt.dot(allsrcs,[means[scope] for scope in logs]))
            
            #photgp_model_x = pm.Deterministic("photgp_model_x", gp.predict(self.lc_fit['flux'][~self.lc_fit['in_trans']], t=self.lc_fit['time'][~self.lc_fit['in_trans']], return_var=False))

            #optimizing:
            start = ootmodel.test_point
            print(ootmodel.check_test_point())
            oot_soln = pmx.optimize(start=start)

        #Sampling:
        with ootmodel: 
            if 'cores' in kwargs:
                self.oot_gp_trace = pm.sample(tune=500, draws=1200, start=oot_soln, 
                                        compute_convergence_checks=False,cores=kwargs['cores'])
            else:
                self.oot_gp_trace = pm.sample(tune=500, draws=1200, start=oot_soln, 
                                        compute_convergence_checks=False)


    def cheops_only_model(self, fk, transittype="fix", force_no_dydt=True, overwrite=False,linpars=None,quadpars=None,**kwargs):
        """Initialising and running a Cheops-only transit model for a given filekey

        Args:
            fk (str): Cheops filekey.
            transittype (str, optional): How to include transit model - "set": set by TESS transits, "loose": allowed to vary, "none": no transit at all. Defaults to "fix".
            force_no_dydt (optional): Do we force the model to avoid using decorrelation with trends? Defaults to None, which mirrors include_transit
            overwrite (bool, optional): Whether to rewrite this initialise model. If not, it will try to reload a pre-run model. Defaults to False.
            linpars (list of strings, optional): Specify the parameters to use for the linear decorrelation. For sin/cos, use cosNphi where N is the harmonic (i.e. normal = 1)
            quadpars (list of strings, optional): Specify the parameters to use for the quadratic decorrelation
        
        Returns:
            PyMC3 trace: The output model trace from the fit.
        """
        #Initialising save name (and then checking if we can re-load the saved model fit):
        savefname="_che_only_fit_"+fk+"_trace"
        che_fk_save_name_dic = {'fix':'_fixtrans','loose':'_loosetrans','none':'_notrans'}
        
        if transittype in che_fk_save_name_dic:
            savefname+=che_fk_save_name_dic[transittype]
        if force_no_dydt: savefname+="_notrend" 

        if not hasattr(self,'cheops_init_trace'):
            self.cheops_init_trace={}

        if linpars is None:
            #Initialising decorrelation parameters:
            self.init_che_linear_decorr_pars=['sin1phi','cos1phi','sin2phi','cos2phi','sin3phi','cos3phi','bg','centroidx','centroidy','time']
            if 'smear' in self.cheops_lc.columns: self.init_che_linear_decorr_pars+=['smear']
            if 'deltaT' in self.cheops_lc.columns and not force_no_dydt: self.init_che_linear_decorr_pars+=['deltaT']
            if force_no_dydt: self.init_che_linear_decorr_pars.remove('time')
        else:
            self.init_che_linear_decorr_pars=linpars
        if quadpars is None:
            self.init_che_quad_decorr_pars=['bg','centroidx','centroidy']
            if 'smear' in self.cheops_lc.columns: self.init_che_quad_decorr_pars+=['smear']
            #if 'deltaT' in self.cheops_lc.columns and not force_no_dydt: self.init_che_quad_decorr_pars+=['deltaT'] 
            #if force_no_dydt: self.init_che_quad_decorr_pars.remove('time')
        else:
            self.init_che_quad_decorr_pars=quadpars
        #Initialising the data specific to each Cheops visit:
        x=self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'].values.astype(np.float64)
        y=self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values.astype(np.float64)
        #Using a robust average (logged) of the point-to-point error & the std as a prior for the decorrelation parameters
        self.cheops_mads[fk]=np.exp(0.5*(np.log(np.std(y))+np.log(np.nanmedian(abs(np.diff(y)))*1.06)))
        
        yerr=self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux_err'].values
        for var in self.init_che_linear_decorr_pars+self.init_che_quad_decorr_pars:
            if var in self.cheops_lc.columns:
                print(var,fk)
                self.norm_cheops_dat[fk][var]=(self.cheops_lc.loc[self.cheops_fk_mask[fk],var].values-np.nanmedian(self.cheops_lc.loc[self.cheops_fk_mask[fk],var].values))/np.nanstd(self.cheops_lc.loc[self.cheops_fk_mask[fk],var].values)
            elif var[:3]=='sin':
                if var[3] not in ['1','2','3','4']:
                    print(var,var[3], "- Must have a number in the cos/sin parameter name to represent the harmonic, e.g. cos1phi or sin3phi")
                    var=var[:3]+"1"+var[3:]
                #self.norm_cheops_dat[fk]
                self.norm_cheops_dat[fk][var]=np.sin(float(int(var[3]))*self.cheops_lc.loc[self.cheops_fk_mask[fk],var[-3:]].values*np.pi/180)
            elif var[:3]=='cos':
                if var[3] not in ['1','2','3','4']:
                    print(var,var[3], "- Must have a number in the cos/sin parameter name to represent the harmonic, e.g. cos1phi or sin3phi")
                    var=var[:3]+"1"+var[3:]
                self.norm_cheops_dat[fk][var]=np.cos(float(int(var[3]))*self.cheops_lc.loc[self.cheops_fk_mask[fk],var[-3:]].values*np.pi/180)
        #print(os.path.join(self.save_file_loc,self.name,self.unq_name+savefname+".pkl"),os.path.exists(os.path.join(self.save_file_loc,self.name,self.unq_name+savefname+".pkl")),overwrite)
        if not overwrite and os.path.exists(os.path.join(self.save_file_loc,self.name,self.unq_name+savefname+".pkl")):
            self.cheops_init_trace[savefname[1:]]=pickle.load(open(os.path.join(self.save_file_loc,self.name,self.unq_name+savefname+".pkl"),"rb"))
            print("Cheops pre-modelled trace exists for filekey=",fk," at",self.unq_name+savefname+".pkl")
            return savefname[1:]
        
        with pm.Model() as self.ichlc_models[fk]:
            #Adding planet model info if there's any transit in the lightcurve
            if transittype!="none" and np.any(self.cheops_lc[(self.cheops_lc['filekey']==fk)&self.cheops_lc['in_trans_all']]):
                Rs = pm.Bound(pm.Normal, lower=0)("Rs",mu=self.Rstar[0], sd=self.Rstar[1])
                #Ms = pm.Bound(pm.Normal, lower=0)("Ms",mu=self.Mstar[0], sd=self.Mstar[1])
                u_star_cheops = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star_cheops", 
                                                mu=np.nanmedian(self.ld_dists['cheops'],axis=0),
                                                sd=np.clip(np.nanstd(self.ld_dists['cheops'],axis=0),0.1,1.0), 
                                                shape=2, testval=np.nanmedian(self.ld_dists['cheops'],axis=0))
                
                logrors={};t0s={};pers={};orbits={};bs={};tdurs={}
                pls=[]
                if np.any(self.cheops_lc[(self.cheops_lc['filekey']==fk)&self.cheops_lc['in_trans_all']]):
                    for pl in self.planets:
                        if np.any(self.cheops_lc[(self.cheops_lc['filekey']==fk)&self.cheops_lc['in_trans_'+pl]]):
                            pls+=[pl]
                            #If this timeseries specifically has this planet in, we need to fit for it
                            if transittype=="fix":
                                logrors[pl] = pm.Normal("logror_"+pl, mu=np.log(np.sqrt(self.planets[pl]['depth'])), sd=0.125, 
                                                        testval=np.log(np.sqrt(self.planets[pl]['depth'])))
                            elif transittype=="loose":
                                logrors[pl] = pm.Normal("logror_"+pl, mu=np.log(np.sqrt(self.planets[pl]['depth'])), sd=3, 
                                                        testval=np.log(np.sqrt(self.planets[pl]['depth'])))
                            #rpl = pm.Deterministic("rpl",109.1*tt.exp(logror)*Rs)
                            bs[pl] = xo.distributions.ImpactParameter("b_"+pl, ror=tt.exp(logrors[pl]),
                                                                      testval=np.clip(self.planets[pl]['b'],0.025,0.975))
                            tdurs[pl]=pm.Normal("tdur_"+pl, mu=self.planets[pl]['tdur'],sd=0.03,testval=self.planets[pl]['tdur'])
                            ntrans=np.round((np.nanmedian(x)-self.planets[pl]['tcen'])/self.planets[pl]['period'])
                            if (self.planets[pl]['tcen_err']+self.planets[pl]['period_err']*ntrans)>2/14: print("Ephemeris potentially lost. Error = ",self.planets[pl]['tcen_err']+self.planets[pl]['period_err']*ntrans,"days")
                            t0s[pl] = pm.Normal("t0_"+pl, mu=self.planets[pl]['tcen']+self.planets[pl]['period']*ntrans,
                                                sd=np.clip(self.planets[pl]['tcen_err']+self.planets[pl]['period_err']*ntrans,0.01,0.2),
                                                testval=self.planets[pl]['tcen']+self.planets[pl]['period']*ntrans)
                            pers[pl] = pm.Normal("per_"+pl, mu=self.planets[pl]['period'],sd=self.planets[pl]['period_err'],testval=self.planets[pl]['period'])
                    if len(pls)>0:
                        orbits={}
                        cheops_planets_x = {}
                        for pl in self.planets:
                            if pl in pls:
                                orbits[pl] = xo.orbits.KeplerianOrbit(r_star=Rs, period=pers[pl],
                                                                t0=t0s[pl],
                                                                duration=tdurs[pl],
                                                                b=bs[pl])#m_star=Ms, p
                                #else:
                                #    orbits[pl] = xo.orbits.KeplerianOrbit(r_star=Rs, period=pers[pls[0]], 
                                #                                    t0=t0s[pls[0]], 
                                #                                    duration=tdurs[pls[0]], 
                                #                                    b=bs[pls[0]])#m_star=Ms, p

                                if np.any(self.cheops_lc[(self.cheops_lc['filekey']==fk)&self.cheops_lc['in_trans_'+pl]]):
                                    cheops_planets_x[pl]=pm.Deterministic("cheops_planets_x_"+pl+"_"+fk, xo.LimbDarkLightCurve(u_star_cheops).get_light_curve(orbit=orbits[pl], 
                                                                                                                                        r=tt.exp(logrors[pl])*Rs,t=x)[:,0]*1000)
                                else:
                                    cheops_planets_x[pl]=tt.zeros(len(x))
                            else:
                                cheops_planets_x[pl]=tt.zeros(len(x))


            cheops_logs = pm.Normal("cheops_logs", mu=np.log(np.nanmedian(abs(np.diff(y))))-3, sd=3)

            #Initialising linear (and quadratic) parameters:
            linear_decorr_dict={};quad_decorr_dict={}
            
            for decorr_1 in self.init_che_linear_decorr_pars:
                if decorr_1=='time':
                    linear_decorr_dict[decorr_1]=pm.Normal("dfd"+decorr_1,mu=0,sd=np.ptp(self.norm_cheops_dat[fk][decorr_1])/self.cheops_mads[fk],testval=np.random.normal(0,0.05))
                else:
                    linear_decorr_dict[decorr_1]=pm.Normal("dfd"+decorr_1,mu=0,sd=self.cheops_mads[fk],testval=np.random.normal(0,0.05))
            for decorr_2 in self.init_che_quad_decorr_pars:
                quad_decorr_dict[decorr_2]=pm.Normal("d2fd"+decorr_2+"2",mu=0,sd=self.cheops_mads[fk],testval=np.random.normal(0,0.05))
            cheops_obs_mean = pm.Normal("cheops_mean",mu=0.0,sd=0.5*np.nanstd(y),testval=0.0)
            cheops_flux_cor = pm.Deterministic("cheops_flux_cor_"+fk,cheops_obs_mean + tt.sum([linear_decorr_dict[param]*self.norm_cheops_dat[fk][param] for param in self.init_che_linear_decorr_pars], axis=0) + \
                                                tt.sum([quad_decorr_dict[param]*self.norm_cheops_dat[fk][param]**2 for param in self.init_che_quad_decorr_pars], axis=0))
            
            #We have a transit, so we need the transit params:
            if transittype!="none" and np.any(self.cheops_lc[(self.cheops_lc['filekey']==fk)&self.cheops_lc['in_trans_all']]):
                #tt.printing.Print("cheops_planets_x")(cheops_planets_x)
                if len(pers)>0:
                    cheops_summodel_x = pm.Deterministic("cheops_summodel_x_"+fk, tt.sum([cheops_planets_x[pl] for pl in self.planets], axis=0) + cheops_flux_cor)
                elif len(pers)==1:
                    tt.printing.Print("cheops_flux_cor")(cheops_flux_cor)
                    #tt.printing.Print("cheops_flux_cor")(cheops_planets_x[list(self.planets.keys())[0]])
                    #print(cheops_planets_x[list(self.planets.keys())[0]].shape,)
                    cheops_summodel_x = pm.Deterministic("cheops_summodel_x_"+fk, cheops_planets_x[list(self.planets.keys())[0]] + cheops_flux_cor)

                else:
                    cheops_summodel_x = pm.Deterministic("cheops_summodel_x_"+fk, cheops_flux_cor)
            else:
                cheops_summodel_x = pm.Deterministic("cheops_summodel_x_"+fk, cheops_flux_cor)
            llk_cheops = pm.Normal("llk_cheops", mu=cheops_summodel_x, sd=tt.sqrt(yerr ** 2 + tt.exp(cheops_logs)**2), observed=y)
            pm.Deterministic("out_llk_cheops",llk_cheops)
            
            #print(self.ichlc_models[fk].check_test_point())
            #Minimizing:
            if transittype!="none" and np.any(self.cheops_lc[(self.cheops_lc['filekey']==fk)&self.cheops_lc['in_trans_all']]):
                comb_soln = pmx.optimize(vars=[Rs,u_star_cheops]+[t0s[pl] for pl in t0s]+[pers[pl] for pl in t0s]+[logrors[pl] for pl in t0s]+[bs[pl] for pl in bs]+[cheops_logs])
                comb_soln = pmx.optimize(start=comb_soln,
                                            vars=[linear_decorr_dict[par] for par in linear_decorr_dict] + \
                                            [cheops_obs_mean,cheops_logs] + \
                                            [quad_decorr_dict[par] for par in quad_decorr_dict])
            else:
                comb_soln = pmx.optimize(vars=[linear_decorr_dict[par] for par in linear_decorr_dict] + \
                                            [cheops_obs_mean,cheops_logs] + \
                                            [quad_decorr_dict[par] for par in quad_decorr_dict])
            comb_soln = pmx.optimize(start=comb_soln)
            self.cheops_init_trace[savefname[1:]]= pmx.sample(tune=300, draws=400, chains=3, cores=3, start=comb_soln)

            pickle.dump(self.cheops_init_trace[savefname[1:]],open(os.path.join(self.save_file_loc,self.name,self.unq_name+savefname+".pkl"),"wb"))
        return savefname[1:]

    def init_cheops(self, force_no_dydt=False, make_detren_params_global=True, force_detrend_pars={}, **kwargs):
        """Initialising the Cheops data.
        This includes running an initial PyMC model on the Cheops data alone to see which detrending parameters to use.

        Args:
            force_no_dydt (bool, optional):  Force the timeseries to be quasi-flat in the case of not introducing an artificial transit (Default is False)
            make_detren_params_global (bool, optional): Whether to globally use the same detrending parameters across all visits.
        Optional Args:
            use_signif (bool, optional):     Determine the detrending factors to use by simply selecting those with significant non-zero coefficients. Defaults to False
            use_bayes_fact (bool, optional): Determine the detrending factors to use with a Bayes Factor (Default is True)
            signif_thresh (float, optional): #Threshold for detrending parameters in sigma (default: 1.25)
            force_detrend_pars (dict, optional): Which parameters to force into the model regardless of bayes_fact/signif (dict in form {'lin':[], 'quad':[]})
        """
        
        self.update(**kwargs) #Updating default settings given kwargs

        #Stolen from pycheops:
        # B = np.exp(-0.5*((v-dfd_priorvalue)/dfd_fitvalue)**2) * dfd_priorsd/dfd_fitsd
        # If B>1, the detrending param is not useful...
        assert hasattr(self,"cheops_lc"), "Must have initialised Cheops LC using `model.add_cheops_lc`"
        assert hasattr(self,"Rstar"), "Must have initialised stellar parameters using `model.init_starpars`"
        assert self.use_signif^self.use_bayes_fact, "Must either use the significant detrending params or use the bayes factors, not both."

        #Initialising Cheops LD dists:
        if not hasattr(self,'ld_dists'):
            self.ld_dists={}
        self.ld_dists['cheops']=get_lds(1200,self.Teff[:2],self.logg[:2],how='cheops')

        #Checking which transits are in which dataset:
        for ipl,pl in enumerate(self.planets):
            self.cheops_lc['in_trans_'+pl]=abs((self.cheops_lc['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.mask_distance*self.planets[pl]['tdur']
        self.cheops_lc['in_trans_all']=np.any(self.cheops_lc.loc[:,['in_trans_'+pl for pl in self.planets]].values,axis=1)
        
        #print(self.cheops_lc['in_trans_all'].values)

        #Generating timeseries which will "fill-in the gaps" when modelling (e.g. for plotting)
        self.cheops_cad = np.nanmedian(np.diff(np.sort(self.cheops_lc['time'].values)))
        self.cheops_gap_timeseries = []
        self.cheops_gap_fks = []
        for fk in self.cheops_filekeys:
            mint=np.min(self.cheops_lc.loc[self.cheops_lc['filekey']==fk,'time'].values)
            maxt=np.max(self.cheops_lc.loc[self.cheops_lc['filekey']==fk,'time'].values)
            ix_gaps=np.min(abs(np.arange(mint,maxt,self.cheops_cad)[:,None]-self.cheops_lc.loc[self.cheops_lc['filekey']==fk,'time'].values[None,:]),axis=1)>0.66*self.cheops_cad
            self.cheops_gap_timeseries+=[np.arange(mint,maxt,self.cheops_cad)[ix_gaps]]
            self.cheops_gap_fks+=[np.tile(fk, np.sum(ix_gaps))]
        self.cheops_gap_timeseries=np.hstack(self.cheops_gap_timeseries)
        self.cheops_gap_fks=np.hstack(self.cheops_gap_fks)
        
        #Making index arrays to allow us to sort/unsort by phi:
        self.cheops_lc['mask_allphi_sorting']=np.tile(-1,len(self.cheops_lc))
        self.cheops_lc['mask_alltime_sorting']=np.tile(-1,len(self.cheops_lc))
        self.cheops_lc.loc[self.cheops_lc['mask'],'mask_allphi_sorting']=np.argsort(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'].values).astype(int)
        self.cheops_lc.loc[self.cheops_lc['mask'],'mask_alltime_sorting']=np.argsort(self.cheops_lc.loc[self.cheops_lc['mask'],'mask_allphi_sorting'].values).astype(int)

        #Making rollangle bin indexes:
        phibins=np.arange(np.nanmin(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'])-1.25,np.nanmax(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'])+1.25,2.5)
        self.cheops_lc['phi_digi']=np.tile(-1,len(self.cheops_lc))
        self.cheops_lc.loc[self.cheops_lc['mask'],'phi_digi']=np.digitize(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'],phibins)

        #Initialising some stuff:
        self.cheops_filekeys = pd.unique(self.cheops_lc['filekey'])
        self.ichlc_models={}
        self.cheops_mads={}
        self.norm_cheops_dat={fk:{} for fk in list(self.cheops_filekeys)+['all']}
        self.init_chefit_summaries={fk:{} for fk in self.cheops_filekeys}
        self.linear_assess={fk:{} for fk in self.cheops_filekeys}
        self.quad_assess={fk:{} for fk in self.cheops_filekeys}
        force_detrend_pars={'lin':[],'quad':[]} if force_detrend_pars is None else force_detrend_pars

        #Looping over all Cheops datasets and building individual models which we can then extract stats for each detrending parameter
        for fk in self.cheops_filekeys:
            print("Performing Cheops-only minimisation with all detrending params for filekey ",fk)
            #Launching a PyMC3 model
            tracename = self.cheops_only_model(fk, include_transit=True, force_no_dydt=force_no_dydt,**kwargs)

            var_names=[var for var in self.cheops_init_trace[tracename].varnames if '__' not in var and np.product(self.cheops_init_trace[tracename][var].shape)<6*np.product(self.cheops_init_trace[tracename]['cheops_logs'].shape)]
            self.init_chefit_summaries[fk]=pm.summary(self.cheops_init_trace[tracename],var_names=var_names,round_to=7)

            for par in self.init_che_linear_decorr_pars:
                dfd_fitvalue=self.init_chefit_summaries[fk].loc["dfd"+par,'mean']
                dfd_fitsd=self.init_chefit_summaries[fk].loc["dfd"+par,'sd']
                dfd_priorsd=1
                if self.use_bayes_fact:
                    self.linear_assess[fk][par] = np.exp(-0.5*((dfd_fitvalue)/dfd_fitsd)**2) * dfd_priorsd/dfd_fitsd
                elif self.use_signif:
                    self.linear_assess[fk][par] = abs(dfd_fitvalue)/dfd_fitsd
            for par in self.init_che_quad_decorr_pars:
                dfd_fitvalue=self.init_chefit_summaries[fk].loc["d2fd"+par+"2",'mean']
                dfd_fitsd=self.init_chefit_summaries[fk].loc["d2fd"+par+"2",'sd']
                dfd_priorsd=0.5
                if self.use_bayes_fact:
                    self.quad_assess[fk][par] = np.exp(-0.5*((dfd_fitvalue)/dfd_fitsd)**2) * dfd_priorsd/dfd_fitsd
                elif self.use_signif:
                    self.quad_assess[fk][par] = abs(dfd_fitvalue)/dfd_fitsd
        
        #Assessing which bayes factors suggest detrending is useful:
        self.cheops_linear_decorrs={}
        self.cheops_quad_decorrs={}
        for fk in self.cheops_filekeys:
            fk_bool=np.array([int(i==fk) for i in self.cheops_filekeys])
            if self.use_bayes_fact:
                #Bayes factor is <1sigma = significant trend = use this in the decorrelation
                self.cheops_linear_decorrs.update({"dfd"+par+"_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_che_linear_decorr_pars if self.linear_assess[fk][par]<1})
                self.cheops_quad_decorrs.update({"d2fd"+par+"2_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_che_quad_decorr_pars if self.quad_assess[fk][par]<1})
            elif self.use_signif:
                #detrend mean is >1sigma = significant trend = use this in the decorrelation
                self.cheops_linear_decorrs.update({"dfd"+par+"_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_che_linear_decorr_pars if self.linear_assess[fk][par]>self.signif_thresh or par in force_detrend_pars['lin']})
                self.cheops_quad_decorrs.update({"d2fd"+par+"2_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_che_quad_decorr_pars if self.quad_assess[fk][par]>self.signif_thresh or par in force_detrend_pars['quad']})
            
                
                self.cheops_linear_decorrs.update({"dfd"+par+"_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_che_linear_decorr_pars if self.linear_assess[fk][par]>self.signif_thresh})
                force_detrend_pars['quad']=self.cheops_quad_decorrs.update({"d2fd"+par+"2_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_che_quad_decorr_pars if self.quad_assess[fk][par]>self.signif_thresh})

        if len(self.cheops_filekeys)>1 and make_detren_params_global:
            #Assessing which detrending parameters we can combine to a global parameter
            all_lin_params=np.unique([self.cheops_linear_decorrs[varname][0] for varname in self.cheops_linear_decorrs if self.cheops_linear_decorrs[varname][0]!='time'])
            all_quad_params=np.unique([self.cheops_quad_decorrs[varname][0] for varname in self.cheops_quad_decorrs if self.cheops_quad_decorrs[varname][0]!='time'])
            #print(all_lin_params)
            #print(all_quad_params)
            for linpar in all_lin_params:
                key='che_only_fit_'+fk+'_trace_fixtrans'+['','_notrend'][int(force_no_dydt)]
                vals=np.column_stack([self.cheops_init_trace[key]["dfd"+linpar] for fk in self.cheops_filekeys])
                dists=[]
                # Let's make a comparison between each val/err and the combined other val/err params.
                # Anomalies will be >x sigma seperate from the group mean, while others will be OK 
                for i in range(len(self.cheops_filekeys)):
                    not_i=np.array([i2!=i for i2 in range(len(self.cheops_filekeys))])
                    #print(linpar, i, not_i, vals.shape, not_i.shape)
                    dists+=[abs(np.nanmedian(vals[:,i])-np.nanmedian(vals[:,not_i]))/np.sqrt(np.nanstd(vals[:,i])**2+np.nanstd(vals[:,not_i])**2)]
                if np.sum(np.array(dists)<2)>1:
                    #Removing the inidividual correlation filekeys from the cheops_linear_decorrs list:
                    #print(self.cheops_filekeys[np.array(dists)<2])
                    for fk in self.cheops_filekeys[np.array(dists)<2]:
                        fk_bool=np.array([int(i==fk) for i in self.cheops_filekeys])
                        varname="dfd"+linpar+"_"+"".join(list(fk_bool.astype(str)))
                        #print(varname,self.cheops_linear_decorrs.keys())
                        if varname in self.cheops_linear_decorrs:
                            _=self.cheops_linear_decorrs.pop(varname)
                    #Replacing them with a combined cheops_linear_decorrs:
                    fk_bool=np.array([int(fk in self.cheops_filekeys[np.array(dists)<2]) for fk in self.cheops_filekeys])
                    varname="dfd"+linpar+"_"+"".join(list(fk_bool.astype(str)))
                    self.cheops_linear_decorrs[varname]=[linpar,list(self.cheops_filekeys[np.array(dists)<2])]
                    if linpar[:3]=="sin":
                        combdat=np.hstack((np.sin(float(int(linpar[3]))*self.cheops_lc.loc[self.cheops_fk_mask[fk],linpar[-3:]].values*np.pi/180) for fk in self.cheops_filekeys[np.array(dists)<2]))
                    elif linpar[:3]=="cos":
                        combdat=np.hstack((np.cos(float(int(linpar[3]))*self.cheops_lc.loc[self.cheops_fk_mask[fk],linpar[-3:]].values*np.pi/180) for fk in self.cheops_filekeys[np.array(dists)<2]))
                    else:
                        combdat=np.hstack((self.cheops_lc.loc[self.cheops_fk_mask[fk],linpar].values for fk in self.cheops_filekeys[np.array(dists)<2]))
                    self.norm_cheops_dat['all'][linpar]=(combdat - np.nanmedian(combdat))/np.nanstd(combdat)

            for quadpar in all_quad_params:
                key='che_only_fit_'+fk+'_trace_fixtrans'+['','_notrend'][int(force_no_dydt)]
                vals=np.column_stack([self.cheops_init_trace[key]["d2fd"+quadpar+"2"] for fk in self.cheops_filekeys])
                dists=[]
                for i in range(len(self.cheops_filekeys)):
                    not_i=[i2!=i for i2 in range(len(self.cheops_filekeys))]
                    dists+=[abs(np.nanmedian(vals[:,i])-np.nanmedian(vals[:,not_i]))/np.sqrt(np.nanstd(vals[:,i])**2+np.nanstd(vals[:,not_i])**2)]
                if np.sum(np.array(dists)<2)>1:
                    #Removing the inidividual correlation filekeys from the cheops_linear_decorrs list:
                    for fk in self.cheops_filekeys[np.array(dists)<2]:
                        fk_bool=np.array([int(i==fk) for i in self.cheops_filekeys])
                        varname="d2fd"+quadpar+"2_"+"".join(list(fk_bool.astype(str)))
                        if varname in self.cheops_quad_decorrs:
                            _=self.cheops_quad_decorrs.pop(varname)
                    #Replacing them with a combined cheops_linear_decorrs:
                    fk_bool=np.array([int(fk in self.cheops_filekeys[np.array(dists)<2]) for fk in self.cheops_filekeys])
                    varname="d2fd"+quadpar+"2_"+"".join(list(fk_bool.astype(str)))
                    self.cheops_quad_decorrs[varname]=[quadpar,list(self.cheops_filekeys[np.array(dists)<2])]
                    if quadpar[:3]=="cos":
                        combdat=np.hstack((np.sin(float(int(quadpar[3]))*self.cheops_lc.loc[self.cheops_fk_mask[fk],quadpar].values*np.pi/180) for fk in self.cheops_filekeys[np.array(dists)<2]))
                    elif quadpar[:3]=="sin":
                        combdat=np.hstack((np.cos(float(int(quadpar[3]))*self.cheops_lc.loc[self.cheops_fk_mask[fk],quadpar].values*np.pi/180) for fk in self.cheops_filekeys[np.array(dists)<2]))
                    else:
                        combdat=np.hstack((self.cheops_lc.loc[self.cheops_fk_mask[fk],quadpar].values for fk in self.cheops_filekeys[np.array(dists)<2]))
                    self.norm_cheops_dat['all'][quadpar]=(combdat - np.nanmedian(combdat))/np.nanstd(combdat)
        
        #Let's iron this out and get a dictionary for each filekey of which detrending parameters are used...
        self.fk_linvars={}
        self.fk_quadvars={}
        for fk in self.cheops_filekeys:
            self.fk_linvars[fk]=[var for var in self.cheops_linear_decorrs if fk in self.cheops_linear_decorrs[var][1]]
            self.fk_quadvars[fk]=[var for var in self.cheops_quad_decorrs if fk in self.cheops_quad_decorrs[var][1]]

        #Stolen from pycheops (TBD):
        # if (dfdsinphi != 0 or dfdsin2phi != 0 or dfdsin3phi != 0 or
        #     dfdcosphi != 0 or dfdcos2phi != 0 or dfdcos3phi != 0):
        #     sinphit = self.sinphi(t)
        #     cosphit = self.cosphi(t)
        #     trend += dfdsinphi*sinphit + dfdcosphi*cosphit
        #     if dfdsin2phi != 0:
        #         trend += dfdsin2phi*(2*sinphit*cosphit)
        #     if dfdcos2phi != 0:
        #         trend += dfdcos2phi*(2*cosphit**2 - 1)
        #     if dfdsin3phi != 0:
        #         trend += dfdsin3phi*(3*sinphit - 4* sinphit**3)
        #     if dfdcos3phi != 0:
        #         trend += dfdcos3phi*(4*cosphit**3 - 3*cosphit)

    def model_comparison_cheops(self,show_detrend=True,**kwargs):
        # For each filekey with a transiting planet, we can perform an equivalent fit with _No_ transit. 
        # This will then allow us to derive a Bayes Factor and assess whether the transit model is justified.
        self.model_comp={}
        self.comp_stats={}
        self.cheops_assess_statements={}
        for fk in self.cheops_filekeys:
            self.model_comp[fk]={}
            #Only doing this comparison on filekeys which have transits (according to prior ephemerides):
            if np.any(self.cheops_lc[(self.cheops_lc['filekey']==fk)&self.cheops_lc['in_trans_all']]):
                trace_w_trans_name = self.cheops_only_model(fk, transittype="loose", force_no_dydt=True, **kwargs)#, linpars=self.cheops_linear_decorrs[fk], quadpars=self.cheops_quad_decorrs[fk])
                #trace_w_trans['log_likelihood']=trace_w_trans.out_llk_cheops
                self.model_comp[fk]['tr_waic']  = pm.stats.waic(self.cheops_init_trace[trace_w_trans_name])
                #notrans_linpars=self.cheops_linear_decorrs[fk]+['time','deltaT'] if 'deltaT' in self.cheops_lc.columns else self.cheops_linear_decorrs[fk]+['time']
                #notrans_quadpars=self.cheops_quad_decorrs[fk]+['time','deltaT'] if 'deltaT' in self.cheops_lc.columns else self.cheops_quad_decorrs[fk]+['time']
                trace_no_trans_name = self.cheops_only_model(fk, transittype="none", force_no_dydt=False, **kwargs)#,linpars=notrans_linpars,quadpars=notrans_quadpars)
                #trace_no_trans['log_likelihood']=trace_no_trans.out_llk_cheops
                self.model_comp[fk]['notr_waic'] = pm.stats.waic(self.cheops_init_trace[trace_no_trans_name])
                self.model_comp[fk]['tr_loglik'] = np.max(self.cheops_init_trace[trace_w_trans_name].out_llk_cheops)
                self.model_comp[fk]['notr_loglik'] = np.max(self.cheops_init_trace[trace_no_trans_name].out_llk_cheops)
                self.model_comp[fk]['delta_loglik'] = (self.model_comp[fk]['tr_loglik'] - self.model_comp[fk]['notr_loglik'])
                
                self.model_comp[fk]['notr_BIC'] = self.model_comp[fk]['notr_waic']['p_waic'] * np.log(np.sum((self.cheops_lc['filekey']==fk)&self.cheops_lc['mask'])) - 2*np.log(np.max(self.cheops_init_trace[trace_no_trans_name].out_llk_cheops))
                self.model_comp[fk]['tr_BIC'] = self.model_comp[fk]['tr_waic']['p_waic'] * np.log(np.sum((self.cheops_lc['filekey']==fk)&self.cheops_lc['mask'])) - 2*np.log(np.max(self.cheops_init_trace[trace_w_trans_name].out_llk_cheops))
                self.model_comp[fk]['deltaBIC'] = self.model_comp[fk]['notr_BIC'] - self.model_comp[fk]['tr_BIC']
                self.model_comp[fk]['BIC_pref_model']="transit" if self.model_comp[fk]['deltaBIC']<0 else "no_transit"
                print(self.model_comp[fk]['notr_waic'].index)
                #print(self.model_comp[fk]['notr_waic'].keys())
                #print(self.model_comp[fk]['notr_waic']['elpd_waic'])
                #print(self.model_comp[fk]['notr_waic'].loc['elpd_waic'],self.model_comp[fk]['tr_waic'].loc['elpd_waic'],self.model_comp[fk]['notr_waic'].shape,self.model_comp[fk]['tr_waic'].shape)
                self.model_comp[fk]['deltaWAIC']=self.model_comp[fk]['tr_waic']['waic']-self.model_comp[fk]['notr_waic']['waic']
                waic_errs=np.sqrt(self.model_comp[fk]['tr_waic']['waic_se']**2+self.model_comp[fk]['notr_waic']['waic_se']**2)
                #self.model_comp[fk]['deltaWAIC']=waic_diffs.loc['waic','self']-waic_diffs.loc['waic','other']
                confidence = np.array(["strongly prefers no transit","weakly prefers no transit","weakly prefers transit","strongly prefers transit"])[np.searchsorted([-1*waic_errs,0,waic_errs],self.model_comp[fk]['deltaWAIC'])]
                self.model_comp[fk]['WAIC_pref_model']="transit" if self.model_comp[fk]['deltaWAIC']>0 else "no_transit"
                #confidence="No detection" if self.model_comp[fk]['deltaWAIC']<2 else "Moderate detection" if (self.model_comp[fk]['deltaWAIC']>=2)&(self.model_comp[fk]['deltaWAIC']<8) else "Strong detection"
                self.cheops_assess_statements[fk]=["For fk="+fk+" WAIC "+confidence+"; Delta WAIC ="+str(np.round(self.model_comp[fk]['deltaWAIC'],2)),"(BIC prefers"+self.model_comp[fk]['BIC_pref_model']+" with deltaBIC ="+str(np.round(self.model_comp[fk]['deltaBIC'],2))+"). "]
                print(self.cheops_assess_statements[fk])
                #print("BIC prefers",self.model_comp[fk]['BIC_pref_model'],"( Delta BIC =",np.round(self.model_comp[fk]['deltaBIC'],2),"). WAIC prefers",self.model_comp[fk]['WAIC_pref_model']," ( Delta WAIC =",np.round(self.model_comp[fk]['deltaWAIC'],2),")")
                
                for pl in self.planets:
                    if 'logror_'+pl in self.cheops_init_trace[trace_w_trans_name].varnames:
                        ror_info=[np.nanmedian(np.exp(self.cheops_init_trace[trace_w_trans_name]['logror_'+pl])),np.nanstd(np.exp(self.cheops_init_trace[trace_w_trans_name]['logror_'+pl]))]
                        sigdiff=abs(np.sqrt(self.planets[pl]['depth'])-ror_info[0])/ror_info[1]
                        pl_statement="For planet "+str(pl)+" the derived radius ratio is "+str(ror_info[0])[:7]+"±"+str(ror_info[1])[:7]+" which is "+str(sigdiff)[:4]+"-sigma from the expected value given TESS depth ("+str(np.sqrt(self.planets[pl]['depth']))[:7]+")"
                        print(pl_statement)
                        self.cheops_assess_statements[fk]+=[pl_statement]
                self.plot_cheops(tracename=trace_no_trans_name, show_detrend=show_detrend, fk=fk, **kwargs)
                self.plot_cheops(tracename=trace_w_trans_name, show_detrend=show_detrend, fk=fk, **kwargs)
            elif np.any(self.cheops_lc[(self.cheops_lc['filekey']==fk)&(~self.cheops_lc['in_trans_all'])]):
                print("No transit event during observation with fk ="+fk)
                self.cheops_assess_statements[fk]=["There appears to be no transit event during observation with fk ="+fk+" according to ephemeris."]

    def init_model(self, **kwargs):
        """Initialising full TESS+CHEOPS model

        """

        self.update(**kwargs)

        assert not self.use_multinest, "Multinest is not currently possible"
        assert not (self.fit_flat&self.fit_gp), "Cannot both flatten data and fit GP. Choose one"
        assert self.use_mstar^self.use_logg, "Must be either use_mstar or use_logg, not both/neither"
        assert not (self.fit_phi_spline&self.fit_phi_gp), "Cannot both fit spline and GP to phi model. Choose one"

        if self.fit_ttvs or self.split_periods is not None:
            self.init_transit_times={}
            self.init_transit_inds={}
            for pl in self.planets:
                #Figuring out how
                if len(self.cheops_filekeys)>0:
                    min_ntr=int(np.floor((np.min(np.hstack([self.lcs[scope]['time'] for scope in self.lcs]+[self.cheops_lc['time']]))-self.planets[pl]['tcen'])/self.planets[pl]['period']))
                    max_ntr=int(np.ceil((np.max(np.hstack([self.lcs[scope]['time'] for scope in self.lcs]+[self.cheops_lc['time']]))-self.planets[pl]['tcen'])/self.planets[pl]['period']))
                else:
                    min_ntr=int(np.floor((np.min(np.hstack([self.lcs[scope]['time'] for scope in self.lcs]))-self.planets[pl]['tcen'])/self.planets[pl]['period']))
                    max_ntr=int(np.ceil((np.max(np.hstack([self.lcs[scope]['time'] for scope in self.lcs]))-self.planets[pl]['tcen'])/self.planets[pl]['period']))
                print(min_ntr,max_ntr,pl)
                if 'tcens' not in self.planets[pl]:
                    tcens=self.planets[pl]['tcen']+np.arange(min_ntr,max_ntr)*self.planets[pl]['period']
                    if len(self.cheops_filekeys)>0:
                        ix=np.min(abs(tcens[:,None]-np.hstack([self.lcs[scope]['time'] for scope in self.lcs]+[self.cheops_lc['time']])[None,:]),axis=1)<self.planets[pl]['tdur']*0.5
                    else:
                        ix=np.min(abs(tcens[:,None]-np.hstack([self.lcs[scope]['time'] for scope in self.lcs])[None,:]),axis=1)<self.planets[pl]['tdur']*0.5
                    self.init_transit_times[pl]=tcens[ix]
                    self.init_transit_inds[pl]=np.arange(min_ntr,max_ntr)[ix]
                    self.planets[pl]['n_trans']=np.sum(ix)
                else:
                    self.init_transit_times[pl]=self.planets[pl]['tcens']
                    self.init_transit_inds[pl]=np.round((self.planets[pl]['tcens']-self.planets[pl]['tcen'])/self.planets[pl]['period']).astype(int)
                    self.planets[pl]['n_trans']=len(self.planets[pl]['tcens'])
                self.init_transit_inds[pl]-=np.min(self.init_transit_inds[pl])            
            
        self.model_params={}
        with pm.Model() as self.model:
            # -------------------------------------------
            #          Stellar parameters
            # -------------------------------------------
            self.model_params['Teff'] = pm.Bound(pm.Normal, lower=0)("Teff",mu=self.Teff[0], sd=self.Teff[1])
            self.model_params['Rs'] = pm.Bound(pm.Normal, lower=0)("Rs",mu=self.Rstar[0], sd=self.Rstar[1])
            if self.use_mstar:
                self.model_params['Ms'] = pm.Bound(pm.Normal, lower=0)("Ms",mu=self.Mstar[0], sd=self.Mstar[1]) #Ms and logg are interchangeably deterministic
                self.model_params['logg'] = pm.Deterministic("logg",tt.log10(self.model_params['Ms']/self.model_params['Rs']**2)+4.41) #Ms and logg are interchangeably deterministic
            elif self.use_logg:
                self.model_params['logg'] = pm.Normal("logg",mu=self.logg[0],sd=self.logg[1]) #Ms and logg are interchangeably deterministic
                self.model_params['Ms'] = pm.Deterministic("Ms",tt.power(10,self.model_params['logg']-4.41)*self.model_params['Rs']**2) #Ms and logg are interchangeably deterministic
            
            # -------------------------------------------
            #             Contamination
            # -------------------------------------------
            # Using the detected companion's I and V mags to constrain Cheops and TESS dilution:
            if self.fit_contam:
                self.model_params['deltaImag_contam'] = pm.Uniform("deltaImag_contam", upper=12, lower=2.5)
                self.model_params['tess_mult'] = pm.Deterministic("tess_mult",(1+tt.power(2.511,-1*self.model_params['deltaImag_contam']))) #Factor to multiply normalised lightcurve by
                if 'k2' in self.lcs or 'kepler' in self.lcs or len(self.cheops_filekeys)>0:
                    self.model_params['deltaVmag_contam'] = pm.Uniform("deltaVmag_contam", upper=12, lower=2.5)
                    self.model_params['cheops_mult'] = pm.Deterministic("cheops_mult",(1+tt.power(2.511,-1*self.model_params['deltaVmag_contam']))) #Factor to multiply normalised lightcurve by
            else:
                for scope in self.lcs:
                    self.model_params[scope+'_mult']=1.0
                if len(self.cheops_filekeys)>0:
                    self.model_params['cheops_mult']=1.0

            self.model_params['u_stars']={}
            for scope in self.ld_dists:
                if self.constrain_lds:
                    self.model_params['u_stars'][scope] = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star_"+scope, 
                                                                    mu=np.clip(np.nanmedian(self.ld_dists[scope],axis=0),0,1),
                                                                    sd=np.clip(np.nanstd(self.ld_dists[scope],axis=0),0.1,1.0), 
                                                                    shape=2, testval=np.clip(np.nanmedian(self.ld_dists[scope],axis=0),0,1))
                else:
                    self.model_params['u_stars'][scope] = xo.distributions.QuadLimbDark("u_star_"+scope, testval=np.array([0.3, 0.2]))
            # -------------------------------------------
            # Initialising parameter dicts for each planet
            # -------------------------------------------
            self.model_params['orbit']={}
            self.model_params['t0']={};self.model_params['P']={};self.model_params['vels']={};self.model_params['tdur']={}
            self.model_params['b']={};self.model_params['rpl']={};self.model_params['logror']={};self.model_params['ror']={}
            self.model_params['a_Rs']={};self.model_params['sma']={};self.model_params['S_in']={};self.model_params['Tsurf_p']={}
            min_ps={pl:self.planets[pl]['period']*(1-1.5*self.timing_sd_durs*self.planets[pl]['tdur']/(np.ptp(np.hstack([self.lc_fit[scope]['time'] for scope in self.lc_fit])))) for pl in self.planets}
            max_ps={pl:self.planets[pl]['period']*(1+1.5*self.timing_sd_durs*self.planets[pl]['tdur']/(np.ptp(np.hstack([self.lc_fit[scope]['time'] for scope in self.lc_fit])))) for pl in self.planets}
            print(min_ps,max_ps,[self.planets[pl]['period'] for pl in self.planets],np.ptp(np.hstack([self.lc_fit[scope]['time'] for scope in self.lc_fit])))
            if self.fit_ttvs or self.split_periods is not None:
                self.model_params['transit_times']={}
            if self.split_periods is not None:
                self.model_params['split_t0']={pl:{} for pl in self.planets}
                self.model_params['split_P']={pl:{} for pl in self.planets}

            if hasattr(self,'rvs'):
                self.model_params['logK']={};self.model_params['K']={};self.model_params['logMp']={};self.model_params['Mp']={};self.model_params['rho_p']={}
                self.model_params['vrad_x']={};self.model_params['vrad_t']={}
            if not self.assume_circ:
                self.model_params['ecc']={};self.model_params['omega']={}

            for pl in self.planets:
                # -------------------------------------------
                #                  Orbits
                # -------------------------------------------
                if not self.fit_ttvs or self.planets[pl]['n_trans']<=2:
                    self.model_params['t0'][pl] = pm.Normal("t0_"+pl, mu=self.planets[pl]['tcen'], sd=2*self.planets[pl]['tcen_err'],
                                                            testval=np.random.normal(self.planets[pl]['tcen'],1e-6))
                    self.model_params['P'][pl] = pm.Bound(pm.Normal, lower=min_ps[pl], upper=max_ps[pl])("P_"+pl,
                                mu=self.planets[pl]['period'],sd=np.clip(self.planets[pl]['period_err'],0,(max_ps[pl]-self.planets[pl]['period'])),
                                testval=np.random.normal(self.planets[pl]['period'],1e-6))
                else:
                    #Initialising transit times:
                    # self.model_params['transit_times'][pl]=pm.Uniform("transit_times_"+pl, 
                    #                                                     upper=self.init_transit_times[pl]+self.planets[pl]['tdur']*self.timing_sd_durs,
                    #                                                     lower=self.init_transit_times[pl]-self.planets[pl]['tdur']*self.timing_sd_durs,
                    #                                                     shape=len(self.init_transit_times[pl]), testval=self.init_transit_times[pl])
                    if self.fit_ttvs:
                        self.model_params['transit_times'][pl]=[]
                        for i in range(len(self.init_transit_times[pl])):
                            if self.ttv_prior.lower()=='uniform':
                                self.model_params['transit_times'][pl].append(pm.Uniform("transit_times_"+pl+"_"+str(i), 
                                                                                upper=self.init_transit_times[pl][i]+self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                                lower=self.init_transit_times[pl][i]-self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                                testval=self.init_transit_times[pl][i]))
                            elif self.ttv_prior.lower()=='normal':
                                self.model_params['transit_times'][pl].append(pm.Normal("transit_times_"+pl+"_"+str(i), 
                                                                                mu=self.init_transit_times[pl][i],sigma=self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                                testval=self.init_transit_times[pl][i]))
                            elif self.ttv_prior.lower()=='boundnormal':
                                self.model_params['transit_times'][pl].append(pm.Bound(pm.Normal, 
                                                                                       lower=self.init_transit_times[pl][i]-self.planets[pl]['tdur']*2*self.timing_sd_durs,
                                                                                       upper=self.init_transit_times[pl][i]+self.planets[pl]['tdur']*2*self.timing_sd_durs)("transit_times_"+pl+"_"+str(i), 
                                                                                                mu=self.init_transit_times[pl][i],
                                                                                                sigma=self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                                                testval=self.init_transit_times[pl][i]))
 

                    elif self.split_periods is not None:
                        #Splitting periods into smaller periods of 3 or 4 transits.
                        #split_periods must be {pl:[[0,1,2],[3,4,5],[6,7,8,9]]} <- i.e. transit indexes split into groups
                        assert np.all(np.hstack(self.split_periods[pl])==np.arange(len(self.planets[pl]['n_trans']))),"split_periods must be a dict where all transits are split into groups by index"
                        for ngroup,group in enumerate(self.split_periods[pl]):
                            self.model_params['split_t0'][pl][ngroup]=pm.Normal("split_t0_"+pl+"_"+str(ngroup), mu=self.init_transit_times[pl][split_periods[pl][0]], 
                                                                  sd=2*self.planets[pl]['tcen_err'])
                            self.model_params['split_P'][pl][ngroup]=pm.Bound(pm.Normal, lower=min_ps[pl], upper=max_ps[pl])("split_P_"+pl+"_"+str(ngroup),
                                                                        mu=self.planets[pl]['period'],
                                                                        sd=np.clip(self.planets[pl]['period_err'],0,(max_ps[pl]-self.planets[pl]['period'])))
                            for ni in group:
                                n_trans_diff=self.init_transit_inds[ni]-self.init_transit_inds[group[0]]
                                self.model_params['transit_times'][pl].append(pm.Deterministic("transit_times_"+pl+"_"+str(i),
                                                                                               self.model_params['split_t0'][pl][ngroup]+self.model_params['split_P'][pl][ngroup]*n_trans_diff))
                                

                    #self.model_params['transit_times'][pl].append(pm.Normal("transit_times_"+pl, mu=self.init_transit_times[pl], sd=self.planets[pl]['tdur']*self.timing_sd_durs,
                    #                                                        shape=len(self.init_transit_times[pl]), testval=self.init_transit_times[pl]))

                # Wide log-normal prior for semi-amplitude
                if hasattr(self,'rvs'):
                    if self.rv_mass_prior=='logK':
                        self.model_params['logK'] = pm.Normal("logK_"+pl, mu=np.log(np.tile(2,len(self.planets))), sd=np.tile(10,len(self.planets)), 
                                                        shape=len(self.planets), testval=np.tile(2.0,len(self.planets)))
                        self.model_params['K'] =pm.Deterministic("K_"+pl,tt.exp(self.model_params['logK']))

                    elif self.rv_mass_prior=='K':
                        self.model_params['K'] = pm.Normal("K_"+pl, mu=np.tile(2,len(self.planets)), sd=np.tile(1,len(self.planets)), 
                                                        shape=len(self.planets), testval=np.tile(2.0,len(self.planets)))
                        self.model_params['logK'] =pm.Deterministic("logK_"+pl,tt.log(self.model_params['K']))
                    elif self.rv_mass_prior=='popMp':
                        if len(self.planets)>1:
                            rads=np.array([109.2*self.planets[pl]['rprs']*self.Rstar[0] for pl in self.planets])
                            mu_mps = 5.75402469 - (rads<=12.2)*(rads>=1.58)*(4.67363091 -0.38348534*rads) - \
                                                                    (rads<1.58)*(5.81943841-3.81604756*np.log(rads))
                            sd_mps= (rads<=8)*(0.07904372*rads+0.24318296) + (rads>8)*(0-0.02313261*rads+1.06765343)
                            self.model_params['logMp'] = pm.Normal('logMp_'+pl,mu=mu_mps,sd=sd_mps)
                        else:
                            rad=109.2*self.planets[list(self.planets.keys())[0]]['rprs']*self.Rstar[0]
                            mu_mps = 5.75402469 - (rad<=12.2)*(rad>=1.58)*(4.67363091 -0.38348534*rad) - \
                                                                    (rad<1.58)*(5.81943841-3.81604756*np.log(rad))
                            sd_mps= (rad<=8)*(0.07904372*rad+0.24318296) + (rad>8)*(0-0.02313261*rad+1.06765343)
                            self.model_params['logMp'] = pm.Normal('logMp_'+pl,mu=mu_mps,sd=sd_mps)
                # Eccentricity & argument of periasteron
                if not self.assume_circ:
                    BoundedBeta = pm.Bound(pm.Beta, lower=1e-5, upper=1-1e-5)
                    self.model_params['ecc'][pl] = BoundedBeta("ecc_"+pl, alpha=0.867 ,beta=3.03, testval=0.05)
                    self.model_params['omega'][pl] = pmx.Angle("omega_"+pl)
                '''
                #This was to model a non-transiting companion:
                P_nontran = pm.Normal("P_nontran", mu=27.386209624, sd=2*0.04947295)
                logK_nontran = pm.Normal("logK_nontran", mu=2,sd=10, testval=2)
                Mpsini_nontran = pm.Deterministic("Mp_nontran", tt.exp(logK_nontran) * 28.439**-1 * Ms**(2/3) * (P_nontran/365.25)**(1/3) * 317.8)
                t0_nontran = pm.Uniform("t0_nontran", lower=np.nanmedian(rv_x)-27.386209624*0.55, upper=np.nanmedian(rv_x)+27.386209624*0.55)
                '''
                self.model_params['logror'][pl] = pm.Uniform("logror_"+pl, lower=np.log(0.001), upper=np.log(0.1), 
                                            testval=np.log(np.sqrt(self.planets[pl]['depth'])))
                self.model_params['ror'][pl] = pm.Deterministic("ror_"+pl,tt.exp(self.model_params['logror'][pl]))
                self.model_params['rpl'][pl] = pm.Deterministic("rpl_"+pl,109.1*self.model_params['ror'][pl]*self.model_params['Rs'])
                self.model_params['b'][pl] = xo.distributions.ImpactParameter("b_"+pl, ror=self.model_params['ror'][pl], testval=self.planets[pl]['b'])
                
                if (self.fit_ttvs or self.split_periods) and self.planets[pl]['n_trans']>2:
                    if self.assume_circ:
                        self.model_params['orbit'][pl] = xo.orbits.TTVOrbit(b=[self.model_params['b'][pl]], 
                                                        transit_times=[self.model_params['transit_times'][pl]], 
                                                        transit_inds=[self.init_transit_inds[pl]], 
                                                        r_star=self.model_params['Rs'], 
                                                        m_star=self.model_params['Ms'])
                    else:
                        self.model_params['orbit'][pl] = xo.orbits.TTVOrbit(b=[self.model_params['b'][pl]], 
                                                        transit_times=[self.model_params['transit_times'][pl]], 
                                                        transit_inds=[self.init_transit_inds[pl]], 
                                                        r_star=self.model_params['Rs'], 
                                                        m_star=self.model_params['Ms'], 
                                                        ecc=[self.model_params['ecc']], 
                                                        omega=[self.model_params['omega']])
                    self.model_params['t0'][pl] = pm.Deterministic("t0_"+pl, self.model_params['orbit'][pl].t0[0])
                    self.model_params['P'][pl] = pm.Deterministic("P_"+pl, self.model_params['orbit'][pl].period[0])
                else:
                    # Then we define the orbit
                    if self.assume_circ:
                        self.model_params['orbit'][pl] = xo.orbits.KeplerianOrbit(r_star=self.model_params['Rs'], m_star=self.model_params['Ms'], 
                                                        period=self.model_params['P'][pl], t0=self.model_params['t0'][pl], b=self.model_params['b'][pl])
                    else:
                        self.model_params['orbit'][pl] = xo.orbits.KeplerianOrbit(r_star=self.model_params['Rs'], m_star=self.model_params['Ms'], period=self.model_params['P'][pl], 
                                                        t0=self.model_params['t0'][pl], b=self.model_params['b'][pl], ecc=self.model_params['ecc'][pl], omega=self.model_params['omega'][pl])
                
                # -------------------------------------------
                #           Derived planet params:
                # -------------------------------------------
                if hasattr(self,'rvs'):
                    
                    if self.rv_mass_prior!='popMp':
                        if self.assume_circ:
                            self.model_params['Mp'][pl] = pm.Deterministic("Mp_"+pl, tt.exp(self.model_params['logK'][pl]) * 28.439**-1 * self.model_params['Ms'][pl]**(2/3) * (self.model_params['P'][pl]/365.25)**(1/3) * 317.8)
                        else:
                            self.model_params['Mp'][pl] = pm.Deterministic("Mp_"+pl, tt.exp(self.model_params['logK'][pl]) * 28.439**-1 * (1-self.model_params['ecc'][pl]**2)**(0.5) * self.model_params['Ms']**(2/3) * (self.model_params['P'][pl]/365.25)**(1/3) * 317.8)
                    else:
                        self.model_params['Mp'][pl] = pm.Deterministic("Mp_"+pl, tt.exp(self.model_params['logMp'][pl]))
                        if self.assume_circ:
                            self.model_params['K'][pl] = pm.Deterministic("K_"+pl, self.model_params['Mp'][pl] / (28.439**-1 * self.model_params['Ms']**(2/3) * (self.model_params['P'][pl]/365.25)**(1/3) * 317.8))
                        else:
                            self.model_params['K'][pl] = pm.Deterministic("K_"+pl, self.model_params['Mp'][pl] / (28.439**-1 * (1-self.model_params['ecc'][pl]**2)**(0.5) * self.model_params['Ms']**(2/3) * (self.model_params['P'][pl]/365.25)**(1/3) * 317.8))
                        self.model_params['logK'][pl] = pm.Deterministic("logK_"+pl, tt.exp(self.model_params['K'][pl]))
                    self.model_params['rho_p'][pl] = pm.Deterministic("rho_p_gcm3_"+pl,5.513*self.model_params['Mp'][pl]/self.model_params['rpl'][pl]**3)
                
                self.model_params['a_Rs'][pl]=pm.Deterministic("a_Rs_"+pl,self.model_params['orbit'][pl].a)
                self.model_params['sma'][pl]=pm.Deterministic("sma_"+pl,self.model_params['a_Rs'][pl]*self.model_params['Rs']*0.00465)
                self.model_params['S_in'][pl]=pm.Deterministic("S_in_"+pl,((695700000*self.model_params['Rs'])**2.*5.67e-8*self.model_params['Teff']**4)/(1.496e11*self.model_params['sma'][pl])**2.)
                self.model_params['Tsurf_p'][pl]=pm.Deterministic("Tsurf_p_"+pl,(((695700000*self.model_params['Rs'])**2.*self.model_params['Teff']**4.*(1.-0.2))/(4*(1.496e11*self.model_params['sma'][pl])**2.))**(1./4.))
                
                #Getting the transit duration:
                self.model_params['vels'][pl] = self.model_params['orbit'][pl].get_relative_velocity(self.model_params['t0'][pl])
                self.model_params['tdur'][pl]=pm.Deterministic("tdur_"+pl,(2*self.model_params['Rs']*tt.sqrt((1+self.model_params['ror'][pl])**2-self.model_params['b'][pl]**2))/tt.sqrt(self.model_params['vels'][pl][0]**2 + self.model_params['vels'][pl][1]**2))

            # -------------------------------------------
            #                    RVs:
            # -------------------------------------------
            if hasattr(self,'rvs'):
                for pl in self.planets:
                    self.model_params['vrad_x'][pl]  = pm.Deterministic("vrad_x_"+pl,self.model_params['orbit'][pl].get_radial_velocity(self.rvs['time'], K=tt.exp(self.model_params['logK'][pl])).dimshuffle(0,'x'))
                    # Also define the model on a fine grid as computed above (for plotting)
                    self.model_params['vrad_t'][pl] = pm.Deterministic("vrad_t",self.model_params['orbit'][pl].get_radial_velocity(self.rv_t, K=tt.exp(self.model_params['logK'][pl])))

                '''orbit_nontran = xo.orbits.KeplerianOrbit(r_star=Rs, m_star=Ms, period=P_nontran, t0=t0_nontran)
                vrad_x = pm.Deterministic("vrad_x",tt.stack([orbit.get_radial_velocity(rv_x, K=tt.exp(logK))[:,0],
                                                                orbit.get_radial_velocity(rv_x, K=tt.exp(logK))[:,1],
                                                                orbit_nontran.get_radial_velocity(rv_x, K=tt.exp(logK_nontran))],axis=1))
                '''

                # Define the background model
                self.model_params['rv_offsets'] = pm.Normal("rv_offsets",
                                        mu=np.array([self.rv_medians[i] for i in self.rv_medians]),
                                        sd=np.array([self.rv_stds[i] for i in self.rv_stds])*5,
                                        shape=len(self.rv_medians))
                #tt.printing.Print("offsets")(tt.sum(offsets*self.rv_instr_ix,axis=1))

                #Only doing npoly-1 coefficients (and adding a leading zero for the vander) as we have seperate telescope offsets.
                if self.npoly_rv>1:
                    self.model_params['rv_trend'] = pm.Normal("rv_trend", mu=0, sd=10.0 ** -np.arange(self.npoly_rv)[::-1], shape=self.npoly_rv)
                    self.model_params['bkg_x'] = pm.Deterministic("bkg_x", tt.sum(self.model_params['rv_offsets']*self.rv_instr_ix,axis=1) + tt.dot(np.vander(self.rvs['time'] - self.rv_x_ref, self.npoly_rv)[:,:-1], self.model_params['rv_trend'][:-1]))
                else:
                    self.model_params['bkg_x'] = pm.Deterministic("bkg_x", tt.sum(self.model_params['rv_offsets']*self.rv_instr_ix,axis=1))

                # Define the RVs at the observed times  
                if len(self.planets)>1:
                    self.model_params['rv_model_x'] = pm.Deterministic("rv_model_x", self.model_params['bkg_x'] + tt.sum([self.model_params['vrad_x'][pl] for pl in self.planets], axis=1))
                else:
                    self.model_params['rv_modelxt'] = pm.Deterministic("rv_model_x", self.model_params['bkg_x'] + self.model_params['vrad_x'][list(self.planets.keys())[0]])

                '''vrad_t = pm.Deterministic("vrad_t",tt.stack([orbit.get_radial_velocity(rv_t, K=tt.exp(logK))[:,0],
                                                                orbit.get_radial_velocity(rv_t, K=tt.exp(logK))[:,1],
                                                                orbit_nontran.get_radial_velocity(rv_t, K=tt.exp(logK_nontran))],axis=1))
                '''
                #orbit.get_radial_velocity(rv_t, K=tt.exp(logK)))
                if self.npoly_rv>1:
                    self.model_params['bkg_t'] = pm.Deterministic("bkg_t", tt.dot(np.vander(self.rv_t - self.rv_x_ref, self.npoly_rv),self.model_params['rv_trend']))
                    if len(self.planets)>1:
                        self.model_params['rv_model_t'] = pm.Deterministic("rv_model_t", self.model_params['bkg_t'] + tt.sum([self.model_params['vrad_t'][pl] for pl in self.planet], axis=1))
                    else:
                        self.model_params['rv_model_t'] = pm.Deterministic("rv_model_t", self.model_params['bkg_t'] + self.model_params['vrad_t'][list(self.planets.keys())[0]])

                else:
                    if len(self.planets)>1:
                        self.model_params['rv_model_t'] = pm.Deterministic("rv_model_t", tt.sum([self.model_params['vrad_t'][pl] for pl in self.planet], axis=1))
                    else:
                        self.model_params['rv_model_t'] = pm.Deterministic("rv_model_t", self.model_params['vrad_t'][list(self.planets.keys())[0]])

            # -------------------------------------------
            #                 PHOT GP:
            # -------------------------------------------
            if self.fit_gp:
                minmax={}
                # Here we interpolate the histograms of the pre-trained GP samples as the input prior for each:
                for scope in self.lcs:
                    minmax[scope+'_logs']=np.percentile(self.oot_gp_trace[scope+"_logs"],[0.5,99.5])
                    self.model_params[scope+'_logs']=pm.Interpolated(scope+'_logs',x_points=np.linspace(minmax[scope+'_logs'][0],minmax[scope+'_logs'][1],201)[1::2],
                                            pdf_points=np.histogram(self.oot_gp_trace[scope+'_logs'],np.linspace(minmax[scope+'_logs'][0],minmax[scope+'_logs'][1],101))[0]
                                            )    
                minmax['S0']=np.percentile(self.oot_gp_trace["S0"],[0.5,99.5])
                self.model_params['phot_S0']=pm.Interpolated("phot_S0",x_points=np.linspace(minmax['S0'][0],minmax['S0'][1],201)[1::2],
                                        pdf_points=np.histogram(self.oot_gp_trace["S0"],np.linspace(minmax['S0'][0],minmax['S0'][1],101))[0]
                                        )
                minmax["w0"]=np.percentile(self.oot_gp_trace["w0"],[0.5,99.5])
                self.model_params['phot_w0']=pm.Interpolated("phot_w0",x_points=np.linspace(minmax["w0"][0],minmax["w0"][1],201)[1::2],
                                            pdf_points=np.histogram(self.oot_gp_trace["w0"],np.linspace(minmax["w0"][0],minmax["w0"][1],101))[0]
                                            )
                self.model_params['phot_kernel'] = theano_terms.SHOTerm(S0=self.model_params['phot_S0'], 
                                                                        w0=self.model_params['phot_w0'], Q=1/np.sqrt(2))#, mean = phot_mean)

                for scope in lcs:
                    minmax[scope+"_mean"]=np.percentile(self.oot_gp_trace[scope+"_mean"],[0.5,99.5])
                    self.model_params[scope+'_mean']=pm.Interpolated(scope+"_mean",
                                            x_points=np.linspace(minmax[scope+'_mean'][0],minmax[scope+'_mean'][1],201)[1::2],
                                            pdf_points=np.histogram(self.oot_gp_trace[scope+'_mean'],np.linspace(minmax[scope+'_mean'][0],minmax[scope+'_mean'][1],101))[0]
                                            )

                    self.model_params[scope+'_gp'] = celerite2.theano.GaussianProcess(self.model_params['phot_kernel'], self.lc_fit[scope]['time'].values, mean=self.model_params[scope+'_mean'],
                                                                                      diag=self.lc_fit[scope]['flux_err'].values ** 2 + tt.exp(self.model_params[scope+'_logs'])**2)
                #tt.dot(self.lc_fit_src_index,tt.exp([logs[scope] for scope in logs])
                #self.model_params['gp_tess'].compute(self.lc_fit['time'].values, , quiet=True)
            else:
                for scope in self.lcs:
                    self.model_params[scope+'_logs']=pm.Normal(scope+'_logs', mu=np.log(np.std(self.lc_fit[scope]['flux'].values)), sd=1)
            # -------------------------------------------
            #         Cheops detrending (linear)
            # -------------------------------------------
            if len(self.cheops_filekeys)>0:
                print("FKS=",self.cheops_filekeys)
                self.model_params['cheops_logs'] = pm.Normal("cheops_logs", mu=np.log(np.nanmedian(abs(np.diff(self.cheops_lc.loc[self.cheops_lc['mask'],'flux'].values)))), sd=3)

                #Initialising linear (and quadratic) parameters:
                self.model_params['linear_decorr_dict']={}#i:{} for i in self.cheops_filekeys}
                
                for decorr in self.cheops_linear_decorrs:
                    varname=self.cheops_linear_decorrs[decorr][0]
                    fks=self.cheops_linear_decorrs[decorr][1]
                    if varname=='time':
                        self.model_params['linear_decorr_dict'][decorr]=pm.Normal(decorr,mu=0,sd=np.nanmedian([np.ptp(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'])/self.cheops_mads[fk] for fk in fks]),
                                                                                    testval=np.nanmedian([self.init_chefit_summaries[fk].loc["dfd"+varname,'mean'] for fk in fks]))
                    else:
                        self.model_params['linear_decorr_dict'][decorr]=pm.Normal(decorr,mu=0,sd=np.nanmedian([self.cheops_mads[fk] for fk in fks]),
                                                                                    testval=np.nanmedian([self.init_chefit_summaries[fk].loc["dfd"+varname,'mean'] for fk in fks]))
                
                self.model_params['quad_decorr_dict']={}#{i:{} for i in self.cheops_filekeys}
                for decorr in self.cheops_quad_decorrs:
                    varname=self.cheops_quad_decorrs[decorr][0]
                    fks=self.cheops_quad_decorrs[decorr][1]
                    if varname=='time':
                        self.model_params['quad_decorr_dict'][decorr]=pm.Normal(decorr,mu=0,sd=np.nanmedian([np.ptp(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'])/self.cheops_mads[fk] for fk in fks]),
                                                                                testval=np.nanmedian([self.init_chefit_summaries[fk].loc["dfd"+varname,'mean'] for fk in fks]))
                    else:
                        self.model_params['quad_decorr_dict'][decorr]=pm.Normal(decorr,mu=0,sd=np.nanmedian([self.cheops_mads[fk] for fk in fks]),
                                                                                testval=np.nanmedian([self.init_chefit_summaries[fk].loc["dfd"+varname,'mean'] for fk in fks]))

                #Creating the flux correction vectors:
                self.model_params['cheops_obs_means']={};self.model_params['cheops_flux_cor']={}
                
                for fk in self.cheops_filekeys:
                    self.model_params['cheops_obs_means'][fk]=pm.Normal("cheops_mean_"+str(fk),mu=np.nanmedian(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values),
                                                                    sd=np.nanstd(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values),testval=0)
                    
                    if len(self.fk_quadvars[fk])>0:
                        #Linear and quadratic detrending
                        self.model_params['cheops_flux_cor'][fk] = pm.Deterministic("cheops_flux_cor_"+str(fk),self.model_params['cheops_obs_means'][fk] + \
                                                                                        tt.sum([self.model_params['linear_decorr_dict'][lvar]*self.norm_cheops_dat[fk][lvar.split('_')[0][3:]] for lvar in self.fk_linvars[fk]], axis=0) + \
                                                                                        tt.sum([self.model_params['quad_decorr_dict'][qvar]*self.norm_cheops_dat[fk][qvar.split('_')[0][4:-1]]**2 for qvar in self.fk_quadvars[fk]], axis=0))
                    elif len(self.fk_linvars[fk])>0:
                        #Linear detrending only
                        tt.printing.Print("obs_mean_"+fk)(self.model_params['cheops_obs_means'][fk])
                        self.model_params['cheops_flux_cor'][fk] = pm.Deterministic("cheops_flux_cor_"+str(fk),self.model_params['cheops_obs_means'][fk] + \
                                                                                        tt.sum([self.model_params['linear_decorr_dict'][lvar]*self.norm_cheops_dat[fk][lvar.split('_')[0][3:]] for lvar in self.fk_linvars[fk]], axis=0))
                    else:
                        #No detrending at all
                        self.model_params['cheops_flux_cor'][fk] = pm.Deterministic("cheops_flux_cor_"+str(fk), tt.tile(self.model_params['cheops_obs_means'][fk],np.sum(self.cheops_fk_mask[fk])))
                # -------------------------------------------
                #      Cheops detrending (roll angle GP)
                # -------------------------------------------
                self.model_params['cheops_summodel_x']={}
                self.model_params['llk_cheops']={}
                if self.fit_phi_gp:
                    self.model_params['rollangle_logpower'] = pm.Normal("rollangle_logpower",mu=-6,sd=1)

                    # self.model_params['rollangle_power'] = pm.InverseGamma("rollangle_power",testval=np.nanmedian(abs(np.diff(self.cheops_lc['flux']))), 
                    #                                   **pmx.estimate_inverse_gamma_parameters(
                    #                                             lower=0.2*np.sqrt(np.nanmedian(abs(np.diff(self.cheops_lc['flux'][self.cheops_lc['mask']])))),
                    #                                             upper=2.5*np.sqrt(np.nanstd(self.cheops_lc['flux'][self.cheops_lc['mask']]))))
                    #self.model_params['rollangle_loglengthscale'] = pm.InverseGamma("rollangle_loglengthscale", testval=np.log(50), 
                    #                                                        **pmx.estimate_inverse_gamma_parameters(lower=np.log(30), upper=np.log(110)))
                    self.model_params['rollangle_logw0'] = pm.Normal('rollangle_logw0',mu=np.log((2*np.pi)/100),sd=1)
                    #self.model_params['rollangle_w0'] = pm.InverseGamma("rollangle_w0", testval=(2*np.pi)/(lowerwl*1.25), **pmx.estimate_inverse_gamma_parameters(lower=(2*np.pi)/100,upper=(2*np.pi)/lowerwl))
                    self.model_params['rollangle_S0'] = pm.Deterministic("rollangle_S0", tt.exp(self.model_params['rollangle_logpower'])/(tt.exp(self.model_params['rollangle_logw0'])**4))
                    self.model_params['gp_rollangle_model_phi']={}
                    if not self.common_phi_model or len(self.cheops_filekeys)==1:
                        self.model_params['rollangle_kernels']={}
                        self.model_params['gp_rollangles']={}
                    else:
                        cheops_sigma2s={}
                elif self.fit_phi_spline:
                    from patsy import dmatrix
                    self.model_params['spline_model_phi']={}
                    if self.common_phi_model:
                        #Fit a single spline to all rollangle data
                        minmax=(np.min(self.cheops_lc.loc[self.cheops_lc['mask'],'phi']),np.max(self.cheops_lc.loc[self.cheops_lc['mask'],'phi']))
                        n_knots=int(np.round((minmax[1]-minmax[0])/self.spline_bkpt_cad))
                        knot_list = np.quantile(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'],np.linspace(0,1,n_knots))
                        B = dmatrix(
                            "bs(phi, knots=knots, degree="+str(int(self.spline_order))+", include_intercept=True) - 1",
                            {"phi": np.sort(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'].values), "knots": knot_list[1:-1]},
                        )

                        self.model_params['splines'] = pm.Normal("splines", mu=0, sd=np.nanmedian(abs(np.diff(self.cheops_lc.loc[self.cheops_lc['mask'],'flux']))), 
                                                                shape=B.shape[1],testval=np.random.normal(0.0,1e-4,B.shape[1]))
                        self.model_params['spline_model_allphi'] = pm.Deterministic("spline_model_allphi", tt.dot(np.asarray(B, order="F"), self.model_params['splines'].T))
                        fk_Bs={}
                        for fk in self.cheops_filekeys:
                            #print(np.sort(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values))
                            #fk_Bs[fk] = np.asarray(B, order="F")[self.cheops_lc.loc[self.cheops_lc['mask'],'filekey']==fk,:]
                            #self.model_params['spline_model_phi'][fk] = pm.Deterministic("spline_model_phi_"+fk, pm.math.dot(fk_Bs[fk], self.model_params['splines'].T))
                            #tt.printing.Print("dotprod phi spline model")(self.model_params['spline_model_phi'][fk])
                            #tt.printing.Print("indexed phi spline model (should be identical)")(self.model_params['spline_model_allphi'][np.array(self.cheops_lc.loc[self.cheops_lc['mask'],'filekey'].values[self.cheops_lc.loc[self.cheops_lc['mask'],'mask_allphi_sorting']]==fk)])
                            self.model_params['spline_model_phi'][fk] = pm.Deterministic("spline_model_phi_"+str(fk), 
                                                                                        self.model_params['spline_model_allphi'][np.array(self.cheops_lc.loc[self.cheops_lc['mask'],'filekey'].values[self.cheops_lc.loc[self.cheops_lc['mask'],'mask_allphi_sorting']]==fk)])
                    else:
                        #Fit splines to each rollangle
                        knot_list={}
                        B={}
                        self.model_params['splines']={}
                        for fk in self.cheops_filekeys:
                            minmax=(np.min(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi']),np.max(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi']))
                            n_knots=int(np.round((minmax[1]-minmax[0])/self.spline_bkpt_cad))
                            knot_list[fk]=np.quantile(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'],np.linspace(0,1,n_knots))
                            print(knot_list[fk],self.spline_order)
                            B[fk] = dmatrix(
                                "bs(phi, knots=knots, degree="+str(int(self.spline_order))+", include_intercept=True) - 1",
                                {"phi": np.sort(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values), "knots": knot_list[fk][1:-1]},
                            )
                            self.model_params['splines'][fk] = pm.Normal("splines_"+fk, mu=0, sd=np.nanmedian(abs(np.diff(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values))), shape=B[fk].shape[1])
                            self.model_params['spline_model_phi'][fk] = pm.Deterministic("spline_model_phi_"+fk, pm.math.dot(np.asarray(B[fk], order="F"), self.model_params['splines'][fk].T))
                    cheops_sigma2s={}
                else:
                    cheops_sigma2s={}
            
                #in the full model, we do a full cheops planet model for all filekeys simultaneously (unlike for the cheops_only_model)
                self.model_params['cheops_planets_x'] = {}
                self.model_params['cheops_planets_gaps'] = {}
                for pl in self.planets:
                    self.model_params['cheops_planets_x'][pl] = pm.Deterministic("cheops_planets_x_"+pl, xo.LimbDarkLightCurve(self.model_params['u_stars']["cheops"]).get_light_curve(orbit=self.model_params['orbit'][pl], r=self.model_params['rpl'][pl]/109.2,
                                                                                                        t=self.cheops_lc['time'].values.astype(np.float64))[:,0]*1000/self.model_params['cheops_mult'])
                    self.model_params['cheops_planets_gaps'][pl] = pm.Deterministic("cheops_planets_gaps_"+pl,xo.LimbDarkLightCurve(self.model_params['u_stars']["cheops"]).get_light_curve(orbit=self.model_params['orbit'][pl], r=self.model_params['rpl'][pl]/109.2,
                                                                                                        t=self.cheops_gap_timeseries.astype(np.float64))[:,0]*1000/self.model_params['cheops_mult'])
                if self.fit_phi_gp:
                    self.model_params['rollangle_kernels'] = theano_terms.SHOTerm(S0=self.model_params['rollangle_S0'], w0=tt.exp(self.model_params['rollangle_logw0']), Q=1/np.sqrt(2))#, mean = phot_mean)

                if self.fit_phi_gp and self.common_phi_model and len(self.cheops_filekeys)>1:
                    #Trying a new tack - binning to 2.5-degree bins.
                    #To do this we also need to hard-wire the indexes & average the fluxes
                    #print(np.nanmin(np.diff(np.sort(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'].values))))
                    self.cheops_binphi_2d_index=np.column_stack(([np.array(self.cheops_lc.loc[self.cheops_lc['mask'],'phi_digi']==n).astype(int)/np.sum(self.cheops_lc['phi_digi']==n) for n in np.unique(self.cheops_lc.loc[self.cheops_lc['mask'],'phi_digi'])]))
                    #plt.plot(np.sum(mod.cheops_lc.loc[mod.cheops_lc['mask'],'phi'][:,None]*cheops_binphi_2d_index,axis=0),
                    #        np.sum(mod.cheops_lc.loc[mod.cheops_lc['mask'],'flux'][:,None]*cheops_binphi_2d_index,axis=0),
                    tt.printing.Print("diag")((np.sum(self.cheops_lc.loc[self.cheops_lc['mask'],'flux_err'][:,None]*self.cheops_binphi_2d_index**1.5,axis=0))** 2 + \
                                                                                            (tt.sum(tt.exp(self.model_params['cheops_logs'])*self.cheops_binphi_2d_index**1.5,axis=0))**2)
                    print(np.sum(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'][:,None]*self.cheops_binphi_2d_index,axis=0))
                    print("flux",np.sum(self.cheops_lc.loc[self.cheops_lc['mask'],'flux'][:,None]*self.cheops_binphi_2d_index,axis=0))
                    self.model_params['gp_rollangles'] = celerite2.theano.GaussianProcess(self.model_params['rollangle_kernels'], 
                                                                                        np.sum(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'][:,None]*self.cheops_binphi_2d_index,axis=0), mean=0.0,
                                                                                        diag=(np.sum(self.cheops_lc.loc[self.cheops_lc['mask'],'flux_err'][:,None]*self.cheops_binphi_2d_index**1.5,axis=0))** 2 + \
                                                                                            (tt.sum(tt.exp(self.model_params['cheops_logs'])*self.cheops_binphi_2d_index**1.5,axis=0))**2)#Adding **1.5 as we want an extra 1/N**0.5 (instead of just 1/N in the average). 
                                                                                            #Also including these 1/N**1.5 terms in the jitter to ensure jitter is per-point
                    #self.model_params['gp_rollangles'].compute(np.sort(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'].values),
                    #                                           diag=..., quiet=True)
                elif self.fit_phi_gp:
                    for fk in self.cheops_filekeys:
                        # Roll angle vs flux GP                
                        self.model_params['gp_rollangles'][fk] = celerite2.theano.GaussianProcess(self.model_params['rollangle_kernels'], 
                                                                                                t=np.sort(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values), mean=0,
                                                                                                diag=(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux_err'].values[self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values]) ** 2 + \
                                                                                                tt.exp(self.model_params['cheops_logs'])**2)
                        #self.model_params['gp_rollangles'][fk].compute(np.sort(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values),
                        #                                               , quiet=True)
                
                for fk in self.cheops_filekeys:
                    #Adding the correlation model:
                    self.model_params['cheops_summodel_x'][fk] = pm.Deterministic("cheops_summodel_x_"+str(fk), tt.sum([self.model_params['cheops_planets_x'][pl][self.cheops_fk_mask[fk]] for pl in self.planets],axis=0) + self.model_params['cheops_flux_cor'][fk])
                    if self.fit_phi_gp and (not self.common_phi_model or len(self.cheops_filekeys)==1):
                            self.model_params['gp_rollangle_model_phi'][fk] = pm.Deterministic("gp_rollangle_model_phi_"+str(fk), 
                                                                self.model_params['gp_rollangles'][fk].predict((self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values[self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values] - \
                                                                                        self.model_params['cheops_summodel_x'][fk][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values]), 
                                                                                    t=np.sort(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values), return_var=False))
                if self.fit_phi_gp and self.common_phi_model and len(self.cheops_filekeys)>1:
                    all_summodels=tt.concatenate([self.model_params['cheops_summodel_x'][fk] for fk in pd.unique(self.cheops_lc['filekey'])])#,axis=0)
                    tt.printing.Print("flux - model")(tt.sum((self.cheops_lc.loc[self.cheops_lc['mask'],'flux'].values - \
                                                                                                all_summodels).dimshuffle(0,'x')*self.cheops_binphi_2d_index,axis=0))
                    self.model_params['gp_rollangle_model_allphi'] = pm.Deterministic("gp_rollangle_model_allphi",
                                                        self.model_params['gp_rollangles'].predict(y=tt.sum((self.cheops_lc.loc[self.cheops_lc['mask'],'flux'].values - \
                                                                                                all_summodels).dimshuffle(0,'x')*self.cheops_binphi_2d_index,axis=0), 
                                                                                                t=np.sort(self.cheops_lc.loc[self.cheops_lc['mask'],'phi'].values), return_var=False))
                    for fk in self.cheops_filekeys:
                        self.model_params['gp_rollangle_model_phi'][fk] = pm.Deterministic("gp_rollangle_model_phi_"+str(fk), 
                                                                                        self.model_params['gp_rollangle_model_allphi'][np.array(self.cheops_lc.loc[self.cheops_lc['mask'],'filekey'].values[self.cheops_lc.loc[self.cheops_lc['mask'],'mask_allphi_sorting']]==fk)])

                # -------------------------------------------
                #      Evaluating log likelihoods            
                # -------------------------------------------
                for fk in self.cheops_filekeys:
                    if self.fit_phi_gp and ((not self.common_phi_model) or len(self.cheops_filekeys)==1):
                        self.model_params['llk_cheops'][fk] = self.model_params['gp_rollangles'][fk].marginal("llk_cheops_"+str(fk), 
                                                                    observed = self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values - self.model_params['cheops_summodel_x'][fk])
                        #print("w rollangle GP",fk)
                        #tt.printing.Print("llk_cheops")(self.model_params['llk_cheops'][fk])
                    else:
                        cheops_sigma2s[fk] = self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux_err'].values ** 2 + tt.exp(self.model_params['cheops_logs'])**2
                        if self.fit_phi_spline:
                            self.model_params['llk_cheops'][fk] = pm.Normal("llk_cheops_"+fk, mu=self.model_params['cheops_summodel_x'][fk] + self.model_params['spline_model_phi'][fk][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting']], 
                                                                            sd=tt.sqrt(cheops_sigma2s[fk]), observed=self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values)
                        if self.fit_phi_gp and self.common_phi_model and len(self.cheops_filekeys)>1:
                            self.model_params['llk_cheops'][fk] = pm.Normal("llk_cheops_"+fk, mu=self.model_params['cheops_summodel_x'][fk] + self.model_params['gp_rollangle_model_phi'][fk][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting']], 
                                                                            sd=tt.sqrt(cheops_sigma2s[fk]), observed=self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values)
                        if not self.fit_phi_gp and not self.fit_phi_spline:
                            #In the case of the common roll angle on binned phi, we cannot use the gp marginal, so we do an "old fashioned" likelihood:
                            self.model_params['llk_cheops'][fk] = pm.Normal("llk_cheops_"+fk, mu=self.model_params['cheops_summodel_x'][fk], sd=tt.sqrt(cheops_sigma2s[fk]), 
                                                                            observed=self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values)
                    
                        #print("no rollangle GP",fk)
                        #tt.printing.Print("llk_cheops")(self.model_params['llk_cheops'][fk])
                        # self.model_params['llk_cheops'][fk] = pm.Potential("llk_cheops_"+str(fk), 
                        #                              -0.5 * tt.sum((self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values - \
                        #                                             self.model_params['cheops_summodel_x'][fk]) ** 2 / \
                        #                                            cheops_sigma2s[fk] + np.log(cheops_sigma2s[fk]))
                        #                            )
                #if self.fit_phi_gp and self.common_phi_model and len(self.cheops_filekeys)>1:
                #    self.model_params['llk_cheops'] = self.model_params['gp_rollangles'].marginal("llk_cheops",
                #                                                    observed = self.cheops_lc.loc[self.cheops_lc['mask'],'flux'].values - all_summodels)
            if not self.fit_gp:
                sigma2s={}
            for scope in self.lcs:
                self.model_params[scope+'_model_x']={}
                for pl in self.planets:
                    self.model_params[scope+'_model_x'][pl] = pm.Deterministic(scope+"_model_x_"+pl, xo.LimbDarkLightCurve(self.model_params['u_stars'][scope]).get_light_curve(orbit=self.model_params['orbit'][pl], r=self.model_params['rpl'][pl]/109.2,
                                                                                                                           t=self.lc_fit[scope]['time'].values)[:,0]*1000/self.model_params[scope+'_mult'])
                self.model_params[scope+'_summodel_x'] = pm.Deterministic(scope+"_summodel_x", tt.sum([self.model_params[scope+'_model_x'][pl] for pl in self.planets],axis=0))
                if self.fit_gp:
                    self.model_params[scope+'_llk'] = self.model_params[scope+'_gp'].marginal(scope+'_llk', observed=self.lc_fit[scope]['flux'].values - self.model_params[scope+'_summodel_x'])
                    self.model_params[scope+'_gp_model_x'] = pm.Deterministic(scope+"_gp_model_x", self.model_params[scope+'_gp'].predict(self.lc_fit[scope]['flux'].values - self.model_params[scope+'_summodel_x'], t=self.lc_fit[scope]['time'].values, return_var=False))
                else:
                    sigma2s[scope] = self.lc_fit[scope]['flux_err'].values ** 2 + tt.exp(self.model_params[scope+'_logs'])**2
                    self.model_params[scope+'_llk'] = pm.Potential(scope+'_llk', -0.5 * tt.sum((self.lc_fit[scope]['flux'].values - self.model_params[scope+'_summodel_x']) ** 2/sigma2s[scope] + np.log(sigma2s[scope])))
                    tt.printing.Print(scope+"_llk")(self.model_params[scope+'_llk'])
            if hasattr(self,"rvs"):
                rv_logjitter = pm.Normal("rv_logjitter",mu=np.nanmin(self.rvs['yerr'].values)-3,sd=3)
                rv_sigma2 = self.rvs['yerr'].values ** 2 + tt.exp(rv_logjitter)**2
                self.model_params['llk_rv'] = pm.Potential("llk_rv", -0.5 * tt.sum((self.rvs['y'].values - self.model_params['rv_model_x']) ** 2 / rv_sigma2 + np.log(rv_sigma2)))

            #print(self.model.check_test_point())

            #First try to find best-fit transit stuff:
            if not self.fit_ttvs:
                #print([self.model_params[par][pl] for pl in self.planets for par in ['logror','P','t0']]+[self.model_params[par] for par in ['logs_tess','cheops_logs']])
                #print(len([self.model_params[par][pl] for pl in self.planets for par in ['logror','P','t0']]+[self.model_params[par] for par in ['logs_tess','cheops_logs']]))
                if len(self.cheops_filekeys)>0:
                    comb_soln = pmx.optimize(vars=[self.model_params[par][pl] for pl in self.planets for par in ['logror','P','t0']]+[self.model_params[scope+"_logs"] for scope in list(self.lcs.keys())+['cheops']])
                else:
                    comb_soln = pmx.optimize(vars=[self.model_params[par][pl] for pl in self.planets for par in ['logror','P','t0']]+[self.model_params[scope+"_logs"] for scope in self.lcs])
            else:
                optvar=[]
                for pl in self.planets:
                    if pl in self.model_params['transit_times']:
                        optvar+=[self.model_params['transit_times'][pl][i] for i in range(len(self.model_params['transit_times'][pl]))]
                    else:
                        optvar+=[self.model_params['P'][pl],self.model_params['t0'][pl]]
                if len(self.cheops_filekeys)>0:
                    optvar+=[self.model_params['cheops_logs']]+[self.model_params['linear_decorr_dict'][par] for par in self.model_params['linear_decorr_dict']]
                comb_soln = pmx.optimize(vars = optvar+[self.model_params['logror'][pl] for pl in self.planets] + \
                                                [self.model_params[scope+'_logs'] for scope in self.lcs])

            #Now let's do decorrelation seperately:
            if len(self.cheops_filekeys)>0:
                decorrvars = [self.model_params['linear_decorr_dict'][par] for par in self.model_params['linear_decorr_dict']] + \
                            [self.model_params['cheops_obs_means'][fk] for fk in self.cheops_filekeys] + [self.model_params['cheops_logs']] + \
                            [self.model_params['quad_decorr_dict'][par] for par in self.model_params['quad_decorr_dict']]
                if self.fit_phi_gp:
                    #decorrvars+=[self.model_params['rollangle_loglengthscale'],self.model_params['rollangle_logpower']]
                    decorrvars+=[self.model_params['rollangle_logw0'],self.model_params['rollangle_logpower']]
                elif self.fit_phi_spline and self.common_phi_model:
                    decorrvars+=[self.model_params['splines']]
                elif self.fit_phi_spline and not self.common_phi_model:
                    decorrvars+=[self.model_params['splines'][fk] for fk in self.cheops_filekeys]
                comb_soln = pmx.optimize(start=comb_soln, vars=decorrvars)
            
            #More complex transit fit. Also RVs:
            ivars=[self.model_params['b'][pl] for pl in self.planets]+[self.model_params['logror'][pl] for pl in self.planets]+[self.model_params[scope+'_logs'] for scope in self.lcs]
            if len(self.cheops_filekeys)>0:
                ivars+=[self.model_params['cheops_logs']]
            ivars+=[self.model_params['u_stars'][u] for u in self.model_params['u_stars']]
            if self.fit_ttvs:
                for pl in self.planets:
                    if pl in self.model_params['transit_times']:
                        ivars+=[self.model_params['transit_times'][pl][i] for i in range(len(self.model_params['transit_times'][pl]))]
                    else:
                        ivars+=[self.model_params['P'][pl],self.model_params['t0'][pl]]
            elif self.split_periods is not None:
                for pl in self.planets:
                    if self.planets[pl]['n_trans']>2:
                        ivars+=[self.model_params['split_P'][pl][i] for i in range(len(self.split_periods[pl]))]
                        ivars+=[self.model_params['split_t0'][pl][i] for i in range(len(self.split_periods[pl]))]
                    else:
                        ivars+=[self.model_params['P'][pl],self.model_params['t0'][pl]]

            else:
                ivars+=[self.model_params['t0'][pl] for pl in self.planets]+[self.model_params['P'][pl] for pl in self.planets]
            if not self.assume_circ:
                ivars+=[self.model_params['ecc'][pl] for pl in self.planets]+[self.model_params['omega'][pl] for pl in self.planets]
            if hasattr(self,'rvs'):
                ivars+=[self.model_params['logK'][pl] for pl in self.planets]+[self.model_params['rv_offsets']]
                if self.npoly_rv>1:
                    ivars+=[self.model_params['rv_trend']]
            print(ivars)
            comb_soln = pmx.optimize(start=comb_soln, vars=ivars)

            #Doing everything:
            self.init_soln = pmx.optimize(start=comb_soln)
        
        if len(self.cheops_filekeys)>0 and self.fit_phi_gp:
            #Checking if the GP is useful in the model:
            self.check_rollangle_gp(**kwargs)
        elif len(self.cheops_filekeys)>0 and self.fit_phi_spline:
            self.check_rollangle_spline(**kwargs)
    
    def sample_model(self,n_tune_steps=1200,n_draws=998,n_cores=3,n_chains=2,cheops_groups="all",save_model=True,**kwargs):
        """Sample model

        Args:
            n_tune_steps (int, optional): Number of steps during tuning. Defaults to 1200.
            n_draws (int, optional): Number of model draws per chain. Defaults to 998.
            n_cores (int, optional): Number of cores. Defaults to 3.
            n_chains (int, optional): Number of chains per core. Defaults to 2.
            save_model (bool, optional): Whether to save the full model to disk. Defaults to True.
        """
        self.update(**kwargs)
        with self.model:
            #As we have detrending which is indepdent from each other, we can vastly improve sampling speed by splitting up these as `parameter_groups` in pmx.sample
            #`regularization_steps` also, apparently, helps for big models with lots of correlation
            
            #+[combined_model['d2fd'+par+'2_'+i+'_interval__'] for par in quad_decorr_dict[i]]
            groups=[]
            if len(self.cheops_filekeys)>0 and cheops_groups=="by fk":
                for fk in self.cheops_filekeys:
                    #Creating groups for each CHEOPS filekey with 
                    groups+=[[self.model_params['cheops_obs_means'][fk]]]
                    fkbool="".join(np.array(self.cheops_filekeys==fk).astype(int).astype(str))
                    for par in self.model_params['linear_decorr_dict']:
                        if par.split("_")[-1]==fkbool:
                            groups[-1]+=self.model_params['linear_decorr_dict'][par]
                    for par in self.model_params['quad_decorr_dict']:
                        if par.split("_")[-1]==fkbool:
                            groups[-1]+=self.model_params['linear_decorr_dict'][par]
                    #groups[-1]+=[self.model_params['dfd'+par+'_'+str(fk)] for par in self.model_params['linear_decorr_dict'][fk]]
                    if self.fit_phi_spline and not self.common_phi_model:
                        groups[-1]+=self.model_params["splines_"+fk]
            elif len(self.cheops_filekeys)>0 and cheops_groups=="all":
                #Putting all CHEOPS parameters into a single group
                groups+=[[self.model_params['cheops_obs_means'][fk] for fk in self.cheops_filekeys]+[self.model_params['linear_decorr_dict'][par] for par in self.model_params['linear_decorr_dict']]]
                if 'quad_decorr_dict' in self.model_params:
                    groups[-1]+=[self.model_params['quad_decorr_dict'][par] for par in self.model_params['quad_decorr_dict']]
                if self.fit_phi_spline and not self.common_phi_model:
                    groups[-1]+=[self.model_params["splines"][fk] for fk in self.cheops_filekeys]
                elif self.fit_phi_spline and self.common_phi_model:
                    groups[-1]+=[self.model_params["splines"]]
                elif self.fit_phi_gp:
                    groups[-1]+=[self.model_params["rollangle_logpower"],self.model_params["rollangle_logw0"]]
                groups[-1]+=[self.model_params["cheops_logs"]]
            if hasattr(self,'rvs'):
                rvgroup=[self.model_params['rv_offsets']]
                if self.rv_mass_prior=='popMp':
                    rvgroup+=[self.model_params['logMp']]
                elif self.rv_mass_prior=='logK':
                    rvgroup+=[self.model_params['logK']]
                elif self.rv_mass_prior=='K':
                    rvgroup+=[self.model_params['K']]
                if self.npoly_rv>1:
                    rvgroup+=[self.model_params['rv_trend']]
                groups+=[rvgroup]
            self.trace = pmx.sample(tune=n_tune_steps, draws=n_draws, 
                                    chains=int(n_chains*n_cores), cores=n_cores, 
                                    start=self.init_soln, target_accept=0.8,
                                    parameter_groups=groups,**kwargs)#**kwargs)
        self.save_trace_summary()
        if save_model:
            self.save_model_to_file()

    def save_trace_summary(self,returndf=True):
        var_names=[var for var in self.trace.varnames if 'gp_' not in var and 'model_' not in var and '__' not in var and (np.product(self.trace[var].shape)<6*np.product(self.trace['Rs'].shape) or 'transit_times' in var)]
        df=pm.summary(self.trace,var_names=var_names,round_to=8,
                      stat_funcs={"5%": lambda x: np.percentile(x, 5),"-$1\sigma$": lambda x: np.percentile(x, 15.87),
                                  "median": lambda x: np.percentile(x, 50),"+$1\sigma$": lambda x: np.percentile(x, 84.13),
                                  "95%": lambda x: np.percentile(x, 95)})
        df.to_csv(os.path.join(self.save_file_loc,self.name,self.unq_name+"_model_summary.csv"))
        if returndf:
            return df
    
    def make_cheops_obs_table(self, dur_unit="orbits"):
        """
        Make table of cheops observations. This table is used to calculate the time difference between observations and the time difference between them
        
        Args:
            dur_unit: duration unit ("orbits" or "hours")
        """

        
        # Headers = Date start, JD start, Duration [orbits], Filekey, cadence, Average efficiency, RMS [ppm], Planets present
        che_latex_tab=pd.DataFrame()
        starts=[]
        latex_tab="\\begin{table}\n\\centering\n\\begin{tabular}{lccccccc}\n"

        dur_mult = 1 if dur_unit=="orbits" else (98.77/60)
        fk_starts=[]
        for nfk,fk in enumerate(np.unique(self.cheops_filekeys)):
            fk_starts+=[np.nanmedian(self.cheops_lc.loc[self.cheops_lc['filekey']==fk,'time'].values)]
        
        for nfk,fk in enumerate(np.array(self.cheops_filekeys)[np.argsort(fk_starts)]):
            lcfk=self.cheops_lc.loc[self.cheops_lc['filekey']==fk]
            cad=np.nanmedian(np.diff(lcfk['time'].values))*86400
            dur=(lcfk['time'].values[-1]-lcfk['time'].values[0])/(98.9/1440)
            floored_orbs=np.floor(dur)*(98.77/1440)
            if hasattr(self,'trace'):
                if 'gp_rollangle_model_phi_'+str(fk) in self.trace.varnames:
                    fkmod=np.nanmedian(self.trace['cheops_summodel_x_'+str(fk)]+self.trace['gp_rollangle_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],axis=0)
                elif 'spline_model_phi_'+str(fk) in self.trace.varnames:
                    fkmod=np.nanmedian(self.trace['cheops_summodel_x_'+str(fk)]+self.trace['spline_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],axis=0)
                else:
                    fkmod=np.nanmedian(self.trace['cheops_summodel_x_'+str(fk)],axis=0)

            else:
                assert hasattr(self,'init_soln'), "must have run `init_model`"
                if 'gp_rollangle_model_phi_'+str(fk) in self.trace.varnames:
                    fkmod=self.init_soln['cheops_flux_cor_'+str(fk)]+np.sum(np.vstack([self.init_soln['cheops_planets_x_'+pl][self.cheops_lc['filekey']==fk] for pl in self.planets]),axis=1)+self.init_soln['gp_rollangle_model_phi_'+str(fk)][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)]
                elif 'spline_model_phi_'+str(fk) in self.trace.varnames:
                    fkmod=self.init_soln['cheops_flux_cor_'+str(fk)]+np.sum(np.vstack([self.init_soln['cheops_planets_x_'+pl][self.cheops_lc['filekey']==fk] for pl in self.planets]),axis=1)+self.init_soln['spline_model_phi_'+str(fk)][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)]
                else:
                    fkmod=self.init_soln['cheops_flux_cor_'+str(fk)]+np.sum(np.vstack([self.init_soln['cheops_planets_x_'+pl][self.cheops_lc['filekey']==fk] for pl in self.planets]),axis=1)
            starts+=[lcfk['time'].values[0]]
            effchecks=np.arange(0,dur-1,1/11)
            aveff=np.nanmedian([(np.sum((lcfk['time'].values>(lcfk['time'][0]+ec*0.0686))&(lcfk['time'].values<(lcfk['time'][0]+(ec+1)*0.0686)))*cad)/(98.77*60) for ec in effchecks])            
            print(fk,int(np.round(aveff*100)))
            info={"Date start":Time(lcfk['time'].values[0],format='jd').isot,
                  "BJD start":"$ "+str(lcfk['time'].values[0])+" $",
                  "Dur ["+dur_unit+"]":"$ "+str(np.round(dur*dur_mult,2))+" $",
                  "Filekey":fk,
                  "Cad. [s]":"$ "+str(np.round(cad,1))+" $",
                  "Av. eff. [%]":"$ "+str(int(aveff*100))+" $",
                  "RMS [ppm]":"$ "+str(int(np.round(np.std(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux']-fkmod)*1e6)))+" $",
                  "Planets":", ".join([pl for pl in self.planets if np.any(self.cheops_lc.loc[self.cheops_fk_mask[fk],"in_trans_"+pl])])}
            #Efficiency is number of actual observations (cutting any final residual orbit) / expected observations at 100% efficiency
            if nfk==0:
                latex_tab+=" & ".join(info.keys())+"\\\\\n"
            latex_tab+=" & ".join(info.values())+"\\\\\n"
        latex_tab+="\\end{tabular}\n\\caption{List of CHEOPS observations.}\\ref{tab:cheops_dat}\n\\end{table}"
        return latex_tab

    def make_lcs_timeseries(self, src, overwrite=False, **kwargs):
        """
        Pandas dataframe with:
         - tess_gpmodel_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - either fitted GP or pre-detrtended spline model
         - tess_[b]model_[p] (where p is +2sig, +1sig, med, -1sig, -2sig; and [b] is for each planet) - fitted planetary models
         - tess_allplmodel_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - combined planetary models
         """
        
        if not hasattr(self,'models_out'):
            self.models_out={}
         
        if src not in self.models_out or overwrite:
            self.models_out[src]=self.lcs[src].loc[self.lcs[src]['mask']]
            if self.fit_gp:
                if self.bin_oot:
                    #Need to interpolate to the smaller values
                    from scipy.interpolate import interp1d
                    if hasattr(self,'trace'):
                        for p in self.percentiles:
                            #print(np.min(self.lcs[src].iloc[0]['time'])-0.5,self.lc_fit['time'][0],self.lc_fit['time'][-1],np.max(self.lcs[src].iloc[-1]['time'])+0.5))
                            interpp=interp1d(np.hstack((np.min(self.lcs[src].iloc[0]['time'])-0.5,self.lc_fit[src]['time'].values,np.max(self.lcs[src].iloc[-1]['time'])+0.5)),
                                                np.hstack((0,np.percentile(self.trace[src+'_gp_model_x'],self.percentiles[p],axis=0),0)))
                            #print(np.min(self.lcs[src].loc[self.lcs[src]['mask'],'time'].values),np.max(self.lcs[src].loc[self.lcs[src]['mask'],'time'].values),self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[0],self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[-1])
                            self.models_out[src][src+"_gpmodel_"+p]=interpp(self.lcs[src].loc[self.lcs[src]['mask'],'time'].values)
                    elif hasattr(self,'init_soln'):
                        interpp=interp1d(np.hstack((np.min(self.lcs[src].iloc[0]['time'])-0.5,self.lc_fit[src]['time'].values,np.max(self.lcs[src].iloc[-1]['time'])+0.5)),
                                            np.hstack((0,self.init_soln[src+'_gp_model_x'],0)))
                        self.models_out[src][src+"_gpmodel_med"]=interpp(self.lcs[src].loc[self.lcs[src]['mask'],'time'].values)
                elif not self.cut_oot:
                    if hasattr(self,'trace'):
                        for p in self.percentiles:
                            self.models_out[src][src+"_gpmodel_"+p]=np.percentile(self.trace[src+'_gp_model_x'],self.percentiles[p],axis=0)
                    elif hasattr(self,'init_soln'):
                        self.models_out[src][src+"_gpmodel_med"]=self.init_soln[src+'_gp_model_x']
                elif self.cut_oot:
                    if hasattr(self,'trace'):
                        for p in self.percentiles:
                            self.models_out[src][src+"_gpmodel_"+p] = np.tile(np.nan,len(self.models_out[src]['time']))
                            self.models_out[src][src+"_gpmodel_"+p][self.lcs[src]['near_trans']&self.lcs[src]['mask']] = np.percentile(self.trace[src+'_gp_model_x'],self.percentiles[p],axis=0)
                    elif hasattr(self,'init_soln'):
                        p="med"
                        self.models_out[src][src+"_gpmodel_"+p] = np.tile(np.nan,len(self.models_out[src]['time']))
                        self.models_out[src][src+"_gpmodel_"+p][self.lcs[src]['near_trans']&self.lcs[src]['mask']] = self.init_soln[src+'_gp_model_x']
            else:
                self.models_out[src][src+"_gpmodel_med"] = self.models_out[src]["spline"].values[:]
            if hasattr(self,'trace'):
                for p in self.percentiles:
                    for pl in self.planets:
                        self.models_out[src][src+'_'+pl+"model_"+p]=np.zeros(np.sum(self.lcs[src]['mask']))
                        self.models_out[src].loc[self.lcs[src].loc[self.lcs[src]['mask'],'near_trans'],src+'_'+pl+"model_"+p]=np.nanpercentile(self.trace[src+'_model_x_'+pl][:,self.lc_fit[src]['near_trans']],self.percentiles[p],axis=0)
                    self.models_out[src][src+"_allplmodel_"+p]=np.zeros(np.sum(self.lcs[src]['mask']))
                    self.models_out[src].loc[self.lcs[src].loc[self.lcs[src]['mask'],'near_trans'],src+"_allplmodel_"+p]=np.nanpercentile(np.sum(np.dstack([self.trace[src+'_model_x_'+pl][:,self.lc_fit[src]['near_trans']] for pl in self.planets]),axis=2),self.percentiles[p],axis=0)
                    #self.models_out[src][src+"_allmodel_"+p]=self.models_out[src][src+"_gpmodel_"+p] if src+"_gpmodel_"+p in self.models_out[src] else self.models_out[src][src+"_gpmodel_med"]
                    #self.models_out[src][src+"_allmodel_"+p]+=self.models_out[src][src+"_allplmodel_"+p].values
            elif hasattr(self,'init_soln'):
                p="med"
                for pl in self.planets:
                    self.models_out[src][src+'_'+pl+"model_"+p]=np.zeros(np.sum(self.lcs[src]['mask']))
                    self.models_out[src].loc[self.lcs[src].loc[self.lcs[src]['mask'],'near_trans'],src+'_'+pl+"model_"+p]=self.init_soln[src+'_model_x_'+pl][self.lc_fit[src]['near_trans']]
                self.models_out[src][src+"_allplmodel_"+p]=np.sum(np.vstack([self.models_out[src][src+'_'+pl+"model_"+p] for pl in self.planets]),axis=0)
                #self.models_out[src].loc[self.lcs[src].loc[self.lcs[src]['mask'],'near_trans'],src+"_allplmodel_"+p]=np.sum(np.vstack([self.init_soln[src+'_model_x_'+pl][self.lc_fit[src]['near_trans']] for pl in self.planets]),axis=1)
                #self.models_out[src][src+"_allmodel_"+p]=self.models_out[src][src+"_gpmodel_"+p]+self.models_out[src][src+"_allplmodel_"+p]

    def make_cheops_timeseries(self, tracename=None, init_trace=None, fk=None, overwrite=False, **kwargs):
        """
        Pandas dataframe (self.models_out['cheops']) with:
         - che_gpmodel_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - Fitted roll angle GP model in time axis
         - che_pred_spline_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - Fitted roll angle spline model in time axis
         - che_lindetrend_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - Linear decorrelation model predictions
         - che_alldetrend_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - Combined spline/gp model with linear decorrelation
         - che_[b]model_[p] (where p is +2sig, +1sig, med, -1sig, -2sig; and [b] is for each planet) - fitted planetary models
         - che_allplmodel_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - combined planetary models
        The final two are also included in the "cheops_gap_models_out" Pandas dataframe
        """
        if not hasattr(self,'models_out'):
            self.models_out={}
        if tracename is None and 'cheops' not in self.models_out or overwrite:
            assert init_trace is None and fk is None, "We will use the default \'self.trace\' for the final CHEOPS model. For an intermediate trace, specify the trace type & filekey)"
            self.models_out['cheops']=pd.DataFrame()
            self.models_out['cheops_gap_models_out']=pd.DataFrame()

            for col in ['time','flux','flux_err','phi','bg','centroidx','centroidy','deltaT','xoff','yoff','filekey']:
                if col in self.cheops_lc.columns:
                    self.models_out['cheops'][col]=np.hstack([self.cheops_lc.loc[self.cheops_fk_mask[fk],col] for fk in self.cheops_filekeys])
            if self.fit_phi_gp:
                if hasattr(self,'trace'):
                    for p in self.percentiles:
                        self.models_out['cheops']['che_pred_gp_'+p]=np.hstack([np.nanpercentile(self.trace['gp_rollangle_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])
                elif hasattr(self,'init_soln'):
                    self.models_out['cheops']['che_pred_gp_med']=np.hstack([self.init_soln['gp_rollangle_model_phi_'+str(fk)][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)] for fk in self.cheops_filekeys])

            elif self.fit_phi_spline:
                if hasattr(self,'trace'):
                    for p in self.percentiles:
                        self.models_out['cheops']['che_pred_spline_'+p]=np.hstack([np.nanpercentile(self.trace['spline_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])
                        #fkmod=np.nanmedian(self.trace['cheops_summodel_x_'+str(fk)]+self.trace['spline_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],axis=0)
                elif hasattr(self,'init_soln'):
                    self.models_out['cheops']['che_pred_spline_med']=np.hstack([self.init_soln['spline_model_phi_'+str(fk)][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)] for fk in self.cheops_filekeys])
            if hasattr(self,'cheops_gap_timeseries'):
                self.models_out['cheops_gap_models_out']['time']=self.cheops_gap_timeseries
            if hasattr(self,'cheops_gap_fks'):
                self.models_out['cheops_gap_models_out']['filekey']=self.cheops_gap_fks
            if hasattr(self,'trace'):
                for p in self.percentiles:
                    self.models_out['cheops']['che_lindetrend_'+p]=np.hstack([np.nanpercentile(self.trace['cheops_flux_cor_'+fk],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])
                    if self.fit_phi_gp:
                        self.models_out['cheops']['che_alldetrend_'+p]=np.hstack([np.nanpercentile(self.trace['cheops_flux_cor_'+fk]+self.trace['gp_rollangle_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])

                    elif self.fit_phi_spline:
                        self.models_out['cheops']['che_alldetrend_'+p]=np.hstack([np.nanpercentile(self.trace['cheops_flux_cor_'+fk]+self.trace['spline_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])
                    
                    for npl,pl in enumerate(self.planets):
                        if 'cheops_planets_x' in self.trace.varnames:
                            self.models_out['cheops']['che_'+pl+"model_"+p]=np.nanpercentile(self.trace['cheops_planets_x'][:,self.cheops_lc['mask'],npl],self.percentiles[p],axis=0)
                        else:
                            self.models_out['cheops']['che_'+pl+"model_"+p]=np.nanpercentile(self.trace['cheops_planets_x_'+pl][:,self.cheops_lc['mask']],self.percentiles[p],axis=0)
                        if 'cheops_planets_gaps_'+pl in self.trace.varnames:
                            self.models_out['cheops_gap_models_out']['che_'+pl+"model_"+p]=np.nanpercentile(self.trace['cheops_planets_gaps_'+pl],self.percentiles[p],axis=0)

                    if 'cheops_planets_x' in self.trace.varnames:
                        self.models_out['cheops']['che_allplmodel_'+p]=np.nanpercentile(np.sum(self.trace['cheops_planets_x'][:,self.cheops_lc['mask'],:],axis=2),self.percentiles[p],axis=0)
                    else:
                        self.models_out['cheops']["che_allplmodel_"+p]=np.nanpercentile(np.sum(np.dstack([self.trace['cheops_planets_x_'+pl][:,self.cheops_lc['mask']] for pl in self.planets]),axis=2),self.percentiles[p],axis=0)
                    self.models_out['cheops_gap_models_out']["che_allplmodel_"+p]=np.nanpercentile(np.sum(np.dstack([self.trace['cheops_planets_gaps_'+pl] for pl in self.planets]),axis=2),self.percentiles[p],axis=0)
            elif hasattr(self,'init_soln'):
                p="med"
                self.models_out['cheops']['che_lindetrend_'+p]=np.hstack([self.init_soln['cheops_flux_cor_'+fk] for fk in self.cheops_filekeys])
                if self.fit_phi_gp:
                    self.models_out['cheops']['che_alldetrend_'+p]=np.hstack([self.init_soln['cheops_flux_cor_'+fk]+self.init_soln['gp_rollangle_model_phi_'+str(fk)][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)] for fk in self.cheops_filekeys])

                elif self.fit_phi_spline:
                    self.models_out['cheops']['che_alldetrend_'+p]=np.hstack([self.init_soln['cheops_flux_cor_'+fk]+self.init_soln['spline_model_phi_'+str(fk)][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)] for fk in self.cheops_filekeys])
                for npl,pl in enumerate(self.planets):
                    if 'cheops_planets_x' in self.init_soln:
                        self.models_out['cheops']['che_'+pl+"model_"+p]=self.init_soln['cheops_planets_x'][self.cheops_lc['mask'],npl]
                    else:
                        self.models_out['cheops']['che_'+pl+"model_"+p]=self.init_soln['cheops_planets_x_'+pl][self.cheops_lc['mask']]
                    if 'cheops_planets_gaps_'+pl in self.init_soln:
                        self.models_out['cheops_gap_models_out']['che_'+pl+"model_"+p]=self.init_soln['cheops_planets_gaps_'+pl]

                if 'cheops_planets_x' in self.init_soln:
                    self.models_out['cheops']['che_allplmodel_'+p]=np.sum(self.init_soln['cheops_planets_x'][self.cheops_lc['mask'],:],axis=1)
                else:
                    self.models_out['cheops']['che_allplmodel_'+p]=np.sum(np.vstack([self.models_out['cheops']['che_'+pl+"model_"+p] for pl in self.planets]),axis=0)
                self.models_out['cheops_gap_models_out']["che_allplmodel_"+p]=np.sum(np.vstack([self.init_soln['cheops_planets_gaps_'+pl] for pl in self.planets]),axis=0)
        elif tracename not in self.models_out or overwrite:
            #We have intermediate CHEOPS trace which we want to save in the same format as the final CHEOPS trace above (i.e. for plotting)
            self.models_out[tracename]=pd.DataFrame()
            #self.models_out[tracename+'_gap_models_out']=pd.DataFrame()
            for col in ['time','flux','flux_err','filekey']:
                if col in self.cheops_lc.columns:
                    self.models_out[tracename][col]=np.hstack([self.cheops_lc.loc[self.cheops_fk_mask[fk],col] for fk in self.cheops_filekeys])
            if type(init_trace)==pm.backends.base.MultiTrace:
                for p in self.percentiles:
                    self.models_out[tracename]['che_lindetrend_'+p]=np.hstack([np.nanpercentile(init_trace['cheops_flux_cor_'+fk],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])
                    self.models_out[tracename]['che_alldetrend_'+p]=self.models_out[tracename]['che_lindetrend_'+p].values[:]
                    self.models_out[tracename]["che_allplmodel_"+p]=np.zeros(len(self.models_out[tracename]['time']))
                    for npl,pl in enumerate(self.planets):
                        if "cheops_planets_x_"+pl+"_"+fk in init_trace.varnames:
                            self.models_out[tracename]['che_'+pl+"model_"+p]=np.hstack([np.nanpercentile(init_trace["cheops_planets_x_"+pl+"_"+fk],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])
                        else:
                            self.models_out[tracename]['che_'+pl+"model_"+p]=np.zeros(len(self.models_out[tracename]['time']))
                        self.models_out[tracename]["che_allplmodel_"+p]+=self.models_out[tracename]['che_'+pl+"model_"+p]
            elif type(init_trace)==dict:
                p="med"
                self.models_out[tracename]['che_lindetrend_'+p]=np.hstack([init_trace['cheops_flux_cor_'+fk] for fk in self.cheops_filekeys])
                self.models_out[tracename]['che_alldetrend_'+p]=self.models_out[tracename]['che_lindetrend_'+p].values[:]
                self.models_out[tracename]["che_allplmodel_"+p]=np.zeros(len(self.models_out[tracename]['time']))
                for npl,pl in enumerate(self.planets):
                    if "cheops_planets_x_"+pl+"_"+fk in init_trace.varnames:
                        self.models_out[tracename]['che_'+pl+"model_"+p]=np.hstack([np.nanpercentile(init_trace["cheops_planets_x_"+pl+"_"+fk],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])
                    else:
                        self.models_out[tracename]['che_'+pl+"model_"+p]=np.zeros(len(self.models_out[tracename]['time']))
                    self.models_out[tracename]["che_allplmodel_"+p]+=self.models_out[tracename]['che_'+pl+"model_"+p]
            #self.models_out[tracename+'_gap_models_out']=self.models_out[tracename] #Setting these to be identical
   #cheops_planets_x

    def make_rv_timeseries(self):

        if not hasattr(self,'models_out'):
            self.models_out={}
        self.models_out['rv']=self.rvs
        self.models_out['rv_t']=pd.DataFrame({'time':self.rv_t})
        for p in self.percentiles:
            self.models_out['rv_t']["rvt_bothmodel_"+p]=np.nanpercentile(self.trace['rv_model_t'], self.percentiles[p], axis=0)
            
            for npl,pl in enumerate(self.planets):
                self.models_out['rv_t']["rvt_"+pl+"model_"+p]=np.nanpercentile(self.trace['vrad_t'][:,:,npl], self.percentiles[p], axis=0)
                self.models_out['rv']["rv_"+pl+"model_"+p]=np.nanpercentile(self.trace['vrad_x'][:,:,npl], self.percentiles[p], axis=0)

            if self.npoly_rv>1:
                self.models_out['rv_t']["rvt_bkgmodel_"+p]=np.nanpercentile(self.trace['bkg_t'][:,:], self.percentiles[p], axis=0)
            self.models_out['rv']["rv_bkgmodel_"+p]=np.nanpercentile(self.trace['bkg_x'][:,:], self.percentiles[p], axis=0)

    def make_prior_posterior_table(self):
        """Making a table of prior & posterior values
        Copying the arctitecture from the init_model function but simply storing a list
        """
        tab=[['Teff','Stellar $T_{\\rm eff}$','[K]','normal',self.Teff[0],self.Teff[1]],
            ['Rs','Stellar Radius, $R_s$','[$R_\\odot$]','normal',self.Rstar[0],self.Rstar[1]]]
        if self.use_mstar:
            tab+=[['Ms','Stellar Mass, $M_s$','$M_\\odot$','normal',self.Mstar[0],self.Mstar[1]]]
        elif self.use_logg:
            tab+=[['logg','Stellar $\\log{\\rm g}$','','normal',self.logg[0],self.logg[1]]]
        if self.fit_contam:
            tab+=[['deltaImag_contam','Contamination $\\Delta I_{\\rm cont}$', 'mag', 'Uniform',2.5,12]]
            tab+=[['deltaVmag_contam','Contamination $\\Delta V_{\\rm cont}$','mag', 'Uniform',2.5,12]]

        for scope in self.ld_dists:
            if self.constrain_lds:
                tab+=[['u_star'+scope+'|0','Quadratic LD param $u_{\\rm '+scope+',0}$', '', 'BoundNormal',np.clip(np.nanmedian(self.ld_dists[scope],axis=0)[0],0,1),
                    np.clip(np.nanstd(self.ld_dists[scope],axis=0)[0],0.1,1.0),0,1]]
                tab+=[['u_star'+scope+'|1','Quadratic LD param $u_{\\rm '+scope+',0}$', '', 'BoundNormal',np.clip(np.nanmedian(self.ld_dists[scope],axis=0)[1],0,1),
                    np.clip(np.nanstd(self.ld_dists[scope],axis=0)[1],0.1,1.0),0,1]]
            else:
                tab+=[['u_star_'+scope+'|0','$q\'_{\\rm '+scope+',0}^\dagger{}$','','Uniform',0,1]]
                tab+=[['u_star_'+scope+'|1','$q\'_{\\rm '+scope+',1}^\dagger{}$','','Uniform',0,1]]

        for npl,pl in enumerate(self.planets):
            if not self.fit_ttvs or self.planets[pl]['n_trans']<=2:
                tab+=[['t0_'+pl,'Epoch, t_{{0,{{\rm '+pl+'}}}}','[BJD]','normal',self.planets[pl]['tcen'],self.planets[pl]['tcen_err']]]
                min_p=self.planets[pl]['period']*(1-1.5*self.timing_sd_durs*self.planets[pl]['tdur']/(np.ptp(np.hstack([self.lc_fit[src]['time'] for src in self.lc_fit]))))
                max_p=self.planets[pl]['period']*(1+1.5*self.timing_sd_durs*self.planets[pl]['tdur']/(np.ptp(np.hstack([self.lc_fit[src]['time'] for src in self.lc_fit]))))
                tab+=[['P_'+pl,'Period, P_{{\rm '+pl+'}}','[d]','BoundNormal',self.planets[pl]['period'],np.clip(self.planets[pl]['period_err'],0,(max_p-self.planets[pl]['period'])),min_p, max_p]]
            else:
                for i in range(len(self.init_transit_times[pl])):
                    tab+=[['transit_times_'+pl+'_'+str(i),'Transit time, $t_{'+str(i)+','+pl+'}$','[BJD]','Uniform',
                           self.init_transit_times[pl][i]-self.planets[pl]['tdur']*self.timing_sd_durs,
                           self.init_transit_times[pl][i]+self.planets[pl]['tdur']*self.timing_sd_durs]]
            if hasattr(self,'rvs'):
                if self.rv_mass_prior=='logK':
                    tab+=[['logK_'+pl,'log semi-amplitude, $\\log{K_{'+pl+'}}$','','normal',np.log(2),10]]
                elif self.rv_mass_prior=='K':
                    tab+=[['K_'+pl,'RV semi-amplitude, $K_{'+pl+'}$','$m.s^{-1}$','normal',2,1]]
                elif self.rv_mass_prior=='popMp':
                    rad=109.2*self.planets[list(self.planets.keys())[0]]['rprs']*self.Rstar[0]
                    mu_mp = 5.75402469 - (rad<=12.2)*(rad>=1.58)*(4.67363091 -0.38348534*rad) - \
                                                            (rad<1.58)*(5.81943841-3.81604756*np.log(rad))
                    sd_mp= (rad<=8)*(0.07904372*rad+0.24318296) + (rad>8)*(0-0.02313261*rad+1.06765343)
                    tab+=[['logMp_'+pl,'log planet mass, $\\log{M_{'+pl+'}$','','normal',mu_mp,sd_mp]]

            # Eccentricity & argument of periasteron
            if not self.assume_circ:
                tab+=[['ecc_'+pl,'$e_{pl,'+pl+'}$','','beta',0.867,3.03]]
                tab+=[['omega_'+pl,'$\\omega_{'+pl+'}$','','Uniform',0,2*np.pi]]
            tab+=[['logror_'+pl,'log Radius Ratio, $\\log{R_{\\rm '+pl+'}/R_s}$','','Uniform',np.log(0.001),np.log(0.1)]]
            tab+=[['b_'+pl,'Impact Param, $b_{\\rm '+pl+'}$','','ImpactPar']]
        if hasattr(self,'rvs'):
            for n in range(len(self.rv_medians)):
                tab+=[['rv_offsets|'+str(n),'RV Offset','$m.s^{-1}$','normal',self.rv_medians[n],self.rv_stds[n]]]
            if self.npoly_rv>1:
                for n in range(self.npoly_rv):
                    if n==0:
                        tab+=[['rv_trend|'+str(n),'RV polynomial, $d{\\rm RV}/dt','normal',0,(10.0 ** -np.arange(self.npoly_rv)[::-1])[n]]]
                    else:
                        tab+=[['rv_trend|'+str(n),'RV polynomial, $d^{'+str(n)+'}{\\rm RV}/dt^{'+str(n)+'}','normal',0,(10.0 ** -np.arange(self.npoly_rv)[::-1])[n]]]
        if self.fit_gp:
            minmax={}
            # Here we interpolate the histograms of the pre-trained GP samples as the input prior for each:
            if 'tess' in self.lcs:
                tab+=[['tess_logs','TESS jitter $\\log{\\sigma_{\\rm TESS}}$','','interp',np.nanmedian(self.oot_gp_trace["tess_logs"]),np.nanstd(self.oot_gp_trace["tess_logs"])]]
                tab+=[['tess_mean','TESS mean $\\mu_{\\rm TESS}}$','','interp',np.nanmedian(self.oot_gp_trace["tess_mean"]),np.nanstd(self.oot_gp_trace["tess_mean"])]]
            if 'k2' in self.lcs:
                tab+=[['k2_logs','K2 jitter $\\log{\\sigma_{\\rm K2}}$','','interp',np.nanmedian(self.oot_gp_trace["k2_logs"]),np.nanstd(self.oot_gp_trace["k2_logs"])]]
                tab+=[['k2_mean','K2 mean $\\mu_{\\rm K2}}$','','interp',np.nanmedian(self.oot_gp_trace["k2_mean"]),np.nanstd(self.oot_gp_trace["k2_mean"])]]
            elif 'kepler' in self.lcs:
                tab+=[['kepler_logs','Kepler jitter $\\log{\\sigma_{\\rm Kepler}}$','','interp',np.nanmedian(self.oot_gp_trace["kepler_logs"]),np.nanstd(self.oot_gp_trace["kepler_logs"])]]
                tab+=[['kepler_mean','Kepler mean $\\mu_{\\rm Kepler}}$','','interp',np.nanmedian(self.oot_gp_trace["kepler_mean"]),np.nanstd(self.oot_gp_trace["kepler_mean"])]]

            tab+=[['phot_S0','Photometric GP term $S_0$','','interp',np.nanmedian(self.oot_gp_trace["S0"]),np.nanstd(self.oot_gp_trace["S0"])]]
            tab+=[['phot_w0','Photometric GP term $\\omega_0$','','interp',np.nanmedian(self.oot_gp_trace["w0"]),np.nanstd(self.oot_gp_trace["w0"])]]
        else:
            if 'tess' in self.lcs:
                tab+=[['tess_logs','TESS jitter $\\log{\\sigma_{\\rm TESS}}$','','normal',np.log(np.std(self.lc_fit['tess']['flux'].values)),1]]
            if 'k2' in self.lcs:
                tab+=[['k2_logs','K2 jitter $\\log{\\sigma_{\\rm K2}}$','','normal',np.log(np.std(self.lc_fit['k2']['flux'].values)),1]]
            elif 'kepler' in self.lcs:
                tab+=[['kepler_logs','Kepler jitter $\\log{\\sigma_{\\rm Kepler}}$','','normal',np.log(np.std(self.lc_fit['kepler']['flux'].values)),1]]

        if len(self.cheops_filekeys)>0:
            tab+=[['cheops_logs','CHEOPS jitter $\\log{\\sigma_{\\rm CHEOPS}}$','','normal',np.log(np.nanmedian(abs(np.diff(self.cheops_lc.loc[self.cheops_lc['mask'],'flux'].values)))),3]]
            for decorr in self.cheops_linear_decorrs:
                varname=self.cheops_linear_decorrs[decorr][0]
                if 'cos' in varname or "sin" in varname:
                    if varname[3] in ['2','3','4','5']:
                        newvarname="\\"+varname[:3]+"{"+varname[3]+"\\Phi}"
                    else:
                        newvarname="\\"+varname[:3]+"{"+varname[3]+"\\Phi}"
                else:
                    newvarname="\\rm "+varname
                newvarname=newvarname.replace("delta","\Delta ") if 'delta' in newvarname else newvarname
                fks=self.cheops_linear_decorrs[decorr][1]
                if len(fks)==len(self.cheops_filekeys):
                    fk_string="Shared"
                else:
                    fk_string=",".join(list(fks)).replace("_","\_").replace("_V0200","")
                if varname=='time':
                    tab+=[[decorr,'Linear decorrelation, '+fk_string+', $dy/d{'+newvarname+'}$','','normal',
                        0,np.nanmedian([np.ptp(self.norm_cheops_dat[fk][varname])/self.cheops_mads[fk] for fk in fks])]]
                else:
                    tab+=[[decorr,'Linear decorrelation, '+fk_string+', $dy/d{'+newvarname+'}$','','normal',
                        0,np.nanmedian([self.cheops_mads[fk] for fk in fks])]]
            for decorr in self.cheops_quad_decorrs:
                varname=self.cheops_quad_decorrs[decorr][0]
                if 'cos' in varname or "sin" in varname:
                    if varname[3] in ['2','3','4','5']:
                        newvarname="\\"+varname[:3]+"{"+varname[3]+"\\Phi}"
                    else:
                        newvarname="\\"+varname[:3]+"{"+varname[3]+"\\Phi}"
                else:
                    newvarname="\\rm "+varname
                fks=self.cheops_quad_decorrs[decorr][1]
                if np.all(np.isin(fks,self.cheops_filekeys)):
                    fk_string="Shared"
                if varname=='time':
                    tab+=[[decorr,'Quadratic decorrelation, '+fk_string+', $d^2y/d{'+newvarname+'}^2$','','normal',
                        0,np.nanmedian([np.ptp(self.norm_cheops_dat['all'][varname])/self.cheops_mads[fk] for fk in fks])]]
                else:
                    tab+=[[decorr,'Quadratic decorrelation, '+fk_string+', $d^2y/d{'+newvarname+'}^2$','','normal',
                        0,np.nanmedian([self.cheops_mads[fk] for fk in fks])]]
            for fk in self.cheops_filekeys:
                tab+=[['cheops_mean_'+fk,"Cheops "+fk.replace("_","\_").replace("_V0200","")+" mean flux","[ppt]","normal",np.nanmedian(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values),np.nanstd(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values)]]
            if self.fit_phi_gp:
                tab+=[['rollangle_logpower',"${\\rm GP}_{\\rm CHEOPS}$, $\log{\\rm power}$","","normal",-6,1]]
                tab+=[['rollangle_logw0',"${\\rm GP}_{\\rm CHEOPS}$, $\log{\\rm \\omega_0}$","","normal",np.log((2*np.pi)/100),1]]
            elif self.fit_phi_spline:
                from patsy import dmatrix
                if self.common_phi_model:
                    #Fit a single spline to all rollangle data
                    minmax=(np.min(self.cheops_lc.loc[self.cheops_lc['mask'],'phi']),np.max(self.cheops_lc.loc[self.cheops_lc['mask'],'phi']))
                    n_knots=int(np.round((minmax[1]-minmax[0])/self.spline_bkpt_cad))
                    for n in np.arange(n_knots):
                        tab+=[['splines|'+str(n),"CHEOPS rollangle B-spline "+str(n),"[ppt]","normal",0,np.nanmedian(abs(np.diff(self.cheops_lc.loc[self.cheops_lc['mask'],'flux'])))]]
                else:
                    #Fit splines to each rollangle
                    knot_list={}
                    B={}
                    self.model_params['splines']={}
                    for fk in self.cheops_filekeys:
                        minmax=(np.min(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi']),np.max(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi']))
                        n_knots=int(np.round((minmax[1]-minmax[0])/self.spline_bkpt_cad))
                        for n in np.arange(n_knots):
                            tab+=[['splines|'+fk+"|"+str(n),"CHEOPS "+fk+" rollangle B-spline "+str(n),"[ppt]","normal",0,np.nanmedian(abs(np.diff(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values)))]]
        table=["\\hline"," & ".join(["Parameter","Unit","Prior","Posterior"])+"\\\\","\\hline"]
        for row in tab:
            print(row)
            newtabrow=[row[1],row[2]] #Name and unit
            if row[3].lower()=='normal':
                round_int=int(-1*np.ceil(np.log10(row[5]))+1)
                mu=str(np.round(row[4],round_int));sd=str(np.round(row[5],round_int))
                newtabrow+=["$\\mathcal{{N}}(\\mu={0},\\sigma={1}) $".format(mu,sd)]
            elif row[3].lower()=='uniform':
                round_int=int(-1*np.ceil(np.log10(row[5]-row[4]))+2)
                a=str(np.round(row[4],round_int));b=str(np.round(row[5],round_int))
                newtabrow+=["$ \\mathcal{{U}}(a={0},b={1}) $".format(a,b)]
            elif row[3].lower()=="impactpar":
                newtabrow+=["$ \\mathcal{{U}}(a=0.0,b=1+R_{p,"+row[0].split("_")[-1]+"}/R_s)^\ddagger{}$"]
            elif row[3].lower()=="boundnormal":
                #print(row[5],np.log10(row[5]),-1*np.ceil(np.log10(row[5])))
                round_int=int(-1*np.ceil(np.log10(row[5]))+1)
                mu=str(np.round(row[4],round_int));sd=str(np.round(row[5],round_int))
                newtabrow+=["$\\mathcal{{N}}_{{\\mathcal{{U}}}}(\\mu={0},\\sigma={1},a={2:0.4f},b={3:0.4f})$".format(mu,sd,row[6],row[7])]
            elif row[3].lower()=="interp":
                newtabrow+=["$\\mathcal{{I}}(\\mu={0:0.4f},\\sigma={1:0.4f})$".format(row[4],row[5])]
            if len(row[0].split("|"))==1:
                posterior=vals_to_latex(np.percentile(self.trace[row[0]],[15.87,50,84.13]))
            elif len(row[0].split("|"))==2:
                posterior=vals_to_latex(np.percentile(self.trace[row[0].split("|")[0]][:,int(row[0].split("|")[1])],[15.87,50,84.13]))

            newtabrow+=[posterior]
            table+=[" & ".join(newtabrow)+" \\\\"]
        table+=["\\hline"]
        with open(os.path.join(self.save_file_loc,self.name,self.unq_name+"_trace_modeltable.tex"),'w') as f:
            f.write('\n'.join(table))

    def make_timeseries(self, overwrite=False, **kwargs):
        #assert hasattr(self,'trace'), "Must have run an MCMC"
        
        if not hasattr(self,'models_out'):
            self.models_out={}
        for src in self.lcs:
            if src not in self.models_out or overwrite:
                self.make_lcs_timeseries(src,**kwargs)
        if len(self.cheops_filekeys)>0 and ('cheops' not in self.models_out or overwrite):
            self.make_cheops_timeseries(**kwargs)
        
        if hasattr(self,'rvs') and ('rvs' not in self.models_out or overwrite):
            self.make_rv_timeseries(**kwargs)

    def save_timeseries(self,**kwargs):
        self.make_timeseries(**kwargs)
        for mod in self.models_out:
            if mod is not None:
                self.models_out[mod].to_csv(os.path.join(self.save_file_loc,self.name,self.unq_name+"_"+mod+"_timeseries.csv"))
    
    def check_rollangle_gp(self, make_change=True, **kwargs):
        """Checking now that the model is initialised whether the rollangle GP improves the loglik or not.
        """
        #self.init_soln['cheops_summo']
        llk_cheops_wgp={}
        
        cheops_sigma2s = self.cheops_lc.loc[self.cheops_lc['mask'],'flux_err'].values ** 2 + np.exp(self.init_soln['cheops_logs'])**2
        llk_cheops_nogp = -0.5 * np.sum((self.cheops_lc.loc[self.cheops_lc['mask'],'flux'].values - \
                                        np.hstack([self.init_soln['cheops_summodel_x_'+fk] for fk in self.cheops_filekeys])) ** 2 / \
                                        cheops_sigma2s + np.log(cheops_sigma2s))
        llk_cheops_wgp = -0.5 * np.sum((self.cheops_lc.loc[self.cheops_lc['mask'],'flux'].values - \
                                        np.hstack([self.init_soln['cheops_summodel_x_'+fk]+self.init_soln['gp_rollangle_model_phi_'+fk] for fk in self.cheops_filekeys])) ** 2 / \
                                        cheops_sigma2s + np.log(cheops_sigma2s))
        #lower BIC values are generally preferred.
        #So (deltabic_wgp - deltabic_nogp)<0 prefers nogp
        delta_bic = 2*np.sum(self.cheops_lc['mask']) + 2*(llk_cheops_nogp - llk_cheops_wgp)
        if delta_bic<0:
            print("Assessment of the rollangle suggests a roll angle GP is not beneficial in this case. (",delta_bic,")")
            if make_change:
                self.update(fit_phi_gp = False)
                self.init_model()
        else:
            print("Rollangle GP is beneficial with DelatBIC =",delta_bic)
    
    def check_rollangle_spline(self, make_change=True, **kwargs):
        """Checking now that the model is initialised whether the rollangle GP improves the loglik or not.
        """
        #self.init_soln['cheops_summo']
        
        cheops_sigma2s = self.cheops_lc.loc[self.cheops_lc['mask'],'flux_err'].values ** 2 + np.exp(self.init_soln['cheops_logs'])**2
        llk_cheops_nospline = -0.5 * np.sum((self.cheops_lc.loc[self.cheops_lc['mask'],'flux'].values - \
                                        np.hstack([self.init_soln['cheops_summodel_x_'+fk] for fk in self.cheops_filekeys])) ** 2 / \
                                        cheops_sigma2s + np.log(cheops_sigma2s))
        llk_cheops_wspline = -0.5 * np.sum((self.cheops_lc.loc[self.cheops_lc['mask'],'flux'].values - \
                                        np.hstack([self.init_soln['cheops_summodel_x_'+fk]+self.init_soln['spline_model_phi_'+fk] for fk in self.cheops_filekeys])) ** 2 / \
                                        cheops_sigma2s + np.log(cheops_sigma2s))
        #lower BIC values are generally preferred.
        #So (deltabic_wspline - deltabic_nospline)<0 prefers nospline
        delta_bic = 2*np.sum(self.cheops_lc['mask']) + 2*(llk_cheops_nospline - llk_cheops_wspline)
        if delta_bic<0:
            print("Assessment of the rollangle suggests a roll angle spline model is not beneficial in this case. (",delta_bic,")")
            if make_change:
                self.update(fit_phi_spline = False)
                self.init_model()
        else:
            print("Rollangle spline model is beneficial with DeltaBIC =",delta_bic)
                        

    def print_settings(self):
        settings=""
        for key in self.defaults:
            settings+=key+"\t\t"+str(getattr(self,key))+"\n"
        print(settings)

    def save_trace(self):
        if not os.path.exists(os.path.join(self.save_file_loc,self.name)):
            os.mkdir(os.path.join(self.save_file_loc,self.name))
        pickle.dump(self.trace,open(os.path.join(self.save_file_loc,self.name,self.unq_name+"_mcmctrace.pkl"),"wb"))

    def load_model_from_file(self, loadfile):
        """Load a chexo_model object direct from file.

        Args:
            loadfile (str): Pickle file to load from.
        """
        assert os.path.exists(loadfile), "No file found to load from"
        #Loading from pickled dictionary
        pick=pickle.load(open(loadfile,'rb'))
        assert not isinstance(pick, chexo_model)
        #Unpickling each of the model objects separately)
        for key in pick:
            setattr(self,key,pick[key])

    def save_model_to_file(self, savefile=None, limit_size=True, remove_all_trace_timeseries=False):
        """Save a chexo_model object direct to file.

        Args:
            savefile (str, optional): File location to save to, otherwise it takes the default location using `GetSavename`. Defaults to None.
            limit_size (bool, optional): If we want to limit size this function can delete unuseful hyperparameters before saving. Defaults to False.
        """
        if savefile is None:
            savefile=os.path.join(self.save_file_loc,self.name,self.unq_name+'_model.pkl')
        
        #First saving GP predictions/etc using save_timeseries:
        self.save_timeseries()
        #Loading from pickle dictionary
        if limit_size and hasattr(self,'trace'):
            #We cannot afford to store full arrays of GP predictions and transit models

            #Let's clip gp and lightcurves and pseudo-variables from the trace:
            #remvars=[var for var in self.trace.varnames if (('_allphi' in var or 'gp_' in var or '_gp' in var or 'light_curve' in var) and np.product(self.trace[var].shape)>6*len(self.trace['Rs'])) or '__' in var]
            remvars=[var for var in self.trace.varnames if '_allphi' in var or '__' in var]
            if remove_all_trace_timeseries:
                remvars=list(np.unique(remvars+[var for var in self.trace.varnames if np.product(self.trace[var].shape)>50*self.trace['Rs'].shape[0]]))
            for key in remvars:
                #Permanently deleting these values from the trace.
                self.trace.remove_values(key)
            #medvars=[var for var in self.trace.varnames if 'gp_' not in var and '_gp' not in var and 'light_curve' not in var]
        n_bytes = 2**31
        max_bytes = 2**31-1
        #print({k:type(self.__dict__[k]) for k in self.__dict__})
        bytes_out = pickle.dumps({k:self.__dict__[k] for k in self.__dict__ if type(self.__dict__[k]) not in [pm.Model,xo.orbits.KeplerianOrbit] and k not in ['model_params']}) 
        #bytes_out = pickle.dumps(self)
        with open(savefile, 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])
        del bytes_out
        #pick=pickle.dump(self.__dict__,open(loadfile,'wb'))

    def plot_rollangle_model(self,save=True,savetype='png',**kwargs):
        
        if not hasattr(self,"models_out") or "cheops" not in self.models_out or (hasattr(self,'trace') and "che_lindetrend_+1sig" not in self.models_out["cheops"]):
            self.make_cheops_timeseries(**kwargs)

        modname="gp" if self.fit_phi_gp else "spline"
        assert "che_pred_"+modname+"_med" in self.models_out["cheops"], "Must have "+modname+" in saved timeseries."

        plt.figure()
        yoffset=5*np.std(self.models_out["cheops"]['flux']-(self.models_out["cheops"]['che_lindetrend_med']+self.models_out["cheops"]['che_allplmodel_med']))
        for ifk,fk in enumerate(self.cheops_filekeys):
            fk_ix=self.models_out["cheops"]['filekey']==fk
            phi=self.models_out['cheops'].loc[fk_ix,'phi']
            plt.plot(phi, yoffset*ifk+self.models_out['cheops'].loc[fk_ix,'flux']-self.models_out['cheops'].loc[fk_ix,'che_lindetrend_med']-self.models_out["cheops"].loc[fk_ix,'che_allplmodel_med'],
                    ".k",markersize=1.33,alpha=0.4)
            if "che_pred_"+modname+"_+1sig" in self.models_out['cheops']:
                plt.fill_between(np.sort(phi),yoffset*ifk+self.models_out['cheops'].loc[fk_ix,"che_pred_"+modname+"_-2sig"].values[np.argsort(phi)],
                                 yoffset*ifk+self.models_out['cheops'].loc[fk_ix,"che_pred_"+modname+"_+2sig"].values[np.argsort(phi)],alpha=0.15,color='C'+str(int(ifk)))
                plt.fill_between(np.sort(phi),yoffset*ifk+self.models_out['cheops'].loc[fk_ix,"che_pred_"+modname+"_-1sig"].values[np.argsort(phi)],
                                 yoffset*ifk+self.models_out['cheops'].loc[fk_ix,"che_pred_"+modname+"_+1sig"].values[np.argsort(phi)],alpha=0.15,color='C'+str(int(ifk)))
            plt.plot(np.sort(phi),yoffset*ifk+self.models_out['cheops'].loc[fk_ix,"che_pred_"+modname+"_med"].values[np.argsort(phi)],'-',alpha=0.45,linewidth=4,color='C'+str(int(ifk)))
        plt.xlabel("roll angle [deg]")
        plt.ylabel("Flux [ppt]")
        plt.ylim(-1*yoffset,(len(self.cheops_filekeys))*yoffset)
        if save:
            plt.savefig(os.path.join(self.save_file_loc,self.name,self.unq_name+"_rollangle"+modname+"_plots."+savetype))

    def plot_cheops(self, save=True, savetype='png', tracename=None, fk='all', show_detrend=False,
                    ylim=None, dynamic_plot_resizing=True, save_suffix="", transparent=False, **kwargs):
        """Plot cheops lightcurves with model

        Args:
            save (bool, optional): Save the figure? Defaults to True.
            savetype (str, optional): What suffix for the plot to be saved to? Defaults to 'png'.
            input_trace (PyMC3 trace, optional): Specify a trace. Defaults to None, which uses the saved full initialised model trace.
            fk (str, optional): Specify a Cheops filekey to plot, otherwise all available are plotted
            show_detrend (bool, optional): Whether to show both the pre-detrending flux and detrending model and the detrended transit+flux. Default is False
            transtype (str, optional): What type of transit prior was used. Must be one of 'set', 'loose', or 'none' (Defaults to 'set', i.e. constraining prior)
            ylim (tuple, optional): Manually set the ylim across all plots
            dynamic_plot_resizing (bool, optional): Whether to resize plots based on observing duration. Default: True
            save_suffix (str, optional): Add suffix when saving. Default is blank.
            transparent (bool, optional): Whether to save pngs with a transparent background
        """
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
        import seaborn as sns
        sns.set_palette("Paired")
        #sns.set_palette("rocket")#, as_cmap=True)

        #plt.plot(cheops_x, ,'.')
        plt.figure(figsize=(6+len(self.cheops_filekeys)*4/3,4))

        if tracename is not None:
            assert fk!="all", "to plot only loose or no transit models, need to only plot individual filekeys (i.e. set \'fk=PR...\')"
            save_suffix+="_"+np.array([f for f in ['fixtrans','loosetrans','notrans'] if f in tracename])[0]
            if tracename not in self.models_out:
                self.make_cheops_timeseries(tracename = tracename, fk=fk, init_trace=self.cheops_init_trace[tracename], **kwargs)
        else:
            tracename="cheops"
            if not hasattr(self,'models_out') or "cheops" not in self.models_out:
                self.make_cheops_timeseries(**kwargs)
        

        if fk=='all':
            fkloop=self.cheops_filekeys
        else:
            save_suffix+='_'+fk
            fkloop=[fk]
        for n,fk in enumerate(fkloop):
            fk_ix=self.models_out[tracename]['filekey']==fk
            fk_gap_ix=self.models_out[tracename+"_gap_models_out"]['filekey']==fk

            yoffset=3*np.std(self.models_out[tracename].loc[fk_ix,'flux']-(self.models_out[tracename].loc[fk_ix,'che_alldetrend_med']+self.models_out[tracename].loc[fk_ix,'che_allplmodel_med']))
            n_pts=np.sum(fk_ix)
            raw_alpha=np.clip(6*(n_pts)**(-0.4),0.02,0.99)
            spacing=10
            if len(fkloop)>2 & len(fkloop)<5:
                spacing=5
            elif len(fkloop)>=5:
                spacing=2
            
            if show_detrend:
                plt.subplot(2,len(fkloop),1+n)
                plt.plot(self.models_out[tracename].loc[fk_ix,"time"],
                         self.models_out[tracename].loc[fk_ix,"flux"], '.k',markersize=3.5,alpha=raw_alpha,zorder=1)
                binlc = bin_lc_segment(np.column_stack((self.models_out[tracename].loc[fk_ix,"time"],
                                                        self.models_out[tracename].loc[fk_ix,"flux"],
                                                        self.models_out[tracename].loc[fk_ix,"flux_err"])),1/120)
                plt.errorbar(binlc[:,0], binlc[:,1], yerr=binlc[:,2], fmt='.',color='C3',markersize=10,zorder=2,alpha=0.75)

                plt.plot(self.models_out[tracename].loc[fk_ix,"time"], yoffset+self.models_out[tracename].loc[fk_ix,"che_alldetrend_med"],'.',
                         markersize=3.5,c='C1',alpha=raw_alpha,zorder=5)
                if "che_alldetrend_+1sig" in self.models_out[tracename]:
                    plt.fill_between(self.models_out[tracename].loc[fk_ix,"time"], yoffset+self.models_out[tracename].loc[fk_ix,"che_alldetrend_-2sig"],
                                     yoffset+self.models_out[tracename].loc[fk_ix,"che_alldetrend_+2sig"],color='C0',alpha=0.15,zorder=3)
                    plt.fill_between(self.models_out[tracename].loc[fk_ix,"time"], yoffset+self.models_out[tracename].loc[fk_ix,"che_alldetrend_-1sig"],
                                     yoffset+self.models_out[tracename].loc[fk_ix,"che_alldetrend_+1sig"],color='C0',alpha=0.15,zorder=4)
                lims=np.nanpercentile(binlc[:,1],[1,99])
                if ylim is None:
                    plt.ylim(lims[0]-0.66*yoffset,lims[1]+1.5*yoffset)
                else:
                    plt.ylim(ylim)

                if n>0:
                    plt.gca().set_yticklabels([])
                else:
                    plt.gca().set_ylabel("Flux [ppt]")
                plt.gca().set_xticklabels([])

                plt.subplot(2,len(fkloop),1+len(fkloop)+n)
                if n==0:
                    plt.ylabel("flux [ppt]")
            else:
                plt.subplot(1,len(fkloop),1+n)
            plt.plot(self.models_out[tracename].loc[fk_ix,"time"],
                     self.models_out[tracename].loc[fk_ix,"flux"]-self.models_out[tracename].loc[fk_ix,"che_alldetrend_med"],
                    '.k',alpha=raw_alpha,markersize=3.5,zorder=1)
            binlc = bin_lc_segment(np.column_stack((self.models_out[tracename].loc[fk_ix,'time'], 
                                                    self.models_out[tracename].loc[fk_ix,'flux']-self.models_out[tracename].loc[fk_ix,"che_alldetrend_med"],
                                                    self.models_out[tracename].loc[fk_ix,'flux_err'])),1/120)
            plt.errorbar(binlc[:,0], binlc[:,1], yerr=binlc[:,2], c='C3', fmt='.',markersize=10, zorder=2, alpha=0.8)
            if n==0:
                plt.ylabel("flux [ppt]")
            for npl,pl in enumerate(self.planets):
                if np.any(self.models_out[tracename].loc[fk_ix,"che_"+pl+"model_med"]<-1e-5):
                    #print(np.shape(pl_time),self.chplmod[fk][pl][2].shape,as_pl_time.shape)
                    modtime=np.hstack([self.models_out[tracename].loc[fk_ix,"time"],self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"time"]])
                    modflux=np.hstack([self.models_out[tracename].loc[fk_ix,"che_"+pl+"model_med"],self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"che_"+pl+"model_med"]])
                    plt.plot(np.sort(modtime),modflux[np.argsort(modtime)],'--', c='C'+str(5+2*npl), linewidth=3, alpha=0.6, zorder=10)
                    if "che_"+pl+"model_+1sig" in self.models_out[tracename+"_gap_models_out"]:
                        modflux2sig=[np.hstack([self.models_out[tracename].loc[fk_ix,"che_"+pl+"model_-2sig"],self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"che_"+pl+"model_-2sig"]]),
                                     np.hstack([self.models_out[tracename].loc[fk_ix,"che_"+pl+"model_+2sig"],self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"che_"+pl+"model_+2sig"]])]
                        plt.fill_between(np.sort(modtime),modflux2sig[0][np.argsort(modtime)],modflux2sig[1][np.argsort(modtime)],
                                         color='C'+str(4+2*npl), alpha=0.15, zorder=6)
                        modflux1sig=[np.hstack([self.models_out[tracename].loc[fk_ix,"che_"+pl+"model_-1sig"],self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"che_"+pl+"model_-1sig"]]),
                                     np.hstack([self.models_out[tracename].loc[fk_ix,"che_"+pl+"model_+1sig"],self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"che_"+pl+"model_+1sig"]])]
                        plt.fill_between(np.sort(modtime),modflux1sig[0][np.argsort(modtime)],modflux1sig[1][np.argsort(modtime)],
                                         color='C'+str(4+2*npl), alpha=0.15, zorder=7)
            if np.all(self.models_out[tracename].loc[fk_ix,"che_allplmodel_med"]>-1e-5):
                #Plotting a flat line only if none of the planets have any transits for this fk:
                plt.plot(self.models_out[tracename].loc[fk_ix,'time'], np.zeros(np.sum(fk_ix)), '--',c='C'+str(4+2*npl),linewidth=3,alpha=0.6,zorder=10)

            if np.sum([np.any(self.models_out[tracename].loc[fk_ix,"che_"+pl+"model_med"]<-1e-5) for pl in self.planets])>1:
                #Multiple transits together - we need a summed model
                modtime=np.hstack([self.models_out[tracename].loc[fk_ix,"time"],self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"time"]])
                modallpl=np.hstack([self.models_out[tracename].loc[fk_ix,"che_allplmodel_med"],self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"che_allplmodel_med"]])
                plt.plot(np.sort(modtime),modallpl[np.argsort(modtime)],'--', linewidth=1.4, alpha=1, zorder=10, color='C9')
            if ylim is None:
                plt.ylim(np.nanmin(self.models_out[tracename].loc[fk_ix,"che_allplmodel_med"])-yoffset*0.33,yoffset*0.33)
            else:
                plt.ylim(ylim)
            plt.xlabel("Time [BJD]")
            if n>0:
                plt.gca().set_yticklabels([])
            else:
                plt.gca().set_ylabel("Flux [ppt]")
            
            plt.gca().set_xticks(np.arange(np.ceil(np.nanmin(self.models_out[tracename].loc[fk_ix,"time"])*spacing)/spacing,np.max(self.models_out[tracename].loc[fk_ix,"time"]),1/spacing))
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        plt.subplots_adjust(wspace=0.05,hspace=0.05)
        if save:
            plt.savefig(os.path.join(self.save_file_loc,self.name,self.unq_name+save_suffix+"_cheops_plots."+savetype),transparent=transparent)

    def init_phot_plot_sects(self,src,**kwargs):
        diffs=np.diff(self.lcs[src].loc[self.lcs[src]['mask'],'time'])
        jumps=diffs>0.66
        total_obs_time = np.sum(diffs[diffs<0.25])
        typ_obs_dur={'tess':27,'k2':90,'kepler':90}
        likely_sects=np.ceil(total_obs_time/typ_obs_dur[src.lower()])
        gap_starts=np.hstack([self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[0]-0.1,self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[:-1][jumps],self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[-1]])
        gap_ends=np.hstack([self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[0],self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[1:][jumps],self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[-1]+0.1])
        sectinfo={1:{'start':self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[0]}}
        nsect=1
        while nsect<(likely_sects+1):
            new_gap_ix=np.argmin(abs(gap_starts-(sectinfo[nsect]['start']+int(0.96*typ_obs_dur[src.lower()]))))
            sectinfo[nsect]['end']=gap_starts[new_gap_ix]
            sectinfo[nsect]['dur']=sectinfo[nsect]['end']-sectinfo[nsect]['start']
            if gap_ends[new_gap_ix]<self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[-1]:
                sectinfo[nsect+1]={'start':gap_ends[new_gap_ix]}
            else:
                break
            nsect+=1
        return sectinfo

    def init_phot_plot(self, src, overwrite=False, **kwargs):
        #Initialising photometric plot info - i.e. flux and planet models from either 
        #We want per-sector arrays which match time[mask]
        
        if not hasattr(self,"models_out") or not (src in self.models_out) or (hasattr(self,'trace') and src+"_allplmodel_+1sig" not in self.models_out[src].columns) or overwrite:
            #Either no saved timeseries at all, or no specific timeseries for this source, or we now have a trace but the saved timeseries have not been updated (no +/-1 sigma regions)
            self.make_lcs_timeseries(src,**kwargs)

        if not hasattr(self,"phot_plot_info"):
            self.phot_plot_info={}
        if src not in self.phot_plot_info or overwrite:
            self.phot_plot_info[src]={}
            self.phot_plot_info[src]['sectinfo'] = self.init_phot_plot_sects(src)
        for ns in self.phot_plot_info[src]['sectinfo']:
            self.phot_plot_info[src]['sectinfo'][ns]['ix'] = (self.models_out[src]['time'].values>=self.phot_plot_info[src]['sectinfo'][ns]['start'])&(self.models_out[src]['time']<=self.phot_plot_info[src]['sectinfo'][ns]['end'])
        
        self.phot_plot_info[src]['transmin']=np.min(np.hstack([self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_med'].values for ns in self.phot_plot_info[src]['sectinfo']]))
        self.phot_plot_info[src]['stdev']=np.nanmedian([np.nanstd(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'flux'].values) for ns in self.phot_plot_info[src]['sectinfo']])
        self.phot_plot_info[src]['flat_stdev']=np.nanmedian([np.nanstd(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'flux'].values - \
                                                                       (self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_med'].values + \
                                                                        self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_gpmodel_med'].values)) for ns in self.phot_plot_info[src]['sectinfo']]
                                                            )

    def plot_phot(self, src='tess', save=True, savetype='png', plot_flat=False, plot_both=False,**kwargs):
        """
        Make plot of photometric observations. 
        
        Args:
            src (str, optional): Which lightcurve source to plot (i.e. telescope). Defaults to 'tess'.
            save (bool, optional): Whether to save the plot. Defaults to True.
            savetype (str, optional): Which format to save the plot. Defaults to 'png'.
            plot_flat (bool, optional): Plot only flat/detrended timeseries (Defaults to False)
            plot_both (bool, optional): Plot both flat and detrended timeseries (Defaults to False)
        """
        sns.set_palette("Paired")   
        
        #Finding sector gaps and sector information
        self.init_phot_plot(src,**kwargs)

        plt.figure(figsize=(11,9))
        for ns in self.phot_plot_info[src]['sectinfo']:
            if not plot_flat or plot_both:
                if plot_both:
                    plt.subplot(2,len(self.phot_plot_info[src]['sectinfo']),2*ns-1)
                else:
                    plt.subplot(len(self.phot_plot_info[src]['sectinfo']),1,ns)
                #Plotting flux
                plt.plot(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                         self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'flux'],
                         '.k',markersize=1.0,alpha=0.4,zorder=1)
                binsect=bin_lc_segment(np.column_stack((self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                                                        self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'flux'],
                                                        self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'flux_err'])),1/48)
                plt.errorbar(binsect[:,0],binsect[:,1],yerr=binsect[:,2],fmt='.',color="C1",ecolor="C0",alpha=0.6,zorder=2)
                if src+'_gpmodel_+1sig' in self.models_out[src]: #_allplmodel_med,_gpmodel_med
                    #flux model regions
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_gpmodel_-2sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_gpmodel_+2sig'],
                                     color='C4',alpha=0.15,zorder=3)
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_gpmodel_-1sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_gpmodel_+1sig'],
                                     color='C4',alpha=0.15,zorder=4)
                if src+'_allplmodel_+1sig' in self.models_out[src]: #_allplmodel_med,_gpmodel_med
                    #planet + flux model regions
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_-2sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_+2sig'],
                                     color='C2',alpha=0.15,zorder=6)
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_-1sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_+1sig'],
                                     color='C2',alpha=0.15,zorder=7)
                plt.plot(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                         self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_gpmodel_med'],
                         linewidth=2,color='C5',alpha=0.75,zorder=5)
                plt.plot(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                         self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_gpmodel_med']+self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_med'],
                         linewidth=2,color='C3',alpha=0.75,zorder=8)

            if plot_flat:
                if plot_both:
                    plt.subplot(2,len(self.phot_plot_info[src]['sectinfo']),2*ns)
                else:
                    plt.subplot(len(self.phot_plot_info[src]['sectinfo']),1,ns)
                #Plotting flux (minus variability model)
                plt.plot(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                         self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'flux']-self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_gpmodel_med'],
                         '.k',markersize=1.0,alpha=0.4,zorder=1)
                binsect=bin_lc_segment(np.column_stack((self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                                                        self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'flux']-self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_gpmodel_med'],
                                                        self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'flux_err'])),1/48)
                plt.errorbar(binsect[:,0],binsect[:,1],yerr=binsect[:,2],fmt='.',color="C1",ecolor="C0",alpha=0.6,zorder=2)
                if src+'_allplmodel_+1sig' in self.models_out[src]: #_allplmodel_med,_gpmodel_med
                    #planet + flux model regions
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_-2sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_+2sig'],
                                     color='C2',alpha=0.15,zorder=6)
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_-1sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_+1sig'],
                                     color='C2',alpha=0.15,zorder=7)
                plt.plot(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'time'],
                         self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_med'],
                         linewidth=2,color='C3',alpha=0.75,zorder=8)
            
            plt.xlim(self.phot_plot_info[src]['sectinfo'][ns]['start']-1,self.phot_plot_info[src]['sectinfo'][ns]['end']+1)
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
            
            if plot_flat:
                plt.ylim(self.phot_plot_info[src]['transmin']-1.66*self.phot_plot_info[src]['flat_stdev'],1.66*self.phot_plot_info[src]['flat_stdev'])
            else:
                plt.ylim(self.phot_plot_info[src]['transmin']-1.66*self.phot_plot_info[src]['stdev'],1.66*self.phot_plot_info[src]['stdev'])

            if ns==len(self.phot_plot_info[src]['sectinfo']):
                plt.xlabel("BJD")
            plt.ylabel("Flux [ppt]")

            # if self.fit_gp:
            #     #Plotting GP
            #     if hasattr(self,'trace'):
            #         #Using MCMC
            #         from scipy.interpolate import interp1d
            #         if self.bin_oot:
            #             bf_gp=[]
            #             for p in self.percentiles:
            #                 interpp=interp1d(np.hstack((self.lcs[src].loc[sect_ix,'time'].values[0]-0.1,self.lc_fit[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['fit_ix'],'time'],self.lcs[src].loc[sect_ix,'time'].values[-1]+0.1)),
            #                                  np.hstack((0,np.percentile(self.trace['photgp_model_x'][:,self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']],self.percentiles[p],axis=0),0)))
            #                 bf_gp+=[interpp(self.lcs[src].loc[sect_ix,'time'].values)]
            #             #print("bin_oot",bf_gp[0].shape)
            #         elif not self.cut_oot:
            #             bf_gp = np.nanpercentile(self.trace['photgp_model_x'][:,self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']], list(self.percentiles.values()), axis=0)
            #         #print(len(bf_gp[0]),len(self.lc_fit[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['fit_ix'],'time']))
            #         if not plot_flat:
            #             plt.fill_between(self.lcs[src].loc[sect_ix,'time'],bf_gp[0],bf_gp[4],color='C4',alpha=0.15,zorder=3)
            #             plt.fill_between(self.lcs[src].loc[sect_ix,'time'],bf_gp[1],bf_gp[3],color='C4',alpha=0.15,zorder=4)
            #             plt.plot(self.lcs[src].loc[sect_ix,'time'],bf_gp[2],linewidth=2,color='C5',alpha=0.75,zorder=5)
            #         else:
            #             plt.plot(self.lcs[src].loc[sect_ix,'time'],self.lcs[src].loc[sect_ix,'flux']-bf_gp[2],'.k',markersize=1.0,alpha=0.4,zorder=1)
            #             binsect=bin_lc_segment(np.column_stack((self.lcs[src].loc[sect_ix,'time'],
            #                                                     self.lcs[src].loc[sect_ix,'flux']-bf_gp[2],
            #                                                     self.lcs[src].loc[sect_ix,'flux_err'])),1/48)
            #             plt.errorbar(binsect[:,0],binsect[:,1],yerr=binsect[:,2],fmt='.',color="C1",ecolor="C0",alpha=0.6,zorder=2)

            #         fluxmod=bf_gp[2]
            #     else:
            #         #Using initial soln
            #         fluxmod=self.init_soln['photgp_model_x'][self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']]
            #         if not plot_flat:
            #             plt.plot(self.lc_fit[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['fit_ix'],'time'],fluxmod,linewidth=2,color='C5',alpha=0.75,zorder=5)
            #         else:
            #             plt.plot(self.lcs[src].loc[sect_ix,'time'],self.lcs[src].loc[sect_ix,'flux']-fluxmod,'.k',markersize=1.0,alpha=0.4,zorder=1)
            #             binsect=bin_lc_segment(np.column_stack((self.lcs[src].loc[sect_ix,'time'],
            #                                                     self.lcs[src].loc[sect_ix,'flux']-fluxmod,
            #                                                     self.lcs[src].loc[sect_ix,'flux_err'])),1/48)
            #             plt.errorbar(binsect[:,0],binsect[:,1],yerr=binsect[:,2],fmt='.',color="C1",ecolor="C0",alpha=0.6,zorder=2)

            # elif not self.fit_gp and self.fit_flat:
            #     #Plotting kepler spline
            #     fluxmod=self.lcs[src].loc[sect_ix,'spline'].values
            #     fitfluxmod=self.lc_fit[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['fit_ix'],'spline'].values
            #     if plot_flat:
            #         plt.plot(self.lcs[src].loc[sect_ix,'time'], fluxmod, 
            #                 linewidth=2,color='C5',alpha=0.75)
            #     else:
            #         plt.plot(self.lcs[src].loc[sect_ix,'time'],self.lcs[src].loc[sect_ix,'flux']-fluxmod,'.k',markersize=1.0,alpha=0.4,zorder=1)
            #         binsect=bin_lc_segment(np.column_stack((self.lcs[src].loc[sect_ix,'time'],
            #                                                 self.lcs[src].loc[sect_ix,'flux']-fluxmod,
            #                                                 self.lcs[src].loc[sect_ix,'flux_err'])),1/48)
            #         plt.errorbar(binsect[:,0],binsect[:,1],yerr=binsect[:,2],fmt='.',color="C1",ecolor="C0",alpha=0.6,zorder=2)
            # else:
            #     fluxmod = np.zeros(np.sum(sect_ix))
            
            # if hasattr(self,'trace'):
            #     if self.cut_oot and self.fit_flat:
            #         pl_mod=np.zeros((5,np.sum(self.lcs[src].loc[sect_ix,'mask'])))
            #         pl_mod[:,self.lcs[src].loc[sect_ix&self.lcs[src]['mask'],'near_trans']]=np.nanpercentile(self.trace[src+'_summodel_x'][:,self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']], list(self.percentiles.values()), axis=0)
            #     elif self.bin_oot:
            #         pl_mod=[]
            #         for p in self.percentiles:
            #             interpp=interp1d(np.hstack((self.lcs[src].loc[sect_ix,'time'].values[0]-0.1,self.lc_fit[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['fit_ix'],'time'].values,self.lcs[src].loc[sect_ix,'time'].values[-1]+0.1)),
            #                             np.hstack((0,np.percentile(self.trace[src+'_summodel_x'][:,self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']],self.percentiles[p],axis=0),0)))
            #             pl_mod+=[interpp(self.lcs[src].loc[sect_ix,'time'].values)]

            #     else:
            #         pl_mod = np.nanpercentile(self.trace[src+'_summodel_x'][:,self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']], list(self.percentiles.values()), axis=0)
            #     #len(fluxmod),len(pl_mod[0]),len(self.lcs[src].loc[sect_ix,'time']))
            #     if plot_flat or self.fit_flat:
            #         plt.fill_between(self.lcs[src].loc[sect_ix,'time'],pl_mod[0],pl_mod[4],color='C2',alpha=0.15,zorder=6)
            #         plt.fill_between(self.lcs[src].loc[sect_ix,'time'],pl_mod[1],pl_mod[3],color='C2',alpha=0.15,zorder=7)
            #         plt.plot(self.lcs[src].loc[sect_ix,'time'],pl_mod[2],linewidth=2,color='C3',alpha=0.75,zorder=8)
            #     else:
            #         plt.fill_between(self.lcs[src].loc[sect_ix,'time'],fluxmod+pl_mod[0],fluxmod+pl_mod[4],color='C2',alpha=0.15,zorder=6)
            #         plt.fill_between(self.lcs[src].loc[sect_ix,'time'],fluxmod+pl_mod[1],fluxmod+pl_mod[3],color='C2',alpha=0.15,zorder=7)
            #         plt.plot(self.lcs[src].loc[sect_ix,'time'],fluxmod+pl_mod[2],linewidth=2,color='C3',alpha=0.75,zorder=8)
            #     transmin=np.min(pl_mod[0])

            # else:
                
            #     if self.cut_oot and self.fit_flat:
            #         pl_mod=np.tile(np.nan,np.sum(self.lcs[src].loc[sect_ix,'mask']))
            #         pl_mod[self.lcs[src].loc[sect_ix&self.lcs[src]['mask'],'near_trans']]=fitfluxmod+self.init_soln[src+'_summodel_x'][self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']]
            #         pl_mod[~self.lcs[src].loc[sect_ix&self.lcs[src]['mask'],'near_trans']]=fluxmod[[~self.lcs[src].loc[sect_ix&self.lcs[src]['mask'],'near_trans']]]
            #         plt.plot(self.lcs[src].loc[sect_ix,'time'],pl_mod,linewidth=2,color='C3',alpha=0.75,zorder=8)
            #     else:
            #         if self.bin_oot:
            #             pl_mod=self.init_soln[src+'_summodel_x'][self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']]
            #         else:
            #             pl_mod[self.lcs[src].loc[sect_ix&self.lcs[src]['mask'],'near_trans']]=self.init_soln[src+'_summodel_x'][self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']]
            #         plt.plot(self.lc_fit[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['fit_ix'],'time'],pl_mod,linewidth=2,color='C3',alpha=0.75,zorder=8)

            #     transmin=np.min(self.init_soln[src+'_summodel_x'][self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']])


            # plt.xlim(sectinfo[ns]['start']-1,sectinfo[ns]['end']+1)
            # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
            # stdev=np.std(self.lcs[src].loc[sect_ix,'flux'])
            # plt.ylim(transmin-1.66*stdev,1.66*stdev)

            # if ns==len(sectinfo):
            #     plt.xlabel("BJD")
            # plt.ylabel("Flux [ppt]")
        
        if save:
            plt.savefig(os.path.join(self.save_file_loc,self.name,self.unq_name+"_"+src+"_plot."+savetype))

    def plot_ttvs(self,save=True,nplots=None,savetype='png'):
        """Plot TTVs

        Args:
            save (bool, optional): Whether to save the plot. Defaults to True.
            nplots (int, optional): Number of seperate plots. Defaults to the length of the planets.
            savetype (str, optional): Type of image to save. Defaults to 'png'.
        """
        assert self.fit_ttvs, "Must have fitted TTVs timing values using `fit_ttvs` flag"
        
        plt.figure(figsize=(5.5,9))
        pls_with_ttvs=[pl for pl in self.planets if self.planets[pl]['n_trans']>2]
        nplots=len(pls_with_ttvs) if nplots is None else nplots
        ttvs={pl:[] for pl in pls_with_ttvs}
        times={pl:[] for pl in pls_with_ttvs}
        for npl,pl in enumerate(pls_with_ttvs):
            if hasattr(self,'trace'):
                new_p=np.nanmedian(self.trace['P_'+pl])
                new_t0=np.nanmedian(self.trace['t0_'+pl])
                #print(out.shape,self.planets[pl]['n_trans'])
            else:
                ttvs[pl]=np.array([self.init_soln['transit_times_'+pl+'_'+str(n)] for n in range(self.planets[pl]['n_trans'])])
                new_p=self.init_soln['P_'+pl]
                new_t0=self.init_soln['t0_'+pl]
                out=1440*ttvs - (new_t0+new_p*self.init_transit_inds[pl])
            for n in range(self.planets[pl]['n_trans']):
                if hasattr(self,'trace'):
                    times[pl]+=[np.nanmedian(self.trace['transit_times_'+pl+'_'+str(n)])]
                    ttvs[pl]+=[np.nanpercentile(1440*(self.trace['transit_times_'+pl+'_'+str(n)] - (self.trace['t0_'+pl]+new_p*self.init_transit_inds[pl][n])),list(self.percentiles.values())[1:-1])]
                else:
                    times[pl]+=[self.init_soln['transit_times_'+pl+'_'+str(n)]]
                    ttvs[pl]=[self.init_soln['transit_times_'+pl+'_'+str(n)]]
                #print(out[1,i],"±",0.5*(out[2,i]-out[0,i]))
            ttvs[pl]=np.vstack((ttvs[pl]))
            for ipl in range(nplots):
                plt.subplot(nplots,1,ipl+1)
                if hasattr(self,'trace'):
                    plt.errorbar(new_t0+new_p*self.init_transit_inds[pl],
                                 ttvs[pl][:,1],yerr=[ttvs[pl][:,1]-ttvs[pl][:,0],ttvs[pl][:,2]-ttvs[pl][:,1]],
                                fmt='.-',label=str(pl),alpha=0.6,markersize=15)
                else:
                    #print(ipl,pl,np.array(ttvs[pl]))
                    plt.plot(new_t0+new_p*self.init_transit_inds[pl],
                             np.array(ttvs[pl]),'.-',label=str(pl),alpha=0.6,markersize=15)
        all_times=np.sort(np.hstack([times[pl] for pl in pls_with_ttvs]))
        splittimes=[list(all_times)]
        for n in range(nplots-1):
            maxdiffs=[]
            maxdiffpos=[]
            for d in range(len(splittimes)):
                diffs=np.diff(splittimes[d])
                maxdiffs+=[np.max(diffs)]
                maxdiffpos+=[np.argmax(diffs)]
            maxalldiffs=np.argmax(maxdiffs)
            newsplittimes=[splittimes[s] for s in range(len(splittimes)) if s!=maxalldiffs]
            newsplittimes+=[splittimes[maxalldiffs][:1+maxdiffpos[maxalldiffs]],
                            splittimes[maxalldiffs][1+maxdiffpos[maxalldiffs]:]]
            splittimes=newsplittimes[:]

        for i in range(nplots):
            plt.subplot(nplots,1,i+1)
            plt.xlim(splittimes[i][0]-5,splittimes[i][-1]+5)
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.ylabel("O-C [mins]")
            plt.legend()
        if save:
            plt.savefig(os.path.join(self.save_file_loc,self.name,self.unq_name+"_ttv_plot."+savetype))

    def plot_rvs(self,save=True,savetype='png'):
        assert hasattr(self,'rvs'), "No RVs found..."
        plt.figure(figsize=(8,6))
        plt.subplot(2,1,1)
        if hasattr(self,'trace'):
            rvt_mods=np.nanpercentile(self.trace['rv_model_t'],list(self.percentiles.values()),axis=0)
            plt.fill_between(self.rv_t,rvt_mods[0],rvt_mods[4],color='C4',alpha=0.15)
            plt.fill_between(self.rv_t,rvt_mods[1],rvt_mods[3],color='C4',alpha=0.15)
            plt.plot(self.rv_t,rvt_mods[2],c='C4',alpha=0.66)
            if self.npoly_rv>1:
                plt.plot(self.rv_t,np.nanmedian(self.trace['bkg_t'],axis=0),c='C2',alpha=0.3,linewidth=3)
            
        else:
            plt.plot(self.rv_t,self.init_soln['rv_model_t'],c='C4',alpha=0.66)
            if self.npoly_rv>1:
                plt.plot(self.rv_t,self.init_soln['bkg_t'],'--',c='C2',alpha=0.3,linewidth=3)
        #plt.plot(rv_t,np.nanmedian(trace_2['vrad_t'][:,:,0],axis=0),':')
        #plt.plot(rv_t,np.nanmedian(trace_2['vrad_t'][:,:,1],axis=0),':')
        
        #plt.fill_between(rv_t,rv_prcnt[:,0],rv_prcnt[:,2],color='C1',alpha=0.05)
        
        labs=['harps_pre','harps_post','pfs']
        for isc,sc in enumerate(pd.unique(self.rvs['scope'])):
            plt.errorbar(self.rvs.loc[self.rvs['scope']==sc,'time'],self.rvs.loc[self.rvs['scope']==sc,'y']-self.init_soln['rv_offsets'][isc],
                         yerr=self.rvs.loc[self.rvs['scope']==sc,'yerr'],
                         fmt='.',ecolor='#aaaaaa',zorder=10,label=sc)
        plt.legend()
        plt.ylabel("RV [ms]")
        #plt.xlim(1850,1870)

        for n,pl in enumerate(self.planets):
            plt.subplot(2,len(self.planets),len(self.planets)+1+n)
            t0 = self.init_soln['t0'][n] if not hasattr(self,'trace') else np.nanmedian(self.trace['t0'][:,n])
            p = self.init_soln['P'][n] if not hasattr(self,'trace') else np.nanmedian(self.trace['P'][:,n])
            rv_phase_x = (self.rvs['time']-t0-0.5*p)%p-0.5*p
            rv_phase_t = (self.rv_t-t0-0.5*p)%p-0.5*p
            if not hasattr(self,'trace'):
                other_pls_bg=self.init_soln['bkg_x']+np.sum([self.init_soln['vrad_x'][:,inpl] for inpl in range(len(self.planets)) if inpl!=n],axis=0)
                for isc,sc in enumerate(pd.unique(self.rvs['scope'])):
                    plt.errorbar(rv_phase_x[self.rvs['scope'].values==sc],
                                self.rvs.loc[self.rvs['scope']==sc,'y'] - other_pls_bg[self.rvs['scope']==sc],
                                yerr=self.rvs.loc[self.rvs['scope']==sc,'yerr'],
                                fmt='.',ecolor='#aaaaaa',zorder=10,label=sc)
                if len(self.planets)>1:
                    plt.plot(np.sort(rv_phase_t),self.init_soln['vrad_t'][np.argsort(rv_phase_t)][:,n],c='C1')
                else:
                    plt.plot(np.sort(rv_phase_t),self.init_soln['vrad_t'][np.argsort(rv_phase_t)],c='C1')

            else:
                other_pls_bg=np.nanmedian(self.trace['bkg_x']+np.sum([self.trace['vrad_x'][:,:,inpl] for inpl in range(len(self.planets)) if inpl!=n],axis=0),axis=0)
                for isc,sc in enumerate(pd.unique(self.rvs['scope'])):
                    plt.errorbar(rv_phase_x[self.rvs['scope'].values==sc],
                                self.rvs.loc[self.rvs['scope']==sc,'y'] - other_pls_bg[self.rvs['scope']==sc],
                                yerr=self.rvs.loc[self.rvs['scope']==sc,'yerr'],
                                fmt='.',ecolor='#aaaaaa',zorder=10,label=sc)
                if len(self.planets)>1:
                    rvt_mods=np.nanpercentile(self.trace['vrad_t'][:,np.argsort(rv_phase_t),n],list(self.percentiles.values()),axis=0)
                else:
                    rvt_mods=np.nanpercentile(self.trace['vrad_t'][:,np.argsort(rv_phase_t)],list(self.percentiles.values()),axis=0)
                plt.fill_between(np.sort(rv_phase_t),rvt_mods[0],rvt_mods[4],color='C1',alpha=0.15)
                plt.fill_between(np.sort(rv_phase_t),rvt_mods[1],rvt_mods[3],color='C1',alpha=0.15)
                plt.plot(np.sort(rv_phase_t),rvt_mods[2],c='C1',alpha=0.65)
                        
            if n==0:
                plt.ylabel("RV [ms]")
            else:
                plt.gca().set_yticklabels([])
            plt.xlabel("Time from t0 [d]")
        if save:
            plt.savefig(os.path.join(self.save_file_loc,self.name,self.unq_name+"_ttvs."+savetype))

    def plot_transits_fold(self,save=True,savetype='png',xlim=None,show_legend=True,sigma_fill=2,yoffsets=None):

        if not hasattr(self,"models_out"):
            self.make_timeseries()

        import seaborn as sns
        sns.set_palette("Paired")
        plt.figure(figsize=(5,3+2*len(self.planets)**0.66))
        if xlim is None:
            xlim=(-1*2/(len(self.planets)**0.5)*np.nanmax([self.planets[pl]['tdur'] for pl in self.planets]),
                  2/(len(self.planets)**0.5)*np.nanmax([self.planets[pl]['tdur'] for pl in self.planets]))
        for npl,pl in enumerate(self.planets):
            plt.subplot(len(self.planets),1,1+npl)
            yoffset=0
            t0 = self.init_soln['t0_'+pl] if not hasattr(self,'trace') else np.nanmedian(self.trace['t0_'+pl])
            p = self.init_soln['P_'+pl] if not hasattr(self,'trace') else np.nanmedian(self.trace['P_'+pl])
            nscope=0
            for scope in self.lc_fit:
                if self.fit_ttvs and self.planets[pl]['n_trans']>2:
                    #subtract nearest fitted transit time for each time value
                    trans_times= np.array([self.init_soln['transit_times_'+pl+'_'+str(n)] for n in range(self.planets[pl]['n_trans'])]) if not hasattr(self,'trace') else np.array([np.nanmedian(self.trace['transit_times_'+pl+'_'+str(n)], axis=0) for n in range(self.planets[pl]['n_trans'])])
                    nearest_times=np.argmin(abs(self.models_out[scope]['time'].values[:,None]-trans_times[None,:]),axis=1)
                    phase = self.models_out[scope]['time'].values - trans_times[nearest_times]
                    #(self.lc_fit[src].loc[self.lc_fit[scope]['near_trans'],'time']-t0-0.5*p)%p-0.5*p
                else:
                    phase = (self.models_out[scope]['time'].values-t0-0.5*p)%p-0.5*p
                    #(self.lc_fit[scope].loc[self.lc_fit[scope]['near_trans'],'time']-t0-0.5*p)%p-0.5*p

                ix = abs(phase)<(3*self.planets[pl]['tdur'])
                phase=phase[ix]
                n_pts=np.sum((phase<xlim[1])&(phase>xlim[0]))
                raw_alpha=np.clip(6*(n_pts)**(-0.4),0.02,0.99)
                #pl2mask=[list(info.keys())[n2] in range(3) if n2!=n]

                #Need to also remove the influence of other planets here:
                transmin =np.min(self.models_out[scope].loc[ix,scope+'_'+pl+'model_med'])
                plflux=self.models_out[scope].loc[ix,'flux'].values-self.models_out[scope].loc[ix,scope+"_gpmodel_med"].values-self.models_out[scope].loc[ix,scope+"_allplmodel_med"].values+self.models_out[scope].loc[ix,scope+"_"+pl+"model_med"].values
                plt.plot(phase, yoffset+plflux,'.',c='C'+str(nscope*2),alpha=raw_alpha,markersize=3,zorder=1)
                binsrclc=bin_lc_segment(np.column_stack((np.sort(phase), plflux[np.argsort(phase)],
                                                         self.models_out[scope].loc[ix,'flux_err'].values[np.argsort(phase)])),self.planets[pl]['tdur']/8)
                plt.errorbar(binsrclc[:,0],yoffset+binsrclc[:,1],yerr=binsrclc[:,2],fmt='.',markersize=8,alpha=0.8,ecolor='#ccc',zorder=2,color='C'+str(1+nscope*2),label=scope)
                if scope+"_"+pl+"model_+1sig" in self.models_out[scope] and sigma_fill>0:
                    if int(sigma_fill)>=2:
                        plt.fill_between(np.sort(phase), 
                                         yoffset+self.models_out[scope].loc[ix,scope+"_"+pl+"model_-2sig"].values[np.argsort(phase)],
                                         yoffset+self.models_out[scope].loc[ix,scope+"_"+pl+"model_+2sig"].values[np.argsort(phase)],
                                         alpha=0.15,zorder=3,color='C'+str(4+2*npl))
                    plt.fill_between(np.sort(phase), 
                                     yoffset+self.models_out[scope].loc[ix,scope+"_"+pl+"model_-1sig"].values[np.argsort(phase)],
                                     yoffset+self.models_out[scope].loc[ix,scope+"_"+pl+"model_+1sig"].values[np.argsort(phase)],
                                     alpha=0.15,zorder=4,color='C'+str(4+2*npl))
                plt.plot(np.sort(phase),yoffset+self.models_out[scope].loc[ix,scope+"_"+pl+"model_med"].values[np.argsort(phase)],':',
                            alpha=0.66,zorder=5,color='C'+str(5+2*npl),linewidth=2.5)
                std=np.nanmedian(abs(np.diff(plflux-self.models_out[scope].loc[ix,scope+"_"+pl+"model_med"])))
                if yoffsets is None:
                    yoffset+= transmin + 3*std
                else:
                    yoffset+= yoffsets
                nscope+=1
            
            if len(self.cheops_filekeys)>0:
                if yoffsets is None:
                    yoffset+=abs(transmin)
                else:
                    yoffset+= yoffsets
                if not self.fit_ttvs or self.planets[pl]['n_trans']<=2:
                    chphase = (self.models_out['cheops']['time'].values-t0-0.5*p)%p-0.5*p
                    #(self.lc_fit[scope].loc[self.lc_fit[scope]['near_trans'],'time']-t0-0.5*p)%p-0.5*p
                else:
                    #subtract nearest fitted transit time for each time value
                    trans_times= np.array([self.init_soln['transit_times_'+pl+'_'+str(n)] for n in range(self.planets[pl]['n_trans'])]) if not hasattr(self,'trace') else np.array([np.nanmedian(self.trace['transit_times_'+pl+'_'+str(n)], axis=0) for n in range(self.planets[pl]['n_trans'])])
                    nearest_times=np.argmin(abs(self.models_out['cheops']['time'].values[:,None]-trans_times[None,:]),axis=1)
                    chphase = self.models_out['cheops']['time'].values - trans_times[nearest_times]
                    #(self.lc_fit[src].loc[self.lc_fit[scope]['near_trans'],'time']-t0-0.5*p)%p-0.5*p
                chix = abs(chphase)<(3*self.planets[pl]['tdur'])
                chphase=chphase[chix]
                ch_n_pts=np.sum((chphase<xlim[1])&(chphase>xlim[0]))
                ch_raw_alpha=np.clip(9*(ch_n_pts)**(-0.4),0.02,0.99)
                #print("alphas=",raw_alpha,ch_raw_alpha)
                cheopsally = self.models_out['cheops'].loc[chix,'flux'].values-self.models_out['cheops'].loc[chix,'che_alldetrend_med'].values-self.models_out['cheops'].loc[chix,'che_allplmodel_med'].values+self.models_out['cheops'].loc[chix,'che_'+pl+'model_med'].values
                plt.plot(chphase, yoffset+cheopsally,'.k',markersize=5,c='C'+str(nscope*2),alpha=ch_raw_alpha,zorder=5)
                binchelc=bin_lc_segment(np.column_stack((np.sort(chphase), cheopsally[np.argsort(chphase)],
                                                            self.models_out['cheops'].loc[chix,'flux_err'])),
                                        self.planets[pl]['tdur']/8)
                plt.errorbar(binchelc[:,0],yoffset+binchelc[:,1],yerr=binchelc[:,2],fmt='.',markersize=8,alpha=0.8,ecolor='#ccc',zorder=6,color='C'+str(1+nscope*2),label="CHEOPS")
                
                chgapphase = (np.hstack([self.models_out['cheops']['time'],self.models_out['cheops_gap_models_out']['time']])-t0-0.5*p)%p-0.5*p
                chgapix    = abs(chgapphase)<(3*self.planets[pl]['tdur'])
                if "che_"+pl+"model_+1sig" in self.models_out[scope] and sigma_fill>0:
                    if int(sigma_fill)>=2:
                        modflux2sig=[np.hstack([self.models_out['cheops']['che_'+pl+'model_-2sig'],self.models_out['cheops_gap_models_out']['che_'+pl+'model_-2sig']]),
                                     np.hstack([self.models_out['cheops']['che_'+pl+'model_+2sig'],self.models_out['cheops_gap_models_out']['che_'+pl+'model_+2sig']])]
                        plt.fill_between(np.sort(chgapphase[chgapix]), yoffset+modflux2sig[0][chgapix][np.argsort(chgapphase[chgapix])],
                                         yoffset+modflux2sig[1][chgapix][np.argsort(chgapphase[chgapix])],
                                         zorder=6,alpha=0.15,color='C'+str(4+2*npl))
                    modflux1sig=[np.hstack([self.models_out['cheops']['che_'+pl+'model_-1sig'],self.models_out['cheops_gap_models_out']['che_'+pl+'model_-1sig']]),
                                 np.hstack([self.models_out['cheops']['che_'+pl+'model_+1sig'],self.models_out['cheops_gap_models_out']['che_'+pl+'model_+1sig']])]
                    plt.fill_between(np.sort(chgapphase[chgapix]), yoffset+modflux1sig[0][chgapix][np.argsort(chgapphase[chgapix])],
                                         yoffset+modflux1sig[1][chgapix][np.argsort(chgapphase[chgapix])],
                                     zorder=7,alpha=0.15,color='C'+str(4+2*npl))
                modflux=np.hstack([self.models_out['cheops']['che_'+pl+'model_med'],self.models_out['cheops_gap_models_out']['che_'+pl+'model_med']])
                plt.plot(np.sort(chgapphase[chgapix]), yoffset+modflux[chgapix][np.argsort(chgapphase[chgapix])],
                        '--',zorder=8,alpha=0.6,color='C'+str(5+2*npl),linewidth=2.5)
                
                yoffset+=5*np.nanmedian(abs(np.diff(self.models_out['cheops']['flux']-self.models_out['cheops']['che_alldetrend_med']-self.models_out['cheops']['che_allplmodel_med'])))
            
            plt.ylabel("Flux [ppt]")
            #print(npl,len(self.planets))
            if npl==len(self.planets)-1:
                plt.xlabel("Time from transit [d]")
            else:
                plt.gca().set_xticklabels([])
            plt.ylim(transmin-std,yoffset)
            plt.xlim(xlim)
        if show_legend:
            plt.legend() 
        plt.subplots_adjust(hspace=0.05)

        if save:
            plt.savefig(os.path.join(self.save_file_loc,self.name,self.unq_name+"_folded_trans."+savetype))

    def MakeExoFopFiles(self, toi_names, desc=None, tsnotes='',plnotes='',table=None,username="osborn",
                        initials="ho",files=["cheops_plots","folded_trans"],
                        upload_loc="/Users/hosborn/Postdoc/Cheops/ChATeAUX/",check_toi_list=True, **kwargs):
        """Make ExoFop Files for upload. There are three things to upload - the lightcurve plot, the timseries table entry, and potentially a meta-description.
        Once this is complete, the outputs must be uploaded to 

        Args:
            toi_names (list): Names of each TOI found in the data. Potentially multiple if many are in each visit.
            desc (str, None): Description of the CHEOPS observation to go into the plot upload table
            tsnotes (str, optional): Brief notes on the observation to go into the timeseries file
            plnotes (str, optional): Brief notes on the planetary parameters to go into the timeseries file
            table (pd.DataFrame, optional): Table of planetary parameters to use. Otherwise we built one from the trace
            username (str, optional): ExoFop username for use in the autotag. Defaults to "osborn"
            initials (str, optional): Initials. Defaults to "HO"
            files (list, optional): Lightcurve plot files to upload - can include "folded_trans", "cheops_plots", "tess_plot". Defaults to ["fold"]
            check_toi_list (bool, optional): Whether to use the TOI list to double check which planet corresponds to which TOI. Defaults to True.
        """
        assert type(toi_names[0])==str, "TOI names must be string"

        
        for t in range(len(toi_names)):
            toi_names[t]="TOI"+toi_names[t] if toi_names[t][0]!="T" else toi_names[t] #Making sure we have only the integer TOI here.
        base_toi_name="TOI"+str(int(float(toi_names[0][3:])))

        if check_toi_list:
            #df=pd.read_csv("https://tev.mit.edu/data/collection/193/csv/5/")
            df=pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
            df=df.loc[df['TOI'].values.astype(int)==int(float(toi_names[0][3:]))]
        else:
            if len(self.planets)>1:
                print("Without using `check_toi_list`, we have to assume the TOIs provided here match the planets in the model, i.e. with periods:",{npl:self.planets[npl]['period'] for npl in self.planets})
            for t in range(len(toi_names)):
                if toi_names[t].find(".")==-1:
                    toi_names[t]+='.01'
        
        for filetype in ["ExoFopTimeSeries","ExoFopFiles","ExoFopComments","ExoFopPlanetParameters"]:
            if not os.path.exists(os.path.join(upload_loc,filetype)):
                os.mkdir(os.path.join(upload_loc,filetype))

        table = self.save_trace_summary(returndf=True) if table is None else table
                
        for npl,pl in enumerate(self.planets):
            p=table.loc['P_'+pl,'mean']
            t0=table.loc['t0_'+pl,'mean']
            dur=table.loc['tdur_'+pl,'mean']
            
            if check_toi_list:
                this_toi=df.iloc[np.argmin((p-df['Period (days)'])**2)]['TOI']
                if type(this_toi)!=str or this_toi[:3]!="TOI":
                    this_toi="TOI"+str(this_toi)
                self.planets[pl]['toi']=this_toi
            else:
                this_toi=npl
                self.planets[pl]['toi']=toi_names[npl]
            
            #Checking for each planet and each filekey the coverage (i.e. in-transit fraction):
            for fk in self.cheops_filekeys:
                ph=((self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'].values-t0-0.5*p)%p-0.5*p)/dur
                #print(self.planets[pl])
                if np.any(ph<=-0.3)&np.any(ph>=0.30):
                    self.planets[pl][fk+"_covindex"]=2#"Full" #We have both first 20% and last 20%
                    self.planets[pl][fk+"_coverage"]="Full"
                    #print("Full")
                elif not np.any(abs(ph)<0.5):
                    self.planets[pl][fk+"_covindex"]=0#"Out of Transit" #We have nothing in-transit
                    self.planets[pl][fk+"_coverage"]="Out of Transit"
                    #print("Out")
                elif np.all(ph>-0.30):
                    self.planets[pl][fk+"_covindex"]=1.1#"Egress" #We only have post-ingress
                    self.planets[pl][fk+"_coverage"]="Egress"
                    #print("Egress")
                elif np.all(ph<0.30):
                    self.planets[pl][fk+"_covindex"]=1.2 #We only have pre-egress
                    self.planets[pl][fk+"_coverage"]="Ingress"
                    #print("Ingress")
                #print(self.planets[pl])

        target_pl={}
        maxcovtoiname={}
        maxcoverage={}
        for fk in self.cheops_filekeys:
            cov_values=np.array([self.planets[pl][fk+"_covindex"] for pl in self.planets])
            target_pl[fk]=list(self.planets.keys())[np.argmax(cov_values)]
            maxcovtoiname[fk]=self.planets[target_pl[fk]]['toi']
            maxcoverage[fk]=self.planets[target_pl[fk]][fk+"_coverage"]
        
        print(tsnotes,type(tsnotes),base_toi_name,type(base_toi_name),desc,type(desc))
        tsnotes+="Public \"CHATEAUX\" filler progam pilot observations of "+base_toi_name+" performed by the CHEOPS GTO. "
        allnotes={fk:tsnotes for fk in self.cheops_filekeys}

        #Adding the specific TOI observed for each observing note:
        obs_starts={}
        for fk in self.cheops_filekeys:
            if np.sum([self.planets[pl][fk+"_covindex"]>0 for pl in self.planets])>1:
                allnotes[fk]+=" Multiple planets seen in this observations - "
                allnotes[fk]+=" & ".join([self.planets[pl]['toi'] for pl in self.planets if self.planets[pl][fk+"_covindex"]>0])
            elif not np.all([self.planets[pl][fk+"_covindex"]==0 for pl in self.planets]):
                allnotes[fk]+=" No planet in-transit during this data"
            elif np.sum([self.planets[pl][fk+"_covindex"]>0 for pl in self.planets])==1:
                allnotes[fk]+=maxcovtoiname[fk]+" seen to transit with coverage: "+maxcoverage[fk]+". "
        
            obs_starts[fk]=Time(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'].values[0],format='jd')
        n_tag=0
        data_tag=Time.now().isot[:10].replace("-","")+"_"+username+"_"+base_toi_name+"CheopsGtoChateaux_"+str(n_tag)
        file_tags={}
        desc_infos={}
        num=str(len(glob.glob(os.path.join(upload_loc,"ExoFopFiles",initials.lower()+Time.now().isot[:10].replace("-","")+"*.txt")))+1)
        desc_name=initials.lower()+Time.now().isot[:10].replace("-","")+"-"+num
        with open(os.path.join(upload_loc,"ExoFopFiles",desc_name+'.txt'),'w') as desctxt:
            for f in files:
                if f=="cheops_plots":
                    for fk in self.cheops_filekeys:
                        #print(obs_starts[fk].isot)
                        date=obs_starts[fk].isot.replace(":","").replace("/","").replace("-","")
                        #Creating a plot for each CHEOPS filekey
                        self.plot_cheops(fk=fk)
                        #self.unq_name+"_"+fk+"_cheops_plots."+savetype
                        print(os.path.join(self.save_file_loc,self.name,self.unq_name+"_"+fk+"_"+f+"*"),
                                glob.glob(os.path.join(self.save_file_loc,self.name,self.unq_name+"_"+fk+"_"+f+"*")))
                        filename=glob.glob(os.path.join(self.save_file_loc,self.name,self.unq_name+"_"+fk+"_"+f+"*"))[0]
                        print(maxcovtoiname,str(maxcovtoiname[fk])+"L-"+initials.lower()+date+"cheops-gto-chateaux."+filename[-3:])
                        file_tags[f+"_"+fk] = str(maxcovtoiname[fk])+"L-"+initials.lower()+date+"cheops-gto-chateaux."+filename[-3:]
                        #Copying file to upload location
                        #print(upload_loc+file_tags[f])
                        os.system("cp "+filename+" "+os.path.join(upload_loc,"ExoFopFiles",file_tags[f+"_"+fk]))
                        if desc==None and len(self.cheops_assess_statements[fk])>2:
                            desc_infos[f+"_"+fk] = file_tags[f+"_"+fk]+"|"+str(data_tag)+"|tfopwg|12|"+",".join([self.cheops_assess_statements[fk][0]]+self.cheops_assess_statements[fk][2:])+"_CheopsPIPEPhotometry_"+fk
                        elif desc==None:
                            desc_infos[f+"_"+fk]=file_tags[f+"_"+fk]+"|"+str(data_tag)+"|tfopwg|12|"+self.cheops_assess_statements[fk][0]+"_CheopsPIPEPhotometry_"+fk
                        else:
                            desc_infos[f+"_"+fk]=file_tags[f+"_"+fk]+"|"+str(data_tag)+"|tfopwg|12|"+desc+"_CheopsPIPEPhotometry_"+fk
                        #desc+"_CheopsPIPEPhotometry_"+fk
                        desctxt.write(desc_infos[f+"_"+fk])
                else:
                    #Creating a single plot for all CHEOPS filekeys
                    #print(os.path.join(self.save_file_loc,self.name,self.unq_name+"_"+f+".*"))
                    filename=glob.glob(os.path.join(self.save_file_loc,self.name,self.unq_name+"_"+f+".*"))[0]
                    file_tags[f] = base_toi_name+"L-"+initials.lower()+str(np.round(obs_starts[fk].jd-2400000,5))+"cheops-gto-chateaux.png"
                    #Copying file to upload location
                    #print(upload_loc+file_tags[f])
                    os.system("cp "+filename+" "+os.path.join(upload_loc,"ExoFopFiles",file_tags[f]))
                    if desc==None and len(self.cheops_assess_statements[fk])>2:
                        desc_infos[f] = file_tags[f]+"|"+str(data_tag)+"|tfopwg|12|"+",".join([self.cheops_assess_statements[fk][0] for fk in self.cheops_filekeys])+", ".join([", ".join(self.cheops_assess_statements[fk][2:]) for fk in self.cheops_filekeys])+"_CheopsAndTess_phasefolded"
                    elif desc==None:
                        desc_infos[f]=file_tags[f]+"|"+str(data_tag)+"|tfopwg|12|"+",".join([self.cheops_assess_statements[fk][0] for fk in self.cheops_filekeys])+"_CheopsAndTess_phasefolded"
                    else:
                        desc_infos[f]=file_tags[f]+"|"+str(data_tag)+"|tfopwg|12|"+desc+"_CheopsAndTess_phasefolded"

                    #desc_infos[f] = file_tags[f]+"|"+str(data_tag)+"|tfopwg|12|"+desc+"_CheopsAndTess_phasefolded"
                    desctxt.write(desc_infos[f])

        #phot_tag = obs_start.isot[:10].replace('/','')+'_'+username+'_'+desc.replace(' ','_')+'_'
        #Using the TOI with maximum coverage as our name in the photometric file:
        for fk in self.cheops_filekeys:
            dat={'Target':str(maxcovtoiname[fk]),
            'Tel':'CHEOPS',
            'TelSize':'0.30',
            'Camera':'CHEOPS',
            'Filter':'CHEOPS',
            'FiltCent':'0.715',
            'FiltWidth':'0.77',
            'FiltUnits':'microns',
            'Pixscale':'1',
            'PSF':'30',
            'PhotApRad':'',
            'ObsDate':obs_starts[fk].isot.replace('T',' '),
            'ObsDur':str(np.round(np.ptp(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'].values)*1440,1)),
            'ObsNum':str(len(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'].values)),
            'ObsType':'Continuous',
            'TransCov':maxcoverage[fk],
            'DeltaMag':'',
            'Tag':data_tag,
            'Group':'tfopwg',
            'Notes':plnotes}
            #saving this timeseries file:
            num=str(len(os.path.join(upload_loc,"ExoFopTimeSeries","obs-timeseries-"+obs_starts[fk].isot[:10].replace("-","")+"-*.txt"))+1)
            with open(os.path.join(upload_loc,"ExoFopTimeSeries","obs-timeseries-"+obs_starts[fk].isot[:10].replace("-","")+"-"+num+".txt"),"w") as timeseriesfile:
                timeseriesfile.write("|".join(list(dat.values())))
        #pd.Series(dat).T.to_csv(os.path.join(upload_loc,"obs-timeseries-"+obs_start.isot[:10].replace("-","")+"-001.txt"),sep="|")
        #Tel|TelSize|Camera|Filter|FiltCent|FiltWidth|FiltUnits|Pixscale|PSF|PhotApRad|ObsDate|ObsDur|ObsNum|ObsType|TransCov|DeltaMag|Tag|Group|Notes}
        #TIC2021557|FLWO|1.2|KeplerCam|V||||6.025|9.85|8|2018-04-18|10|100|Continuous|Full|2.3|261|myGroup|suspected NEB

        #New Planet Parameters:
        plhd="target|flag|disp|period|period_unc|epoch|epoch_unc|depth|depth_unc|duration|duration_unc|inc|inc_unc|imp|imp_unc|r_planet|r_planet_err|ar_star|ar_star_err|radius|radius_unc|\
              mass|mass_unc|temp|temp_unc|insol|insol_unc|dens|dens_unc|sma|sma_unc|ecc|ecc_unc|arg_peri|arg_peri_err|time_peri|time_peri_unc|vsa|vsa_unc|tag|group|prop_period|notes"
        plandat=[]
        for pl in self.planets:
            npl=list(self.planets.keys()).index(pl)
            nlcs=np.sum([self.planets[pl][fk+"_covindex"]>0 for fk in self.cheops_filekeys])
            if nlcs==0:
                newnote="Only used TESS photometry;"
            else:
                newnote="Combined TESS-CHEOPS model with "+str(int(nlcs))+" chateaux obs ("+",".join([self.planets[pl][fk+"_coverage"][:4] for fk in self.cheops_filekeys if self.planets[pl][fk+"_covindex"]>0])+")"
            #data_tag="20220726_osborn_toi4417cheopsgtochateauxfull_0"
            plandat+=[[str(self.planets[pl]['toi']),"newparams","PC",table.loc["P_"+pl,"mean"],table.loc["P_"+pl,"sd"],table.loc["t0_"+pl,"mean"],
                     table.loc["t0_"+pl,"sd"],table.loc["ror_"+pl,"mean"]**2*1e6,table.loc["ror_"+pl,"mean"]**2*(2*table.loc["ror_"+pl,"sd"]/table.loc["ror_"+pl,"mean"])*1e6,
                     24*table.loc["tdur_"+pl,"mean"],24*table.loc["tdur_"+pl,"sd"],180/np.pi*np.arccos(table.loc["b_"+pl,"mean"]/table.loc["a_Rs_"+pl,"mean"]),"",
                     table.loc["b_"+pl,"mean"],table.loc["b_"+pl,"sd"],table.loc["ror_"+pl,"mean"],table.loc["ror_"+pl,"sd"],
                     table.loc["a_Rs_"+pl,"mean"],table.loc["a_Rs_"+pl,"sd"],table.loc["rpl_"+pl,"mean"],table.loc["rpl_"+pl,"sd"],
                     "","",table.loc["Tsurf_p_"+pl,"mean"],table.loc["Tsurf_p_"+pl,"sd"],table.loc["S_in_"+pl,"mean"]/1370,table.loc["S_in_"+pl,"sd"]/1370,
                     table.loc["Ms","mean"]/table.loc["Rs","mean"]**3,"",table.loc["sma_"+pl,"mean"],table.loc["sma_"+pl,"sd"],
                     "","","","","","","","",data_tag,"tfopwg",0,newnote+plnotes]]
            num=str(len(glob.glob(os.path.join(upload_loc,"ExoFopPlanetParameters","params_planet_"+Time.now().isot[:10].replace("-","")+"_*.txt")))+1).zfill(3)
        with open(os.path.join(upload_loc,"ExoFopPlanetParameters","params_planet_"+Time.now().isot[:10].replace("-","")+"_"+num+".txt"),"w") as plandatfile:
            for p in plandat:
                outstr="|".join(list(np.array(p).astype(str)))
                assert(len(outstr.split("|"))==len(plhd.split("|")))
                plandatfile.write(outstr+"\n")

    def MakeLatexMacros(self,):
        print("TBD")

    def MakeLatexAllParamTable(self,use_macros=True):
        print("TBD")

    def MakePlanetPropertiesTable(self,DR2ID,data_ref="",table=None):
        """
         Make a table of planet properties for each planet.
         
         Args:
         	 DR2ID: Gaia DR2 ID
         	 data_ref: Data reference to use for data analysis
         	 table: If None ( default ) the trace summary table is used (save_trace_summary)
        """
        assert hasattr(self,'trace'), "Must have already sampled the model"
       
        table = self.save_trace_summary(returndf=True) if table is None else table
        allpl_dats=[]
        for pl in self.planets:
            pldic={"obj_id_catname":self.name+" "+pl,"obj_id_gaiadr2":DR2ID,
                   "obj_trans_t0_bjd":table.loc["t0_"+pl,"mean"],"obj_trans_t0_bjd_err":table.loc["t0_"+pl,"sd"],
                   "obj_trans_period_days":table.loc["P_"+pl,"mean"],"obj_trans_period_days_err":table.loc["P_"+pl,"sd"]}
            if self.assume_circ:
                pldic.update({"obj_trans_ecosw":0,"obj_trans_ecosw_err":0,"obj_trans_esinw":0,"obj_trans_esinw_err":0})
            else:
                pldic.update({"obj_trans_ecosw":np.nanmedian(self.trace["ecc_"+pl]*np.cos(self.trace["omega_"+pl])),
                            "obj_trans_ecosw_err":np.nanstd(self.trace["ecc_"+pl]*np.cos(self.trace["omega_"+pl])),
                            "obj_trans_esinw":np.nanmedian(self.trace["ecc_"+pl]*np.sin(self.trace["omega_"+pl])),
                            "obj_trans_esinw_err":np.nanstd(self.trace["ecc_"+pl]*np.sin(self.trace["omega_"+pl]))})
            pldic.update({"obj_trans_depth_ppm":1e6*table.loc["ror_"+pl,"mean"]**2,"obj_trans_depth_ppm_err":table.loc["ror_"+pl,"mean"]**2*2*table.loc["ror_"+pl,"sd"]/table.loc["ror_"+pl,"mean"],
                          "obj_trans_duration_days":table.loc["tdur_"+pl,"mean"],"obj_trans_duration_days":table.loc["tdur_"+pl,"sd"]})
            if hasattr(self, 'rvs'):
                pldic.update({'obj_rv_k_mps':table.loc['K_'+pl,'mean'],'obj_rv_k_mps_err':table.loc['K_'+pl,'sd']})
            else:
                pldic.update({'obj_rv_k_mps':0,'obj_rv_k_mps_err':0})
            pldic.update({'db_info_reference':data_ref,
                          'db_info_remarks':'chexoplanet-generated planet properties table.'})
            allpl_dats+=[pldic]
        colstr="obj_id_catname	obj_id_gaiadr2	obj_id_planet_catname	obj_trans_t0_bjd	obj_trans_t0_bjd_err	obj_trans_period_days	obj_trans_period_days_err	obj_trans_ecosw	obj_trans_ecosw_err	obj_trans_esinw	obj_trans_esinw_err	obj_trans_depth_ppm	obj_trans_depth_ppm_err	obj_trans_duration_days	obj_trans_duration_days_err	obj_rv_k_mps	obj_rv_k_mps_err	db_info_reference	db_info_remarks".split("\t")
        with open(os.path.join(self.save_file_loc,self.name,self.unq_name+"_"+mod+"_PlanetProperties.tsv"),"w") as plandatfile:
            for dicdat in allpl_dats:
                outstr="\t".join([pldic[col] for col in colstr])
                assert(len(outstr.split("|"))==len(colstr.split("\t")))
                plandatfile.write(outstr+"\n")
        for mod in self.models_out:
            self.models_out[mod].to_csv(os.path.join(self.save_file_loc,self.name,self.unq_name+"_"+mod+"_timeseries.csv"))

    
    def MakeModelDescription(self,use_macros=True,GTO_prog=56):
        descr=f"{{\\it CHEOPS}} is a 30-cm ESA space telescope which was launched into a sun-synchronous low-Earth orbit in 2019 and focusses on transiting exoplanet science \\citep{{Benz2021}}."
        
        n_occ=['one','two','three','four','five','six','seven'][len(self.cheops_filekeys)-1]
        occ_desc="on "+n_occ+" occasions " if n_occ!='one' else ""
        if GTO_prog==56:
            descr+=f"{self.name} was observed {occ_desc}through {{\it CHEOPS}} GTO program \\#56 \"CHATEAUX: CHeops And TEss vAlidate Unconfirmed eXoplanets\". "
            descr+="CHATEAUX is a filler program designed to observe the transits of TESS candidate planets using highly flexible and short-duration observations. "
            descr+="We cut and filter TOIs to remove likely false positives, and to make sure that only those for which Cheops obervations would be beneficial \& superior to e.g. ground-based photometry are observed. "
            descr+="In order to ensure the most precise ephemerides, we re-fit all targets using all available {\\it TESS} data. "
        elif GTO_prog==48:
            descr+=f"{self.name} was observed {occ_desc}through {{\it CHEOPS}} GTO program \\#48 \"Duos: Recovering long period duo-transiting planets with CHEOPS\" which is specifically designed to target the period aliases of long-period planets observed by TESS. "

        exptimes="The visits had average exposure times" if n_occ!='one' else "The visit had an exposure time"
        avexp=str(int(np.round(np.nanmedian([self.descr_dict[fk]['cad'] for fk in self.cheops_filekeys])*86400)))
        durs=str(np.round(self.descr_dict[self.cheops_filekeys[0]]['len']*24,1)) if n_occ=='one' or np.nanstd([self.descr_dict[fk]['len']*24 for fk in self.cheops_filekeys])<1 else "between "+str(np.round(np.max([self.descr_dict[fk]['len'] for fk in self.cheops_filekeys()])*24,1))+" and "+str(np.round(np.min([self.descr_dict[fk]['len'] for fk in self.cheops_filekeys()])*24))
        descr+=f"{exptimes} of {avexp} and lasted {durs}hrs. "
        descr+="We downloaded the Cheops data from \\texttt{DACE} \citep{Buchschacher2015}. "
        if np.any([self.descr_dict[fk]['src']=='PIPE' for fk in self.cheops_filekeys]):
            descr+="In order to maximise the photometric precision, we used the \\texttt{PIPE} module \\footnote{\\url{https://github.com/alphapsa/pipe}} which performs fits to the image sub-arrays using the measured point-spread function of Cheops. "
        if n_occ=='one':
            descr+="The visit achieved an RMS of "+str(int(np.round(self.descr_dict[self.cheops_filekeys[0]]['rms'])))+"ppm. "
        else:
            descr+="The mean RMS across all visits was "+str(int(np.round(np.nanmedian([self.descr_dict[fk]['rms'] for fk in self.cheops_filekeys]))))+"ppm. "
        if np.any([len(self.cheops_linear_decorrs[fk])>0 for fk in self.cheops_filekeys]):
            descr+="Due to its position in low-Earth orbit, and the rotation of the field during each 98min orbit, {\\it CHEOPS} data needs to be detrended using various metadata to achieve maximum photometric precision. "
            descr+="For each {{\it CHEOPS}} visit we perform a fit using an \\texttt{exoplanet} transit model \citep{exoplanet:joss} informed by the expected transit ephemeris as well as using all available decorrelation vectors. "
            descr+="This includes linear and quadratic terms for the roll angle $\\Phi$, trigonometic functions of the roll angle $\\cos{\\Phi}$ and $\\sin{\\Phi}$, centroid positions $x$ \\& $y$, and estimation of background contamination, \\& smear done by the {\\it CHEOPS} data reduction pipeline), all of which are normalised to have $\\mu=0.0$ and $\\sigma=1.0$."
            if self.use_bayes_fact:
                descr+="We then sampled this {{\it CHEOPS}}-only model using \\texttt{PyMC3} and calculated which detrending parameters improved the fit taking all parameters with a Bayes Factors less than "+str(np.round(self.signif_thresh,1))+". "
            elif self.use_signif:
                descr+="We then sampled this {{\it CHEOPS}}-only model using \\texttt{PyMC3} and calculated which detrending parameters improved the fit taking all parameters with significant non-zero correlation coefficients (i.e. $>"+str(np.round(self.signif_thresh,1))+"\\sigma$ from 0). "
            descr+="\nWe perform two models - one a comparison with and without a transit model to the {{\it CHEOPS}} data to assess the presence of a transit in the photometry, and a second including available TESS data in order to assess the improvement in radius and ephemeris precision."
            descr+="Both models used \\texttt{PyMC3} and \\texttt{exoplanet} and included the TOI parameters (e.g. ephemeris) as priors. "
            if self.fit_phi_gp:
                descr+="We also included a \\texttt{celerite} Gaussian Process model \\citep{celerite} to fit variations in flux as a function of roll angle not removed by the decorrelations. "
            descr+="The resulting best-fit model for the {{\it CHEOPS}} data is shown in Figure \ref{fig:cheops}.\n\n"
            descr+="\begin{figure}\n\includegraphics[width=\columnwidth]{"+self.unq_name+"_cheops_plots.png}\n\caption{"
            descr+="Cheops photometry. Upper panel shows raw and binned {{\it CHEOPS}} photometry, and the modelled flux variations due to decorrelation "
            if self.fit_phi_gp: descr+="and Gaussian process "
            descr+="models offset above, with a best-fit line as well as 1- \& 2-$\\sigma$ regions. Lower panel shows the detrended {{\it CHEOPS}} photometry as well as the best-fit line as well as 1- \& 2-$\\sigma$ regions of a combined {\\it TESS} and {\\it CHEOPS} transit model."
            descr+="}\n\label{fig:CheopsOther}\n\end{figure}\n\n"
            descr+=""

"""
@INPROCEEDINGS{Buchschacher2015,
       author = {{Buchschacher}, N. and {S{\'e}gransan}, D. and {Udry}, S. and {D{\'\i}az}, R.},
        title = "{Data and Analysis Center for Exoplanets}",
    booktitle = {Astronomical Data Analysis Software an Systems XXIV (ADASS XXIV)},
         year = 2015,
       editor = {{Taylor}, A.~R. and {Rosolowsky}, E.},
       series = {Astronomical Society of the Pacific Conference Series},
       volume = {495},
        month = sep,
        pages = {7},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2015ASPC..495....7B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@ARTICLE{Benz2021,
       author = {{Benz}, W. and {Broeg}, C. and {Fortier}, A. and {Rando}, N. and {Beck}, T. and {Beck}, M. and {Queloz}, D. and {Ehrenreich}, D. and {Maxted}, P.~F.~L. and {Isaak}, K.~G. and {Billot}, N. and {Alibert}, Y. and {Alonso}, R. and {Ant{\'o}nio}, C. and {Asquier}, J. and {Bandy}, T. and {B{\'a}rczy}, T. and {Barrado}, D. and {Barros}, S.~C.~C. and {Baumjohann}, W. and {Bekkelien}, A. and {Bergomi}, M. and {Biondi}, F. and {Bonfils}, X. and {Borsato}, L. and {Brandeker}, A. and {Busch}, M. -D. and {Cabrera}, J. and {Cessa}, V. and {Charnoz}, S. and {Chazelas}, B. and {Collier Cameron}, A. and {Corral Van Damme}, C. and {Cortes}, D. and {Davies}, M.~B. and {Deleuil}, M. and {Deline}, A. and {Delrez}, L. and {Demangeon}, O. and {Demory}, B.~O. and {Erikson}, A. and {Farinato}, J. and {Fossati}, L. and {Fridlund}, M. and {Futyan}, D. and {Gandolfi}, D. and {Garcia Munoz}, A. and {Gillon}, M. and {Guterman}, P. and {Gutierrez}, A. and {Hasiba}, J. and {Heng}, K. and {Hernandez}, E. and {Hoyer}, S. and {Kiss}, L.~L. and {Kovacs}, Z. and {Kuntzer}, T. and {Laskar}, J. and {Lecavelier des Etangs}, A. and {Lendl}, M. and {L{\'o}pez}, A. and {Lora}, I. and {Lovis}, C. and {L{\"u}ftinger}, T. and {Magrin}, D. and {Malvasio}, L. and {Marafatto}, L. and {Michaelis}, H. and {de Miguel}, D. and {Modrego}, D. and {Munari}, M. and {Nascimbeni}, V. and {Olofsson}, G. and {Ottacher}, H. and {Ottensamer}, R. and {Pagano}, I. and {Palacios}, R. and {Pall{\'e}}, E. and {Peter}, G. and {Piazza}, D. and {Piotto}, G. and {Pizarro}, A. and {Pollaco}, D. and {Ragazzoni}, R. and {Ratti}, F. and {Rauer}, H. and {Ribas}, I. and {Rieder}, M. and {Rohlfs}, R. and {Safa}, F. and {Salatti}, M. and {Santos}, N.~C. and {Scandariato}, G. and {S{\'e}gransan}, D. and {Simon}, A.~E. and {Smith}, A.~M.~S. and {Sordet}, M. and {Sousa}, S.~G. and {Steller}, M. and {Szab{\'o}}, G.~M. and {Szoke}, J. and {Thomas}, N. and {Tschentscher}, M. and {Udry}, S. and {Van Grootel}, V. and {Viotto}, V. and {Walter}, I. and {Walton}, N.~A. and {Wildi}, F. and {Wolter}, D.},
        title = "{The CHEOPS mission}",
      journal = {Experimental Astronomy},
     keywords = {Exoplanets, CHEOPS, Small mission, High-precision transit photometry, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics},
         year = 2021,
        month = feb,
       volume = {51},
       number = {1},
        pages = {109-151},
          doi = {10.1007/s10686-020-09679-4},
archivePrefix = {arXiv},
       eprint = {2009.11633},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ExA....51..109B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@article{exoplanet:joss,
       author = {{Foreman-Mackey}, Daniel and {Luger}, Rodrigo and {Agol}, Eric
                and {Barclay}, Thomas and {Bouma}, Luke G. and {Brandt},
                Timothy D. and {Czekala}, Ian and {David}, Trevor J. and
                {Dong}, Jiayin and {Gilbert}, Emily A. and {Gordon}, Tyler A.
                and {Hedges}, Christina and {Hey}, Daniel R. and {Morris},
                Brett M. and {Price-Whelan}, Adrian M. and {Savel}, Arjun B.},
        title = "{exoplanet: Gradient-based probabilistic inference for
                  exoplanet data \& other astronomical time series}",
      journal = {arXiv e-prints},
         year = 2021,
        month = may,
          eid = {arXiv:2105.01994},
        pages = {arXiv:2105.01994},
archivePrefix = {arXiv},
       eprint = {2105.01994},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210501994F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@article{celerite,
    author = {{Foreman-Mackey}, D. and {Agol}, E. and {Angus}, R. and
              {Ambikasaran}, S.},
     title = {Fast and scalable Gaussian process modeling
              with applications to astronomical time series},
      year = {2017},
   journal = {ArXiv},
       url = {https://arxiv.org/abs/1703.09710}
}

"""