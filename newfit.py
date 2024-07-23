import exoplanet as xo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter    
import pandas as pd

from astropy.io import fits
import astropy.units as u
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord, get_body

import pickle
import os.path
from datetime import date
import os
import glob
import time
import arviz as az
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

#Setting up logging - INFO level stuff to console and DEBUG level stuff to file:
import logging 
# logging.getLogger("filelock").setLevel(logging.ERROR)
# logging.getLogger("theano").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.WARNING)

import sys
floattype=np.float64
tablepath = os.path.join(os.path.dirname(__file__),'tables')

import pymc as pm
import pymc_ext as pmx
from celerite2.pymc import terms as pymc_terms
import celerite2.pymc

from .tools import *

class chexo_model():
    """Fit Cheops with Exoplanet core model class"""

    def __init__(self, targetname, overwrite=True, radec=None, comment="", **kwargs):
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
                       'n_cores':6,             # Number of cores with which to sample
                       'use_bayes_fact':True,   # Determine the detrending factors to use with a Bayes Factor
                       'use_signif':False,      # Determine the detrending factors to use by simply selecting those with significant non-zero coefficients
                       'signif_thresh':1.25,    # Threshold for detrending parameters in sigma
                       'use_multinest':False,   # use_multinest - bool - currently not supported
                       'use_pymc':True,         # use_pymc - bool
                       'use_PIPE':True,         # use_PIPE - bool
                       'assume_circ':False,     # assume_circ - bool - Assume circular orbits (no ecc & omega)?
                       'timing_sd_durs':0.33,   # timing_sd_durs - float - The standard deviation to use (in units of transit duration) when setting out timing priors
                       'fit_ttvs':False,        # Fit a TTVorbit exoplanet model which searches for TTVs
                       'split_periods':None,    # Fit for multiple split periods. Input must be None or a dict matching mod.planets with grouped indexes for those transits to group
                       'ttv_prior':'Normal',    # What prior to have for individual transit times. Possibilities: "Normal","Uniform","BoundNormal"
                       'fit_phi_gp':False,      # fit_phi_gp - bool - co-fit a GP to the roll angle.
                       'fit_phi_spline':True,   # fit_phi_spline - bool - co-fit a spline model to the roll angle
                       'spline_bkpt_cad':9.,    # spline_bkpt_cad - float - The spline breakpoint cadence in degrees. Default is 9deg
                       'spline_order':3.,       # spline_order - int - Thespline order. Defaults to 3 (cubic)
                       'phi_model_type':"common",# phi_model_type - str - How to fit the same roll angle GP trend. Either "individual" (different model for each visit), "common" (same model for each visit), or "split_2" (different models for each N season); formerly common_phi_model
                       'ecc_prior':'auto',      # ecc_prior - string - 'uniform', 'kipping' or 'vaneylen'. If 'auto' we decide based on multiplicity
                       'tight_depth_prior':False,# tight_depth_prior - bool - Whether to tightly constrain depth to the observed value. May be useful in e.g. TTV fits with few transits which are 
                       'npoly_rv':2,            # npoly_rv - int - order of polynomial fit to RVs
                       'rv_mass_prior':'logK',  # rv_mass_prior - str - What mass prior to use. "logK" = normal prior on log amplitude of K, "popMp" = population-derived prior on logMp, "K" simple normal prior on K.
                       'spar_param':'Mstar',    # spar_param - str - The stellar parameter to use when modelling. Either Mstar, logg or rhostar
                       'spar_prior':'constr',   # spar_prior - str - The prior to use on the second stellar parameter. Either constr, loose of logloose
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
        #Setting up logging
        self.init_logging()

        self.percentiles={'-2sig':2.2750132, '-1sig':15.8655254, 'med':50., '+1sig':84.1344746, '+2sig':97.7249868}
    
    def update(self,**kwargs):
        """Update global parameters
        """
        #Updating settings
        for param in kwargs:
            if param in self.defaults:
                setattr(self,param,kwargs[param])
        if self.save_file_loc is None:
            self.save_file_loc=os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(self.save_file_loc,self.name.replace(" ","_"))):
            os.mkdir(os.path.join(self.save_file_loc,self.name.replace(" ","_")))
        if not os.path.exists(os.path.join(self.save_file_loc,self.name.replace(" ","_"),'logs')):
            os.mkdir(os.path.join(self.save_file_loc,self.name.replace(" ","_"),"logs"))
        bools=['debug','load_from_file','fit_gp','fit_flat','train_gp','cut_oot','bin_oot','pred_all','use_bayes_fact','use_signif',
               'use_multinest','use_pymc','assume_circ','fit_ttvs','fit_phi_gp','fit_phi_spline',
               'constrain_lds','fit_contam','tight_depth_prior']
        boolstr=''
        for i in bools:
            boolstr+=str(int(getattr(self,i)))
            
        nonbools=['flat_knotdist','cut_distance','mask_distance','bin_size','signif_thresh','ecc_prior','npoly_rv','ld_mult','timing_sd_durs','rv_mass_prior','spline_order','spline_bkpt_cad','phi_model_type','spar_param','spar_prior']
        nonboolstrs=[]
        for i in nonbools:
            nonboolstrs+=[str(getattr(self,i)) if len(str(getattr(self,i)))<5 else str(getattr(self,i))[:5]]
        comm="" if self.comment is None else "_"+self.comment
        self.unq_name=self.name.replace(" ","_")+"_"+date.today().strftime("%Y%m%d")+"_"+str(int(boolstr, 2))+"_"+"_".join(nonboolstrs)+comm

    def init_logging(self):
        self.logger = logging.getLogger(__name__)
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)

        fileHandler = logging.FileHandler(os.path.join(self.save_file_loc,self.name.replace(" ","_"),"logs",self.comment.replace(" ","_")+"_model.log"))
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)
        
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Initialised logger")
        self.logger.debug("Checking if logger prints to file or stream")

    def get_tess(self,tic=None,**kwargs):
        """Automatically download archival photometric (i.e. TESS) data using MonoTools.lightcurve
        This also automatically initialises the stellar parameters from the TIC catalogue info.
        tic (int, optional): TIC ID. If not present, we can get it from the TOI list
        """
        try:
            from . import lightcurve, tools
        except:
            raise ImportError("Cannot import MonoTools. Check it is installed, or initialise the class with `get_tess=False` and add a lightcurve with `mod.add_lc`")
        lcloc=os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.name.replace(" ","_")+'_lightcurves.pkl.gz')
        #First we need the TIC ID
        if tic is not None:
            self.monotools_lc=lightcurve.multilc(int(tic),'tess',savefileloc=lcloc, **kwargs)
        elif "TIC" in self.name:
            self.monotools_lc=lightcurve.multilc(int(self.name.replace('-','')[3:]),'tess',savefileloc=lcloc,**kwargs)
        elif "TOI" in self.name:
            self.filter_TOI()
            self.monotools_lc=lightcurve.multilc(self.init_toi_data.iloc[0]['TIC ID'],'tess',savefileloc=lcloc,**kwargs)
        if not hasattr(self,'radec') and hasattr(self.monotools_lc,'radec'):
            self.radec=self.monotools_lc.radec
        self.monotools_lc.sort_timeseries()
        
        Rstar, Teff, logg = starpars_from_MonoTools_lc(self.monotools_lc)
        self.init_starpars(Rstar=Rstar,Teff=Teff,logg=logg)
        
        #Getting cadences for each mission and adding them to the model:
        lc_dic_flipped={v: k for k, v in lightcurve.tools.lc_dic.items()}
        all_missions=np.unique([cad.split("_")[0] for cad in self.monotools_lc.cadence_list])
        misss=np.array([c[:2] for c in self.monotools_lc.cadence])
        for unq_miss_id in all_missions:
            self.logger.info("Lightcurves found with mission ID:"+unq_miss_id)
            cad_ix=(self.monotools_lc.mask)&(misss==unq_miss_id)
            self.add_lc(self.monotools_lc.time[cad_ix]+2457000,self.monotools_lc.flux[cad_ix],self.monotools_lc.flux_err[cad_ix],source=lc_dic_flipped[unq_miss_id])
        self.monotools_lc.plot()

    def get_cheops(self, do_search=True, catname=None, n_prog=None, distthresh=3, download=True, use_PIPE=True, fks=None, start_date=None,end_date=None, vmag=None, **kwargs):
        """Automatically download CHEOPS data using Dace.

        Args:
        catname - Dace catalogue name if different from the intiialised object name. Default = None
        n_prog - Number of GTO programme. Default = None
        distthresh - Distance in arcsec to call a target a match to the observed CHEOPS observation table. Default = 3
        download - Whether to download the data or simply load/use on the fly. Default = True
        use_PIPE - Whether to use PIPE for the CHEOPS data. Default = True
        fks - List of CHEOPS filekeys to use. Default = None
        start_date - Date from which to access CHEOPS data
        end_date - Date until when to access CHEOPS data
        vmag - If we are only taking the data from file, we need the vmag for PIPE
        """
        self.update(**kwargs)

        catname= self.name if catname is None else catname
        if do_search and fks is not None:
            #Checking the required filekeys aren't already downloaded...
            localfks=[f.split("/")[-1] for f in glob.glob(os.path.join(self.save_file_loc,self.name.replace(" ","_"),"PR??????_????????_V0?00"))]
            if np.all(np.isin(fks,localfks)):
                do_search=False
        if do_search:
            these_cheops_obs=[]
            n_try=0
            while n_try<3 and len(these_cheops_obs)==0:
                if n_try>0:
                    self.logger.warning("DACE non-responsive... waiting 15secs. (NB - try logging in on DACE)")
                    time.sleep(15)
                try:
                    from dace_query.cheops import Cheops
                except:
                    raise ImportError("Cannot import Dace. Check it is installed, or initialise the class with `get_cheops=False` and add a lightcurve with `mod.add_cheops_lc`")
                self.logger.debug("Attempting to get objects via the catalogue name")
                if fks is None:
                    these_cheops_obs=pd.DataFrame(Cheops.query_database(limit=50,filters={"obj_id_catname":{"equal":[catname]},"file_key":{"contains":"_V030"}}))
                else:
                    these_cheops_obs=pd.DataFrame(Cheops.query_database(limit=50,filters={"file_key":{"equal":["CH_"+f if f[:2]=="PR" else f for f in fks]}}))
                try:
                    if len(these_cheops_obs)==0:
                        from dace_query.cheops import Cheops
                        raise ValueError("No objects returned for name="+catname)
                    if hasattr(self,'radec'):
                        obs_radecs=SkyCoord([rd.split(" / ")[0] for rd in these_cheops_obs['obj_pos_coordinates_hms_dms'].values], 
                                            [rd.split(" / ")[1] for rd in these_cheops_obs['obj_pos_coordinates_hms_dms'].values],
                                            unit=(u.hourangle,u.deg))
                        self.logger.debug("Checking separation between object in Dace catalogue and target RA/Dec "+str(self.radec.ra.deg)+" "+str(self.radec.dec.deg))
                        assert np.all(self.radec.separation(obs_radecs).arcsec<distthresh), "The RA/DEC of CHEOPS visits does not match the Ra/DEC included above"
                except:
                    self.logger.debug("Zero entries found when searching by name. Trying with coordinate")
                    assert hasattr(self,'radec'), "If indexing by name does not work, we must have an RA/Dec coordinate"
                    #Could not get it using the name, trying with the programme and the coordinates
                    if n_prog is not None:
                        self.logger.debug("Limiting search via programme ID; "+str(n_prog))
                        all_cheops_obs=pd.DataFrame(Cheops.query_database(limit=50,filters={"prog_id":{"contains":str(int(n_prog))},"file_key":{"contains":"_V030"}}))
                    else:
                        #Getting all data/all programmes
                        all_cheops_obs=pd.DataFrame(Cheops.query_database(limit=50,filters={"prog_id":{"contains":"CHEOPS"},"file_key":{"contains":"_V030"}}))
                    #Finding which target we have:
                    self.logger.debug("All CHEOPS observations found:")
                    self.logger.debug(all_cheops_obs)
                    all_radecs=SkyCoord([rd.split(" / ")[0] for rd in all_cheops_obs['obj_pos_coordinates_hms_dms'].values], 
                                        [rd.split(" / ")[1] for rd in all_cheops_obs['obj_pos_coordinates_hms_dms'].values],
                                        unit=(u.hourangle,u.deg))
                    these_cheops_obs=all_cheops_obs.loc[self.radec.separation(all_radecs).arcsec<distthresh]
                    self.logger.debug("Minimum CHEOPS object distance: "+str(np.min(self.radec.separation(all_radecs).arcsec))+" | new nearby objs:")
                    self.logger.debug(all_cheops_obs)
                n_try+=1
        else:
            #Purely getting the filekeys from the output directory and using those...
            localfks=[f.split("/")[-1] for f in glob.glob(os.path.join(self.save_file_loc,self.name.replace(" ","_"),"PR??????_????????_V0?00"))]
            these_cheops_obs=pd.DataFrame({'file_key':localfks})
            #these_cheops_obs=these_cheops_obs.loc[np.in1d(fks,these_cheops_obs),:]
        
        if fks is not None:
            fks=[f.replace("CH_","") for f in fks]
            these_cheops_obs=these_cheops_obs.loc[np.array([f.replace("CH_","") in fks for f in these_cheops_obs['file_key'].values])]
            self.logger.debug("filtering by pre-set filekeys: "+",".join(fks))
        #print(these_cheops_obs['file_key'].values)

        assert these_cheops_obs.shape[0]>0, "No matches found in the CHEOPS database. Are you logged in via your .dacerc file?"
        for fk in these_cheops_obs['file_key'].values:
            
            if "V020" in fk and fk.replace("V020","V030") in these_cheops_obs['file_key'].values:
                self.logger.debug("Skipping "+fk+" as there's a better/newer reduction available.")
                continue
            if 'date_mjd_start' in these_cheops_obs.columns:
                t_obs=Time(these_cheops_obs.loc[these_cheops_obs['file_key']==fk,'date_mjd_start'].values,format='mjd').jd
                if (end_date is not None and t_obs<end_date) or (start_date is not None and t_obs>start_date):
                    self.logger.debug("Skipping "+fk+" as the observation is not within the start/end date: "+str(start_date)+"-"+str(end_date))
                    continue
            
            ifk=fk[3:] if fk[:3]=="CH_" else fk
            if hasattr(self,'monotools_lc'):
                self.logger.debug("Adding CHEOPS lc for "+fk+"; using magnitude from MonoTools file")
                self.add_cheops_lc(filekey=ifk, fileloc=None, download=download, use_PIPE=use_PIPE, mag=self.monotools_lc.all_ids['tess']['data']['GAIAmag'], **kwargs)
            elif 'obj_mag_v' in these_cheops_obs.columns:
                self.logger.debug("Adding CHEOPS lc for "+fk+"; using magnitude from Dace catalogue")
                self.add_cheops_lc(filekey=ifk, fileloc=None, download=download, use_PIPE=use_PIPE, 
                                   mag=these_cheops_obs.loc[these_cheops_obs['file_key']==fk,'obj_mag_v'], **kwargs)
                                   #self.monotools_lc.all_ids['tess']['data'][`'GAIAmag']
            elif vmag is not None:
                self.logger.debug("Adding CHEOPS lc for "+fk+"; using magnitude from Dace catalogue")
                self.add_cheops_lc(filekey=ifk, fileloc=None, download=download, use_PIPE=use_PIPE,
                                   mag=vmag, **kwargs)
                                   #self.monotools_lc.all_ids['tess']['data'][`'GAIAmag']

            else:
                self.logger.debug("Adding CHEOPS lc for "+fk+"; without magnitude")
                self.add_cheops_lc(filekey=ifk, fileloc=None, download=download, use_PIPE=use_PIPE, **kwargs)
        if not use_PIPE:
            #Need to make sure all the DRP apertures are the same
            self.assess_cheops_drp_apertures()

    def filter_TOI(self,threshdist=3,**kwargs):
        """Load the TOI list (using `get_TOI`) and then find the specific case which either refers to the name given to this target, or to the RA/Dec (within threshdic arcsec)

        Args:
        threshdic - Threshhold in arcsecs below which to associate the target with a given TOI.
        """
        if not hasattr(self,'toi_cat'):
            self.get_TOI(**kwargs)
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

    def get_TOI(self,**kwargs):
        """Get TOI info. We either download the TOI catalogue from ExoFop, or if this was recently accessed, we load the TOI catalogue from the data/tables folder in cheoxplanet"""
        round_date=int(np.round(Time.now().jd,-1))
        
        if not os.path.exists(os.path.join(tablepath,"TOI_tab_jd_"+str(round_date)+".csv")):
            self.logger.debug("Downloading TOI catalogue from ExoFop")
            self.toi_cat=pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
            self.toi_cat['star_TOI']=self.toi_cat['TOI'].values.astype(int)
            self.toi_cat.to_csv(os.path.join(tablepath,"TOI_tab_jd_"+str(round_date)+".csv"))
        else:
            self.logger.debug("Accessing recent TOI catalogue from file (round date="+str(round_date)+")")
            self.toi_cat=pd.read_csv(os.path.join(tablepath,"TOI_tab_jd_"+str(round_date)+".csv"),index_col=0)

    def add_lc(self, time, flux, flux_err, mask=None, source='tess'):
        """Add photometric lightcurve to the model.
        
        Args:
        time - np.array - Time in BJD
        flux - np.array - Flux in ppt (normalised to zero)
        flux_err - np.array - Flux errors in ppt """
        if not hasattr(self,'lcs'):
            self.lcs={}
        self.logger.debug("Adding lightcurve to file from time,flux,flux_error")
        self.lcs[source]=pd.DataFrame({'time':time,'flux':flux,'flux_err':flux_err})
        self.lcs[source]['mask']=np.tile(True,len(time)) if mask is None else mask

    def run_PIPE(self,out_dir,fk,mag=None,overwrite=False,make_psf=False,binary=False,optimise_klim=True,use_past_optimsation=True,**kwargs):
        """
         Run Pipe to extract PSFs. This is a wrapper around the : py : func : ` ~psf. pipeline. PipeParam ` class and it's subarray extraction function
         
         Args:
         	 out_dir: directory to save the results
         	 fk: Filenames of the PSF subarray
         	 mag: Magnitude of the subarray in Hz
         	 overwrite: Whether to overwrite the files or not. If True they will be overwritten
             make_psf: Use PIPE to make a specific PSF for this target (long)
             binary: Whether there is a nearby contaminant within the PSF that we want to model using the BG-star-fits
             optimise_klim: Whether to optimise PIPE
             use_past_optimsation: Whether to use the optimised PIPE parameters found in a previous PIPE run. Default is True
         Returns: 
         	 A list of : py : class : ` ~psf. pipeline. PipelineParam `
        """
        self.logger.debug("Running PIPE on filekey "+fk)
        try:
            from pipe import PipeParam, PipeControl, config
        except:
            raise ImportError("PIPE not importable.")

        mag=10.5 if mag is None else mag

        out_dir=os.path.join(self.save_file_loc,self.name.replace(" ","_"))
        self.logger.debug(["PIPE data location:",out_dir,config.get_conf_paths()])
        pipe_refdataloc=config.get_conf_paths()[1]
        
        # folds = glob.glob(os.path.join(out_dir,fk,"Outdata","000??"))
        # folds = {int(pf[0].split('/')[-1]):pf for pf in folds[0] if len(glob.glob(os.path.join(pf,"*.fits")))>0}
        # n_max = list(folds.keys())[np.argmax(list(folds.keys()))]
        
        #Checking if we have an Outdata file but no PIPE outputs (in which case we delete)
        #print(os.path.exists(os.path.join(out_dir,fk,"Outdata")),os.path.join(out_dir,fk,"Outdata"),overwrite)
        if (os.path.exists(os.path.join(out_dir,fk,"Outdata","00000")) and len(glob.glob(os.path.join(out_dir,fk,"Outdata","00000","*.fits")))==0) or overwrite:
            self.logger.debug("Overwriting stored PIPE data as either overwrite=True or no fits files generated in previous PIPE run. Filekey="+fk)
            os.system("rm -r "+os.path.join(out_dir,fk,"Outdata"))
        #print(glob.glob(os.path.join(out_dir,fk,"CH_PR*SCI_COR_Lightcurve-RINF_V0?00.fits")))
        fitslist=glob.glob(os.path.join(out_dir,fk,"CH_PR*SCI_COR_Lightcurve-RINF_V0?00.fits"))
        self.logger.debug("Looking for lightcurve files in "+os.path.join(out_dir,fk,"CH_PR*SCI_COR_Lightcurve-RINF_V0?00.fits")+" = "+",".join(fitslist))
        ifitfile=fits.open(fitslist[0])
        exptime = float(ifitfile[1].header['EXPTIME'])
        im_thresh=22.65 #Threshold in EXPTIME below which imagettes (and not just sub-arrays) are generated
        #print(os.path.exists(os.path.join(out_dir,fk,"Outdata")),os.path.join(out_dir,fk,"Outdata"),overwrite)
        if not os.path.exists(os.path.join(out_dir,fk,"Outdata")) or overwrite:
            #os.system("mkdir "+os.path.join(out_dir,fk,"Outdata"))
            #os.system("mkdir "+os.path.join(out_dir,fk,"Outdata","00000"))
            #Running PIPE:
            from pipe import PipeParam, PipeControl
            self.logger.debug("PIPE file locations: "+self.name.replace(" ","_")+fk+" | "+os.path.join(out_dir,fk,"Outdata","00000")+" | "+os.path.join(out_dir,fk))
            pps = PipeParam(self.name.replace(" ","_"), fk, 
                            outdir=os.path.join(out_dir,fk,"Outdata","00000"),
                            datapath=os.path.join(out_dir,fk))
            #pps.bgstars = True
            pps.fit_bgstars = False
            
            if hasattr(self,'Teff') and self.Teff is not None:
                pps.Teff = int(np.round(self.Teff[0]))
            elif 'Teff' in kwargs:
                pps.Teff = int(np.round(kwargs['Teff']))

            if use_past_optimsation:
                past_params = check_past_PIPE_params(out_dir)
            
            #pps.limflux = 1e-5
            pps.darksub = True
            #pps.dark_level = 2
            #pps.remove_static = True
            #pps.save_static = False
            #pps.static_psf_rad = False
            if use_past_optimsation and past_params is not None and past_params['im']['exists'] and exptime<im_thresh:
                pps.im_optimise = False
                #Setting key paramaters here:
                for kpar in past_params['im']:
                    if kpar!='exists':
                        setattr(pps,kpar,past_params['im'][kpar])

            elif exptime<im_thresh:
                pps.im_optimise = optimise_klim
            
            if use_past_optimsation and past_params is not None and past_params['sa']['exists'] and exptime>=im_thresh:
                pps.sa_optimise = False
                #Setting key paramaters here:
                for kpar in past_params['sa']:
                    if kpar!='exists':
                        setattr(pps,kpar,past_params['sa'][kpar])

            elif exptime>=im_thresh:
                pps.sa_optimise = optimise_klim
            
            #pps.smear_fact = 5.5
            pps.psf_score = None
            pps.psf_min_num = 12
            pps.cti_corr = True
            #pps.smear_resid = False
            #pps.smear_resid_sa = True
            pps.non_lin_tweak = True

            if optimise_klim:
                if exptime<im_thresh:
                    pps.im_test_klips = [int(np.clip(2.5**(12-mag)*0.66666,1,7)),int(np.clip(2.5**(12-mag),2,10)),int(np.clip(1.3333*2.5**(12-mag),3,15))]
                elif exptime>=im_thresh:
                    pps.sa_test_klips = [int(np.clip(2.5**(12-mag)*0.66666,1,7)),int(np.clip(2.5**(12-mag),2,10)),int(np.clip(1.3333*2.5**(12-mag),3,15))]
                self.logger.debug("Setting number of klip models to test from magnitude: "+",".join([str(c) for c in pps.sa_test_klips])+". Filekey="+fk)
            else:
                if not use_past_optimisation:
                    self.logger.debug("Setting klip from magnitude."+str(pps.klip)+". Filekey="+fk)
                    pps.klip = int(np.clip(2.5**(12-mag),1,10))
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
                self.logger.debug("Running PIPE with a binary/companion star. Filekey="+fk)
                pps.fit_bgstars = True
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
            self.logger.debug("PIPE already been run. Finding most recent to return. Filekey="+fk)
            for folder in np.sort(np.array(glob.glob(os.path.join(out_dir,fk,"Outdata","000*"))))[::-1]:
                # Returns the name of the file in the folder.
                if len(glob.glob(os.path.join(folder,self.name.replace(" ","_")+"*_im.fits")))>0:
                    return glob.glob(os.path.join(folder,self.name.replace(" ","_")+"*_im.fits"))[0]
                elif len(glob.glob(os.path.join(folder,self.name.replace(" ","_")+"*_sa.fits")))>0:
                    return glob.glob(os.path.join(folder,self.name.replace(" ","_")+"*_sa.fits"))[0]
                else:
                    continue
            raise ValueError("Unable to either run PIPE extractions or return PIPE fits file")

    def assess_cheops_drp_apertures(self):
        """Given all the potential DRP apertures, we want to make sure that the selection is uniform across ALL cheoops filekeys.
        So this function reassess all the lightcurves, saving the flux statistics, and then

        Args:
            """
        all_aps=np.unique(np.hstack([list(self.chlcstats[f].keys()) for f in self.cheops_filekeys]))
        best_ap_array=np.vstack([[1/(1/self.chlcstats[filekey][a]['rel_std']**2+1/self.chlcstats[filekey][a]['rel_med_abs_diff']**2)**0.5 for a in all_aps] for filekey in self.cheops_filekeys])
        best_ap = all_aps[np.argmin(np.nanmedian(best_ap_array,axis=1))]
        for fk in self.cheops_filekeys:
            ix=self.lcs['cheops']['filekey']==fk
            self.lcs['cheops'].loc[ix,'raw_flux']=self.chlcstats[fk][best_ap]['flux']
            self.lcs['cheops'].loc[ix,'raw_flux_err']=self.chlcstats[fk][best_ap]['flux_err']
            self.lcs['cheops'].loc[ix,'flux']=1e3*(self.lcs['cheops'].loc[ix,'raw_flux']/np.nanmedian(self.lcs['cheops'].loc[ix&self.lcs['cheops']['mask'].values,'raw_flux'])-1.0)
            self.lcs['cheops'].loc[ix,'flux_err']=1e3*self.lcs['cheops'].loc[ix,'raw_flux_err']/np.nanmedian(self.lcs['cheops'].loc[ix&self.lcs['cheops']['mask'].values,'raw_flux'])

    def add_cheops_lc(self, filekey, fileloc=None, download=True, ylims=(-15,15), overwrite=False, bg_percentile_thresh=80,
                      use_PIPE=True, PIPE_bin_src=None, mag=None, **kwargs):
        """AI is creating summary for add_cheops_lc

        Args:
            filekey (str): Unique filekey for this Cheops lightcurve
            fileloc (str, optional): Location of lightcurve fits file
            download (bool, optional): Should we download from DACE?
            ylims (tuple, optional): Limits below/above which to cut (-15,15)
            overwrite (bool, optional): Whether to refit CHEOPs PIPE extraction
            bg_percentile_thresh (float, optional): The pervcentile value in background flux to use as a limit above which data is masked (plus 35% margin)
            use_PIPE (bool, optional): Is this a PIPE file? Defaults to False.
            PIPE_bin_src (int, optional): If this is a PIPE file with two stars (ie binary model) which should we model? Defaults to None.
            mag (float, optional): Magnitude needed by PIPE to guess fit radius? Defaults to 10.5
        """

        filekey=filekey[3:] if filekey[:3]=="CH_" else filekey
        self.logger.debug("Adding "+filekey+" CHEOPS lc")
        self.logger.debug([overwrite,filekey,self.cheops_filekeys])
        assert overwrite or (filekey not in self.cheops_filekeys), "Duplicated CHEOPS filekeys"

        self.update(**kwargs)

        out_dir=os.path.join(self.save_file_loc,self.name.replace(" ","_"))
        
        if fileloc is None and download:
            from dace_query.cheops import Cheops
            n_attempts=0
            while not os.path.isdir(os.path.join(out_dir,filekey)) and n_attempts<5:
                try:
                    self.logger.debug("Trying to download "+filekey+" from DACE")
                    if not os.path.isdir(os.path.join(out_dir,filekey)):
                        #Downloading Cheops data:
                        if use_PIPE:
                            self.logger.info("Downloading "+filekey+" with Dace to "+out_dir)
                            Cheops.download('all', {'file_key': {'equal':["CH_"+str(filekey)]}},
                                            output_directory=out_dir, 
                                            output_filename=self.name.replace(" ","_")+'_'+filekey+'_dace_download.tar.gz')
                            self.logger.info("Succeeded downloading "+filekey+" with Dace to "+out_dir+"/"+self.name.replace(" ","_")+'_'+filekey+'_dace_download.tar.gz')
                        else:
                            #Assume DRP
                            Cheops.download('lightcurves', {'file_key': {'equal':["CH_"+str(filekey)]}},
                                            output_directory=out_dir, output_filename=self.name.replace(" ","_")+'_'+filekey+'_dace_download.tar.gz')
                        time.sleep(30)
                        os.system("tar -xvf "+out_dir+'/'+self.name.replace(" ","_")+'_'+filekey+'_dace_download.tar.gz -C '+out_dir)
                        #Deleting it
                        self.logger.debug("Command:rm "+out_dir+'/'+self.name.replace(" ","_")+'_'+filekey+'_dace_download.tar.gz')
                        os.system("rm "+out_dir+'/'+self.name.replace(" ","_")+'_'+filekey+'_dace_download.tar.gz') #Deleting .tar file
                        self.logger.debug("Command:rm "+out_dir+'/'+filekey+'/*.mp4')
                        os.system("rm "+out_dir+'/'+filekey+'/*.mp4') #Deleting videos
                except:
                    self.logger.debug("Waiting a bit... maybe Dace needs a break? Filekey="+filekey)
                    time.sleep(15)
                self.logger.debug("new dir exists?"+str(os.path.isdir(os.path.join(out_dir,filekey))))
                if not os.path.isdir(os.path.join(out_dir,filekey)):
                    time.sleep(15)
                    n_attempts+=1
            assert os.path.isdir(os.path.join(out_dir,filekey)), "Unable to download filekey "+filekey+" using Dace."
        
        if use_PIPE and ((fileloc is None) or (os.path.isdir(fileloc)) or overwrite) :
            self.logger.debug("Running PIPE on "+filekey)
            fileloc = self.run_PIPE(out_dir, filekey, mag, **kwargs)
            self.logger.debug("fileloc="+fileloc)

        if not hasattr(self,"lcs"):
            self.lcs = {}
        if "cheops" not in self.lcs:
            self.lcs["cheops"] = pd.DataFrame()
        
        if self.use_PIPE:
            binchar=str(int(PIPE_bin_src)) if PIPE_bin_src is not None else ''
            sources={'time':'BJD_TIME', 'flux':'FLUX'+binchar, 'flux_err':'FLUXERR'+binchar, 
                     'bg':'BG', 'centroidx':'XC'+binchar,'phi':'ROLL', 
                     'centroidy':'YC'+binchar, 'deltaT':'thermFront_2','smear':None}
            self.logger.debug("Adding PCs from PIPE PC. Filekey="+filekey)
            
            sources.update({'U'+str(int(nPC)):'U'+str(int(nPC)) for nPC in range(0,9)})
        else:
            if not hasattr(self,'chlcstats'):
                self.chlcstats={}
            self.logger.debug("Getting DRP LCs. Filekey="+filekey)
            v3list=glob.glob(os.path.join(out_dir,filekey,"*SCI_COR_Lightcurve-*V0300.fits"))
            if len(v3list)==0:
                #No V0300 - need to use 
                v2list=glob.glob(os.path.join(out_dir,filekey,"*SCI_COR_Lightcurve-DEFAULT_V0200.fits"))
                assert len(v2list)>0, "Either V0200 and V0300 lightcurve files must be found within "+os.path.join(out_dir,filekey,"*SCI_COR_Lightcurve-DEFAULT_V0200.fits")
                fileloc = v2list[0]
            else:
                self.logger.debug("Need to find the optimal aperture... Using 90th-10th percentile of flux to find lowest variability. Filekey="+filekey)
                v3dic={}
                self.chlcstats[filekey]={}
                aps=[]
                self.logger.debug("Looping through DRP lcs="+",".join(v3list))
                for v in v3list:
                    try:
                        ap=v.split("-R")[-1].split("_")[0]
                        aps+=[ap]
                        v3dic[ap]=v
                        f=Table.read(v,format='fits').to_pandas()
                        highlow=np.nanpercentile(f['FLUX'],[2.5,97.5])
                        highlowcutmask=(f['FLUX']>highlow[0])&(f['FLUX']<highlow[1])&np.isfinite(f['FLUX'])
                        self.chlcstats[filekey][ap] = {'file':v,'flux':f['FLUX'],'flux_err':f['FLUXERR'],'aperture':ap,
                                                    'medflux':np.nanmedian(f['FLUX'][highlowcutmask]), 'std':np.nanstd(f['FLUX'][highlowcutmask])}
                        self.chlcstats[filekey][ap]['rel_std'] = np.nanstd(f['FLUX'][highlowcutmask]/self.chlcstats[filekey][ap]['medflux'])
                        self.chlcstats[filekey][ap]['rel_med_abs_diff'] = np.nanmedian(abs(np.diff(f['FLUX'][highlowcutmask]/self.chlcstats[filekey][ap]['medflux'])))
                    except:
                        print(v)
                best=aps[np.argmin([1/(1/self.chlcstats[filekey][a]['rel_std']**2+1/self.chlcstats[filekey][a]['rel_med_abs_diff']**2)**0.5 for a in v3dic])]
                fileloc = v3dic[best]
                
            sources={'time':'BJD_TIME', 'flux':'FLUX', 'flux_err':'FLUXERR', 
                     'bg':'BACKGROUND', 'centroidx':'CENTROID_X', 
                     'centroidy':'CENTROID_Y', 'deltaT':None, 'smear':'SMEARING_LC','phi':'ROLL_ANGLE', }
        f=Table.read(fileloc,format='fits').to_pandas()
        #fits.open(fileloc)
        iche=pd.DataFrame()
        for s in sources:
            self.logger.debug("Processing derived data. Source="+str(s)+" Filekey="+filekey)
            
            #print(s,iche.columns)
            if sources[s] is not None:
                if s=='flux_err' and sources[s] not in f.columns:
                    iche[s]=np.sqrt([sources['flux']])
                elif sources[s] in f.columns:
                    iche[s]=f[sources[s]]
            if sources[s] in f.columns:
                if s=='flux':
                    iche['raw_flux']=iche[s].values
                    iche[s]=(iche[s].values/np.nanmedian(f[sources['flux']])-1)*1000
                if s=='flux_err':
                    iche['raw_flux_err']=iche[s].values
                    iche[s]=(iche[s].values/np.nanmedian(f[sources['flux']]))*1000
                if s=='bg':
                    bgthresh=np.percentile(iche['bg'].values,bg_percentile_thresh)*1.35
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
                    #Performing simple anomaly masking using background limit, nans, and flux outliers:
                    iche['mask']=(~np.isnan(iche['flux']))&(~np.isnan(iche['flux_err']))&cut_anom_diff(iche['flux'].values)&(iche['bg']<bgthresh)&(iche['flux']>ylims[0])&(iche['flux']<ylims[-1])
                    #iche['mask']=cut_high_rollangle_scatter(iche['mask'].values,iche[s].values,iche['flux'].values,iche['raw_flux_err'].values,**kwargs)
                    #iche[s]=roll_rollangles(iche[s].values,mask=iche['mask'])
        iche['xoff']=iche['centroidx']-np.nanmedian(iche['centroidx'])
        iche['yoff']=iche['centroidy']-np.nanmedian(iche['centroidy'])
        iche['phi_sorting']=np.argsort(iche['phi'].values)
        iche['time_sorting']=np.argsort(iche['time'].values[iche['phi_sorting'].values])

        #Getting moon-object angle:

        iche['filekey']=np.tile(filekey,len(f[sources['time']]))
        self.logger.debug("Applying anomaly mask. Filekey="+filekey)
        iche.loc[iche['mask'],'mask']&=cut_anom_diff(iche['flux'].values[iche['mask']])
        
        iche['mask_phi_sorting']=np.tile(-1,len(iche['mask']))
        iche['mask_time_sorting']=np.tile(-1,len(iche['mask']))
        iche.loc[iche['mask'],'mask_phi_sorting']=np.argsort(iche.loc[iche['mask'],'phi'].values).astype(int)
        iche.loc[iche['mask'],'mask_time_sorting']=np.argsort(iche.loc[iche['mask'],'mask_phi_sorting'].values)

        if 'filekey' in self.lcs['cheops'].columns and filekey in self.lcs["cheops"]['filekey'] and overwrite:
            #Removing
            self.lcs["cheops"]=self.lcs["cheops"].loc[self.lcs["cheops"]['filekey']!=filekey]
        elif 'filekey' in self.lcs['cheops'].columns and filekey in self.lcs["cheops"]['filekey']:
            assert "Filekey already in CHEOPS LC - must `overwrite`"
        self.lcs["cheops"]=pd.concat([self.lcs["cheops"],iche])
        self.lcs["cheops"]=self.lcs["cheops"].sort_values("time") #Sorting by time, in case we are not already

        
        if not hasattr(self,'cheops_filekeys'):
            self.cheops_filekeys=[filekey]
        elif filekey not in self.cheops_filekeys:
            self.cheops_filekeys+=[filekey]
        self.logger.debug("Added filekey ("+filekey+") to global list (Now "+",".join(self.cheops_filekeys)+")")
        
        if not hasattr(self,'cheops_fk_mask'):
            self.cheops_fk_mask={}
        for fk in self.cheops_filekeys:
            self.cheops_fk_mask[fk]=(self.lcs["cheops"]['filekey'].values==fk)&(self.lcs["cheops"]['mask'].values)
        
        self.descr_dict[filekey]={'src':["DRP","PIPE"][int(self.use_PIPE)],
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

        self.logger.debug("Adding RVs")

        #initialising stored dicts:
        if not hasattr(self,"rvs") or overwrite:
            self.rvs = pd.DataFrame()
        if not hasattr(self,"rv_medians") or overwrite:
            self.rv_medians={}
        if not hasattr(self,"rv_stds") or overwrite:
            self.rv_stds={}

        irv=pd.DataFrame({'time':x,'y':y,'yerr':yerr,'scope':np.tile(name,len(x))})
        self.rvs=pd.concat([self.rvs,irv])
        
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
            
        """
        self.update(**kwargs)

        self.logger.debug("Adding starpars")

        if Rstar is None and hasattr(self.monotools_lc,'all_ids') and 'tess' in self.monotools_lc.all_ids and 'data' in self.monotools_lc.all_ids['tess'] and 'rad' in self.monotools_lc.all_ids['tess']['data']:
            #Radius info from lightcurve data (TIC)
            if 'eneg_Rad' in self.monotools_lc.all_ids['tess']['data'] and self.monotools_lc.all_ids['tess']['data']['eneg_Rad'] is not None and self.monotools_lc.all_ids['tess']['data']['eneg_Rad']>0:
                self.logger.debug("Getting starpars from MonoTools lightcurve/TIC data - epos and eneg exists")
                Rstar=self.monotools_lc.all_ids['tess']['data'][['rad','eneg_Rad','epos_Rad']].values
            else:
                self.logger.debug("Getting starpars from MonoTools lightcurve/TIC data - e_rad exists")
                Rstar=self.monotools_lc.all_ids['tess']['data'][['rad','e_rad','e_rad']].values
        if (Rstar is None)|(np.all(np.isnan(np.array(Rstar).astype(float)))):
            Rstar=None
        elif np.isnan(Rstar[1]) or np.isnan(Rstar[2]):
            Rstar=[Rstar[0],0.2*Rstar[0],0.2*Rstar[0]]
        if Teff is None and hasattr(self.monotools_lc,'all_ids') and 'tess' in self.monotools_lc.all_ids and 'data' in self.monotools_lc.all_ids['tess'] and 'Teff' in self.monotools_lc.all_ids['tess']['data']:
            if 'eneg_Teff' in self.monotools_lc.all_ids['tess']['data'] and self.monotools_lc.all_ids['tess']['data']['eneg_Teff'] is not None and self.monotools_lc.all_ids['tess']['data']['eneg_Teff']>0:
                Teff=self.monotools_lc.all_ids['tess']['data'][['Teff','eneg_Teff','epos_Teff']].values
            else:
                Teff=self.monotools_lc.all_ids['tess']['data'][['Teff','e_Teff','e_Teff']].values
        self.logger.debug(["Teff:",Teff,type(Teff),type(Teff[0])])
        if (Teff is None)|(np.all(np.isnan(np.array(Teff).astype(float)))):
            Teff=None
        elif np.isnan(Teff[1]) or np.isnan(Teff[2]):
            Teff=[Teff[0],250,250]

        if logg is None and hasattr(self.monotools_lc,'all_ids') and 'tess' in self.monotools_lc.all_ids and 'data' in self.monotools_lc.all_ids['tess'] and 'logg' in self.monotools_lc.all_ids['tess']['data']:
            if 'eneg_logg' in self.monotools_lc.all_ids['tess']['data'] and self.monotools_lc.all_ids['tess']['data']['eneg_logg'] is not None and self.monotools_lc.all_ids['tess']['data']['eneg_logg']>0:
                logg=self.monotools_lc.all_ids['tess']['data'][['logg','eneg_logg','epos_logg']].values
            else:
                logg=self.monotools_lc.all_ids['tess']['data'][['logg','e_logg','e_logg']].values
        if (logg is None )|( np.all(np.isnan(np.array(logg).astype(float)))):
            logg=None
        elif np.isnan(logg[1]) or np.isnan(logg[2]):
            logg=[logg[0],0.25,0.25]

        if Rstar is None and Teff is not None and not np.isnan(Teff[0]) and hasattr(self,'monotools_lc'):
            #Approximating from Teff
            Mstar=[(Teff[0]/5770)**(7/4)]
            Mstar+=[0.5*Mstar[0],0.5*Mstar[0]]
            Rstar=[Mstar[0]**(3/7)]
            Rstar+=[0.5*Rstar[0],0.5*Rstar[0]]
        if logg is None and Mstar is not None and Rstar is not None:
            logg=[np.log(Mstar[0]/(Rstar[0]**2))+4.41]
            logg+=[np.log((Mstar[0]-Mstar[1])/((Rstar[0]+Rstar[2])**2))+4.41-logg[0]]
            logg+=[np.log((Mstar[0]+Mstar[2])/((Rstar[0]-Rstar[1])**2))+4.41-logg[0]]

        #Stellar coordinates still None so using roughly stellar parameters
        if Rstar is None:
            Rstar=[1.0,0.15,0.15]
        if Teff is None:
            Teff=[5777,300,300]
        if logg is None:
            logg=[4.3,1.0,1.0]

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

    def add_planets_from_toi(self,add_starpars=True,overwrite=False,**kwargs):
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
                            period_err=float(row[1]['Period (days) err']),**kwargs)
            
    def add_planet(self, name, tcen, period, tdur, depth, tcen_err=None, period_err=None, b=None, 
                   rprs=None, K=None, overwrite=False,check_per=False,force_check_per=False,**kwargs):
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
            check_per (bool, optional): Whether to check the lightcurve data to see if period can be improved...
            force_check_per (bool, optional): Insist that we run TLS to check, using lightcurve data, if period can be improved.
        """
        assert name not in self.planets or overwrite, "Name is already stored as a planet"
        
        if period_err is None:
            if 'tess' in self.lcs:
                span=np.ptp(self.lcs['tess']['time']) 
            else:
                span=365.25 #Guess 1yr span.
            #period_err=0.4*pl[1]['duration']*pl[1]['true period']/730
            period_err = self.timing_sd_durs*tdur*period/span

        if rprs is None:
            assert depth<0.25 #Depth must be a ratio (not in mmags)
            rprs=np.sqrt(depth)

        if b is None:
            rho_S=self.rhostar[0] if hasattr(self,'rhostar') else 1.0
            #Estimating b from simple geometry:

            b=np.clip((1+rprs)**2 - (tdur*86400)**2 * \
                                ((3*period*86400) / (np.pi**2*6.67e-11*rho_S*1410))**(-2/3),
                                0.01,2.0)**0.5
        
        if hasattr(self,'monotools_lc') and check_per:
            ntrans=np.round((np.nanmedian(np.max(self.monotools_lc.time))-tcen)/period)
            if (tcen+period_err*ntrans)>tdur*0.666 or force_check_per:
                self.monotools_lc.flatten(transit_mask=((self.monotools_lc.time-tcen-0.5*period)%period-0.5*period)<0.5*tdur)
                period=update_period_w_tls(self.monotools_lc.time[self.monotools_lc.mask],
                                           self.monotools_lc.flux_flat[self.monotools_lc.mask],period)
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

        assert ~(self.cut_oot&self.bin_oot), "Cannot both cut and bin out of transit data. Pick one."
        assert ~(self.fit_flat&self.fit_gp), "Cannot both flatten data and fit GP. Choose one"        
        
        #masking, binning, flattening light curve

        if not hasattr(self,'binlc'):
            self.binlc={}
        if not hasattr(self,'lc_fit') or self.overwrite:
            self.lc_fit={scope:pd.DataFrame() for scope in self.lcs}
        for src in self.lcs:
            if src!='cheops':
                self.lcs[src]['mask']=~np.isnan(self.lcs[src]['flux'].values)&~np.isnan(self.lcs[src]['flux_err'].values)
                self.lcs[src]['mask'][self.lcs[src]['mask']]=cut_anom_diff(self.lcs[src]['flux'].values[self.lcs[src]['mask']])
                self.lcs[src]['mask'][self.lcs[src]['mask']]=cut_anom_diff(self.lcs[src]['flux'].values[self.lcs[src]['mask']])
                
                self.lcs[src]['near_trans'] = np.tile(False,len(self.lcs[src]['mask']))
                if hasattr(self,'planets'):
                    for pl in self.planets:
                        self.lcs[src]['in_trans_'+pl]=abs((self.lcs[src]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<(self.mask_distance*self.planets[pl]['tdur'])
                        self.lcs[src]['near_trans']+=abs((self.lcs[src]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.cut_distance*self.planets[pl]['tdur']
                    self.lcs[src]['in_trans_all'] = np.any(np.vstack([self.lcs[src]['in_trans_'+pl] for pl in self.planets]),axis=0)
                else:
                    self.lcs[src]['in_trans_all'] = np.tile(False,len(self.lcs[src]['mask']))
                #FLATTENING
                if self.fit_flat:
                    spline, newmask = kepler_spline(self.lcs[src]['time'].values[self.lcs[src]['mask']],
                                                    self.lcs[src]['flux'].values[self.lcs[src]['mask']], 
                                                    transit_mask=~self.lcs[src]['in_trans_all'][self.lcs[src]['mask']],bk_space=self.flat_knotdist)
                    self.lcs[src]['spline']=np.tile(np.nan,len(self.lcs[src]['time']))
                    self.lcs[src].loc[self.lcs[src]['mask'],'spline']=spline
                    self.lcs[src]['flux_flat']=self.lcs[src]['flux'].values
                    self.lcs[src]['flux_flat'][self.lcs[src]['mask']]-=self.lcs[src]['spline']


                #BINNING
                ibinlc=bin_lc_segment(np.column_stack((self.lcs[src]['time'].values[self.lcs[src]['mask']],
                                                        self.lcs[src]['flux'].values[self.lcs[src]['mask']],
                                                        self.lcs[src]['flux_err'].values[self.lcs[src]['mask']])),
                                    self.bin_size)
                self.logger.debug("bin lc")
                self.logger.debug(ibinlc)
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
                self.binlc[src]['near_trans'] = np.tile(False,len(self.binlc[src]['time']))
                if hasattr(self,'planets'):
                    for pl in self.planets:
                        self.binlc[src]['in_trans_'+pl]=abs((self.binlc[src]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.mask_distance*self.planets[pl]['tdur']
                        self.binlc[src]['near_trans']+=abs((self.binlc[src]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.cut_distance*self.planets[pl]['tdur']
                    self.binlc[src]['in_trans_all']=np.any(np.vstack([self.binlc[src]['in_trans_'+pl] for pl in self.planets]),axis=0)
                else:
                    self.binlc[src]['in_trans_all']=np.tile(False,len(self.binlc[src]['time']))

                vals=['time','flux','flux_err','in_trans_all','near_trans']
                if hasattr(self,'planets'):
                    vals+=['in_trans_'+pl for pl in self.planets]
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
            else:
                self.logger.warning("Run `init_cheops` in order to intialise the CHEOPS lightcurves")
            #     #Adding source to the array:
            #     if 'src' not in self.lc_fit.columns:
            #         self.lc_fit['src']=np.tile(src,len(newvals))
            #     else:
            #         self.lc_fit['src']=np.hstack((self.lc_fit['src'],np.tile(src,len(newvals))))
            #    self.lc_fit[src]=self.lc_fit[src].sort_values('time')          
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

    def init_gp(self, logprior_func='InverseGamma', **kwargs):
        """Initiliasing photometry GP on e.g. TESS

        Optional
        """
        self.update(**kwargs)

        from celerite2.pymc import terms as pymc_terms
        import celerite2.pymc
        
        lcrange=27
        av_dur = np.average([self.planets[key]['tdur'] for key in self.planets])
        exps=np.array([np.log((2*np.pi)/(av_dur)), np.log((2*np.pi)/(0.1*lcrange))])
        #Max power as half the 1->99th percentile in flux
        maxpowers=[0.5*np.ptp(np.percentile(self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'flux'].values,[2,98])) for scope in self.lcs if scope!='cheops']
        logmaxpowers=[np.log(0.5*np.ptp(np.percentile(self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'flux'].values,[1,99]))) for scope in self.lcs if scope!='cheops']
        #([np.nanstd(self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'flux'].values) for scope in self.lcs]))
        
        #Min power as 2x the average point-to-point displacement
        logminpowers=[np.log(2*np.nanmedian(abs(np.diff(self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'flux'].values)))) for scope in self.lcs if scope!='cheops']
        minpowers=[0.5*np.nanmedian(abs(np.diff(self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'flux'].values))) for scope in self.lcs if scope!='cheops']
        span=abs(np.min(logmaxpowers)-np.max(logminpowers))
        
        allt=np.hstack([self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'time'].values for scope in self.lcs if scope!="cheops"])
        ally=np.hstack([self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'flux'].values for scope in self.lcs if scope!="cheops"])[np.argsort(allt)]
        allyerr=np.hstack([self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'flux_err'].values for scope in self.lcs if scope!="cheops"])[np.argsort(allt)]
        che_ix = list(self.lcs.keys()).index('cheops')
        allsrcs=np.hstack([np.tile(iscope,len(self.lc_fit[list(self.lcs.keys())[iscope]]['time'])) for iscope in np.arange(len(self.lcs)) if iscope!=che_ix])[np.argsort(allt)]
        allsrcs=np.column_stack([np.isin(allsrcs,i) for i in np.arange(len(self.lcs)) if i!=che_ix])
        allt=np.sort(allt)
        self.logger.debug(allyerr)

        with pm.Model() as ootmodel:
            logs={}
            for scope in self.lcs:
                if scope!="cheops":
                    logs[scope] = pm.Normal(scope+"_logs", 
                                            mu=np.log(np.std(self.lc_fit[scope]['flux']))+2, 
                                            sigma=1,initval=np.log(np.std(self.lc_fit[scope]['flux']))+1)
            
            #Initialising the SHO frequency
            if logprior_func.lower()=='pareto':
                log_w0 = pm.Pareto("log_w0", m=exps[1], alpha=0.1*np.ptp(exps), initval=exps[1]+0.15*np.ptp(exps))
                self.logger.debug("w0 m: "+str(exps[1])+"  alpha: "+str(np.ptp(exps)/3)+"  testval: "+str(exps[1]+0.45*np.ptp(exps))+"  test per "+str(np.pi*2/(exps[1]+0.45*np.ptp(exps))))
                w0 = pm.Deterministic("w0", pm.math.exp(log_w0))
                log_sigma = pm.Pareto("log_sigma", m=np.max(logminpowers), alpha=0.1*span, initval=np.max(logminpowers)+0.5*span)
                self.logger.debug("logsigma m: "+str(np.max(logminpowers))+"   alpha: "+str(00.2*span)+"  start: "+str(np.max(logminpowers)+0.5*span))
                sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

            elif logprior_func.lower()=='normal':
                log_w0 = pm.Normal("log_w0", mu=exps[1]+0.3*np.ptp(exps), sigma=0.05*np.ptp(exps), initval=exps[1]+0.15*np.ptp(exps))
                self.logger.debug("w0 mu: "+str((exps[0]+exps[1])/2)+"  sigma: "+str(np.ptp(exps)/5)+"  testval: "+str(exps[1]+0.4*np.ptp(exps))+"  test per "+str(np.pi*2/(exps[1]+0.2*np.ptp(exps))))
                w0 = pm.Deterministic("w0", pm.math.exp(log_w0))
                log_sigma = pm.Normal("log_sigma", mu=(np.min(logmaxpowers)+np.max(logminpowers))/2, sigma=0.2*abs(np.min(logmaxpowers)-np.max(logminpowers)),initval=np.min(logmaxpowers)-0.1)
                self.logger.debug("logsigma mu"+str((np.min(logmaxpowers)+np.max(logminpowers))/2)+"   sigma: "+str(0.2*abs(np.min(logmaxpowers)-np.max(logminpowers)))+"  start: "+str(np.min(logmaxpowers)-0.1))
                sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

            elif logprior_func.lower()=='inversegamma':
                target=0.01
                success=np.array([False,False]);target=0.01
                while np.any(~success) and target<0.2:
                    if not success[0]:
                        try:
                            low=(2*np.pi)/(abs(np.random.normal(3,1)))
                            #itarg=abs(np.random.normal(target,0.5*target))
                            w0 = pm.InverseGamma("w0", **pmx.utils.estimate_inverse_gamma_parameters(lower=low,
                                                                                                upper=(2*np.pi)/(av_dur*((0.03/target)**0.5)),
                                                                                                target=0.01))
                            success[0]=True
                            self.logger.debug("w0 InverseGamma: "+str(success)+" low "+str((2*np.pi)/(5))+" up "+str((2*np.pi)/(av_dur*(0.03/target)))+" target "+str(target))
                        except:
                           success[0]=False
                            
                    if not success[1]:
                        try:
                            sigma = pm.InverseGamma("sigma",initval=np.max(minpowers)*5,
                                                    **pmx.utils.estimate_inverse_gamma_parameters(lower=np.min(minpowers),
                                                                                        upper=np.min(maxpowers)*np.sqrt(target/0.1),
                                                                                        target=0.01))
                            success[1]=True
                            self.logger.debug("sigma InverseGamma: "+str(success)+" min "+str(np.max(logminpowers))+" max "+str(np.min(logmaxpowers)/np.sqrt(target/0.01))+" target "+str(target))
                        except:
                            success[1]=False
                    target*=1.15
                assert np.all(success), "InverseGamma estimation of "+"&".join(list(np.array(["w0","sigma"])[~success]))+" failed"
            else:
                print("No log prior func selected...")
            # power=None
            # while not success and target<0.25:
            #     try:
            #         power = pm.InverseGamma("power",initval=minpower*5,
            #                                 **pmx.estimate_inverse_gamma_parameters(lower=minpower,
            #                                                                         upper=maxpower/(target/0.01),
            #                                                                         target=0.1))
            #         success=True
            #     except:
            #         target*=1.15
            #         success=False
            # self.logger.debug("power:",minpower,maxpower/(target/0.01),target,success)
            # #print("power",success,target)
            # if power is None:
            #     logspan=np.log(maxpower)-np.log(minpower)
            #     logpower = pm.Normal("logpower", mu=np.log(minpower)+0.3333*logspan, sigma=logspan/3)
            #     power = pm.Deterministic("power", pm.math.exp(logpower))
            # self.logger.debug("logpower mu:", np.log(minpower)+0.3333*logspan,"sigma:",logspan/3)
            # GP model for the light curve
            kernel = pymc_terms.SHOTerm(sigma=sigma, w0=w0, Q=1/np.sqrt(2))
            means={}
            gps={}
            for scope in self.lcs:
                if scope!="cheops":
                    means[scope] = pm.Normal(scope+"_mean", mu=0.0, sigma=10.0, initval=np.nanmedian(self.lcs[scope]['flux']))
                    gps[scope] = celerite2.pymc.GaussianProcess(kernel, mean=means[scope])
                    gps[scope].compute(allt, yerr=np.sqrt(self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'flux_err'].values ** 2 + pm.math.exp(logs[scope])**2), quiet=True)
                    loglik=gps[scope].marginal("loglik",observed=self.lc_fit[scope].loc[~self.lc_fit[scope]['in_trans_all'],'flux'].values)
            oot_soln = pmx.optimize()#start=start)
            self.logger.debug(ootmodel.debug())

        #Sampling:
        with ootmodel: 
            if 'cores' in kwargs:
                self.oot_gp_trace = pm.sample(tune=500, draws=1200, start=oot_soln, 
                                        compute_convergence_checks=False,cores=kwargs['cores'],return_inferencedata=True)
            else:
                self.oot_gp_trace = pm.sample(tune=500, draws=1200, start=oot_soln, 
                                        compute_convergence_checks=False,return_inferencedata=True)


    def cheops_only_model(self, fk, transittype="fix", force_no_dydt=True, overwrite=False, load_similar_past_model=True, include_PIPE_PCs=True, 
                          linpars=None, quadpars=None, split_spline_fit_vars=['deltaT','smear','bg'], split_spline_dt=9.8/1440, **kwargs):
        """Initialising and running a Cheops-only transit model for a given filekey

        Args:
            fk (str): Cheops filekey.
            transittype (str, optional): How to include transit model - "set": set by TESS transits, "loose": allowed to vary, "none": no transit at all. Defaults to "fix".
            force_no_dydt (optional): Do we force the model to avoid using decorrelation with trends? Defaults to None, which mirrors include_transit
            overwrite (bool, optional): Whether to rewrite this initialise model. If not, it will try to reload a pre-run model. Defaults to False.
            load_similar_past_model (bool, optional): Whether we take any similar past model save (ignoring precise date and parameters) instead of overwriting every time. Default: true
            include_PIPE_PCs (bool, optional): Whether to include PIPE PCA of model residuals as decorrelation parameter 
            linpars (list of strings, optional): Specify the parameters to use for the linear decorrelation. For sin/cos, use cosNphi where N is the harmonic (i.e. normal = 1)
            split_spline_fit_vars (list, optional): The specific parameters which we can use a spline to split into high and low frequency variability
            split_spline_dt (float, optional): The time span to be used as spline knots; default = 10 knots per orbit/1 per ~10mins
            quadpars (list of strings, optional): Specify the parameters to use for the quadratic decorrelation
        
        Returns:
            pymc trace: The output model trace from the fit.
        """
        #Initialising save name (and then checking if we can re-load the saved model fit):
        savefname="_cheops_only_fit_"+fk+"_trace"
        cheops_fk_save_name_dic = {'fix':'_fixtrans','loose':'_loosetrans','none':'_notrans'}
        
        if transittype in cheops_fk_save_name_dic:
            savefname+=cheops_fk_save_name_dic[transittype]
        if force_no_dydt: savefname+="_notrend" 

        if not hasattr(self,'cheops_init_trace'):
            self.cheops_init_trace={}
        
        include_PIPE_PCs=False if not self.use_PIPE else include_PIPE_PCs

        if linpars is None:
            #Initialising decorrelation parameters:
            self.init_cheops_linear_decorr_pars=['sin1phi','cos1phi','sin2phi','cos2phi','sin3phi','cos3phi','bg','centroidx','centroidy','time']
            if 'smear' in self.lcs["cheops"].columns: self.init_cheops_linear_decorr_pars+=['smear']
            if 'deltaT' in self.lcs["cheops"].columns and not force_no_dydt: self.init_cheops_linear_decorr_pars+=['deltaT']
            if include_PIPE_PCs:
                for nPC in range(9):
                    #Checking if we have this array in the CHEOPS lightcurve, and if not all the values are the same...
                    if "U"+str(int(nPC)) in self.lcs["cheops"].columns:
                        #Making any nans into zeros:
                        self.lcs["cheops"].loc[np.isnan(self.lcs["cheops"]["U"+str(int(nPC))]),"U"+str(int(nPC))]=0.0
                        if np.sum(np.diff(self.lcs["cheops"]["U"+str(int(nPC))])==0.0)<0.2*self.lcs["cheops"].shape[0]: 
                            self.init_cheops_linear_decorr_pars+=["U"+str(int(nPC))]
            if force_no_dydt: self.init_cheops_linear_decorr_pars.remove('time')
        else:
            self.init_cheops_linear_decorr_pars=linpars
        if quadpars is None:
            self.init_cheops_quad_decorr_pars=['bg','centroidx','centroidy']
            if 'smear' in self.lcs["cheops"].columns: self.init_cheops_quad_decorr_pars+=['smear']
            #if 'deltaT' in self.lcs["cheops"].columns and not force_no_dydt: self.init_cheops_quad_decorr_pars+=['deltaT'] 
            if include_PIPE_PCs:
                for nPC in range(9):
                    #Checking if we have this array in the CHEOPS lightcurve, and if not all the values are the same...
                    if "U"+str(int(nPC)) in self.lcs["cheops"].columns and np.sum(np.diff(self.lcs["cheops"]["U"+str(int(nPC))])==0.0)<0.2*self.lcs["cheops"].shape[0]: self.init_cheops_quad_decorr_pars+=["U"+str(int(nPC))]
            #if force_no_dydt: self.init_cheops_quad_decorr_pars.remove('time')
        else:
            self.init_cheops_quad_decorr_pars=quadpars
        #Initialising the data specific to each Cheops visit:
        x=self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'time'].values.astype(np.float64)
        y=self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values.astype(np.float64)
        #Using a robust average (logged) of the point-to-point error & the std as a prior for the decorrelation parameters
        self.cheops_mads[fk]=np.exp(0.5*(np.log(np.std(y))+np.log(np.nanmedian(abs(np.diff(y)))*1.06)))
        
        #Splitting some params into two decorrelation paramaters - one encompassing long trends with time, one with short-term variation.
        for var in split_spline_fit_vars:
            if var in self.init_cheops_linear_decorr_pars or var in self.init_cheops_quad_decorr_pars:
                if var+"slow" not in self.lcs["cheops"].columns:
                    self.lcs["cheops"][var+"slow"]=np.zeros(len(self.lcs["cheops"]))
                if var+"fast" not in self.lcs["cheops"].columns:
                    self.lcs["cheops"][var+"fast"]=self.lcs["cheops"][var].values[:]
                # Do a spline fit using split_spline_dt
                self.lcs["cheops"].loc[self.lcs["cheops"]['filekey']==fk,var+"slow"]=kepler_spline(self.lcs["cheops"].loc[self.lcs["cheops"]['filekey']==fk,"time"],
                                                                                          self.lcs["cheops"].loc[self.lcs["cheops"]['filekey']==fk,var],
                                                                                          np.isfinite(self.lcs["cheops"].loc[self.lcs["cheops"]['filekey']==fk,var]),
                                                                                          bk_space=split_spline_dt)[0]
                self.lcs["cheops"].loc[self.cheops_fk_mask[fk],var+"fast"]-=self.lcs["cheops"].loc[self.cheops_fk_mask[fk],var+"slow"]
            #Removing variable and adding two new variables:
            if var in self.init_cheops_linear_decorr_pars:
                self.init_cheops_linear_decorr_pars.remove(var)
                self.init_cheops_linear_decorr_pars+=[var+"slow",var+"fast"]
            if var in self.init_cheops_quad_decorr_pars:
                self.init_cheops_quad_decorr_pars.remove(var)
                self.init_cheops_quad_decorr_pars+=[var+"slow",var+"fast"]

        yerr=self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux_err'].values
        for var in self.init_cheops_linear_decorr_pars+self.init_cheops_quad_decorr_pars:
            if var in self.lcs["cheops"].columns:
                self.logger.debug(str(var)+fk)
                self.norm_cheops_dat[fk][var]=np.nan_to_num((self.lcs["cheops"].loc[self.cheops_fk_mask[fk],var].values-np.nanmedian(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],var].values))/np.nanstd(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],var].values))
            elif var[:3]=='sin':
                if var[3] not in ['1','2','3','4']:
                    self.logger.warning(str(var)+str(var[3])+"- Must have a number in the cos/sin parameter name to represent the harmonic, e.g. cos1phi or sin3phi")
                    var=var[:3]+"1"+var[3:]
                #self.norm_cheops_dat[fk]
                self.norm_cheops_dat[fk][var]=np.nan_to_num(np.sin(float(int(var[3]))*self.lcs["cheops"].loc[self.cheops_fk_mask[fk],var[-3:]].values*np.pi/180))
            elif var[:3]=='cos':
                if var[3] not in ['1','2','3','4']:
                    self.logger.warning(str(var)+str(var[3])+"- Must have a number in the cos/sin parameter name to represent the harmonic, e.g. cos1phi or sin3phi")
                    var=var[:3]+"1"+var[3:]
                self.norm_cheops_dat[fk][var]=np.nan_to_num(np.cos(float(int(var[3]))*self.lcs["cheops"].loc[self.cheops_fk_mask[fk],var[-3:]].values*np.pi/180))
        #self.logger.debug(os.path.join(self.save_file_loc,self.name,self.unq_name+savefname+".pkl"),os.path.exists(os.path.join(self.save_file_loc,self.name,self.unq_name+savefname+".pkl")),overwrite)
        if not overwrite and os.path.exists(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+savefname+".pkl")):
            self.cheops_init_trace[savefname[1:]]=pickle.load(open(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+savefname+".pkl"),"rb"))
            self.logger.info("Exact match cheops pre-modelled trace exists for filekey="+fk+" at "+self.unq_name+savefname+".pkl")
            return savefname[1:]
        elif not overwrite and load_similar_past_model:
            pastfiles=glob.glob(os.path.join(self.save_file_loc,self.name.replace(" ","_"),"*"+savefname+".pkl"))
            if len(pastfiles)>0:
                latest_file = max(pastfiles, key=os.path.getctime)
                try:
                    self.cheops_init_trace[savefname[1:]]=pickle.load(open(latest_file,"rb"))
                    self.logger.info("Near-match cheops pre-modelled trace exists for filekey="+fk+" at "+latest_file.split('/')[-1]+".pkl")
                    return savefname[1:]
                except:
                    self.logger.warning("Loading Cheops pre-modelled trace for filekey="+fk+" at "+latest_file.split('/')[-1]+".pkl FAILS - likely due to being generated by a (now incompitible) previous version. Continuing with individual CHEOPS filekey fitting.")
        
        with pm.Model() as self.ichlc_models[fk]:
            #Adding planet model info if there's any transit in the lightcurve
            if transittype!="none" and np.any(self.lcs["cheops"][(self.lcs["cheops"]['filekey']==fk)&self.lcs["cheops"]['in_trans_all']]):
                Rs = pm.TruncatedNormal("Rs", lower=0, mu=self.Rstar[0], sigma=self.Rstar[1])
                #Ms = pm.TruncatedNormal("Ms", lower=0, mu=self.Mstar[0], sigma=self.Mstar[1])
                u_star_cheops = pm.TruncatedNormal("u_star_cheops", lower=0.0, upper=1.0,
                                                mu=np.nanmedian(self.ld_dists['cheops'],axis=0),
                                                sigma=np.clip(np.nanstd(self.ld_dists['cheops'],axis=0),0.1,1.0), 
                                                shape=2, initval=np.nanmedian(self.ld_dists['cheops'],axis=0))
                
                logrors={};t0s={};pers={};orbits={};bs={};tdurs={}
                pls=[]
                if np.any(self.lcs["cheops"][(self.lcs["cheops"]['filekey']==fk)&self.lcs["cheops"]['in_trans_all']]):
                    for pl in self.planets:
                        if np.any(self.lcs["cheops"][(self.lcs["cheops"]['filekey']==fk)&self.lcs["cheops"]['in_trans_'+pl]]):
                            pls+=[pl]
                            #If this timeseries specifically has this planet in, we need to fit for it
                            if transittype=="fix":
                                logrors[pl] = pm.Normal("logror_"+pl, mu=np.log(np.sqrt(self.planets[pl]['depth'])), sigma=0.125, 
                                                        initval=np.log(np.sqrt(self.planets[pl]['depth'])))
                            elif transittype=="loose":
                                logrors[pl] = pm.Normal("logror_"+pl, mu=np.log(np.sqrt(self.planets[pl]['depth'])), sigma=3, 
                                                        initval=np.log(np.sqrt(self.planets[pl]['depth'])))
                            #rpl = pm.Deterministic("rpl",109.1*pm.math.exp(logror)*Rs)
                            bs[pl] = xo.distributions.ImpactParameter("b_"+pl, ror=pm.math.exp(logrors[pl]),
                                                                      initval=np.clip(self.planets[pl]['b'],0.025,0.975))
                            tdurs[pl]=pm.Normal("tdur_"+pl, mu=self.planets[pl]['tdur'],sigma=0.03,initval=self.planets[pl]['tdur'])
                            ntrans=np.round((np.nanmedian(x)-self.planets[pl]['tcen'])/self.planets[pl]['period'])
                            if (self.planets[pl]['tcen_err']+self.planets[pl]['period_err']*ntrans)>2/14: self.logger.warning("Ephemeris potentially lost. Error = ",self.planets[pl]['tcen_err']+self.planets[pl]['period_err']*ntrans,"days")
                            t0s[pl] = pm.Normal("t0_"+pl, mu=self.planets[pl]['tcen']+self.planets[pl]['period']*ntrans,
                                                sigma=np.clip(self.planets[pl]['tcen_err']+self.planets[pl]['period_err']*ntrans,0.01,0.2),
                                                initval=self.planets[pl]['tcen']+self.planets[pl]['period']*ntrans)
                            pers[pl] = pm.Normal("per_"+pl, mu=self.planets[pl]['period'],sigma=self.planets[pl]['period_err'],initval=self.planets[pl]['period'])
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

                                if np.any(self.lcs["cheops"][(self.lcs["cheops"]['filekey']==fk)&self.lcs["cheops"]['in_trans_'+pl]]):
                                    cheops_planets_x[pl]=pm.Deterministic("cheops_planets_x_"+pl+"_"+fk, xo.LimbDarkLightCurve(u_star_cheops).get_light_curve(orbit=orbits[pl], 
                                                                                                                                        r=pm.math.exp(logrors[pl])*Rs,t=x)[:,0]*1000)
                                else:
                                    cheops_planets_x[pl]=np.zeros(len(x))
                            else:
                                cheops_planets_x[pl]=np.zeros(len(x))


            cheops_logs = pm.Normal("cheops_logs", mu=np.log(np.nanmedian(abs(np.diff(y))))-3, sigma=3)

            #Initialising linear (and quadratic) parameters:
            linear_decorr_dict={};quad_decorr_dict={}
            
            for decorr_1 in self.init_cheops_linear_decorr_pars:
                if decorr_1=='time':
                    linear_decorr_dict[decorr_1]=pm.Normal("dfd"+decorr_1,mu=0,sigma=np.ptp(self.norm_cheops_dat[fk][decorr_1])/self.cheops_mads[fk],initval=np.random.normal(0,0.05))
                else:
                    linear_decorr_dict[decorr_1]=pm.Normal("dfd"+decorr_1,mu=0,sigma=self.cheops_mads[fk],initval=np.random.normal(0,0.05))
            for decorr_2 in self.init_cheops_quad_decorr_pars:
                quad_decorr_dict[decorr_2]=pm.Normal("d2fd"+decorr_2+"2",mu=0,sigma=self.cheops_mads[fk],initval=np.random.normal(0,0.05))
            cheops_obs_mean = pm.Normal("cheops_mean",mu=0.0,sigma=0.5*np.nanstd(y),initval=0.0)
            cheops_flux_cor = pm.Deterministic("cheops_flux_cor_"+fk,cheops_obs_mean + pm.math.sum([linear_decorr_dict[param]*self.norm_cheops_dat[fk][param] for param in self.init_cheops_linear_decorr_pars], axis=0) + \
                                                pm.math.sum([quad_decorr_dict[param]*self.norm_cheops_dat[fk][param]**2 for param in self.init_cheops_quad_decorr_pars], axis=0))
            
            #We have a transit, so we need the transit params:
            if transittype!="none" and np.any(self.lcs["cheops"][(self.lcs["cheops"]['filekey']==fk)&self.lcs["cheops"]['in_trans_all']]):
                ##pm.math.printing.Print("cheops_planets_x")(cheops_planets_x)
                if len(pers)>0:
                    cheops_summodel_x = pm.Deterministic("cheops_summodel_x_"+fk, pm.math.sum([cheops_planets_x[pl] for pl in self.planets], axis=0) + cheops_flux_cor)
                elif len(pers)==1:
                    ##pm.math.printing.Print("cheops_flux_cor")(cheops_flux_cor)
                    ##pm.math.printing.Print("cheops_flux_cor")(cheops_planets_x[list(self.planets.keys())[0]])
                    #print(cheops_planets_x[list(self.planets.keys())[0]].shape,)
                    cheops_summodel_x = pm.Deterministic("cheops_summodel_x_"+fk, cheops_planets_x[list(self.planets.keys())[0]] + cheops_flux_cor)

                else:
                    cheops_summodel_x = pm.Deterministic("cheops_summodel_x_"+fk, cheops_flux_cor)
            else:
                cheops_summodel_x = pm.Deterministic("cheops_summodel_x_"+fk, cheops_flux_cor)
            cheops_llk = pm.Normal("cheops_llk", mu=cheops_summodel_x, sigma=pm.math.sqrt(yerr ** 2 + pm.math.exp(cheops_logs)**2), observed=y)
            pm.Deterministic("out_cheops_llk",cheops_llk)
            
            #print(self.ichlc_models[fk].check_test_point())
            #Minimizing:
            if transittype!="none" and np.any(self.lcs["cheops"][(self.lcs["cheops"]['filekey']==fk)&self.lcs["cheops"]['in_trans_all']]):
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
            #self.logger.debug(mod.)
            self.cheops_init_trace[savefname[1:]]= pm.sample(tune=300, draws=400, chains=self.n_cores, cores=self.n_cores, start=comb_soln, return_inferencedata=True)

            pickle.dump(self.cheops_init_trace[savefname[1:]],open(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+savefname+".pkl"),"wb"))
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
        # B = np.exp(-0.5*((v-dfd_priorvalue)/dfd_fitvalue)**2) * dfd_priorsigma/dfd_fitsigma
        # If B>1, the detrending param is not useful...
        assert hasattr(self,"lcs") and 'cheops' in self.lcs, "Must have initialised Cheops LC using `model.add_cheops_lc`"
        assert hasattr(self,"Rstar"), "Must have initialised stellar parameters using `model.init_starpars`"
        assert self.use_signif^self.use_bayes_fact, "Must either use the significant detrending params or use the bayes factors, not both."

        #self.lcs["cheops"]['phi']=roll_all_rollangles(self.lcs["cheops"]['phi'].values) #Performing a coherent "gap detection" of roll angles across all filekeys
        
        #Initialising Cheops LD dists:
        if not hasattr(self,'ld_dists'):
            self.ld_dists={}
        self.ld_dists['cheops']=get_lds(1200,self.Teff[:2],self.logg[:2],how='cheops')

        #Checking which transits are in which dataset:
        for ipl,pl in enumerate(self.planets):
            self.lcs["cheops"]['in_trans_'+pl]=abs((self.lcs["cheops"]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.mask_distance*self.planets[pl]['tdur']
        self.lcs["cheops"]['in_trans_all']=np.any(self.lcs["cheops"].loc[:,['in_trans_'+pl for pl in self.planets]].values,axis=1)
        
        #print(self.lcs["cheops"]['in_trans_all'].values)

        #Generating timeseries which will "fill-in the gaps" when modelling (e.g. for plotting)
        self.cheops_cad = np.nanmedian(np.diff(np.sort(self.lcs["cheops"]['time'].values)))
        self.cheops_gap_timeseries = []
        self.cheops_gap_fks = []
        for fk in self.cheops_filekeys:
            mint=np.min(self.lcs["cheops"].loc[self.lcs["cheops"]['filekey']==fk,'time'].values)
            maxt=np.max(self.lcs["cheops"].loc[self.lcs["cheops"]['filekey']==fk,'time'].values)
            self.logger.debug([mint,maxt,self.cheops_cad])
            ix_gaps=np.min(abs(np.arange(mint,maxt,self.cheops_cad)[:,None]-self.lcs["cheops"].loc[self.lcs["cheops"]['filekey']==fk,'time'].values[None,:]),axis=1)>0.66*self.cheops_cad
            self.cheops_gap_timeseries+=[np.arange(mint,maxt,self.cheops_cad)[ix_gaps]]
            self.cheops_gap_fks+=[np.tile(fk, np.sum(ix_gaps))]
        self.cheops_gap_timeseries=np.hstack(self.cheops_gap_timeseries)
        self.cheops_gap_fks=np.hstack(self.cheops_gap_fks)
        
        #Making index arrays to allow us to sort/unsort by phi:
        self.lcs["cheops"]['mask_allphi_sorting']=np.tile(-1,len(self.lcs["cheops"]))
        self.lcs["cheops"]['mask_alltime_sorting']=np.tile(-1,len(self.lcs["cheops"]))
        self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'mask_allphi_sorting']=np.argsort(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'].values).astype(int)
        self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'mask_alltime_sorting']=np.argsort(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'mask_allphi_sorting'].values).astype(int)

        #Making rollangle bin indexes:
        phibins=np.arange(np.nanmin(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'])-1.25,np.nanmax(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'])+1.25,2.5)
        self.lcs["cheops"]['phi_digi']=np.tile(-1,len(self.lcs["cheops"]))
        self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi_digi']=np.digitize(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'],phibins)

        #Initialising some stuff:
        self.cheops_filekeys = pd.unique(self.lcs["cheops"]['filekey'])
        self.ichlc_models={}
        self.cheops_mads={}
        self.norm_cheops_dat={fk:{} for fk in list(self.cheops_filekeys)+['all']}
        self.init_chefit_summaries={fk:{} for fk in self.cheops_filekeys}
        self.linear_assess={fk:{} for fk in self.cheops_filekeys}
        self.quad_assess={fk:{} for fk in self.cheops_filekeys}
        force_detrend_pars={'lin':[],'quad':[]} if force_detrend_pars is None else force_detrend_pars

        #Looping over all Cheops datasets and building individual models which we can then extract stats for each detrending parameter
        for fk in self.cheops_filekeys:
            self.logger.info("Performing Cheops-only minimisation with all detrending params for filekey "+fk)
            #Launching a pymc model
            tracename = self.cheops_only_model(fk, include_transit=self.planets!={}, force_no_dydt=force_no_dydt,**kwargs)

            var_names=[var for var in self.cheops_init_trace[tracename].posterior if '__' not in var and np.product(self.cheops_init_trace[tracename].posterior[var].shape)<6*np.product(self.cheops_init_trace[tracename].posterior['cheops_logs'].shape)]
            self.init_chefit_summaries[fk]=pm.summary(self.cheops_init_trace[tracename], var_names=var_names,round_to=7)

            #New idea: split into spline fit v time AND short-period parameter - spline. Should help with e.g. deltaT
            for par in self.init_cheops_linear_decorr_pars:
                dfd_fitvalue=self.init_chefit_summaries[fk].loc["dfd"+par,'mean']
                dfd_fitsigma=self.init_chefit_summaries[fk].loc["dfd"+par,'sd']
                dfd_priorsigma=1
                if self.use_bayes_fact:
                    self.linear_assess[fk][par] = np.exp(-0.5*((dfd_fitvalue)/dfd_fitsigma)**2) * dfd_priorsigma/dfd_fitsigma
                elif self.use_signif:
                    self.linear_assess[fk][par] = abs(dfd_fitvalue)/dfd_fitsigma
            for par in self.init_cheops_quad_decorr_pars:
                dfd_fitvalue=self.init_chefit_summaries[fk].loc["d2fd"+par+"2",'mean']
                dfd_fitsigma=self.init_chefit_summaries[fk].loc["d2fd"+par+"2",'sd']
                dfd_priorsigma=0.5
                if self.use_bayes_fact:
                    self.quad_assess[fk][par] = np.exp(-0.5*((dfd_fitvalue)/dfd_fitsigma)**2) * dfd_priorsigma/dfd_fitsigma
                elif self.use_signif:
                    self.quad_assess[fk][par] = abs(dfd_fitvalue)/dfd_fitsigma
        
        #Assessing which bayes factors suggest detrending is useful:
        self.cheops_linear_decorrs={}
        self.cheops_quad_decorrs={}
        for fk in self.cheops_filekeys:
            fk_bool=np.array([int(i==fk) for i in self.cheops_filekeys])
            if self.use_bayes_fact:
                #Bayes factor is <1sigma = significant trend = use this in the decorrelation
                self.cheops_linear_decorrs.update({"dfd"+par+"_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_cheops_linear_decorr_pars if self.linear_assess[fk][par]<1})
                self.cheops_quad_decorrs.update({"d2fd"+par+"2_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_cheops_quad_decorr_pars if self.quad_assess[fk][par]<1})
            elif self.use_signif:
                #detrend mean is >1sigma = significant trend = use this in the decorrelation
                self.cheops_linear_decorrs.update({"dfd"+par+"_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_cheops_linear_decorr_pars if self.linear_assess[fk][par]>self.signif_thresh or par in force_detrend_pars['lin']})
                self.cheops_quad_decorrs.update({"d2fd"+par+"2_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_cheops_quad_decorr_pars if self.quad_assess[fk][par]>self.signif_thresh or par in force_detrend_pars['quad']})
            
                
                self.cheops_linear_decorrs.update({"dfd"+par+"_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_cheops_linear_decorr_pars if self.linear_assess[fk][par]>self.signif_thresh})
                force_detrend_pars['quad']=self.cheops_quad_decorrs.update({"d2fd"+par+"2_"+"".join(list(fk_bool.astype(str))):[par,[fk]] for par in self.init_cheops_quad_decorr_pars if self.quad_assess[fk][par]>self.signif_thresh})

        if len(self.cheops_filekeys)>1 and make_detren_params_global:
            #Assessing which detrending parameters we can combine to a global parameter
            all_lin_params=np.unique([self.cheops_linear_decorrs[varname][0] for varname in self.cheops_linear_decorrs if self.cheops_linear_decorrs[varname][0]!='time'])
            all_quad_params=np.unique([self.cheops_quad_decorrs[varname][0] for varname in self.cheops_quad_decorrs if self.cheops_quad_decorrs[varname][0]!='time'])
            #self.logger.debug(all_lin_params)
            #self.logger.debug(all_quad_params)
            self.logger.debug([v for v in self.cheops_init_trace['cheops_only_fit_'+fk+'_trace_fixtrans'+['','_notrend'][int(force_no_dydt)]].posterior])
            for linpar in all_lin_params:
                vals=np.column_stack([self.cheops_init_trace['cheops_only_fit_'+fk+'_trace_fixtrans'+['','_notrend'][int(force_no_dydt)]].posterior["dfd"+linpar].values.ravel() for fk in self.cheops_filekeys])
                dists=[]
                # Let's make a comparison between each val/err and the combined other val/err params.
                # Anomalies will be >x sigma seperate from the group mean, while others will be OK 
                for i in range(len(self.cheops_filekeys)):
                    not_i=np.array([i2!=i for i2 in range(len(self.cheops_filekeys))])
                    #self.logger.debug(linpar, i, not_i, vals.shape, not_i.shape)
                    dists+=[abs(np.nanmedian(vals[:,i])-np.nanmedian(vals[:,not_i]))/np.sqrt(np.nanstd(vals[:,i])**2+np.nanstd(vals[:,not_i])**2)]
                if np.sum(np.array(dists)<2)>1:
                    #Removing the inidividual correlation filekeys from the cheops_linear_decorrs list:
                    #self.logger.debug(self.cheops_filekeys[np.array(dists)<2])
                    for fk in self.cheops_filekeys[np.array(dists)<2]:
                        fk_bool=np.array([int(i==fk) for i in self.cheops_filekeys])
                        varname="dfd"+linpar+"_"+"".join(list(fk_bool.astype(str)))
                        #self.logger.debug(varname,self.cheops_linear_decorrs.keys())
                        if varname in self.cheops_linear_decorrs:
                            _=self.cheops_linear_decorrs.pop(varname)
                    #Replacing them with a combined cheops_linear_decorrs:
                    fk_bool=np.array([int(fk in self.cheops_filekeys[np.array(dists)<2]) for fk in self.cheops_filekeys])
                    varname="dfd"+linpar+"_"+"".join(list(fk_bool.astype(str)))
                    self.cheops_linear_decorrs[varname]=[linpar,list(self.cheops_filekeys[np.array(dists)<2])]
                    if linpar[:3]=="sin":
                        combdat=np.hstack([np.sin(float(int(linpar[3]))*self.lcs["cheops"].loc[self.cheops_fk_mask[fk],linpar[-3:]].values*np.pi/180) for fk in self.cheops_filekeys[np.array(dists)<2]])
                    elif linpar[:3]=="cos":
                        combdat=np.hstack([np.cos(float(int(linpar[3]))*self.lcs["cheops"].loc[self.cheops_fk_mask[fk],linpar[-3:]].values*np.pi/180) for fk in self.cheops_filekeys[np.array(dists)<2]])
                    else:
                        combdat=np.hstack([self.lcs["cheops"].loc[self.cheops_fk_mask[fk],linpar].values for fk in self.cheops_filekeys[np.array(dists)<2]])
                    self.norm_cheops_dat['all'][linpar]=(combdat - np.nanmedian(combdat))/np.nanstd(combdat)

            for quadpar in all_quad_params:
                vals=np.column_stack([self.cheops_init_trace['cheops_only_fit_'+fk+'_trace_fixtrans'+['','_notrend'][int(force_no_dydt)]].posterior["d2fd"+quadpar+"2"].values.ravel() for fk in self.cheops_filekeys])
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
                        combdat=np.hstack([np.sin(float(int(quadpar[3]))*self.lcs["cheops"].loc[self.cheops_fk_mask[fk],quadpar].values*np.pi/180) for fk in self.cheops_filekeys[np.array(dists)<2]])
                    elif quadpar[:3]=="sin":
                        combdat=np.hstack([np.cos(float(int(quadpar[3]))*self.lcs["cheops"].loc[self.cheops_fk_mask[fk],quadpar].values*np.pi/180) for fk in self.cheops_filekeys[np.array(dists)<2]])
                    else:
                        combdat=np.hstack([self.lcs["cheops"].loc[self.cheops_fk_mask[fk],quadpar].values for fk in self.cheops_filekeys[np.array(dists)<2]])
                    self.norm_cheops_dat['all'][quadpar]=(combdat - np.nanmedian(combdat))/np.nanstd(combdat)
        
        self.phi_model_ix=self.make_cheops_phi_model_ix()

        #Let's iron this out and get a dictionary for each filekey of which detrending parameters are used...
        self.fk_linvars={}
        self.fk_quadvars={}
        self.logger.debug(self.cheops_linear_decorrs)
        for fk in self.cheops_filekeys:
            self.fk_linvars[fk]=[var for var in self.cheops_linear_decorrs if fk in self.cheops_linear_decorrs[var][1]]
            self.fk_quadvars[fk]=[var for var in self.cheops_quad_decorrs if fk in self.cheops_quad_decorrs[var][1]]

        if not hasattr(self,'lc_fit'):
            self.lc_fit={}

        #making the masked cheops lightcurve in the lc_fit array.
        self.lc_fit['cheops']=self.lcs["cheops"].loc[self.lcs["cheops"]["mask"].values]
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
    
    def make_cheops_phi_model_ix(self):
        """Using the keywords for the phi model (e.g. phi_model_type) to create a (N_fk x N_pts) index for how to treat each filekey"""
        if self.phi_model_type=="common":
            return np.tile(True,(1,len(self.lcs["cheops"]["time"])))
        elif self.phi_model_type=="individual":
            return np.column_stack([self.lcs["cheops"]["time"]==fk for fk in self.cheops_filekeys])
        elif "split" in self.phi_model_type:
            split_info=self.init_phot_plot_sects_noprior("cheops",n_gaps=int(self.phi_model_type.split("_")[1])-1)
            return np.column_stack([split_info[ns]['data_ix'] for ns in split_info])

    def model_comparison_cheops(self,show_detrend=True,**kwargs):
        """
        # For each filekey with a transiting planet, this perform an equivalent fit with _No_ transit. 
        # This will then allow us to derive a Bayes Factor and assess whether the transit model is justified.
        """
        self.model_comp={}
        self.comp_stats={}
        self.cheops_assess_statements={}
        for fk in self.cheops_filekeys:
            self.model_comp[fk]={}
            #Only doing this comparison on filekeys which have transits (according to prior ephemerides):
            if np.any(self.lcs["cheops"][(self.lcs["cheops"]['filekey']==fk)&self.lcs["cheops"]['in_trans_all']]):
                trace_w_trans_name = self.cheops_only_model(fk, transittype="loose", force_no_dydt=True, **kwargs)#, linpars=self.cheops_linear_decorrs[fk], quadpars=self.cheops_quad_decorrs[fk])
                #trace_w_trans['log_likelihood']=trace_w_trans.out_llk_cheops
                self.model_comp[fk]['tr_waic']  = az.waic(self.cheops_init_trace[trace_w_trans_name])
                #notrans_linpars=self.cheops_linear_decorrs[fk]+['time','deltaT'] if 'deltaT' in self.lcs["cheops"].columns else self.cheops_linear_decorrs[fk]+['time']
                #notrans_quadpars=self.cheops_quad_decorrs[fk]+['time','deltaT'] if 'deltaT' in self.lcs["cheops"].columns else self.cheops_quad_decorrs[fk]+['time']
                trace_no_trans_name = self.cheops_only_model(fk, transittype="none", force_no_dydt=False, **kwargs)#,linpars=notrans_linpars,quadpars=notrans_quadpars)
                #trace_no_trans['log_likelihood']=trace_no_trans.out_llk_cheops
                self.model_comp[fk]['notr_waic'] = az.waic(self.cheops_init_trace[trace_no_trans_name])
                self.model_comp[fk]['tr_loglik'] = np.max(self.cheops_init_trace[trace_w_trans_name].posterior['out_cheops_llk'])
                self.model_comp[fk]['notr_loglik'] = np.max(self.cheops_init_trace[trace_no_trans_name].posterior['out_cheops_llk'])
                self.model_comp[fk]['delta_loglik'] = (self.model_comp[fk]['tr_loglik'] - self.model_comp[fk]['notr_loglik'])
                
                self.model_comp[fk]['notr_BIC'] = self.model_comp[fk]['notr_waic']['p_waic'] * np.log(np.sum((self.lcs["cheops"]['filekey']==fk)&self.lcs["cheops"]['mask'])) - 2*np.log(np.max(self.cheops_init_trace[trace_no_trans_name].out_cheops_llk))
                self.model_comp[fk]['tr_BIC'] = self.model_comp[fk]['tr_waic']['p_waic'] * np.log(np.sum((self.lcs["cheops"]['filekey']==fk)&self.lcs["cheops"]['mask'])) - 2*np.log(np.max(self.cheops_init_trace[trace_w_trans_name].out_cheops_llk))
                self.model_comp[fk]['deltaBIC'] = self.model_comp[fk]['notr_BIC'] - self.model_comp[fk]['tr_BIC']
                self.model_comp[fk]['BIC_pref_model']="transit" if self.model_comp[fk]['deltaBIC']<0 else "no_transit"
                self.logger.debug(self.model_comp[fk]['notr_waic'].index)
                #self.logger.debug(self.model_comp[fk]['notr_waic'].keys())
                #self.logger.debug(self.model_comp[fk]['notr_waic']['elpd_waic'])
                #self.logger.debug(self.model_comp[fk]['notr_waic'].loc['elpd_waic'],self.model_comp[fk]['tr_waic'].loc['elpd_waic'],self.model_comp[fk]['notr_waic'].shape,self.model_comp[fk]['tr_waic'].shape)
                if "elpd_waic" in self.model_comp[fk]['tr_waic']:
                    self.model_comp[fk]['deltaWAIC']=self.model_comp[fk]['tr_waic']['elpd_waic']-self.model_comp[fk]['notr_waic']['elpd_waic']
                    waic_errs=np.sqrt(self.model_comp[fk]['tr_waic'][1]**2+self.model_comp[fk]['notr_waic'][1]**2)
                elif 'waic' in self.model_comp[fk]['tr_waic']:
                    self.model_comp[fk]['deltaWAIC']=self.model_comp[fk]['tr_waic']['waic']-self.model_comp[fk]['notr_waic']['waic']
                    waic_errs=np.sqrt(self.model_comp[fk]['tr_waic']['waic_se']**2+self.model_comp[fk]['notr_waic']['waic_se']**2)
                
                #self.model_comp[fk]['deltaWAIC']=waic_diffs.loc['waic','self']-waic_diffs.loc['waic','other']
                confidence = np.array(["strongly prefers no transit","weakly prefers no transit","weakly prefers transit","strongly prefers transit"])[np.searchsorted([-1*waic_errs,0,waic_errs],self.model_comp[fk]['deltaWAIC'])]
                self.model_comp[fk]['WAIC_pref_model']="transit" if self.model_comp[fk]['deltaWAIC']>0 else "no_transit"
                #confidence="No detection" if self.model_comp[fk]['deltaWAIC']<2 else "Moderate detection" if (self.model_comp[fk]['deltaWAIC']>=2)&(self.model_comp[fk]['deltaWAIC']<8) else "Strong detection"
                self.cheops_assess_statements[fk]=["For fk="+fk+" WAIC "+confidence+"; Delta WAIC ="+str(np.round(self.model_comp[fk]['deltaWAIC'],2)),"(BIC prefers"+self.model_comp[fk]['BIC_pref_model']+" with deltaBIC ="+str(np.round(self.model_comp[fk]['deltaBIC'],2))+"). "]
                self.logger.info(self.cheops_assess_statements[fk])
                #self.logger.info("BIC prefers",self.model_comp[fk]['BIC_pref_model'],"( Delta BIC =",np.round(self.model_comp[fk]['deltaBIC'],2),"). WAIC prefers",self.model_comp[fk]['WAIC_pref_model']," ( Delta WAIC =",np.round(self.model_comp[fk]['deltaWAIC'],2),")")
                
                for pl in self.planets:
                    if 'logror_'+pl in self.cheops_init_trace[trace_w_trans_name].posterior:
                        ror_info=[np.nanmedian(np.exp(self.cheops_init_trace[trace_w_trans_name].posterior['logror_'+pl])),np.nanstd(np.exp(self.cheops_init_trace[trace_w_trans_name].posterior['logror_'+pl]))]
                        sigdiff=abs(np.sqrt(self.planets[pl]['depth'])-ror_info[0])/ror_info[1]
                        pl_statement="For planet "+str(pl)+" the derived radius ratio is "+str(ror_info[0])[:7]+"±"+str(ror_info[1])[:7]+" which is "+str(sigdiff)[:4]+"-sigma from the expected value given TESS depth ("+str(np.sqrt(self.planets[pl]['depth']))[:7]+")"
                        self.logger.info(pl_statement)
                        self.cheops_assess_statements[fk]+=[pl_statement]

                self.plot_cheops(tracename=trace_no_trans_name, show_detrend=show_detrend, fk=fk, **kwargs)
                self.plot_cheops(tracename=trace_w_trans_name, show_detrend=show_detrend, fk=fk, **kwargs)
            elif np.any(self.lcs["cheops"][(self.lcs["cheops"]['filekey']==fk)&(~self.lcs["cheops"]['in_trans_all'])]):
                self.logger.info("No transit event during observation with fk ="+fk)
                self.cheops_assess_statements[fk]=["There appears to be no transit event during observation with fk ="+fk+" according to ephemeris."]
    
    def init_ttvs(self,**kwargs):
        """Initialise a model which uses the exoplanet TTVOrbit, which requires the integer of transit past t0."""
        for pl in self.planets:
            #Figuring out how
            min_ntr=int(np.floor((np.min(np.hstack([self.lcs[scope]['time'] for scope in self.lcs]))-self.planets[pl]['tdur']*0.5-self.planets[pl]['tcen'])/self.planets[pl]['period']))
            max_ntr=int(np.ceil((np.max(np.hstack([self.lcs[scope]['time'] for scope in self.lcs]))+self.planets[pl]['tdur']*0.5-self.planets[pl]['tcen'])/self.planets[pl]['period']))
            if 'tcens' not in self.planets[pl]:
                tcens=self.planets[pl]['tcen']+np.arange(min_ntr,max_ntr)*self.planets[pl]['period']
                ix=np.min(abs(tcens[:,None]-np.hstack([self.lcs[scope]['time'] for scope in self.lcs])[None,:]),axis=1)<self.planets[pl]['tdur']*0.5
                self.planets[pl]['init_transit_times']=tcens[ix]
                self.planets[pl]['init_transit_inds']=np.arange(min_ntr,max_ntr)[ix]
                self.planets[pl]['n_trans']=np.sum(ix)
            else:
                self.planets[pl]['init_transit_times']=self.planets[pl]['tcens']
                self.planets[pl]['init_transit_inds']=np.round((self.planets[pl]['tcens']-self.planets[pl]['tcen'])/self.planets[pl]['period']).astype(int)
                self.planets[pl]['n_trans']=len(self.planets[pl]['tcens'])
            self.planets[pl]['init_transit_inds']-=np.min(self.planets[pl]['init_transit_inds'])
            

    def init_model(self, **kwargs):
        """Initialising full TESS+CHEOPS model.
        Important global inputs include:
        - assume_circ - bool - Assume circular orbits (no ecc & omega)? Default: False
        - timing_sd_durs - timing_sd_durs - float - The standard deviation to use (in units of transit duration) when setting out timing priors. Default: 0.33
        - fit_ttvs - bool - Fit a TTVorbit exoplanet model which searches for TTVs. Default: False
        - split_periods - dict - Fit for multiple split periods. Input must be None or a dict matching mod.planets with grouped indexes for those transits to group. Default: None
        - ttv_prior - str - What prior to have for individual transit times. Possibilities: "Normal","Uniform","BoundNormal". Default: 'Normal'
        - fit_phi_gp - bool - co-fit a GP to the roll angle. Default: False
        - fit_phi_spline' - bool - co-fit a spline model to the roll angle. Default: True
        - spline_bkpt_cad - float - The spline breakpoint cadence in degrees. Default is 9deg. Default: 9.
        - spline_order - bool - Thespline order. Defaults to 3 (cubic). Default: 3
        - phi_model_type - bool - How to fit the same roll angle GP trend. Either "individual" (different model for each visit), "common" (same model for each visit), or "split_2" (different models for each N season); formerly common_phi_model. Default: 'common'
        - ecc_prior - str - 'uniform', 'kipping' or 'vaneylen'. If 'auto' we decide based on multiplicity. Default: 'auto'
        - spar_param - str - The stellar parameter to use when modelling. Either Mstar, logg or rhostar. Default: 'Mstar'
        - spar_prior - str - The prior to use on the second stellar parameter. Either constr, loose of logloose. Default: 'constr'
        - constrain_lds - bool - Use constrained LDs from model or unconstrained? Default: True
        - ld_mult - float - How much to multiply theoretical LD param uncertainties. Default: 3
        - fit_contam - bool - Fit for "second light" (i.e. a binary or planet+blend). Default: False

        """

        self.update(**kwargs)

        assert not self.use_multinest, "Multinest is not currently possible"
        assert not (self.fit_flat&self.fit_gp), "Cannot both flatten data and fit GP. Choose one"
        assert not (self.fit_phi_spline&self.fit_phi_gp), "Cannot both fit spline and GP to phi model. Choose one"
        
        self.init_ttvs()#Doing this even if we're not modelling TTVs as "n_trans" is a useful quantity to know
            
        self.model_params={}
        with pm.Model() as self.model:
            # -------------------------------------------
            #          Stellar parameters
            # -------------------------------------------
            self.model_params['Teff']=pm.TruncatedNormal("Teff",lower=0,mu=self.Teff[0],sigma=self.Teff[1])
            self.model_params['Rs']=pm.TruncatedNormal("Rs", lower=0,mu=self.Rstar[0],sigma=self.Rstar[1])
            if self.spar_param=='Mstar':
                if self.spar_prior=='constr':
                    self.model_params['Ms'] = pm.TruncatedNormal("Ms", lower=0,mu=self.Mstar[0], sigma=self.Mstar[1]) #Ms and logg are interchangeably deterministic
                elif self.spar_prior=='loose':
                    self.model_params['Ms'] = pm.TruncatedNormal("Ms", lower=0,mu=self.Mstar[0],sigma=0.33*self.Mstar[0]) #Ms and logg are interchangeably deterministic
                elif self.spar_prior=='logloose':
                    self.model_params['logMs'] = pm.Bound("logMs",lower=np.log(self.Mstar[0])-2,upper=np.log(self.Mstar[0])+2) #Ms and logg are interchangeably deterministic
                    self.model_params['Ms'] = pm.Deterministic("Ms", pm.math.exp(self.model_params['logMs']))
                optim_spar=self.model_params['logMs'] if self.spar_prior=='logloose' else self.model_params['Ms']
                self.model_params['logg'] = pm.Deterministic("logg",pm.math.log(self.model_params['Ms']/self.model_params['Rs']**2)/pm.math.log(10)+4.41) #Ms and logg are interchangeably deterministic
                self.model_params['rhostar'] = pm.Deterministic("rhostar",self.model_params['Ms']/self.model_params['Rs']**3) #Ms and logg are interchangeably deterministic
            elif self.spar_param=='logg':
                if self.spar_prior=='constr':
                    self.model_params['logg'] = pm.TruncatedNormal("logg", lower=0,mu=self.logg[0],sigma=self.logg[1]) #Ms and logg are interchangeably deterministic
                elif self.spar_prior=='loose':
                    self.model_params['logg'] = pm.TruncatedNormal("logg", lower=0,mu=self.logg[0],sigma=1) #Ms and logg are interchangeably deterministic
                optim_spar=self.model_params['logg']
                self.model_params['Ms'] = pm.Deterministic("Ms",pm.math.power(10,self.model_params['logg']-4.41)*self.model_params['Rs']**2) #Ms and logg are interchangeably deterministic
                self.model_params['rhostar'] = pm.Deterministic("rhostar", self.model_params['Ms']/self.model_params['Rs']**3) #Ms and logg are interchangeably deterministic
            elif self.spar_param=='rhostar':
                if self.spar_prior=='logloose':
                    self.model_params['logrhostar'] = pm.Uniform("logrhostar",lower=np.log(self.rhostar[0])-2,upper=np.log(self.rhostar[0])+2,initval=self.rhostar[0]) #Ms and logg are interchangeably deterministic
                    self.model_params['rhostar'] = pm.Deterministic("rhostar", pm.math.exp(self.model_params['logrhostar']))
                elif self.spar_prior=='constr':
                    self.model_params['rhostar'] = pm.TruncatedNormal("rhostar", lower=0, mu=self.rhostar[0],sigma=self.rhostar[1]) #Ms and logg are interchangeably deterministic
                elif self.spar_prior=='loose':
                    self.model_params['rhostar'] = pm.TruncatedNormal("rhostar", lower=0, mu=self.rhostar[0],sigma=0.33*self.rhostar[0]) #Ms and logg are interchangeably deterministic
                optim_spar=self.model_params['logrhostar'] if self.spar_prior=='logloose' else self.model_params['rhostar']
                self.model_params['Ms'] = pm.Deterministic("Ms",self.model_params['Rs']**3*self.model_params['rhostar']) #Ms and logg are interchangeably deterministic
                self.model_params['logg'] = pm.Deterministic("logg",np.log10(self.model_params['Ms']/self.model_params['Rs']**2)+4.41) #Ms and logg are interchangeably deterministic

            # -------------------------------------------
            #             Contamination
            # -------------------------------------------
            # Using the detected companion's I and V mags to constrain Cheops and TESS dilution:
            if len(self.planets)>0:
                if self.fit_contam:
                    self.model_params['deltaImag_contam'] = pm.Uniform("deltaImag_contam", upper=12, lower=2.5)
                    self.model_params['tess_mult'] = pm.Deterministic("tess_mult",(1+pm.math.power(2.511,-1*self.model_params['deltaImag_contam']))) #Factor to multiply normalised lightcurve by
                    if 'k2' in self.lcs or 'kepler' in self.lcs or "cheops" in self.lcs:
                        self.model_params['deltaVmag_contam'] = pm.Uniform("deltaVmag_contam", upper=12, lower=2.5)
                        self.model_params['cheops_mult'] = pm.Deterministic("cheops_mult",(1+pm.math.power(2.511,-1*self.model_params['deltaVmag_contam']))) #Factor to multiply normalised lightcurve by
                else:
                    for scope in self.lcs:
                        self.model_params[scope+'_mult']=1.0

                self.model_params['u_stars']={}
                for scope in self.ld_dists:
                    if self.constrain_lds:
                        self.model_params['u_stars'][scope] = pm.TruncatedNormal("u_star_"+scope, lower=0.0, upper=1.0,
                                                                        mu=np.clip(np.nanmedian(self.ld_dists[scope],axis=0),0,1),
                                                                        sigma=np.clip(np.nanstd(self.ld_dists[scope],axis=0),0.1,1.0), 
                                                                        shape=2, initval=np.clip(np.nanmedian(self.ld_dists[scope],axis=0),0,1))
                    else:
                        self.model_params['u_stars'][scope] = xo.distributions.QuadLimbDark("u_star_"+scope, initval=np.array([0.3, 0.2]))
                # -------------------------------------------
                # Initialising parameter dicts for each planet
                # -------------------------------------------
                self.model_params['orbit']={}
                self.model_params['t0']={};self.model_params['P']={};self.model_params['vels']={};self.model_params['tdur']={}
                self.model_params['b']={};self.model_params['rpl']={};self.model_params['logror']={};self.model_params['ror']={}
                self.model_params['a_Rs']={};self.model_params['sma']={};self.model_params['S_in']={};self.model_params['Tsurf_p']={}
                min_ps={pl:self.planets[pl]['period']*(1-1.5*self.timing_sd_durs*self.planets[pl]['tdur']/(np.ptp(np.hstack([self.lc_fit[scope]['time'] for scope in self.lc_fit])))) for pl in self.planets}
                max_ps={pl:self.planets[pl]['period']*(1+1.5*self.timing_sd_durs*self.planets[pl]['tdur']/(np.ptp(np.hstack([self.lc_fit[scope]['time'] for scope in self.lc_fit])))) for pl in self.planets}
                self.logger.debug([min_ps,max_ps,[self.planets[pl]['period'] for pl in self.planets],np.ptp(np.hstack([self.lc_fit[scope]['time'] for scope in self.lc_fit]))])
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

                    if self.planets[pl]['n_trans']>=2 and (self.fit_ttvs or (self.split_periods is not None and pl in self.split_periods and len(self.split_periods[pl])>1 and self.split_periods[pl]!=range(self.planets[pl]['n_trans']))):
                        #To model TTVS, we need >2 transits. Only split period planets which do not have all the transits in one bin must be considered here:

                        #Initialising transit times:
                        # self.model_params['transit_times'][pl]=pm.Uniform("transit_times_"+pl, 
                        #                                                     upper=self.planets[pl]['init_transit_times']+self.planets[pl]['tdur']*self.timing_sd_durs,
                        #                                                     lower=self.planets[pl]['init_transit_times']-self.planets[pl]['tdur']*self.timing_sd_durs,
                        #                                                     shape=len(self.planets[pl]['init_transit_times']), initval=self.planets[pl]['init_transit_times'])
                        if self.fit_ttvs:
                            self.model_params['transit_times'][pl]=[]
                            for i in range(len(self.planets[pl]['init_transit_times'])):
                                if self.ttv_prior.lower()=='uniform':
                                    self.model_params['transit_times'][pl].append(pm.Uniform("transit_times_"+pl+"_"+str(i), 
                                                                                    upper=self.planets[pl]['init_transit_times'][i]+self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                                    lower=self.planets[pl]['init_transit_times'][i]-self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                                    initval=self.planets[pl]['init_transit_times'][i]))
                                elif self.ttv_prior.lower()=='normal':
                                    self.model_params['transit_times'][pl].append(pm.Normal("transit_times_"+pl+"_"+str(i), 
                                                                                    mu=self.planets[pl]['init_transit_times'][i],sigma=self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                                    initval=self.planets[pl]['init_transit_times'][i]))
                                elif self.ttv_prior.lower()=='boundnormal':
                                    self.model_params['transit_times'][pl].append(pm.TruncatedNormal("transit_times_"+pl+"_"+str(i),
                                                                                        lower=self.planets[pl]['init_transit_times'][i]-self.planets[pl]['tdur']*2*self.timing_sd_durs,
                                                                                        upper=self.planets[pl]['init_transit_times'][i]+self.planets[pl]['tdur']*2*self.timing_sd_durs,
                                                                                                    mu=self.planets[pl]['init_transit_times'][i],
                                                                                                    sigma=self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                                                    initval=self.planets[pl]['init_transit_times'][i]))
    

                        elif self.split_periods is not None:
                            self.model_params['transit_times'][pl]=[]
                            #Splitting periods into smaller periods of 3 or 4 transits.
                            #split_periods must be {pl:[[0,1,2],[3,4,5],[6,7,8,9]]} <- i.e. transit indexes split into groups
                            assert np.all(np.hstack(self.split_periods[pl])==np.arange(self.planets[pl]['n_trans'])),"split_periods must be a dict where all transits are split into groups by index - we have "+",".join(list(np.hstack(self.split_periods[pl]).astype(str)))+" but we need "+str(self.planets[pl]['n_trans'])+" transits at epochs "+",".join(list(self.planets[pl]['init_transit_times'].astype(str)))
                            for ngroup,group in enumerate(self.split_periods[pl]):
                                self.model_params['split_t0'][pl][ngroup]=pm.Normal("split_t0_"+pl+"_"+str(ngroup), mu=self.planets[pl]['init_transit_times'][self.split_periods[pl][ngroup][0]], 
                                                                    sigma=2*self.planets[pl]['tcen_err'])
                                self.model_params['split_P'][pl][ngroup]=pm.TruncatedNormal("split_P_"+pl+"_"+str(ngroup), 
                                                                            lower=min_ps[pl], upper=max_ps[pl],
                                                                            mu=self.planets[pl]['period'],
                                                                            sigma=np.clip(self.planets[pl]['period_err'],0,(max_ps[pl]-self.planets[pl]['period'])))
                                for ni in group:
                                    n_trans_diff=self.planets[pl]['init_transit_inds'][ni]-self.planets[pl]['init_transit_inds'][group[0]]
                                    self.model_params['transit_times'][pl].append(pm.Deterministic("transit_times_"+pl+"_"+str(ni),
                                                                                                self.model_params['split_t0'][pl][ngroup]+self.model_params['split_P'][pl][ngroup]*n_trans_diff))
                                    

                        #self.model_params['transit_times'][pl].append(pm.Normal("transit_times_"+pl, mu=self.planets[pl]['init_transit_times'], sigma=self.planets[pl]['tdur']*self.timing_sd_durs,
                        #                                                        shape=len(self.planets[pl]['init_transit_times']), initval=self.planets[pl]['init_transit_times']))
                    else:
                        self.model_params['t0'][pl] = pm.Normal("t0_"+pl, mu=self.planets[pl]['tcen'], sigma=2*self.planets[pl]['tcen_err'],
                                                                initval=np.random.normal(self.planets[pl]['tcen'],1e-6))
                        self.model_params['P'][pl] = pm.TruncatedNormal("P_"+pl,lower=min_ps[pl], upper=max_ps[pl],
                                    mu=self.planets[pl]['period'],sigma=np.clip(self.planets[pl]['period_err'],0,(max_ps[pl]-self.planets[pl]['period'])),
                                    initval=np.random.normal(self.planets[pl]['period'],1e-6))

                    # Wide log-normal prior for semi-amplitude
                    if hasattr(self,'rvs'):
                        if self.rv_mass_prior=='logK':
                            self.model_params['logK'][pl] = pm.Normal("logK_"+pl, mu=-1, sigma=10, initval=1.5)
                            self.model_params['K'][pl] =pm.Deterministic("K_"+pl,pm.math.exp(self.model_params['logK'][pl]))
                        elif self.rv_mass_prior=='K':
                            self.model_params['K'][pl] = pm.Normal("K_"+pl, mu=2, sigma=1, initval=1.5)
                            self.model_params['logK'][pl] =pm.Deterministic("logK_"+pl,pm.math.log(self.model_params['K'][pl]))
                        elif self.rv_mass_prior=='popMp':
                            if len(self.planets)>1:
                                rads=np.array([109.2*self.planets[pl]['rprs']*self.Rstar[0] for pl in self.planets])
                                mu_mps = 5.75402469 - (rads<=12.2)*(rads>=1.58)*(4.67363091 -0.38348534*rads) - \
                                                                        (rads<1.58)*(5.81943841-3.81604756*np.log(rads))
                                sd_mps= (rads<=8)*(0.07904372*rads+0.24318296) + (rads>8)*(0-0.02313261*rads+1.06765343)
                                self.model_params['logMp'][pl] = pm.Normal('logMp_'+pl,mu=mu_mps,sigma=sd_mps)
                            else:
                                rad=109.2*self.planets[pl]['rprs']*self.Rstar[0]
                                mu_mps = 5.75402469 - (rad<=12.2)*(rad>=1.58)*(4.67363091 -0.38348534*rad) - \
                                                                        (rad<1.58)*(5.81943841-3.81604756*np.log(rad))
                                sd_mps= (rad<=8)*(0.07904372*rad+0.24318296) + (rad>8)*(0-0.02313261*rad+1.06765343)
                                self.model_params['logMp'][pl] = pm.Normal('logMp_'+pl,mu=mu_mps,sigma=sd_mps)
                    # Eccentricity & argument of periasteron
                    if not self.assume_circ:
                        #BoundedBeta = pm.Bound(pm.Beta, lower=1e-5, upper=1-1e-5)
                        self.model_params['ecc'][pl] = pm.Beta("ecc_"+pl, alpha=0.867 ,beta=3.03, initval=0.05)
                        self.model_params['omega'][pl] = pmx.angle("omega_"+pl)
                    '''
                    #This was to model a non-transiting companion:
                    P_nontran = pm.Normal("P_nontran", mu=27.386209624, sigma=2*0.04947295)
                    logK_nontran = pm.Normal("logK_nontran", mu=2,sigma=10, initval=2)
                    Mpsini_nontran = pm.Deterministic("Mp_nontran", pm.math.exp(logK_nontran) * 28.439**-1 * Ms**(2/3) * (P_nontran/365.25)**(1/3) * 317.8)
                    t0_nontran = pm.Uniform("t0_nontran", lower=np.nanmedian(rv_x)-27.386209624*0.55, upper=np.nanmedian(rv_x)+27.386209624*0.55)
                    '''
                    if self.tight_depth_prior:
                        self.model_params['logror'][pl] = pm.Normal("logror_"+pl, mu=np.log(np.sqrt(self.planets[pl]['depth'])), sigma=0.2)
                    else:
                        self.model_params['logror'][pl] = pm.Uniform("logror_"+pl, lower=np.log(0.001), upper=np.log(0.1), 
                                                                    initval=np.log(np.sqrt(self.planets[pl]['depth'])))

                    self.model_params['ror'][pl] = pm.Deterministic("ror_"+pl,pm.math.exp(self.model_params['logror'][pl]))
                    self.model_params['rpl'][pl] = pm.Deterministic("rpl_"+pl,109.1*self.model_params['ror'][pl]*self.model_params['Rs'])
                    self.model_params['b'][pl] = xo.distributions.ImpactParameter("b_"+pl, ror=self.model_params['ror'][pl], initval=self.planets[pl]['b'])
                    
                    if (self.fit_ttvs or self.split_periods is not None) and self.planets[pl]['n_trans']>2 and pl in self.split_periods and len(self.split_periods[pl])>1 and self.split_periods[pl]!=range(self.planets[pl]['n_trans']):
                        if self.assume_circ:
                            self.model_params['orbit'][pl] = xo.orbits.TTVOrbit(b=[self.model_params['b'][pl]], 
                                                            transit_times=[self.model_params['transit_times'][pl]], 
                                                            transit_inds=[self.planets[pl]['init_transit_inds']], 
                                                            r_star=self.model_params['Rs'], 
                                                            m_star=self.model_params['Ms'])
                        else:
                            self.model_params['orbit'][pl] = xo.orbits.TTVOrbit(b=[self.model_params['b'][pl]], 
                                                            transit_times=[self.model_params['transit_times'][pl]], 
                                                            transit_inds=[self.planets[pl]['init_transit_inds']], 
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
                                self.model_params['Mp'][pl] = pm.Deterministic("Mp_"+pl, pm.math.exp(self.model_params['logK'][pl]) * 28.439**-1 * self.model_params['Ms']**(2/3) * (self.model_params['P'][pl]/365.25)**(1/3) * 317.8)
                            else:
                                self.model_params['Mp'][pl] = pm.Deterministic("Mp_"+pl, pm.math.exp(self.model_params['logK'][pl]) * 28.439**-1 * (1-self.model_params['ecc'][pl]**2)**(0.5) * self.model_params['Ms']**(2/3) * (self.model_params['P'][pl]/365.25)**(1/3) * 317.8)
                        else:
                            self.model_params['Mp'][pl] = pm.Deterministic("Mp_"+pl, pm.math.exp(self.model_params['logMp'][pl]))
                            if self.assume_circ:
                                self.model_params['K'][pl] = pm.Deterministic("K_"+pl, self.model_params['Mp'][pl] / (28.439**-1 * self.model_params['Ms']**(2/3) * (self.model_params['P'][pl]/365.25)**(1/3) * 317.8))
                            else:
                                self.model_params['K'][pl] = pm.Deterministic("K_"+pl, self.model_params['Mp'][pl] / (28.439**-1 * (1-self.model_params['ecc'][pl]**2)**(0.5) * self.model_params['Ms']**(2/3) * (self.model_params['P'][pl]/365.25)**(1/3) * 317.8))
                            self.model_params['logK'][pl] = pm.Deterministic("logK_"+pl, pm.math.exp(self.model_params['K'][pl]))
                        self.model_params['rho_p'][pl] = pm.Deterministic("rho_p_gcm3_"+pl,5.513*self.model_params['Mp'][pl]/self.model_params['rpl'][pl]**3)
                    
                    self.model_params['a_Rs'][pl]=pm.Deterministic("a_Rs_"+pl,self.model_params['orbit'][pl].a/self.model_params['Rs'])
                    self.model_params['sma'][pl]=pm.Deterministic("sma_"+pl,self.model_params['a_Rs'][pl]*0.00465)
                    self.model_params['S_in'][pl]=pm.Deterministic("S_in_"+pl,((695700000*self.model_params['Rs'])**2.*5.67e-8*self.model_params['Teff']**4)/(1.496e11*self.model_params['sma'][pl])**2.)
                    self.model_params['Tsurf_p'][pl]=pm.Deterministic("Tsurf_p_"+pl,(((695700000*self.model_params['Rs'])**2.*self.model_params['Teff']**4.*(1.-0.2))/(4*(1.496e11*self.model_params['sma'][pl])**2.))**(1./4.))
                    
                    #Getting the transit duration:
                    self.model_params['vels'][pl] = self.model_params['orbit'][pl].get_relative_velocity(self.model_params['t0'][pl])
                    self.model_params['tdur'][pl]=pm.Deterministic("tdur_"+pl,(2*self.model_params['Rs']*pm.math.sqrt((1+self.model_params['ror'][pl])**2-self.model_params['b'][pl]**2))/pm.math.sqrt(self.model_params['vels'][pl][0]**2 + self.model_params['vels'][pl][1]**2))

            # -------------------------------------------
            #                    RVs:
            # -------------------------------------------
            if hasattr(self,'rvs'):
                self.logger.debug(self.planets.keys())
                for pl in self.planets:
                    self.model_params['vrad_x'][pl]  = pm.Deterministic("vrad_x_"+pl,self.model_params['orbit'][pl].get_radial_velocity(self.rvs['time'], K=pm.math.exp(self.model_params['logK'][pl]))) #
                    #pm.math.printing.Print("vrad_x_"+pl)(self.model_params['vrad_x'][pl])
                    # Also define the model on a fine grid as computed above (for plotting)
                    self.model_params['vrad_t'][pl] = pm.Deterministic("vrad_t_"+pl,self.model_params['orbit'][pl].get_radial_velocity(self.rv_t, K=pm.math.exp(self.model_params['logK'][pl])))

                '''orbit_nontran = xo.orbits.KeplerianOrbit(r_star=Rs, m_star=Ms, period=P_nontran, t0=t0_nontran)
                vrad_x = pm.Deterministic("vrad_x",pm.math.stack([orbit.get_radial_velocity(rv_x, K=pm.math.exp(logK))[:,0],
                                                                orbit.get_radial_velocity(rv_x, K=pm.math.exp(logK))[:,1],
                                                                orbit_nontran.get_radial_velocity(rv_x, K=pm.math.exp(logK_nontran))],axis=1))
                '''

                # Define the background model
                self.model_params['rv_offsets'] = pm.Normal("rv_offsets",
                                        mu=np.array([self.rv_medians[i] for i in self.rv_medians]),
                                        sigma=np.array([self.rv_stds[i] for i in self.rv_stds])*5,
                                        shape=len(self.rv_medians))
                ##pm.math.printing.Print("offsets")(pm.math.sum(offsets*self.rv_instr_ix,axis=1))

                #Only doing npoly-1 coefficients (and adding a leading zero for the vander) as we have seperate telescope offsets.
                if self.npoly_rv>1:
                    self.model_params['rv_trend'] = pm.Normal("rv_trend", mu=0, sigma=10.0 ** -np.arange(self.npoly_rv)[::-1], shape=self.npoly_rv)
                    self.model_params['bkg_x'] = pm.Deterministic("bkg_x", pm.math.sum(self.model_params['rv_offsets']*self.rv_instr_ix,axis=1) + pm.math.dot(np.vander(self.rvs['time'] - self.rv_x_ref, self.npoly_rv)[:,:-1], self.model_params['rv_trend'][:-1]))
                else:
                    self.model_params['bkg_x'] = pm.Deterministic("bkg_x", pm.math.sum(self.model_params['rv_offsets']*self.rv_instr_ix,axis=1))

                # Define the RVs at the observed times  
                if len(self.planets)>1:
                    self.model_params['rv_model_x'] = pm.Deterministic("rv_model_x", self.model_params['bkg_x'] + pm.math.sum([self.model_params['vrad_x'][pl] for pl in self.planets], axis=0))
                else:
                    self.model_params['rv_model_x'] = pm.Deterministic("rv_model_x", self.model_params['bkg_x'] + self.model_params['vrad_x'][list(self.planets.keys())[0]])

                '''vrad_t = pm.Deterministic("vrad_t",pm.math.stack([orbit.get_radial_velocity(rv_t, K=pm.math.exp(logK))[:,0],
                                                                orbit.get_radial_velocity(rv_t, K=pm.math.exp(logK))[:,1],
                                                                orbit_nontran.get_radial_velocity(rv_t, K=pm.math.exp(logK_nontran))],axis=1))
                '''
                #orbit.get_radial_velocity(rv_t, K=pm.math.exp(logK)))
                if self.npoly_rv>1:
                    self.model_params['bkg_t'] = pm.Deterministic("bkg_t", pm.math.dot(np.vander(self.rv_t - self.rv_x_ref, self.npoly_rv),self.model_params['rv_trend']))
                    if len(self.planets)>1:
                        #pm.math.printing.Print("rv_sum")(pm.math.sum([self.model_params['vrad_t'][pl] for pl in self.planets], axis=0))
                        #pm.math.printing.Print("rv_bkg")(self.model_params['bkg_t'])
                        self.model_params['rv_model_t'] = pm.Deterministic("rv_model_t", self.model_params['bkg_t'] + pm.math.sum([self.model_params['vrad_t'][pl] for pl in self.planets], axis=0))
                    else:
                        self.model_params['rv_model_t'] = pm.Deterministic("rv_model_t", self.model_params['bkg_t'] + self.model_params['vrad_t'][list(self.planets.keys())[0]])

                else:
                    if len(self.planets)>1:
                        self.model_params['rv_model_t'] = pm.Deterministic("rv_model_t", pm.math.sum([self.model_params['vrad_t'][pl] for pl in self.planets], axis=0))
                    else:
                        self.model_params['rv_model_t'] = pm.Deterministic("rv_model_t", self.model_params['vrad_t'][list(self.planets.keys())[0]])

            # -------------------------------------------
            #                 PHOT GP:
            # -------------------------------------------
            if self.fit_gp and ('tess' in self.lcs or 'k2' in self.lcs or 'kepler' in self.lcs):
                minmax={}
                # Here we interpolate the histograms of the pre-trained GP samples as the input prior for each:
                for scope in self.lcs:
                    if scope!="cheops":
                        minmax[scope+'_logs']=np.percentile(self.oot_gp_trace.posterior[scope+"_logs"],[0.5,99.5])
                        self.model_params[scope+'_logs']=pm.Interpolated(scope+'_logs',x_points=np.linspace(minmax[scope+'_logs'][0],minmax[scope+'_logs'][1],201)[1::2],
                                                pdf_points=np.histogram(self.oot_gp_trace.posterior[scope+'_logs'],np.linspace(minmax[scope+'_logs'][0],minmax[scope+'_logs'][1],101))[0]
                                                )    
                    #Already defined below:
                    # else:
                    #    self.model_params[scope+'_logs'] = pm.Normal(scope+'_logs', mu=np.log(np.nanmedian(abs(np.diff(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'].values)))), sigma=3)

                minmax['sigma']=np.percentile(self.oot_gp_trace.posterior["sigma"],[0.5,99.5])
                self.model_params['phot_sigma']=pm.Interpolated("phot_sigma",x_points=np.linspace(minmax['sigma'][0],minmax['sigma'][1],201)[1::2],
                                        pdf_points=np.histogram(self.oot_gp_trace.posterior["sigma"],np.linspace(minmax['sigma'][0],minmax['sigma'][1],101))[0]
                                        )
                minmax["w0"]=np.percentile(self.oot_gp_trace.posterior["w0"],[0.5,99.5])
                self.model_params['phot_w0']=pm.Interpolated("phot_w0",x_points=np.linspace(minmax["w0"][0],minmax["w0"][1],201)[1::2],
                                            pdf_points=np.histogram(self.oot_gp_trace.posterior["w0"],np.linspace(minmax["w0"][0],minmax["w0"][1],101))[0]
                                            )
                self.model_params['phot_kernel'] = pymc_terms.SHOTerm(sigma=self.model_params['phot_sigma'], 
                                                                        w0=self.model_params['phot_w0'], Q=1/np.sqrt(2))#, mean = phot_mean)

                for scope in self.lcs:
                    if scope=='cheops':
                        self.model_params[scope+'_logs']=pm.Normal(scope+'_logs', mu=np.log(np.std(self.lc_fit[scope]['flux'].values)), sigma=1)
                    else:
                        minmax[scope+"_mean"]=np.percentile(self.oot_gp_trace.posterior[scope+"_mean"],[0.5,99.5])
                        self.model_params[scope+'_mean']=pm.Interpolated(scope+"_mean",
                                                x_points=np.linspace(minmax[scope+'_mean'][0],minmax[scope+'_mean'][1],201)[1::2],
                                                pdf_points=np.histogram(self.oot_gp_trace.posterior[scope+'_mean'],np.linspace(minmax[scope+'_mean'][0],minmax[scope+'_mean'][1],101))[0]
                                                )
                        self.model_params[scope+'_gp'] = celerite2.pymc.GaussianProcess(self.model_params['phot_kernel'], self.lc_fit[scope]['time'].values, mean=self.model_params[scope+'_mean'])#,
                        #                                                                yerr=np.sqrt(self.lc_fit[scope]['flux_err'].values ** 2 + pm.math.exp(self.model_params[scope+'_logs'])**2))
                        self.model_params[scope+'_gp'].compute(self.lc_fit[scope]['time'].values,yerr=np.sqrt(self.lc_fit[scope]['flux_err'].values ** 2 + pm.math.exp(self.model_params[scope+'_logs'])**2))
                #pm.math.dot(self.lc_fit_src_index,pm.math.exp([logs[scope] for scope in logs])
                #self.model_params['gp_tess'].compute(self.lc_fit['time'].values, , quiet=True)
            else:
                for scope in self.lcs:
                    logmad=np.log(np.nanmedian(abs(np.diff(self.lc_fit[scope]['flux'].values))))
                    self.model_params[scope+'_logs']=pm.TruncatedNormal(scope+'_logs', mu=logmad+0.5, sigma=1, lower=logmad-30,upper=logmad+5,initval=logmad+0.5)
            # -------------------------------------------
            #         Cheops detrending (linear)
            # -------------------------------------------
            if "cheops" in self.lcs:
                self.logger.debug("FKS="+",".join(self.cheops_filekeys))

                #Initialising linear (and quadratic) parameters:
                self.model_params['linear_decorr_dict']={}#i:{} for i in self.cheops_filekeys}
                
                for decorr in self.cheops_linear_decorrs:
                    varname=self.cheops_linear_decorrs[decorr][0]
                    fks=self.cheops_linear_decorrs[decorr][1]
                    if varname=='time':
                        self.model_params['linear_decorr_dict'][decorr]=pm.Normal(decorr,mu=0,sigma=np.nanmedian([np.ptp(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'time'])/self.cheops_mads[fk] for fk in fks]),
                                                                                    initval=np.nanmedian([self.init_chefit_summaries[fk].loc["dfd"+varname,'mean'] for fk in fks]))
                    else:
                        self.model_params['linear_decorr_dict'][decorr]=pm.Normal(decorr,mu=0,sigma=np.nanmedian([self.cheops_mads[fk] for fk in fks]),
                                                                                    initval=np.nanmedian([self.init_chefit_summaries[fk].loc["dfd"+varname,'mean'] for fk in fks]))
                
                self.model_params['quad_decorr_dict']={}#{i:{} for i in self.cheops_filekeys}
                for decorr in self.cheops_quad_decorrs:
                    varname=self.cheops_quad_decorrs[decorr][0]
                    fks=self.cheops_quad_decorrs[decorr][1]
                    if varname=='time':
                        self.model_params['quad_decorr_dict'][decorr]=pm.Normal(decorr,mu=0,sigma=np.nanmedian([np.ptp(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'time'])/self.cheops_mads[fk] for fk in fks]),
                                                                                initval=np.nanmedian([self.init_chefit_summaries[fk].loc["dfd"+varname,'mean'] for fk in fks]))
                    else:
                        self.model_params['quad_decorr_dict'][decorr]=pm.Normal(decorr,mu=0,sigma=np.nanmedian([self.cheops_mads[fk] for fk in fks]),
                                                                                initval=np.nanmedian([self.init_chefit_summaries[fk].loc["dfd"+varname,'mean'] for fk in fks]))

                #Creating the flux correction vectors:
                self.model_params['cheops_obs_means']={};self.model_params['cheops_flux_cor']={}
                
                for fk in self.cheops_filekeys:
                    self.model_params['cheops_obs_means'][fk]=pm.Normal("cheops_mean_"+str(fk),mu=np.nanmedian(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values),
                                                                    sigma=np.nanstd(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values),initval=0)
                    
                    if len(self.fk_quadvars[fk])>0:
                        #Linear and quadratic detrending
                        self.model_params['cheops_flux_cor'][fk] = pm.Deterministic("cheops_flux_cor_"+str(fk),self.model_params['cheops_obs_means'][fk] + \
                                                                                        pm.math.sum([self.model_params['linear_decorr_dict'][lvar]*self.norm_cheops_dat[fk][lvar.split('_')[0][3:]] for lvar in self.fk_linvars[fk]], axis=0) + \
                                                                                        pm.math.sum([self.model_params['quad_decorr_dict'][qvar]*self.norm_cheops_dat[fk][qvar.split('_')[0][4:-1]]**2 for qvar in self.fk_quadvars[fk]], axis=0))
                    elif len(self.fk_linvars[fk])>0:
                        #Linear detrending only
                        #pm.math.printing.Print("obs_mean_"+fk)(self.model_params['cheops_obs_means'][fk])
                        self.model_params['cheops_flux_cor'][fk] = pm.Deterministic("cheops_flux_cor_"+str(fk),self.model_params['cheops_obs_means'][fk] + \
                                                                                        pm.math.sum([self.model_params['linear_decorr_dict'][lvar]*self.norm_cheops_dat[fk][lvar.split('_')[0][3:]] for lvar in self.fk_linvars[fk]], axis=0))
                    else:
                        #No detrending at all
                        self.model_params['cheops_flux_cor'][fk] = pm.Deterministic("cheops_flux_cor_"+str(fk), pm.math.tile(self.model_params['cheops_obs_means'][fk],np.sum(self.cheops_fk_mask[fk])))
                # -------------------------------------------
                #      Cheops detrending (roll angle GP)
                # -------------------------------------------
                self.model_params['cheops_summodel_x']={}
                self.model_params['cheops_llk']={}
                if self.fit_phi_gp:
                    self.model_params['rollangle_logsigma'] = pm.Normal("rollangle_logsigma",mu=-6,sigma=1)

                    # self.model_params['rollangle_power'] = pm.InverseGamma("rollangle_power",initval=np.nanmedian(abs(np.diff(self.lcs["cheops"]['flux']))), 
                    #                                   **pmx.estimate_inverse_gamma_parameters(
                    #                                             lower=0.2*np.sqrt(np.nanmedian(abs(np.diff(self.lcs["cheops"]['flux'][self.lcs["cheops"]['mask']])))),
                    #                                             upper=2.5*np.sqrt(np.nanstd(self.lcs["cheops"]['flux'][self.lcs["cheops"]['mask']]))))
                    #self.model_params['rollangle_loglengthscale'] = pm.InverseGamma("rollangle_loglengthscale", initval=np.log(50), 
                    #                                                        **pmx.estimate_inverse_gamma_parameters(lower=np.log(30), upper=np.log(110)))
                    self.model_params['rollangle_logw0'] = pm.Normal('rollangle_logw0',mu=np.log((2*np.pi)/100),sigma=1)
                    #self.model_params['rollangle_w0'] = pm.InverseGamma("rollangle_w0", initval=(2*np.pi)/(lowerwl*1.25), **pmx.estimate_inverse_gamma_parameters(lower=(2*np.pi)/100,upper=(2*np.pi)/lowerwl))
                    self.model_params['rollangle_sigma'] = pm.Deterministic("rollangle_sigma", pm.math.exp(self.model_params['rollangle_logsigma']))
                    self.model_params['gp_rollangle_model_phi']={}
                    if self.phi_model_type=='individual' or len(self.cheops_filekeys)==1:
                        self.model_params['rollangle_kernels']={}
                        self.model_params['gp_rollangles']={}
                    else:
                        cheops_newsigmas={}
                elif self.fit_phi_spline:
                    from patsy import dmatrix
                    self.model_params['spline_model']={}
                    if self.phi_model_type=="common":
                        self.lcs["cheops"]['n_phi_model']=np.tile(0,len(self.lcs["cheops"]['time']))
                        # #Fit a single spline to all rollangle data
                        # minmax=(np.min(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi']),np.max(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi']))
                        # n_knots=int(np.round((minmax[1]-minmax[0])/self.spline_bkpt_cad))
                        # knot_list = np.quantile(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'],np.linspace(0,1,n_knots))
                        # B = dmatrix(
                        #     "bs(phi, knots=knots, degree="+str(int(self.spline_order))+", include_intercept=True) - 1",
                        #     {"phi": np.sort(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'].values), "knots": knot_list[1:-1]},
                        # )

                        # self.model_params['splines'] = pm.Normal("splines", mu=0, sigma=np.nanmedian(abs(np.diff(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux']))), 
                        #                                         shape=B.shape[1],initval=np.random.normal(0.0,1e-4,B.shape[1]))
                        # self.model_params['spline_model_allphi'] = pm.Deterministic("spline_model_allphi", pm.math.dot(np.asarray(B, order="F"), self.model_params['splines'].T))
                        # fk_Bs={}
                        # for fk in self.cheops_filekeys:
                        #     #print(np.sort(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'phi'].values))
                        #     #fk_Bs[fk] = np.asarray(B, order="F")[self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'filekey']==fk,:]
                        #     #self.model_params['spline_model_phi'][fk] = pm.Deterministic("spline_model_phi_"+fk, pm.math.dot(fk_Bs[fk], self.model_params['splines'].T))
                        #     ##pm.math.printing.Print("dotprod phi spline model")(self.model_params['spline_model_phi'][fk])
                        #     ##pm.math.printing.Print("indexed phi spline model (should be identical)")(self.model_params['spline_model_allphi'][np.array(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'filekey'].values[self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'mask_allphi_sorting']]==fk)])
                        #     self.model_params['spline_model_phi'][fk] = pm.Deterministic("spline_model_phi_"+str(fk), 
                        #                                                                 self.model_params['spline_model_allphi'][np.array(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'filekey'].values[self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'mask_allphi_sorting']]==fk)])

                    elif "split" in self.phi_model_type:
                        #Splitting into subdivisions:
                        n_split=int(self.phi_model_type.split("_")[1])
                        split_av_times=np.sort([np.nanmedian(self.lcs["cheops"]['time'][self.lcs["cheops"]['filekey']==fk]) for fk in self.cheops_filekeys])
                        assert n_split>=2 and len(split_av_times)>n_split, "Must split into at least 2 parts ("+str(n_split)+"), and there must be at least N splits in CHEOPS observations ("+str(len(split_av_times))+")"
                        self.logger.debug([n_split,split_av_times,np.sort(np.diff(split_av_times))[::-1]])
                        split_limit=np.sort(np.diff(split_av_times))[::-1][n_split-1]
                        n_jumps=np.where(np.diff(split_av_times)>split_limit)[0]
                        self.logger.debug([split_limit,split_av_times,np.diff(split_av_times)>split_limit,n_jumps])
                        split_times=[np.min(self.lcs["cheops"]['time'])-0.1]+[0.5*(split_av_times[:-1][n_j]+split_av_times[1:][n_j]) for n_j in n_jumps]+[np.max(self.lcs["cheops"]['time'])+0.1]
                        self.lcs["cheops"]['n_phi_model']=np.tile(np.nan,len(self.lcs['cheops']['time']))
                        for n_st in range(n_split):
                            self.lcs["cheops"]['n_phi_model'][(self.lcs["cheops"]['time']>split_times[n_st])&(self.lcs["cheops"]['time']<split_times[n_st+1])]=n_st
                    else:
                        #Splitting by filekey
                        self.lcs["cheops"]['n_phi_model']=np.tile(np.nan,len(self.lcs["cheops"]['time']))
                        for nfk,fk in enumerate(self.cheops_filekeys):
                            self.lcs["cheops"]['n_phi_model'][self.lcs["cheops"]['filekey']==fk]=nfk
                    #Fitting splines to each time region
                    B={}
                    self.model_params['splines']={}
                    self.model_params['spline_model']={}

                    minmax=(np.min(self.lcs["cheops"]['phi']),np.max(self.lcs["cheops"]['phi']))
                    n_knots=int(np.round((minmax[1]-minmax[0])/self.spline_bkpt_cad))
                    spline_index_arr=np.zeros((len(self.lcs["cheops"]['time']),len(np.unique(self.lcs["cheops"]['n_phi_model']))))
                    self.knots_per_model={'all':np.quantile(self.lcs["cheops"]['phi'].values,np.linspace(0,1,n_knots))}
                    self.logger.debug(spline_index_arr.shape)
                    for nreg in np.unique(self.lcs["cheops"]['n_phi_model']).astype(int):
                        spline_index_arr[self.lcs["cheops"]['n_phi_model']==nreg,nreg]=1.0
                        self.logger.debug(np.shape(spline_index_arr[self.lcs["cheops"]['n_phi_model']==nreg,nreg]))
                        ix=(self.lcs["cheops"]['n_phi_model']==nreg)
                        #self.knots_per_model[nreg]=np.quantile(self.lcs["cheops"].loc[ix,'phi'].values,np.linspace(0,1,n_knots))
                        B[nreg], self.knots_per_model[nreg] = create_angle_spline_dmatrix(self.lcs["cheops"].loc[ix,'phi'].values, bkpt_cad=self.spline_bkpt_cad)
                        # B[nreg] = dmatrix(
                        #     "bs(phi, knots=knots, degree="+str(int(self.spline_order))+", include_intercept=True) - 1",
                        #     {"phi": np.sort(self.lcs["cheops"].loc[ix,'phi'].values), "knots": self.knots_per_model[nreg][1:-1]},
                        # )
                        self.model_params['splines'][nreg] = pm.Normal("splines_"+str(nreg), mu=0, sigma=np.nanmedian(abs(np.diff(self.lcs["cheops"].loc[ix,'flux'].values))), shape=B[nreg].shape[1],initval=np.random.normal(0,1e-5,B[nreg].shape[1]))
                        self.model_params['spline_model'][nreg] = pm.Deterministic("spline_model_"+str(nreg), pm.math.dot(B[nreg], self.model_params['splines'][nreg].T))
                        
                        #Simply indexing the spline flux from the above models for each filekey (and for all the)
                        for fk in pd.unique(self.lcs["cheops"].loc[self.lcs["cheops"]['n_phi_model']==nreg,'filekey']):
                            fkix=ix&(self.lcs["cheops"]['filekey'].values==fk)
                            minmax=[np.min(self.lcs["cheops"].loc[fkix,'phi'].values),np.max(self.lcs["cheops"].loc[fkix,'phi'].values)]
                            self.model_params['spline_model'][fk] = pm.Deterministic("spline_model_"+str(fk), 
                                                                                          self.model_params['spline_model'][nreg][(self.lcs["cheops"].loc[ix,'filekey'].values==fk)&self.lcs['cheops'].loc[ix,'mask'].values])
                            #pm.math.printing.Print(str(fk))(self.model_params['spline_model'][nreg])
                            self.logger.debug([fk,"flux",self.lcs["cheops"].loc[fkix,'flux'].values])
                            self.logger.debug([fk,"time",self.lcs["cheops"].loc[fkix,'time'].values])
                            self.logger.debug([fk,"sort phi",np.sort(self.lcs["cheops"].loc[fkix,'phi'].values)])


                    #Applying the splines to all times, by stack/index/summing into get a single phi array:
                    for nreg in pd.unique(self.lcs["cheops"]['n_phi_model']):
                        #pm.math.printing.Print(str(nreg)+" splines")(self.model_params['splines'][nreg].T)
                        self.logger.debug([str(nreg)+" splines",np.asarray(B[nreg], order="F")])
                        #pm.math.printing.Print(str(nreg)+" dot prod")(pm.math.dot(np.asarray(B[nreg], order="F"), self.model_params['splines'][nreg].T))
                    self.logger.debug(spline_index_arr[np.argsort(self.lcs["cheops"]['phi'].values),:])
                    #pm.math.printing.Print("stacked")(pm.math.concatenate([pm.math.dot(np.asarray(B[nreg], order="F"), self.model_params['splines'][nreg].T) for nreg in pd.unique(self.lcs["cheops"]['n_phi_model'])]))
                    self.model_params['spline_model_alltime'] = pm.Deterministic("spline_model_alltime", pm.math.sum(pm.math.concatenate([pm.math.dot(np.asarray(B[nreg], order="F"), self.model_params['splines'][nreg]) for nreg in pd.unique(self.lcs["cheops"]['n_phi_model'])])*spline_index_arr.T,axis=0))
                    #pm.math.printing.Print("spline_model_alltime")(self.model_params['spline_model_alltime'])

                    cheops_newsigmas={}
                else:
                    cheops_newsigmas={}
                
                #in the full model, we do a full cheops planet model for all filekeys simultaneously (unlike for the cheops_only_model)
                self.model_params['cheops_planets_x'] = {}
                self.model_params['cheops_planets_gaps'] = {}
                for pl in self.planets:
                    self.model_params['cheops_planets_x'][pl] = pm.Deterministic("cheops_planets_x_"+pl, xo.LimbDarkLightCurve(self.model_params['u_stars']["cheops"]).get_light_curve(orbit=self.model_params['orbit'][pl], r=self.model_params['rpl'][pl]/109.2,
                                                                                                        t=self.lcs["cheops"]['time'].values.astype(np.float64))[:,0]*1000/self.model_params['cheops_mult'])
                    #print(self.cheops_gap_timeseries.astype(np.float64))
                    #print(self.model_params['cheops_mult'])
                    ##pm.math.printing.Print("u")(self.model_params['u_stars']["cheops"])
                    ##pm.math.printing.Print("rpl")(self.model_params['rpl'][pl]/109.2)
                    self.model_params['cheops_planets_gaps'][pl] = pm.Deterministic("cheops_planets_gaps_"+pl,xo.LimbDarkLightCurve(self.model_params['u_stars']["cheops"]).get_light_curve(orbit=self.model_params['orbit'][pl], r=self.model_params['rpl'][pl]/109.2,
                                                                                                        t=self.cheops_gap_timeseries.astype(np.float64))[:,0]*1000/self.model_params['cheops_mult'])
                if self.fit_phi_gp:
                    self.model_params['rollangle_kernels'] = pymc_terms.SHOTerm(sigma=self.model_params['rollangle_sigma'], w0=pm.math.exp(self.model_params['rollangle_logw0']), Q=1/np.sqrt(2))#, mean = phot_mean)

                if self.fit_phi_gp and self.phi_model_type=="common" and len(self.cheops_filekeys)>1:
                    #Trying a new tack - binning to 2.5-degree bins.
                    #To do this we also need to hard-wire the indexes & average the fluxes
                    #print(np.nanmin(np.diff(np.sort(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'].values))))
                    self.cheops_binphi_2d_index=np.column_stack(([np.array(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi_digi']==n).astype(int)/np.sum(self.lcs["cheops"]['phi_digi']==n) for n in np.unique(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi_digi'])]))
                    #plt.plot(np.sum(mod.cheops_lc.loc[mod.cheops_lc['mask'],'phi'][:,None]*cheops_binphi_2d_index,axis=0),
                    #        np.sum(mod.cheops_lc.loc[mod.cheops_lc['mask'],'flux'][:,None]*cheops_binphi_2d_index,axis=0),
                    #pm.math.printing.Print("diag")((np.sum(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux_err'][:,None]*self.cheops_binphi_2d_index**1.5,axis=0))** 2 + \
                    #                                                                        (pm.math.sum(pm.math.exp(self.model_params['cheops_logs'])*self.cheops_binphi_2d_index**1.5,axis=0))**2)
                    self.logger.debug(np.sum(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'][:,None]*self.cheops_binphi_2d_index,axis=0))
                    self.logger.debug("flux:")
                    self.logger.debug(np.sum(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'][:,None]*self.cheops_binphi_2d_index,axis=0))
                    self.model_params['gp_rollangles'] = celerite2.pymc.GaussianProcess(self.model_params['rollangle_kernels'], 
                                                                                        np.sum(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'][:,None]*self.cheops_binphi_2d_index,axis=0), mean=0.0,
                                                                                        diag=(np.sum(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux_err'][:,None]*self.cheops_binphi_2d_index**1.5,axis=0))** 2 + \
                                                                                             (pm.math.sum(pm.math.exp(self.model_params['cheops_logs'])*self.cheops_binphi_2d_index**1.5,axis=0))**2)#Adding **1.5 as we want an extra 1/N**0.5 (instead of just 1/N in the average). 
                                                                                             #Also including these 1/N**1.5 terms in the jitter to ensure jitter is per-point
                    #self.model_params['gp_rollangles'].compute(np.sort(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'].values),
                    #                                           diag=..., quiet=True)
                elif self.fit_phi_gp and "split" in self.phi_model_type and len(self.cheops_filekeys)>1:
                    self.logger.debug("Not yet implemented")
                elif self.fit_phi_gp:
                    for fk in self.cheops_filekeys:
                        # Roll angle vs flux GP                
                        self.model_params['gp_rollangles'][fk] = celerite2.pymc.GaussianProcess(self.model_params['rollangle_kernels'], 
                                                                                                t=np.sort(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'phi'].values), mean=0,
                                                                                                diag=(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux_err'].values[self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values]) ** 2 + \
                                                                                                pm.math.exp(self.model_params['cheops_logs'])**2)
                        #self.model_params['gp_rollangles'][fk].compute(np.sort(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'phi'].values),
                        #                                               , quiet=True)
                
                for fk in self.cheops_filekeys:
                    #Adding the correlation model:
                    self.model_params['cheops_summodel_x'][fk] = pm.Deterministic("cheops_summodel_x_"+str(fk), pm.math.sum([self.model_params['cheops_planets_x'][pl][self.cheops_fk_mask[fk]] for pl in self.planets],axis=0) + self.model_params['cheops_flux_cor'][fk])
                    if self.fit_phi_gp and (self.phi_model_type=="individual" or len(self.cheops_filekeys)==1):
                        self.model_params['gp_rollangle_model_phi'][fk] = pm.Deterministic("gp_rollangle_model_phi_"+str(fk), 
                                                                self.model_params['gp_rollangles'][fk].predict((self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values[self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values] - \
                                                                                        self.model_params['cheops_summodel_x'][fk][self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values]), 
                                                                                    t=np.sort(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'phi'].values), return_var=False))
                if self.fit_phi_gp and self.phi_model_type=="common" and len(self.cheops_filekeys)>1:
                    all_summodels=pm.math.concatenate([self.model_params['cheops_summodel_x'][fk] for fk in pd.unique(self.lcs["cheops"]['filekey'])])#,axis=0)
                    #pm.math.printing.Print("flux - model")(pm.math.sum((self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'].values - \
                    #                                                                            all_summodels).dimshuffle(0,'x')*self.cheops_binphi_2d_index,axis=0))
                    self.model_params['gp_rollangle_model_allphi'] = pm.Deterministic("gp_rollangle_model_allphi",
                                                        self.model_params['gp_rollangles'].predict(y=pm.math.sum((self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'].values - \
                                                                                                all_summodels).dimshuffle(0,'x')*self.cheops_binphi_2d_index,axis=0), 
                                                                                                t=np.sort(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi'].values), return_var=False))
                    for fk in self.cheops_filekeys:
                        self.model_params['gp_rollangle_model_phi'][fk] = pm.Deterministic("gp_rollangle_model_phi_"+str(fk), 
                                                                                        self.model_params['gp_rollangle_model_allphi'][np.array(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'filekey'].values[self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'mask_allphi_sorting']]==fk)])
                elif self.fit_phi_gp and self.phi_model_type=="split" and len(self.cheops_filekeys)>1:
                    self.logger.debug("SPLIT GP NOT IMPLEMENTED YET")
                # -------------------------------------------
                #      Evaluating log likelihoods            
                # -------------------------------------------
                for fk in self.cheops_filekeys:
                    # if self.fit_phi_gp and ((self.phi_model_type=="individual") or len(self.cheops_filekeys)==1):
                    #     self.model_params['cheops_llk'][fk] = pm.Potential("cheops_llk_"+str(fk),self.model_params['gp_rollangles'][fk].log_likelihood(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values - self.model_params['cheops_summodel_x'][fk]))
                    #     self.model_params['cheops_llk'][fk] = pm.Normal("cheops_llk_"+str(fk),mu=self.model_params['gp_rollangle_model_phi'][fk],
                    #                                                     sigma = cheops_newsigmas[fk],
                    #                                                     observed = self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values - self.model_params['cheops_summodel_x'][fk]
                    #                                                     self.model_params['gp_rollangles'][fk].log_likelihood())
                    #     #self.model_params['cheops_llk'][fk] = self.model_params['gp_rollangles'][fk].marginal("cheops_llk_"+str(fk),   observed = self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values - self.model_params['cheops_summodel_x'][fk])
                    #     #print("w rollangle GP",fk)
                    #     ##pm.math.printing.Print("llk_cheops")(self.model_params['llk_cheops'][fk])
                    #     #else:
                    cheops_newsigmas[fk] = pm.math.sqrt(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux_err'].values ** 2 + pm.math.exp(self.model_params['cheops_logs'])**2)
                    if self.fit_phi_spline:
                        self.model_params['cheops_llk'][fk] = pm.Normal("cheops_llk_"+fk, mu=self.model_params['cheops_summodel_x'][fk] + self.model_params['spline_model'][fk], 
                                                                        sigma=cheops_newsigmas[fk], observed=self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values)
                    elif self.fit_phi_gp and self.phi_model_type in ["common","split"] and len(self.cheops_filekeys)>1:
                        self.model_params['cheops_llk'][fk] = pm.Potential("cheops_llk_"+fk, self.model_params['gp_rollangles'].log_likelihood(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values[self.cheops_fk_mask[fk],'mask_phi_sorting']-self.model_params['cheops_summodel_x'][fk][self.cheops_fk_mask[fk],'mask_phi_sorting']))
                        #self.model_params['cheops_llk'][fk] = pm.Normal("cheops_llk_"+fk, mu=self.model_params['cheops_summodel_x'][fk] + self.model_params['gp_rollangle_model_phi'][fk][self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'mask_time_sorting']], 
                        #                                                sigma=pm.math.sqrt(cheops_newsigmas[fk]), observed=self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values)
                    elif not self.fit_phi_gp and not self.fit_phi_spline:
                        #In the case of the common roll angle on binned phi, we cannot use the gp marginal, so we do an "old fashioned" likelihood:
                        self.model_params['cheops_llk'][fk] = pm.Normal("cheops_llk_"+fk, mu=self.model_params['cheops_summodel_x'][fk], sigma=cheops_newsigmas[fk], 
                                                                        observed=self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values)
                
                        #print("no rollangle GP",fk)
                        ##pm.math.printing.Print("llk_cheops")(self.model_params['llk_cheops'][fk])
                        # self.model_params['llk_cheops'][fk] = pm.Potential("llk_cheops_"+str(fk), 
                        #                              -0.5 * pm.math.sum((self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values - \
                        #                                             self.model_params['cheops_summodel_x'][fk]) ** 2 / \
                        #                                            cheops_newsigmas[fk] + np.log(cheops_newsigmas[fk]))
                        #                            )
                #if self.fit_phi_gp and self.phi_model_type in ["common","split"] and len(self.cheops_filekeys)>1:
                #    self.model_params['llk_cheops'] = self.model_params['gp_rollangles'].marginal("llk_cheops",
                #                                                    observed = self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'].values - all_summodels)
            if not self.fit_gp:
                newsigmas={}
            for scope in self.lcs:
                self.model_params[scope+'_model_x']={}
                for pl in self.planets:
                    self.model_params[scope+'_model_x'][pl] = pm.Deterministic(scope+"_model_x_"+pl, xo.LimbDarkLightCurve(self.model_params['u_stars'][scope]).get_light_curve(orbit=self.model_params['orbit'][pl], r=self.model_params['rpl'][pl]/109.2,
                                                                                                                           t=self.lc_fit[scope]['time'].values)[:,0]*1000/self.model_params[scope+'_mult'])
                self.model_params[scope+'_summodel_x'] = pm.Deterministic(scope+"_summodel_x", pm.math.sum([self.model_params[scope+'_model_x'][pl] for pl in self.planets],axis=0))
                newsigmas[scope] = pm.math.sqrt(self.lc_fit[scope]['flux_err'].values ** 2 + pm.math.exp(self.model_params[scope+'_logs'])**2)
                if self.fit_gp and scope!="cheops":
                    self.model_params[scope+'_gp_model_x'] = pm.Deterministic(scope+"_gp_model_x", self.model_params[scope+'_gp'].predict(self.lc_fit[scope]['flux'].values - self.model_params[scope+'_summodel_x'], t=self.lc_fit[scope]['time'].values, return_var=False))
                    #self.model_params[scope+'_llk'] = pm.Potential(scope+'_llk', self.model_params[scope+'_gp'].log_likelihood(self.lc_fit[scope]['flux'].values-self.model_params[scope+'_summodel_x']))
                    self.model_params[scope+'_llk'] = pm.Normal(scope+'_llk', 
                                                               mu=self.model_params[scope+'_gp_model_x']+self.model_params[scope+'_summodel_x'],
                                                               sigma=newsigmas[scope],
                                                               observed=self.lc_fit[scope]['flux'].values)
                    #self.model_params[scope+'_llk'] = self.model_params[scope+'_gp'].marginal(scope+'_llk', observed = self.lc_fit[scope]['flux'].values - self.model_params[scope+'_summodel_x'])
                elif scope!="cheops":
                    
                    self.model_params[scope+'_llk'] = pm.Normal(scope+'_llk', mu=self.model_params[scope+'_summodel_x'],sigma=newsigmas[scope],observed=self.lc_fit[scope]['flux'].values)
                    #pm.math.printing.Print(scope+"_llk")(self.model_params[scope+'_llk'])
            
            #Combined 
            if 'cheops' in self.lcs and len(self.lcs)>1:
                self.model_params['log_likelihood']=pm.Deterministic("log_likelihood",pm.math.sum([pm.math.sum(self.model_params[scope+"_llk"]) for scope in self.lcs if scope!='cheops'])+pm.math.sum([pm.math.sum(self.model_params["cheops_llk"][fk]) for fk in self.model_params["cheops_llk"]]))
            elif 'cheops' in self.lcs and len(self.lcs)==1:
                self.model_params['log_likelihood']=pm.Deterministic("log_likelihood",pm.math.sum([pm.math.sum(self.model_params["cheops_llk"][fk]) for fk in self.model_params["cheops_llk"]]))
            else:
                assert 'cheops' not in self.lcs, "We are assuming there is no CHEOPS lightcurve here"
                self.model_params['log_likelihood']=pm.Deterministic("log_likelihood",pm.math.sum([pm.math.sum(self.model_params[scope+"_llk"]) for scope in self.lcs]))


                #elif scope=="cheops":
                #    #Doing cheops-specific stuff here.
                #    self.logger.debug("NA")

            if hasattr(self,"rvs"):
                rv_logjitter = pm.Normal("rv_logjitter",mu=np.nanmin(self.rvs['yerr'].values)-3,sigma=3)
                rv_sigma2 = self.rvs['yerr'].values ** 2 + pm.math.exp(rv_logjitter)**2
                self.model_params['rv_llk'] = pm.Potential("rv_llk", -0.5 * (self.rvs['y'].values - self.model_params['rv_model_x']) ** 2 / rv_sigma2 + np.log(rv_sigma2))
            #print(self.model.check_test_point())
            # if 'cheops' in self.lcs:
            #     # for npar,par in enumerate([self.model_params['cheops_llk'][par] for par in self.model_params['cheops_llk']]+[self.model_params[par] for par in self.model_params if "_llk" in par and par!="cheops_llk"]):
            #     #     #pm.math.printing.Print(str(npar))(par.shape)
            #     # try:
            #     #     print("axis=0")
            #     #     #pm.math.printing.Print("axis=1")(pm.math.stack([self.model_params['cheops_llk'][par] for par in self.model_params['cheops_llk']]+[self.model_params[par] for par in self.model_params if "_llk" in par and par!="cheops_llk"],axis=1))
            #     #     print("worked?")
            #     # except:
            #     #     try:
            #     #         print("axis=1")
            #     #         #pm.math.printing.Print("axis=0")(pm.math.stack([self.model_params['cheops_llk'][par] for par in self.model_params['cheops_llk']]+[self.model_params[par] for par in self.model_params if "_llk" in par and par!="cheops_llk"],axis=0))
            #     #         print("worked?")
            #     #     except:
            #     #         try:
            #     #             print("join?")
            #     #             #pm.math.printing.Print("axis=?")(pm.math.join([self.model_params['cheops_llk'][par] for par in self.model_params['cheops_llk']]+[self.model_params[par] for par in self.model_params if "_llk" in par and par!="cheops_llk"]))
            #     #             print("worked?")
            #     #         except:
            #     #             print("concat?")
            #     #             #pm.math.printing.Print("axis=?")(pm.math.concatenate([self.model_params['cheops_llk'][par] for par in self.model_params['cheops_llk']]+[self.model_params[par] for par in self.model_params if "_llk" in par and par!="cheops_llk"]))
            #     #             print("worked?")

            #     self.model_params['log_likelihood'] = pm.Deterministic("log_likelihood",pm.math.stack([self.model_params['cheops_llk'][par] for par in self.model_params['cheops_llk']]+[self.model_params[par] for par in self.model_params if "_llk" in par and par!="cheops_llk"]))
            # else:
            #     # for npar,par in enumerate([self.model_params[par] for par in self.model_params if "_llk" in par and par!="cheops_llk"]):
            #     #     #pm.math.printing.Print(str(npar))(par.shape)
            #     # try:
            #     #     #pm.math.printing.Print(str(npar))(pm.math.stack([self.model_params[par] for par in self.model_params if "_llk" in par ],axis=0))
            #     # except:
            #     #     #pm.math.printing.Print(str(npar))(pm.math.stack([self.model_params[par] for par in self.model_params if "_llk" in par ],axis=1))

            #     self.model_params['log_likelihood'] = pm.Deterministic("log_likelihood",pm.math.stack([self.model_params[par] for par in self.model_params if "_llk" in par ]))

            self.pre_model_soln = pmx.optimize(vars = self.model_params[list(self.lcs.keys())[0]+'_logs'])
            #First try to find best-fit transit stuff:
            if not self.fit_ttvs:
                #print([self.model_params[par][pl] for pl in self.planets for par in ['logror','P','t0']]+[self.model_params[par] for par in ['logs_tess','cheops_logs']])
                #print(len([self.model_params[par][pl] for pl in self.planets for par in ['logror','P','t0']]+[self.model_params[par] for par in ['logs_tess','cheops_logs']]))
                comb_soln = pmx.optimize(vars=[self.model_params[par][pl] for pl in self.planets for par in ['logror','P','t0']]+[self.model_params[scope+"_logs"] for scope in self.lcs])
            else:
                optvar=[]
                for pl in self.planets:
                    if pl in self.model_params['transit_times']:
                        optvar+=[self.model_params['transit_times'][pl][i] for i in range(len(self.model_params['transit_times'][pl]))]
                    else:
                        optvar+=[self.model_params['P'][pl],self.model_params['t0'][pl]]
                if len(self.cheops_filekeys)>0:
                    optvar+=[self.model_params['linear_decorr_dict'][par] for par in self.model_params['linear_decorr_dict']]
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
                elif self.fit_phi_spline and self.phi_model_type in ["common","split"]:
                    decorrvars+=[self.model_params['splines'][nreg] for nreg in self.model_params['splines']]
                elif self.fit_phi_spline and self.phi_model_type=="individual":
                    decorrvars+=[self.model_params['splines'][fk] for fk in self.cheops_filekeys]
                comb_soln = pmx.optimize(start=comb_soln, vars=decorrvars)
            
            #More complex transit fit. Also RVs:
            ivars=[self.model_params['b'][pl] for pl in self.planets]+[self.model_params['logror'][pl] for pl in self.planets]+[self.model_params[scope+'_logs'] for scope in self.lcs]
            if len(self.cheops_filekeys)>0:
                ivars+=[self.model_params['cheops_logs']]
            if len(self.planets)>0:
                ivars+=[self.model_params['u_stars'][u] for u in self.model_params['u_stars']]
                if self.fit_ttvs:
                    for pl in self.planets:
                        if pl in self.model_params['transit_times']:
                            ivars+=[self.model_params['transit_times'][pl][i] for i in range(len(self.model_params['transit_times'][pl]))]
                        else:
                            ivars+=[self.model_params['P'][pl],self.model_params['t0'][pl]]
                elif self.split_periods is not None:
                    for pl in self.planets:
                        if self.planets[pl]['n_trans']>2 and pl in self.split_periods and len(self.split_periods[pl])>1 and self.split_periods[pl]!=range(self.planets[pl]['n_trans']):
                            ivars+=[self.model_params['split_P'][pl][i] for i in range(len(self.split_periods[pl]))]
                            ivars+=[self.model_params['split_t0'][pl][i] for i in range(len(self.split_periods[pl]))]
                        else:
                            ivars+=[self.model_params['P'][pl],self.model_params['t0'][pl]]

                else:
                    ivars+=[self.model_params['t0'][pl] for pl in self.planets]+[self.model_params['P'][pl] for pl in self.planets]
                if not self.assume_circ:
                    ivars+=[self.model_params['ecc'][pl] for pl in self.planets]+[self.model_params['omega'][pl] for pl in self.planets]
            if hasattr(self,'rvs'):
                if len(self.planets)>0:
                    ivars+=[self.model_params['logK'][pl] for pl in self.planets]
                ivars+=[self.model_params['rv_offsets']]
                if self.npoly_rv>1:
                    ivars+=[self.model_params['rv_trend']]
            self.logger.debug(ivars)
            comb_soln = pmx.optimize(start=comb_soln, vars=ivars)

            #Doing everything:
            self.init_soln = pmx.optimize(start=comb_soln)
        
        if len(self.cheops_filekeys)>0 and self.fit_phi_gp:
            #Checking if the GP is useful in the model:
            self.check_rollangle_gp(**kwargs)
        elif len(self.cheops_filekeys)>0 and self.fit_phi_spline:
            self.check_rollangle_spline(**kwargs)
    
    def sample_model(self,n_tune_steps=1200,n_draws=998,cheops_groups="all",save_model=True,**kwargs):
        """Sample model

        Args:
            n_tune_steps (int, optional): Number of steps during tuning. Defaults to 1200.
            n_draws (int, optional): Number of model draws per chain. Defaults to 998.
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
                    if self.fit_phi_spline and self.phi_model_type=="individual":
                        groups[-1]+=self.model_params["splines_"+fk]
            elif len(self.cheops_filekeys)>0 and cheops_groups=="all":
                #Putting all CHEOPS parameters into a single group
                groups+=[[self.model_params['cheops_obs_means'][fk] for fk in self.cheops_filekeys]+[self.model_params['linear_decorr_dict'][par] for par in self.model_params['linear_decorr_dict']]]
                if 'quad_decorr_dict' in self.model_params:
                    groups[-1]+=[self.model_params['quad_decorr_dict'][par] for par in self.model_params['quad_decorr_dict']]
                if self.fit_phi_spline and self.phi_model_type=="individual":
                    groups[-1]+=[self.model_params["splines"][fk] for fk in self.cheops_filekeys]
                # elif self.fit_phi_spline and self.phi_model_type in ["common","split"]:
                #     groups[-1]+=[self.model_params["splines"]]
                # elif self.fit_phi_gp:
                #     groups[-1]+=[self.model_params["rollangle_logpower"],self.model_params["rollangle_logw0"]]
                groups[-1]+=[self.model_params["cheops_logs"]]
            if hasattr(self,'rvs'):
                rvgroup=[self.model_params['rv_offsets']]
                if self.rv_mass_prior=='popMp':
                    rvgroup+=[self.model_params['logMp'][pl] for pl in self.planets]
                elif self.rv_mass_prior=='logK':
                    rvgroup+=[self.model_params['logK'][pl] for pl in self.planets]
                elif self.rv_mass_prior=='K':
                    rvgroup+=[self.model_params['K'][pl] for pl in self.planets]
                if self.npoly_rv>1:
                    rvgroup+=[self.model_params['rv_trend']]
                groups+=[rvgroup]
            self.trace = pm.sample(tune=n_tune_steps, draws=n_draws, 
                                    chains=self.n_cores, cores=self.n_cores, 
                                    start=self.init_soln, target_accept=0.8,
                                    return_inferencedata=True, 
                                    idata_kwargs=dict(log_likelihood=True), #Adding these for large model sizes
                                    **kwargs)#**kwargs)

            # self.trace = pmx.sample(tune=n_tune_steps, draws=n_draws, 
            #                         chains=int(n_chains*n_cores), cores=n_cores, 
            #                         start=self.init_soln, target_accept=0.8,
            #                         parameter_groups=groups,
            #                         return_inferencedata=True, 
            #                         idata_kwargs=dict(log_likelihood=True), #Adding these for large model sizes
            #                         **kwargs)#**kwargs)
        self.save_trace_summary()
        if save_model:
            self.save_model_to_file()

    def run_slim_ttv_model(self,n_tune_steps=1200,n_draws=998,**kwargs):
        """Running a second model with extremely constrained in number of parameters in order to allow TTV modelling without large parameter correlations"""
        assert hasattr(self,'trace'), "Must have already sampled a classic model"
        assert self.spar_param in ['logg','Mstar','rhostar'], "Must be one of 'logg', 'Mstar', or 'rhostar'"
        assert not (self.spar_param=='logg')&(self.spar_prior=='logloose'), "shouldnt set a log prior for a logged quantity"

        if not hasattr(self,'init_transit_times'):
            self.init_ttvs()

        self.ttv_model_params={}
        cheops_newsigmas={}
        with pm.Model() as self.ttv_model:
            # -------------------------------------------
            #          Stellar parameters
            # -------------------------------------------
            self.ttv_model_params['Rs'] = pm.TruncatedNormal("Rs", lower=0,mu=self.Rstar[0], sigma=self.Rstar[1])
            if self.spar_param=='Mstar':
                if self.spar_prior=='constr':
                    self.ttv_model_params['Ms'] = pm.TruncatedNormal("Ms", lower=0, mu=self.Mstar[0], sigma=self.Mstar[1]) #Ms and logg are interchangeably deterministic
                elif self.spar_prior=='loose':
                    self.ttv_model_params['Ms'] = pm.TruncatedNormal("Ms", lower=0, mu=self.Mstar[0],sigma=0.33*self.Mstar[0]) #Ms and logg are interchangeably deterministic
                elif self.spar_prior=='logloose':
                    self.ttv_model_params['logMs'] = pm.Uniform("logMs",lower=np.log(self.Mstar[0])-2,upper=np.log(self.Mstar[0])+2) #Ms and logg are interchangeably deterministic
                    self.ttv_model_params['Ms'] = pm.Deterministic("Ms", pm.math.exp(self.ttv_model_params['logMs']))
                self.ttv_model_params['logg'] = pm.Deterministic("logg",pm.math.log(self.ttv_model_params['Ms']/self.ttv_model_params['Rs']**2)/pm.math.log(10)+4.41) #Ms and logg are interchangeably deterministic
                self.ttv_model_params['rhostar'] = pm.Deterministic("rhostar",self.ttv_model_params['Ms']/self.ttv_model_params['Rs']**3) #Ms and logg are interchangeably deterministic
            elif self.spar_param=='logg':
                if self.spar_prior=='constr':
                    self.ttv_model_params['logg'] = pm.TruncatedNormal("logg", lower=0,mu=self.logg[0],sigma=self.logg[1]) #Ms and logg are interchangeably deterministic
                elif self.spar_prior=='loose':
                    self.ttv_model_params['logg'] = pm.TruncatedNormal("logg", lower=0, mu=self.logg[0],sigma=1) #Ms and logg are interchangeably deterministic
                self.ttv_model_params['Ms'] = pm.Deterministic("Ms",pm.math.power(10,self.ttv_model_params['logg']-4.41)*self.ttv_model_params['Rs']**2) #Ms and logg are interchangeably deterministic
                self.ttv_model_params['rhostar'] = pm.Deterministic("rhostar", self.ttv_model_params['Ms']/self.ttv_model_params['Rs']**3) #Ms and logg are interchangeably deterministic
            elif self.spar_param=='rhostar':
                if self.spar_prior=='constr':
                    self.ttv_model_params['rhostar'] = pm.TruncatedNormal("rhostar", lower=0, mu=self.rhostar[0],sigma=self.rhostar[1]) #Ms and logg are interchangeably deterministic
                elif self.spar_prior=='loose':
                    self.ttv_model_params['rhostar'] = pm.TruncatedNormal("rhostar", lower=0, mu=self.rhostar[0],sigma=0.33*self.rhostar[0]) #Ms and logg are interchangeably deterministic
                elif self.spar_prior=='logloose':
                    self.ttv_model_params['logrhostar'] = pm.Uniform("logrhostar",lower=np.log(self.rhostar[0])-2,upper=np.log(self.rhostar[0])+2,initval=self.rhostar[0]) #Ms and logg are interchangeably deterministic
                    self.ttv_model_params['rhostar'] = pm.Deterministic("rhostar", pm.math.exp(self.ttv_model_params['logrhostar']))
                self.ttv_model_params['Ms'] = pm.Deterministic("Ms",self.ttv_model_params['Rs']**3*self.ttv_model_params['rhostar']) #Ms and logg are interchangeably deterministic
                self.ttv_model_params['logg'] = pm.Deterministic("logg",pm.math.log(self.ttv_model_params['Ms']/self.ttv_model_params['Rs']**2)/pm.math.log(10)+4.41) #Ms and logg are interchangeably deterministic

            self.ttv_model_params['u_stars']={}
            for scope in self.ld_dists:
                self.ttv_model_params['u_stars'][scope] = pm.TruncatedNormal("u_star_"+scope, lower=0.0, upper=1.0,
                                                                mu=np.nanmedian(self.trace.posterior["u_star_"+scope].values,axis=(0,1)),
                                                                sigma=np.nanstd(self.trace.posterior["u_star_"+scope].values,axis=(0,1)), 
                                                                shape=2, initval=np.nanmedian(self.trace.posterior["u_star_"+scope].values,axis=(0,1)))
            # -------------------------------------------
            # Initialising parameter dicts for each planet
            # -------------------------------------------
            self.ttv_model_params['orbit']={}
            self.ttv_model_params['t0']={};self.ttv_model_params['P']={};self.ttv_model_params['vels']={};self.ttv_model_params['tdur']={}
            self.ttv_model_params['b']={};self.ttv_model_params['rpl']={};self.ttv_model_params['logror']={};self.ttv_model_params['ror']={}
            self.ttv_model_params['a_Rs']={};self.ttv_model_params['sma']={};self.ttv_model_params['S_in']={};self.ttv_model_params['Tsurf_p']={}
            min_ps={pl:self.planets[pl]['period']*(1-1.5*self.timing_sd_durs*self.planets[pl]['tdur']/(np.ptp(np.hstack([self.lc_fit[scope]['time'] for scope in self.lc_fit])))) for pl in self.planets}
            max_ps={pl:self.planets[pl]['period']*(1+1.5*self.timing_sd_durs*self.planets[pl]['tdur']/(np.ptp(np.hstack([self.lc_fit[scope]['time'] for scope in self.lc_fit])))) for pl in self.planets}
            self.logger.debug([min_ps,max_ps,[self.planets[pl]['period'] for pl in self.planets],np.ptp(np.hstack([self.lc_fit[scope]['time'] for scope in self.lc_fit]))])
            self.ttv_model_params['transit_times']={}

            if not self.assume_circ:
                self.ttv_model_params['ecc']={};self.ttv_model_params['omega']={}

            #Using pre-decorrelated CHEOPS flux &  pre-detrended TESS flux
            cor_lcs={}
            if not hasattr(self,'models_out'):
                self.save_timeseries()
            for scope in self.lcs:
                if scope=='cheops':
                    cor_lcs[scope]=np.column_stack((self.models_out[scope]['time'].values,
                                                    self.models_out[scope]['flux'].values-self.models_out[scope]['cheops_alldetrend_med'].values,
                                                    self.models_out[scope]['flux_err'].values))
                else:
                    cor_lcs[scope]=np.column_stack((self.models_out[scope]['time'].values,
                                                    self.models_out[scope]['flux'].values-self.models_out[scope][scope+"_gpmodel_med"].values,
                                                    self.models_out[scope]['flux_err'].values))

            for npl,pl in enumerate(self.planets):
                # -------------------------------------------
                #                  Orbits
                # -------------------------------------------
                self.ttv_model_params['transit_times'][pl]=[]
                for i in range(len(self.planets[pl]['init_transit_times'])):
                    if self.ttv_prior.lower()=='uniform':
                        self.ttv_model_params['transit_times'][pl].append(pm.Uniform("transit_times_"+pl+"_"+str(i), 
                                                                        upper=self.planets[pl]['init_transit_times'][i]+self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                        lower=self.planets[pl]['init_transit_times'][i]-self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                        initval=self.planets[pl]['init_transit_times'][i]))
                    elif self.ttv_prior.lower()=='normal':
                        self.ttv_model_params['transit_times'][pl].append(pm.Normal("transit_times_"+pl+"_"+str(i), 
                                                                        mu=self.planets[pl]['init_transit_times'][i],sigma=self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                        initval=self.planets[pl]['init_transit_times'][i]))
                    elif self.ttv_prior.lower()=='boundnormal':
                        self.ttv_model_params['transit_times'][pl].append(pm.TruncatedNormal("transit_times_"+pl+"_"+str(i), 
                                                                                lower=self.planets[pl]['init_transit_times'][i]-self.planets[pl]['tdur']*2*self.timing_sd_durs,
                                                                                upper=self.planets[pl]['init_transit_times'][i]+self.planets[pl]['tdur']*2*self.timing_sd_durs,
                                                                                        mu=self.planets[pl]['init_transit_times'][i],
                                                                                        sigma=self.planets[pl]['tdur']*self.timing_sd_durs,
                                                                                        initval=self.planets[pl]['init_transit_times'][i]))
                # Eccentricity & argument of periasteron
                if not self.assume_circ:
                    #BoundedBeta = pm.Uniform(pm.Beta, lower=1e-5, upper=1-1e-5)
                    self.ttv_model_params['ecc'][pl] = pm.TruncatedNormal("ecc_"+pl, lower=0.001,upper=1-0.001, mu=np.nanmedian(self.trace.posterior["ecc_"+pl]),sigma=np.nanstd(self.trace.posterior["ecc_"+pl]))
                    self.ttv_model_params['omega'][pl] = pm.TruncatedNormal("omega_"+pl, lower=0,upper=2*np.pi, mu=np.nanmedian(self.trace.posterior["omega_"+pl]),sigma=np.nanstd(self.trace.posterior["omega_"+pl]))
                '''
                #This was to model a non-transiting companion:
                P_nontran = pm.Normal("P_nontran", mu=27.386209624, sigma=2*0.04947295)
                logK_nontran = pm.Normal("logK_nontran", mu=2,sigma=10, initval=2)
                Mpsini_nontran = pm.Deterministic("Mp_nontran", pm.math.exp(logK_nontran) * 28.439**-1 * Ms**(2/3) * (P_nontran/365.25)**(1/3) * 317.8)
                t0_nontran = pm.Uniform("t0_nontran", lower=np.nanmedian(rv_x)-27.386209624*0.55, upper=np.nanmedian(rv_x)+27.386209624*0.55)
                '''
                self.ttv_model_params['logror'][pl] = pm.Normal("logror_"+pl, mu=np.nanmedian(self.trace.posterior["logror_"+pl].values)+0.2,
                                                                sigma=np.clip(3*np.nanstd(self.trace.posterior["logror_"+pl].values),0.2,1.0))
                self.ttv_model_params['ror'][pl] = pm.Deterministic("ror_"+pl,pm.math.exp(self.ttv_model_params['logror'][pl]))
                self.ttv_model_params['rpl'][pl] = pm.Deterministic("rpl_"+pl,109.1*self.ttv_model_params['ror'][pl]*self.ttv_model_params['Rs'])
                self.ttv_model_params['b'][pl] = pm.TruncatedNormal("b_"+pl, lower=0,upper=1+1.25*np.nanmedian(self.trace.posterior["ror_"+pl].values), mu=np.clip(1.1*np.nanmedian(self.trace.posterior["b_"+pl].values),0,1),
                                                                                                                          sigma=np.clip(np.nanstd(self.trace.posterior["b_"+pl].values),0.1,0.5))
                
                if self.assume_circ:
                    self.ttv_model_params['orbit'][pl] = xo.orbits.TTVOrbit(b=[self.ttv_model_params['b'][pl]], 
                                                    transit_times=[self.ttv_model_params['transit_times'][pl]], 
                                                    transit_inds=[self.planets[pl]['init_transit_inds']], 
                                                    r_star=self.ttv_model_params['Rs'], 
                                                    m_star=self.ttv_model_params['Ms'])
                else:
                    self.ttv_model_params['orbit'][pl] = xo.orbits.TTVOrbit(b=[self.ttv_model_params['b'][pl]], 
                                                    transit_times=[self.ttv_model_params['transit_times'][pl]], 
                                                    transit_inds=[self.planets[pl]['init_transit_inds']], 
                                                    r_star=self.ttv_model_params['Rs'], 
                                                    m_star=self.ttv_model_params['Ms'], 
                                                    ecc=[self.ttv_model_params['ecc']], 
                                                    omega=[self.ttv_model_params['omega']])
                self.ttv_model_params['t0'][pl] = pm.Deterministic("t0_"+pl, self.ttv_model_params['orbit'][pl].t0[0])
                self.ttv_model_params['P'][pl] = pm.Deterministic("P_"+pl, self.ttv_model_params['orbit'][pl].period[0])
            newsigmas={}
            for scope in list(self.lcs.keys()):
                self.ttv_model_params[scope+"_logs"] = pm.Normal(scope+"_logs", mu=np.log(np.nanmedian(abs(np.diff(cor_lcs[scope][:,1]))))-3,sigma=3)
                self.ttv_model_params[scope+'_planets_x']={}
                for pl in self.planets:
                    self.ttv_model_params[scope+'_planets_x'][pl] = pm.Deterministic(scope+"_planets_x_"+pl, xo.LimbDarkLightCurve(self.ttv_model_params['u_stars'][scope]).get_light_curve(orbit=self.ttv_model_params['orbit'][pl], r=self.ttv_model_params['rpl'][pl]/109.2,
                                                                                                                                   t=cor_lcs[scope][:,0])[:,0]*1000)
                self.ttv_model_params[scope+'_allplanets_x'] = pm.Deterministic(scope+"_allplanets_x", pm.math.sum([self.ttv_model_params[scope+'_planets_x'][pl] for pl in self.planets],axis=0))
                newsigmas[scope] = pm.math.sqrt(cor_lcs[scope][:,2] ** 2 + pm.math.exp(self.ttv_model_params[scope+'_logs'])**2)
                #self.ttv_model_params[scope+'_llk'] = pm.Normal(scope+'_llk', mu=self.ttv_model_params[scope+'_allplanets_x'], sigma=newsigmas[scope]**2, observed=cor_lcs[scope][:,1])
                #-0.5 * pm.math.sum((cor_lcs[scope][:,1] - self.ttv_model_params[scope+'_allplanets_x']) ** 2/newsigmas[scope] + np.log(newsigmas[scope])))
            self.ttv_model_params['log_likelihood'] = pm.Normal("log_likelihood", mu=pm.math.concatenate([self.ttv_model_params[scope+'_allplanets_x'] for scope in self.lcs]), 
                                                                sigma=pm.math.concatenate([newsigmas[scope] for scope in self.lcs]), 
                                                                observed=pm.math.concatenate([cor_lcs[scope][:,1] for scope in self.lcs]))
            #pm.Deterministic('log_likelihood', pm.math.sum([pm.math.sum(self.ttv_model_params[scope+'_llk']) for scope in self.lcs]))
            #First try to find best-fit transit stuff:
            optvar=[]
            for pl in self.planets:
                if pl in self.ttv_model_params['transit_times']:
                    optvar+=[self.ttv_model_params['transit_times'][pl][i] for i in range(len(self.ttv_model_params['transit_times'][pl]))]
                else:
                    optvar+=[self.ttv_model_params['P'][pl],self.ttv_model_params['t0'][pl]]
            comb_soln = pmx.optimize(vars = optvar+[self.ttv_model_params['logror'][pl] for pl in self.planets] + \
                                            [self.ttv_model_params[scope+'_logs'] for scope in list(self.lcs.keys())])
            
            #More complex transit fit
            ivars=[self.ttv_model_params['b'][pl] for pl in self.planets]+[self.ttv_model_params['logror'][pl] for pl in self.planets]+[self.ttv_model_params[scope+'_logs'] for scope in list(self.lcs.keys())]
            ivars+=[self.ttv_model_params['u_stars'][u] for u in self.ttv_model_params['u_stars']]
            for pl in self.planets:
                if pl in self.ttv_model_params['transit_times']:
                    ivars+=[self.ttv_model_params['transit_times'][pl][i] for i in range(len(self.ttv_model_params['transit_times'][pl]))]
                else:
                    ivars+=[self.ttv_model_params['P'][pl],self.ttv_model_params['t0'][pl]]
            if not self.assume_circ:
                ivars+=[self.ttv_model_params['ecc'][pl] for pl in self.planets]+[self.ttv_model_params['omega'][pl] for pl in self.planets]
            self.logger.debug(ivars)
            comb_soln = pmx.optimize(start=comb_soln, vars=ivars)

            #Doing everything:
            self.ttv_init_soln = pmx.optimize(start=comb_soln)
            
            self.ttv_trace = pm.sample(tune=n_tune_steps, draws=n_draws, 
                                    chains=self.n_cores, cores=self.n_cores, 
                                    start=self.ttv_init_soln, target_accept=0.8, return_inferencedata=True)#**kwargs)
            self.save_trace_summary(trace=self.ttv_trace,suffix="_ttvfit",returndf=False)
        if not hasattr(self,'model_comp'):
            self.model_comp={'ttv':{}}
        self.model_comp["ttv"]['wttv_waic']=az.waic(self.ttv_trace)
        self.model_comp["ttv"]['no_ttv_waic']=az.waic(self.trace,var_name="log_likelihood")
        self.model_comp["ttv"]['deltaWAIC']=self.model_comp["ttv"]['wttv_waic']['elpd_waic']-self.model_comp["ttv"]['no_ttv_waic']['elpd_waic']
        self.model_comp["ttv"]['WAIC_pref_model']="ttvs" if self.model_comp["ttv"]['deltaWAIC']>0 else "no_ttvs"
        
    def save_trace_summary(self, trace=None, suffix="", returndf=True):
        """Make a csv of the pymc model """
        trace=self.trace if trace==None else trace
        assert not (suffix=="" and trace is None), "If you're using a non-standard trace, please include a distinct file suffix."

        var_names=[var for var in trace.posterior if 'gp_' not in var and 'model_' not in var and '__' not in var and (np.product(trace.posterior[var].shape)<6*np.product(trace.posterior['Rs'].shape) or 'transit_times' in var)]
        self.trace_summary=pm.summary(trace,var_names=var_names,round_to=8,
                                        stat_funcs={"5%": lambda x: np.percentile(x, 5),"-$1\sigma$": lambda x: np.percentile(x, 15.87),
                                                    "median": lambda x: np.percentile(x, 50),"+$1\sigma$": lambda x: np.percentile(x, 84.13),
                                                    "95%": lambda x: np.percentile(x, 95)})
        self.trace_summary.to_csv(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_model_summary"+suffix+".csv"))
        if returndf:
            return self.trace_summary
    
    def make_cheops_obs_table(self, dur_unit="orbits", incl_cad=False, incl_planets=True, incl_rms=True,incl_aveff=True):
        """
        Make table of cheops observations. This table is used to calculate the time difference between observations and the time difference between them
        
        Args:
            dur_unit: duration unit ("orbits" or "hours")
        """
        
        # Headers = Date start, JD start, Duration [orbits], Filekey, cadence, Average efficiency, RMS [ppm], Planets present
        cheops_latex_tab=pd.DataFrame()
        starts=[]
        latex_tab="\\begin{table}\n\\centering\n\\begin{tabular}{lccccccc}\n"

        dur_mult = 1 if dur_unit=="orbits" else (98.77/60)
        fk_starts=[]
        for nfk,fk in enumerate(np.unique(self.cheops_filekeys)):
            fk_starts+=[np.nanmedian(self.lcs["cheops"].loc[self.lcs["cheops"]['filekey']==fk,'time'].values)]
        
        for nfk,fk in enumerate(np.array(self.cheops_filekeys)[np.argsort(fk_starts)]):
            lcfk=self.lcs["cheops"].loc[self.lcs["cheops"]['filekey']==fk]
            cad=np.nanmedian(np.diff(lcfk['time'].values))*86400
            dur=(lcfk['time'].values[-1]-lcfk['time'].values[0])/(98.9/1440)
            floored_orbs=np.floor(dur)*(98.77/1440)
            if hasattr(self,'trace'):
                if 'gp_rollangle_model_phi_'+str(fk) in self.trace.posterior:
                    fkmod=np.nanmedian(self.trace.posterior['cheops_summodel_x_'+str(fk)]+self.trace.posterior['gp_rollangle_model_phi_'+str(fk)],axis=(0,1))
                elif 'spline_model_'+str(fk) in self.trace.posterior:
                    fkmod=np.nanmedian(self.trace.posterior['cheops_summodel_x_'+str(fk)].values+self.trace.posterior['spline_model_'+str(fk)].values,axis=(0,1))
                    print(fkmod,fkmod.shape)
                else:
                    fkmod=np.nanmedian(self.trace.posterior['cheops_summodel_x_'+str(fk)],axis=(0,1))
            else:
                assert hasattr(self,'init_soln'), "must have run `init_model`"
                if 'gp_rollangle_model_phi_'+str(fk) in self.init_soln:
                    self.logger.debug(np.sum(np.vstack([self.init_soln['cheops_planets_x_'+pl][self.cheops_fk_mask[fk]] for pl in self.planets]),axis=0).shape)
                    fkmod=self.init_soln['cheops_flux_cor_'+str(fk)]+np.sum(np.vstack([self.init_soln['cheops_planets_x_'+pl][self.cheops_fk_mask[fk]] for pl in self.planets]),axis=0)+self.init_soln['gp_rollangle_model_phi_'+str(fk)][self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)]
                elif 'spline_model_'+str(fk) in self.init_soln:
                    self.logger.debug(np.sum(np.vstack([self.init_soln['cheops_planets_x_'+pl][self.cheops_fk_mask[fk]] for pl in self.planets]),axis=0).shape)
                    fkmod=self.init_soln['cheops_flux_cor_'+str(fk)]+np.sum(np.vstack([self.init_soln['cheops_planets_x_'+pl][self.cheops_fk_mask[fk]] for pl in self.planets]),axis=0)+self.init_soln['spline_model_'+str(fk)]
                else:
                    fkmod=self.init_soln['cheops_flux_cor_'+str(fk)]+np.sum(np.vstack([self.init_soln['cheops_planets_x_'+pl][self.lcs["cheops"]['filekey']==fk] for pl in self.planets]),axis=0)
            starts+=[lcfk['time'].values[0]]
            effchecks=np.arange(0,dur-1,1/11)
            aveff=np.nanmedian([(np.sum((lcfk['time'].values>(lcfk['time'][0]+ec*0.0686))&(lcfk['time'].values<(lcfk['time'][0]+(ec+1)*0.0686)))*cad)/(98.77*60) for ec in effchecks])            
            self.logger.debug(fk+str(int(np.round(aveff*100))))
            info={"Date start":Time(lcfk['time'].values[0],format='jd').isot,
                  "BJD start":"$ "+str(lcfk['time'].values[0])+" $",
                  "Dur ["+dur_unit+"]":"$ "+str(np.round(dur*dur_mult,2))+" $",
                  "Filekey":fk}
            if incl_cad:
                info.update({"Cad. [s]":"$ "+str(np.round(cad,1))+" $"})
            if incl_aveff:
                info.update({"Av. eff. [%]":"$ "+str(int(aveff*100))+" $"})
            if incl_rms:
                info.update({"RMS [ppm]":"$ "+str(int(np.round(np.std(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux']-fkmod)*1e3)))+" $"})
            if incl_planets:
                info.update({"Planets":", ".join([pl for pl in self.planets if np.any(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],"in_trans_"+pl])])})
            #Efficiency is number of actual observations (cutting any final residual orbit) / expected observations at 100% efficiency
            if nfk==0:
                latex_tab+=" & ".join(info.keys())+"\\\\\n"
            latex_tab+=" & ".join(info.values())+"\\\\\n"
        latex_tab+="\\end{tabular}\n\\caption{List of CHEOPS observations.}\\ref{tab:cheops_dat}\n\\end{table}"
        with open(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_cheops_table.tex"),'w') as f:
            f.write(latex_tab)
        return latex_tab

    def make_lcs_timeseries(self, src, tracename="trace", overwrite=False, **kwargs):
        """
        Pandas dataframe with:
         - tess_gpmodel_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - either fitted GP or pre-detrtended spline model
         - tess_[b]model_[p] (where p is +2sig, +1sig, med, -1sig, -2sig; and [b] is for each planet) - fitted planetary models
         - tess_allplmodel_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - combined planetary models
         """
        
        if not hasattr(self,'models_out'):
            self.models_out={}

        if src=="cheops":
            self.make_cheops_timeseries(overwrite=overwrite,**kwargs)
         
        #Checking if there is already this source in the models_out dataframe AND whether we now have a trace (rather than best-fit) input timeseres...
        if src not in self.models_out or overwrite or (src in self.models_out and hasattr(self,'trace') and not np.any(['1sig' in col for col in self.models_out[src].columns])):
            #Firstly taking the masked lightcurve as input:
            self.models_out[src]=self.lcs[src].loc[self.lcs[src]['mask']]
            if self.fit_gp:
                if self.bin_oot:
                    #Need to interpolate to the smaller values
                    from scipy.interpolate import interp1d
                    if hasattr(self,tracename):
                        for p in self.percentiles:
                            #print(np.min(self.lcs[src].iloc[0]['time'])-0.5,self.lc_fit['time'][0],self.lc_fit['time'][-1],np.max(self.lcs[src].iloc[-1]['time'])+0.5))
                            interpp=interp1d(np.hstack((np.min(self.lcs[src].iloc[0]['time'])-0.5,self.lc_fit[src]['time'].values,np.max(self.lcs[src].iloc[-1]['time'])+0.5)),
                                                np.hstack((0,np.percentile(getattr(self,tracename).posterior[src+'_gp_model_x'],self.percentiles[p],axis=(0,1)),0)))
                            #print(np.min(self.lcs[src].loc[self.lcs[src]['mask'],'time'].values),np.max(self.lcs[src].loc[self.lcs[src]['mask'],'time'].values),self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[0],self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[-1])
                            self.models_out[src][src+"_gpmodel_"+p]=interpp(self.lcs[src].loc[self.lcs[src]['mask'],'time'].values)
                    elif hasattr(self,'init_soln'):
                        interpp=interp1d(np.hstack((np.min(self.lcs[src].iloc[0]['time'])-0.5,self.lc_fit[src]['time'].values,np.max(self.lcs[src].iloc[-1]['time'])+0.5)),
                                            np.hstack((0,self.init_soln[src+'_gp_model_x'],0)))
                        self.models_out[src][src+"_gpmodel_med"]=interpp(self.lcs[src].loc[self.lcs[src]['mask'],'time'].values)
                elif not self.cut_oot:
                    if hasattr(self,tracename):
                        for p in self.percentiles:
                            self.models_out[src][src+"_gpmodel_"+p]=np.percentile(getattr(self,tracename).posterior[src+'_gp_model_x'],self.percentiles[p],axis=(0,1))
                    elif hasattr(self,'init_soln'):
                        self.models_out[src][src+"_gpmodel_med"]=self.init_soln[src+'_gp_model_x']
                elif self.cut_oot:
                    if hasattr(self,tracename):
                        for p in self.percentiles:
                            self.models_out[src][src+"_gpmodel_"+p] = np.tile(np.nan,len(self.models_out[src]['time']))
                            self.models_out[src][src+"_gpmodel_"+p][self.lcs[src]['near_trans']&self.lcs[src]['mask']] = np.percentile(getattr(self,tracename).posterior[src+'_gp_model_x'],self.percentiles[p],axis=0)
                    elif hasattr(self,'init_soln'):
                        p="med"
                        self.models_out[src][src+"_gpmodel_"+p] = np.tile(np.nan,len(self.models_out[src]['time']))
                        self.models_out[src][src+"_gpmodel_"+p][self.lcs[src]['near_trans']&self.lcs[src]['mask']] = self.init_soln[src+'_gp_model_x']
            else:
                self.models_out[src][src+"_gpmodel_med"] = self.models_out[src]["spline"].values[:]
            if hasattr(self,tracename):
                for p in self.percentiles:
                    for pl in self.planets:
                        self.models_out[src][src+'_'+pl+"model_"+p]=np.zeros(np.sum(self.lcs[src]['mask']))
                        self.models_out[src].loc[self.lcs[src].loc[self.lcs[src]['mask'],'near_trans'],src+'_'+pl+"model_"+p]=np.nanpercentile(getattr(self,tracename).posterior[src+'_model_x_'+pl].values[:,:,self.lc_fit[src]['near_trans']],self.percentiles[p],axis=(0,1))
                    self.models_out[src][src+"_allplmodel_"+p]=np.zeros(np.sum(self.lcs[src]['mask']))
                    self.models_out[src].loc[self.lcs[src].loc[self.lcs[src]['mask'],'near_trans'],src+"_allplmodel_"+p]=np.nanpercentile(np.sum(np.stack([getattr(self,tracename).posterior[src+'_model_x_'+pl].values[:,:,self.lc_fit[src]['near_trans']] for pl in self.planets]),axis=0),self.percentiles[p],axis=(0,1))
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
         - cheops_gpmodel_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - Fitted roll angle GP model in time axis
         - cheops_pred_spline_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - Fitted roll angle spline model in time axis
         - cheops_lindetrend_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - Linear decorrelation model predictions
         - cheops_alldetrend_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - Combined spline/gp model with linear decorrelation
         - cheops_[b]model_[p] (where p is +2sig, +1sig, med, -1sig, -2sig; and [b] is for each planet) - fitted planetary models
         - cheops_allplmodel_[p] (where p is +2sig, +1sig, med, -1sig, -2sig) - combined planetary models
        The final two are also included in the "cheops_gap_models_out" Pandas dataframe
        """
        if not hasattr(self,'models_out'):
            self.models_out={}
        self.logger.debug([type(init_trace),type(init_trace)==pm.backends.base.MultiTrace])
        
        if tracename is None and 'cheops' not in self.models_out or overwrite or ('cheops' in self.models_out and hasattr(self,'trace') and not np.any(['1sig' in col for col in self.models_out['cheops'].columns])):
            assert init_trace is None and fk is None, "We will use the default \'self.trace\' for the final CHEOPS model. For an intermediate trace, specify the trace type & filekey)"
            self.models_out['cheops']=pd.DataFrame()
            self.models_out['cheops_gap_models_out']=pd.DataFrame()

            for col in ['time','flux','flux_err','phi','bg','centroidx','centroidy','deltaT','xoff','yoff','filekey','n_phi_model','raw_flux_medium_offset','raw_flux_medium_offset_centroidcorr']:
                if col in self.lcs["cheops"].columns:
                    self.models_out['cheops'][col]=np.hstack([self.lcs["cheops"].loc[self.cheops_fk_mask[fk],col] for fk in self.cheops_filekeys])
            if self.fit_phi_gp:
                if hasattr(self,'trace'):
                    for p in self.percentiles:
                        self.models_out['cheops']['cheops_pred_gp_'+p]=np.hstack([np.nanpercentile(self.trace.posterior['gp_rollangle_model_phi_'+str(fk)][:,:,self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],self.percentiles[p],axis=(0,1)) for fk in self.cheops_filekeys])
                elif hasattr(self,'init_soln'):
                    self.models_out['cheops']['cheops_pred_gp_med']=np.hstack([self.init_soln['gp_rollangle_model_phi_'+str(fk)][self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)] for fk in self.cheops_filekeys])

            elif self.fit_phi_spline:
                if hasattr(self,'trace'):
                    for p in self.percentiles:
                        self.models_out['cheops']['cheops_pred_spline_'+p]=np.hstack([np.nanpercentile(self.trace.posterior['spline_model_'+str(fk)],self.percentiles[p],axis=(0,1)) for fk in self.cheops_filekeys])
                        #fkmod=np.nanmedian(self.trace.posterior['cheops_summodel_x_'+str(fk)]+self.trace.posterior['spline_model_phi_'+str(fk)][:,self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],axis=0)
                elif hasattr(self,'init_soln'):
                    self.models_out['cheops']['cheops_pred_spline_med']=np.hstack([self.init_soln['spline_model_'+str(fk)] for fk in self.cheops_filekeys])
            if hasattr(self,'cheops_gap_timeseries'):
                self.models_out['cheops_gap_models_out']['time']=self.cheops_gap_timeseries
            if hasattr(self,'cheops_gap_fks'):
                self.models_out['cheops_gap_models_out']['filekey']=self.cheops_gap_fks
            if hasattr(self,'trace'):
                for p in self.percentiles:
                    self.models_out['cheops']['cheops_lindetrend_'+p]=np.hstack([np.nanpercentile(self.trace.posterior['cheops_flux_cor_'+fk],self.percentiles[p],axis=(0,1)) for fk in self.cheops_filekeys])
                    if self.fit_phi_gp:
                        self.models_out['cheops']['cheops_alldetrend_'+p]=np.hstack([np.nanpercentile(self.trace.posterior['cheops_flux_cor_'+fk].values+self.trace.posterior['gp_rollangle_model_phi_'+str(fk)].values[:,:,self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],self.percentiles[p],axis=(0,1)) for fk in self.cheops_filekeys])

                    elif self.fit_phi_spline:
                        self.models_out['cheops']['cheops_alldetrend_'+p]=np.hstack([np.nanpercentile(self.trace.posterior['cheops_flux_cor_'+fk].values+self.trace.posterior['spline_model_'+str(fk)].values,self.percentiles[p],axis=(0,1)) for fk in self.cheops_filekeys])
                    
                    for npl,pl in enumerate(self.planets):
                        if 'cheops_planets_x' in self.trace.posterior:
                            self.models_out['cheops']['cheops_'+pl+"model_"+p]=np.nanpercentile(self.trace.posterior['cheops_planets_x'].values[:,:,self.lcs["cheops"]['mask'],npl],self.percentiles[p],axis=(0,1))
                        else:
                            self.models_out['cheops']['cheops_'+pl+"model_"+p]=np.nanpercentile(self.trace.posterior['cheops_planets_x_'+pl].values[:,:,self.lcs["cheops"]['mask']],self.percentiles[p],axis=(0,1))
                        if 'cheops_planets_gaps_'+pl in self.trace.posterior:
                            self.models_out['cheops_gap_models_out']['cheops_'+pl+"model_"+p]=np.nanpercentile(self.trace.posterior['cheops_planets_gaps_'+pl],self.percentiles[p],axis=(0,1))

                    if 'cheops_planets_x' in self.trace.posterior:
                        self.models_out['cheops']['cheops_allplmodel_'+p]=np.nanpercentile(np.sum(self.trace.posterior['cheops_planets_x'][:,:,self.lcs["cheops"]['mask'],:],axis=2),self.percentiles[p],axis=(0,1))
                    elif len(self.planets)>0:
                        self.models_out['cheops']["cheops_allplmodel_"+p]=np.nanpercentile(np.sum(np.stack([self.trace.posterior['cheops_planets_x_'+pl].values[:,:,self.lcs["cheops"]['mask']] for pl in self.planets]),axis=0),self.percentiles[p],axis=(0,1))
                    if len(self.planets)>0:
                        self.models_out['cheops_gap_models_out']["cheops_allplmodel_"+p]=np.nanpercentile(np.sum(np.dstack([self.trace.posterior['cheops_planets_gaps_'+pl].values for pl in self.planets]),axis=2),self.percentiles[p],axis=(0,1))
            elif hasattr(self,'init_soln'):
                p="med"
                self.models_out['cheops']['cheops_lindetrend_'+p]=np.hstack([self.init_soln['cheops_flux_cor_'+fk] for fk in self.cheops_filekeys])
                if self.fit_phi_gp:
                    self.models_out['cheops']['cheops_alldetrend_'+p]=np.hstack([self.init_soln['cheops_flux_cor_'+fk]+self.init_soln['gp_rollangle_model_phi_'+str(fk)][self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)] for fk in self.cheops_filekeys])

                elif self.fit_phi_spline:
                    self.models_out['cheops']['cheops_alldetrend_'+p]=np.hstack([self.init_soln['cheops_flux_cor_'+fk]+self.init_soln['spline_model_'+str(fk)] for fk in self.cheops_filekeys])
                for npl,pl in enumerate(self.planets):
                    if 'cheops_planets_x' in self.init_soln:
                        self.models_out['cheops']['cheops_'+pl+"model_"+p]=self.init_soln['cheops_planets_x'][self.lcs["cheops"]['mask'],npl]
                    else:
                        self.models_out['cheops']['cheops_'+pl+"model_"+p]=self.init_soln['cheops_planets_x_'+pl][self.lcs["cheops"]['mask']]
                    if 'cheops_planets_gaps_'+pl in self.init_soln:
                        self.models_out['cheops_gap_models_out']['cheops_'+pl+"model_"+p]=self.init_soln['cheops_planets_gaps_'+pl]

                if 'cheops_planets_x' in self.init_soln:
                    self.models_out['cheops']['cheops_allplmodel_'+p]=np.sum(self.init_soln['cheops_planets_x'][self.lcs["cheops"]['mask'],:],axis=1)
                elif len(self.planets)>0:
                    self.models_out['cheops']['cheops_allplmodel_'+p]=np.sum(np.vstack([self.models_out['cheops']['cheops_'+pl+"model_"+p] for pl in self.planets]),axis=0)
                else:
                    self.models_out['cheops']['cheops_allplmodel_'+p]=np.zeros_like(self.models_out['cheops']['time'])

                
                if len(self.planets)>0:
                    self.models_out['cheops_gap_models_out']["cheops_allplmodel_"+p]=np.sum(np.vstack([self.init_soln['cheops_planets_gaps_'+pl] for pl in self.planets]),axis=0)
                else:
                    self.models_out['cheops_gap_models_out']["cheops_allplmodel_"+p]=np.zeros_like(self.models_out['cheops_gap_models_out']['time'])
        elif tracename not in self.models_out or overwrite:
            #We have intermediate CHEOPS trace which we want to save in the same format as the final CHEOPS trace above (i.e. for plotting)
            self.models_out[tracename]=pd.DataFrame()
            #self.models_out[tracename+'_gap_models_out']=pd.DataFrame()
            fks=self.cheops_filekeys if fk is None else [fk]
            for col in ['time','flux','flux_err','filekey','raw_flux_medium_offset','raw_flux_medium_offset_centroidcorr']:
                if col in self.lcs["cheops"].columns:
                    self.models_out[tracename][col]=np.hstack([self.lcs["cheops"].loc[self.cheops_fk_mask[fk],col] for fk in fks])
            if type(init_trace)==pm.backends.base.MultiTrace:
                for p in self.percentiles:
                    self.models_out[tracename]['cheops_lindetrend_'+p]=np.hstack([np.nanpercentile(init_trace['cheops_flux_cor_'+fk],self.percentiles[p],axis=0) for fk in fks])
                    self.models_out[tracename]['cheops_alldetrend_'+p]=self.models_out[tracename]['cheops_lindetrend_'+p].values[:]
                    if len(self.planets)>0:
                        self.models_out[tracename]["cheops_allplmodel_"+p]=np.zeros(len(self.models_out[tracename]['time']))
                    for npl,pl in enumerate(self.planets):
                        if "cheops_planets_x_"+pl+"_"+fk in init_trace.varnames:
                            self.models_out[tracename]['cheops_'+pl+"model_"+p]=np.hstack([np.nanpercentile(init_trace["cheops_planets_x_"+pl+"_"+fk],self.percentiles[p],axis=0) for fk in fks])
                        else:
                            self.models_out[tracename]['cheops_'+pl+"model_"+p]=np.zeros(len(self.models_out[tracename]['time']))
                        self.models_out[tracename]["cheops_allplmodel_"+p]+=self.models_out[tracename]['cheops_'+pl+"model_"+p]
            elif type(init_trace)==dict:
                p="med"
                self.models_out[tracename]['cheops_lindetrend_'+p]=np.hstack([init_trace['cheops_flux_cor_'+fk] for fk in fks])
                self.models_out[tracename]['cheops_alldetrend_'+p]=self.models_out[tracename]['cheops_lindetrend_'+p].values[:]
                self.models_out[tracename]["cheops_allplmodel_"+p]=np.zeros(len(self.models_out[tracename]['time']))

                for npl,pl in enumerate(self.planets):
                    if "cheops_planets_x_"+pl+"_"+fk in init_trace.varnames:
                        self.models_out[tracename]['cheops_'+pl+"model_"+p]=np.hstack([np.nanpercentile(init_trace["cheops_planets_x_"+pl+"_"+fk],self.percentiles[p],axis=0) for fk in fks])
                    else:
                        self.models_out[tracename]['cheops_'+pl+"model_"+p]=np.zeros(len(self.models_out[tracename]['time']))
                    self.models_out[tracename]["cheops_allplmodel_"+p]+=self.models_out[tracename]['cheops_'+pl+"model_"+p]
            #self.models_out[tracename+'_gap_models_out']=self.models_out[tracename] #Setting these to be identical
   #cheops_planets_x

    def make_rv_timeseries(self,**kwargs):
        """Make the RV model timeseries"""

        if not hasattr(self,'models_out'):
            self.models_out={}
        self.models_out['rv']=self.rvs
        self.models_out['rv_t']=pd.DataFrame({'time':self.rv_t})
        if hasattr(self,'trace'):
            for p in self.percentiles:
                self.models_out['rv_t']["rvt_bothmodel_"+p]=np.nanpercentile(self.trace.posterior['rv_model_t'], self.percentiles[p], axis=(0,1))
                
                for npl,pl in enumerate(self.planets):
                    self.models_out['rv_t']["rvt_"+pl+"model_"+p]=np.nanpercentile(self.trace.posterior['vrad_t_'+pl], self.percentiles[p], axis=(0,1))
                    self.models_out['rv']["rv_"+pl+"model_"+p]=np.nanpercentile(self.trace.posterior['vrad_x_'+pl], self.percentiles[p], axis=(0,1))

                if self.npoly_rv>1:
                    self.models_out['rv_t']["rvt_bkgmodel_"+p]=np.nanpercentile(self.trace.posterior['bkg_t'], self.percentiles[p], axis=(0,1))
                self.models_out['rv']["rv_bkgmodel_"+p]=np.nanpercentile(self.trace.posterior['bkg_x'], self.percentiles[p], axis=(0,1))
        else:
            p='med'
            self.models_out['rv_t']["rvt_bothmodel_"+p]=self.init_soln['rv_model_t']
            
            for npl,pl in enumerate(self.planets):
                self.models_out['rv_t']["rvt_"+pl+"model_"+p]=self.init_soln['vrad_t_'+pl]
                self.models_out['rv']["rv_"+pl+"model_"+p]=self.init_soln['vrad_x_'+pl]

            if self.npoly_rv>1:
                self.models_out['rv_t']["rvt_bkgmodel_"+p]=self.init_soln['bkg_t']
            self.models_out['rv']["rv_bkgmodel_"+p]=self.init_soln['bkg_x']


    def make_prior_posterior_table(self,**kwargs):
        """Making a table of prior & posterior values
        Copying the arctitecture from the init_model function but simply storing a list
        """
        tab=[['Teff','Stellar $T_{\\rm eff}$','[K]','normal',self.Teff[0],self.Teff[1]],
            ['Rs','Stellar Radius, $R_s$','[$R_\\odot$]','normal',self.Rstar[0],self.Rstar[1]]]
        if (hasattr(self,'use_mstar') and self.use_mstar) or self.spar_param=='Mstar':
            if self.spar_prior=='constr':
                tab+=[['Ms','Stellar Mass, $M_s$','$M_\\odot$','normal',self.Mstar[0],self.Mstar[1]]]
            elif self.spar_prior=='loose':
                tab+=[['Ms','Stellar Mass, $M_s$','$M_\\odot$','normal',self.Mstar[0],0.33*self.Mstar[0]]]
            elif self.spar_prior=='logloose':
                tab+=[['logMs','Stellar $\\log{M_s}$','','Uniform',np.log(self.Mstar[0])-2,np.log(self.Mstar[0])+2]]

        elif (hasattr(self,'use_logg') and self.use_logg) or self.spar_param=='logg':
            if self.spar_prior=='constr':
                tab+=[['logg','Stellar $\\log{\\rm g}$','','normal',self.logg[0],self.logg[1]]]
            elif self.spar_prior=='loose':
                tab+=[['logg','Stellar $\\log{\\rm g}$','','normal',self.logg[0],0.33*self.logg[0]]]
                
        elif self.spar_param=='rhostar':
            if self.spar_prior=='constr':
                tab+=[['rhostar','Stellar $\\rho$','$\\rho_\\odot$','normal',self.rhostar[0],self.rhostar[1]]]
            elif self.spar_prior=='loose':
                tab+=[['rhostar','Stellar $\\rho$','$\\rho_\\odot$','normal',self.rhostar[0],self.rhostar[0]*0.33]]
            elif self.spar_prior=='logloose':
                tab+=[['logrhostar','Stellar $\\log{\\rho}$','','Uniform',np.log(self.rhostar[0])-2,np.log(self.rhostar[0])+2]]

        if self.fit_contam:
            tab+=[['deltaImag_contam','Contamination $\\Delta I_{\\rm cont}$', 'mag', 'Uniform',2.5,12]]
            tab+=[['deltaVmag_contam','Contamination $\\Delta V_{\\rm cont}$','mag', 'Uniform',2.5,12]]

        for scope in self.ld_dists:
            if self.constrain_lds:
                tab+=[['u_star_'+scope+'|0','Quadratic LD param $u_{\\rm '+scope+',0}$', '', 'BoundNormal',np.clip(np.nanmedian(self.ld_dists[scope],axis=0)[0],0,1),
                    np.clip(np.nanstd(self.ld_dists[scope],axis=0)[0],0.1,1.0),0,1]]
                tab+=[['u_star_'+scope+'|1','Quadratic LD param $u_{\\rm '+scope+',0}$', '', 'BoundNormal',np.clip(np.nanmedian(self.ld_dists[scope],axis=0)[1],0,1),
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
            elif self.split_periods is not None and pl in self.split_periods and len(self.split_periods[pl])>1 and self.split_periods[pl]!=range(self.planets[pl]['n_trans']):
                for split in range(len(self.split_periods[pl])):
                    tab+=[['P_'+pl+"_"+str(int(split)),'Period from '+str(int(np.round(int(self.planets[pl]['init_transit_times'][self.split_periods[pl][split[0]]]))))+' to '+str(int(np.round(int(self.planets[pl]['init_transit_times'][self.split_periods[pl][split[-1]]]))))+', $P_{{\rm '+pl+', '+str(int(split))+'}}$ ','[d]','BoundNormal',self.planets[pl]['period'],np.clip(self.planets[pl]['period_err'],0,(max_p-self.planets[pl]['period'])),min_p, max_p]]
                    tab+=[['t0_'+pl+"_"+str(int(split)),'Transit epoch from '+str(int(np.round(int(self.planets[pl]['init_transit_times'][self.split_periods[pl][split[0]]]))))+' to '+str(int(np.round(int(self.planets[pl]['init_transit_times'][self.split_periods[pl][split[-1]]]))))+', $t_{{\rm 0, '+pl+', '+str(int(split))+'}}$ ','[d]','Normal',self.planets[pl]['init_transit_times'][self.split_periods[pl][split[0]]],2*self.planets[pl]['tcen_err']]]
            else:
                for i in range(len(self.planets[pl]['init_transit_times'])):
                    tab+=[['transit_times_'+pl+'_'+str(i),'Transit time, $t_{'+str(i)+','+pl+'}$','[BJD]','Uniform',
                           self.planets[pl]['init_transit_times'][i]-self.planets[pl]['tdur']*self.timing_sd_durs,
                           self.planets[pl]['init_transit_times'][i]+self.planets[pl]['tdur']*self.timing_sd_durs]]
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
                tab+=[['tess_logs','TESS jitter $\\log{\\sigma_{\\rm TESS}}$','','interp',np.nanmedian(self.oot_gp_trace.posterior["tess_logs"]),np.nanstd(self.oot_gp_trace.posterior["tess_logs"])]]
                tab+=[['tess_mean','TESS mean $\\mu_{\\rm TESS}}$','','interp',np.nanmedian(self.oot_gp_trace.posterior["tess_mean"]),np.nanstd(self.oot_gp_trace.posterior["tess_mean"])]]
            if 'k2' in self.lcs:
                tab+=[['k2_logs','K2 jitter $\\log{\\sigma_{\\rm K2}}$','','interp',np.nanmedian(self.oot_gp_trace["k2_logs"]),np.nanstd(self.oot_gp_trace.posterior["k2_logs"])]]
                tab+=[['k2_mean','K2 mean $\\mu_{\\rm K2}}$','','interp',np.nanmedian(self.oot_gp_trace["k2_mean"]),np.nanstd(self.oot_gp_trace.posterior["k2_mean"])]]
            elif 'kepler' in self.lcs:
                tab+=[['kepler_logs','Kepler jitter $\\log{\\sigma_{\\rm Kepler}}$','','interp',np.nanmedian(self.oot_gp_trace.posterior["kepler_logs"]),np.nanstd(self.oot_gp_trace.posterior["kepler_logs"])]]
                tab+=[['kepler_mean','Kepler mean $\\mu_{\\rm Kepler}}$','','interp',np.nanmedian(self.oot_gp_trace.posterior["kepler_mean"]),np.nanstd(self.oot_gp_trace.posterior["kepler_mean"])]]

            if "S0" in self.oot_gp_trace.posterior:
                tab+=[['phot_S0','Photometric GP term $S_0$','','interp',np.nanmedian(self.oot_gp_trace.posterior["S0"]),np.nanstd(self.oot_gp_trace.posterior["S0"])]]
            elif "sigma" in self.oot_gp_trace.posterior:
                tab+=[['phot_sigma','Photometric GP term $\sigma_0$','','interp',np.nanmedian(self.oot_gp_trace.posterior["sigma"]),np.nanstd(self.oot_gp_trace.posterior["sigma"])]]
            tab+=[['phot_w0','Photometric GP term $\\omega_0$','','interp',np.nanmedian(self.oot_gp_trace.posterior["w0"]),np.nanstd(self.oot_gp_trace.posterior["w0"])]]
        else:
            if 'tess' in self.lcs:
                tab+=[['tess_logs','TESS jitter $\\log{\\sigma_{\\rm TESS}}$','','normal',np.log(np.std(self.lc_fit['tess']['flux'].values)),1]]
            if 'k2' in self.lcs:
                tab+=[['k2_logs','K2 jitter $\\log{\\sigma_{\\rm K2}}$','','normal',np.log(np.std(self.lc_fit['k2']['flux'].values)),1]]
            elif 'kepler' in self.lcs:
                tab+=[['kepler_logs','Kepler jitter $\\log{\\sigma_{\\rm Kepler}}$','','normal',np.log(np.std(self.lc_fit['kepler']['flux'].values)),1]]

        if len(self.cheops_filekeys)>0:
            tab+=[['cheops_logs','CHEOPS jitter $\\log{\\sigma_{\\rm CHEOPS}}$','','normal',np.log(np.nanmedian(abs(np.diff(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'].values)))),3]]
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
                tab+=[['cheops_mean_'+fk,"Cheops "+fk.replace("_","\_").replace("_V0200","")+" mean flux","[ppt]","normal",np.nanmedian(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values),np.nanstd(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values)]]
            if self.fit_phi_gp:
                tab+=[['rollangle_logpower',"${\\rm GP}_{\\rm CHEOPS}$, $\log{\\rm power}$","","normal",-6,1]]
                tab+=[['rollangle_logw0',"${\\rm GP}_{\\rm CHEOPS}$, $\log{\\rm \\omega_0}$","","normal",np.log((2*np.pi)/100),1]]
            elif self.fit_phi_spline:
                if self.phi_model_type in ["common","split"]:
                    #Fit a single spline to all rollangle data
                    minmax=(np.min(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi']),np.max(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'phi']))
                    n_knots=int(np.round((minmax[1]-minmax[0])/self.spline_bkpt_cad))
                    nsplit=1 if self.phi_model_type=='common' else int(self.phi_model_type.split("_")[1])
                    splittext=", split="+nsplit if nsplit>1 else ""
                    for i in range(nsplit):
                        for n in np.arange(n_knots):
                            tab+=[['splines_'+str(i)+"|"+str(n),"CHEOPS rollangle B-spline"+splittext+" "+str(n),"[ppt]","normal",0,np.nanmedian(abs(np.diff(self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'])))]]
                else:
                    #Fit splines to each rollangle
                    knot_list={}
                    B={}
                    self.model_params['splines']={}
                    for fk in self.cheops_filekeys:
                        minmax=(np.min(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'phi']),np.max(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'phi']))
                        n_knots=int(np.round((minmax[1]-minmax[0])/self.spline_bkpt_cad))
                        for n in np.arange(n_knots):
                            tab+=[['splines_'+fk+"_"+str(n),"CHEOPS "+fk+" rollangle B-spline "+str(n),"[ppt]","normal",0,np.nanmedian(abs(np.diff(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'flux'].values)))]]
        table=["\\hline"," & ".join(["Parameter","Unit","Prior","Posterior"])+"\\\\","\\hline"]
        for row in tab:
            self.logger.debug(row)
            newtabrow=[row[1],row[2]] #Name and unit
            if row[3].lower()=='normal':
                round_int=int(-1*np.ceil(np.log10(row[5]))+1)
                mu=str(np.round(row[4],round_int));sigma=str(np.round(row[5],round_int))
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
                mu=str(np.round(row[4],round_int));sigma=str(np.round(row[5],round_int))
                newtabrow+=["$\\mathcal{{N}}_{{\\mathcal{{U}}}}(\\mu={0},\\sigma={1},a={2:0.4f},b={3:0.4f})$".format(mu,sd,row[6],row[7])]
            elif row[3].lower()=="interp":
                newtabrow+=["$\\mathcal{{I}}(\\mu={0:0.4f},\\sigma={1:0.4f})$".format(row[4],row[5])]
            if len(row[0].split("|"))==1:
                posterior=vals_to_latex(np.percentile(self.trace.posterior[row[0]],[15.87,50,84.13]))
            elif len(row[0].split("|"))==2:
                posterior=vals_to_latex(np.percentile(self.trace.posterior[row[0].split("|")[0]][:,int(row[0].split("|")[1])],[15.87,50,84.13]))

            newtabrow+=[posterior]
            table+=[" & ".join(newtabrow)+" \\\\"]
        table+=["\\hline"]
        with open(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_trace_modeltable.tex"),'w') as f:
            f.write('\n'.join(table))

    def make_timeseries(self, overwrite=False, **kwargs):
        """Make the model timeseries dictionaries for all input data - photometry & RVs"""
        #assert hasattr(self,'trace'), "Must have run an MCMC"
        
        if not hasattr(self,'models_out'):
            self.models_out={}
        for src in self.lcs:
            if (src not in self.models_out or overwrite or (src in self.models_out and hasattr(self,'trace') and not np.any(['1sig' in col for col in self.models_out[src].columns]))) and src!="cheops":
                self.make_lcs_timeseries(src,overwrite=overwrite,**kwargs)
            elif (src not in self.models_out or overwrite or (src in self.models_out and hasattr(self,'trace') and not np.any(['1sig' in col for col in self.models_out[src].columns]))) and src=="cheops":
                self.make_cheops_timeseries(overwrite=overwrite,**kwargs)            
        
        if hasattr(self,'rvs') and ('rvs' not in self.models_out or overwrite):
            self.make_rv_timeseries(overwrite=overwrite,**kwargs)

    def save_timeseries(self,outfile=None,suffix="",**kwargs):
        """Save the model timeseries to a csv file"""
        self.make_timeseries(**kwargs)
        for mod in self.models_out:
            if mod is not None:
                if outfile is None:
                    self.models_out[mod].to_csv(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_"+mod+suffix+"_timeseries.csv"))
                else:
                    self.models_out[mod].to_csv(outfile)
    
    def check_rollangle_gp(self, make_change=False, **kwargs):
        """Checking now that the model is initialised whether the rollangle GP improves the loglik or not.
        """
        #self.init_soln['cheops_summo']
        llk_cheops_wgp={}
        
        cheops_sigma2s = self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux_err'].values ** 2 + np.exp(self.init_soln['cheops_logs'])**2
        llk_cheops_nogp = -0.5 * np.sum((self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'].values - \
                                        np.hstack([self.init_soln['cheops_summodel_x_'+fk] for fk in self.cheops_filekeys])) ** 2 / \
                                        cheops_sigma2s + np.log(cheops_sigma2s))
        llk_cheops_wgp = -0.5 * np.sum((self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'].values - \
                                        np.hstack([self.init_soln['cheops_summodel_x_'+fk]+self.init_soln['gp_rollangle_model_phi_'+fk] for fk in self.cheops_filekeys])) ** 2 / \
                                        cheops_sigma2s + np.log(cheops_sigma2s))
        #lower BIC values are generally preferred.
        #So (deltabic_wgp - deltabic_nogp)<0 prefers nogp
        delta_bic = 2*np.sum(self.lcs["cheops"]['mask']) + 2*(llk_cheops_nogp - llk_cheops_wgp)
        if delta_bic<0:
            self.logger.info("Assessment of the rollangle suggests a roll angle GP is not beneficial in this case. ("+str(delta_bic)+")")
            if make_change:
                self.update(fit_phi_gp = False)
                self.init_model()
        else:
            self.logger.info("Rollangle GP is beneficial with DelatBIC ="+str(delta_bic))
    
    def check_rollangle_spline(self, make_change=True, **kwargs):
        """Checking now that the model is initialised whether the rollangle GP improves the loglik or not.
        """
        #self.init_soln['cheops_summo']
        
        cheops_sigma2s = self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux_err'].values ** 2 + np.exp(self.init_soln['cheops_logs'])**2
        llk_cheops_nospline = -0.5 * np.sum((self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'].values - \
                                        np.hstack([self.init_soln['cheops_summodel_x_'+fk] for fk in self.cheops_filekeys])) ** 2 / \
                                        cheops_sigma2s + np.log(cheops_sigma2s))
        llk_cheops_wspline = -0.5 * np.sum((self.lcs["cheops"].loc[self.lcs["cheops"]['mask'],'flux'].values - \
                                        np.hstack([self.init_soln['cheops_summodel_x_'+fk]+self.init_soln['spline_model_'+fk] for fk in self.cheops_filekeys])) ** 2 / \
                                        cheops_sigma2s + np.log(cheops_sigma2s))
        #lower BIC values are generally preferred.
        #So (deltabic_wspline - deltabic_nospline)<0 prefers nospline
        delta_bic = 2*np.sum(self.lcs["cheops"]['mask']) + 2*(llk_cheops_nospline - llk_cheops_wspline)
        if delta_bic<0:
            self.logger.info("Assessment of the rollangle suggests a roll angle spline model is not beneficial in this case. ("+str(delta_bic)+")")
            if make_change:
                self.update(fit_phi_spline = False)
                self.init_model()
        else:
            self.logger.info("Rollangle spline model is beneficial with DeltaBIC ="+str(delta_bic))
                        

    def print_settings(self):
        """Print all the global settings"""
        settings=""
        for key in self.defaults:
            settings+=key+"\t\t"+str(getattr(self,key))+"\n"
        self.logger.info(settings)
        print(settings)

    def save_trace(self):
        """Save the pymc trace to file with pickle. Save location default is \'NAME_mcmctrace.pk\'"""
        if not os.path.exists(os.path.join(self.save_file_loc,self.name.replace(" ","_"))):
            os.mkdir(os.path.join(self.save_file_loc,self.name.replace(" ","_")))
        pickle.dump(self.trace,open(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_mcmctrace.pkl"),"wb"))

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
        if os.path.exists(loadfile.replace("_model.pkl","_trace.nc")):
            self.trace=az.from_netcdf(loadfile.replace("_model.pkl","_trace.nc"))

    def save_model_to_file(self, savefile=None, limit_size=True, suffix=None, remove_all_trace_timeseries=False):
        """Save a chexo_model object direct to file.

        Args:
            savefile (str, optional): File location to save to, otherwise it takes the default location using `GetSavename`. Defaults to None.
            limit_size (bool, optional): If we want to limit size this function can delete unuseful hyperparameters before saving. Defaults to False.
        """
        if savefile is None:
            savefile=os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+'_model.pkl')
        
        
        #First saving GP predictions/etc using save_timeseries:
        self.save_timeseries(overwrite=True)

        #Saving the trace summary
        if hasattr(self,'trace') and not hasattr(self, 'trace_summary'):
            self.save_trace_summary
        elif hasattr(self, 'trace'):
            suffix='' if suffix is None else suffix
            self.trace_summary.to_csv(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_model_summary"+suffix+".csv"))

        #Loading from pickle dictionary
        if limit_size and hasattr(self,'trace'):
            #We cannot afford to store full arrays of GP predictions and transit models

            #Let's clip gp and lightcurves and pseudo-variables from the trace:
            #remvars=[var for var in self.trace.varnames if (('_allphi' in var or 'gp_' in var or '_gp' in var or 'light_curve' in var) and np.product(self.trace.posterior[var].shape)>6*len(self.trace.posterior['Rs'])) or '__' in var]
            remvars=[var for var in self.trace.posterior if '_allphi' in var or '__' in var]
            if remove_all_trace_timeseries:
                remvars=list(np.unique(remvars+[var for var in self.trace.posterior if np.product(self.trace.posterior[var].shape)>50*self.trace.posterior['Rs'].shape[0]]))
            for key in remvars:
                #Permanently deleting these values from the trace.
                del self.trace.posterior[key]
            #medvars=[var for var in self.trace.varnames if 'gp_' not in var and '_gp' not in var and 'light_curve' not in var]
        n_bytes = 2**31
        max_bytes = 2**31-1
        #print({k:type(self.__dict__[k]) for k in self.__dict__})
        bytes_out = pickle.dump({k:self.__dict__[k] for k in self.__dict__ if type(self.__dict__[k]) not in [pm.Model,xo.orbits.KeplerianOrbit,az.data.inference_data.InferenceData] and k not in ['model_params']},open(savefile,'wb')) 
        #bytes_out = pickle.dumps(self)
        with open(savefile, 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])
        self.trace.to_netcdf(savefile.replace("_model.pkl","_trace.nc"))
        del bytes_out
        #pick=pickle.dump(self.__dict__,open(loadfile,'wb'))
    
    def plot_all(self,**kwargs):
        if not hasattr(self,"models_out") or "cheops" not in self.models_out or (hasattr(self,'trace') and "cheops_lindetrend_+1sig" not in self.models_out["cheops"]):
            self.make_cheops_timeseries(**kwargs)
        for scope in self.lcs:
            if scope=='cheops':
                if self.fit_phi_gp:
                    self.plot_rollangle_model(**kwargs)
                self.plot_cheops(**kwargs)
            else:
                self.plot_phot(scope,**kwargs)
        self.plot_transits_fold(**kwargs)


    def plot_rollangle_model(self,save=True,savetype='png',save_suffix=None,**kwargs):
        """Plot the CHEOPS model as a function of rollangle. Works for either spline or GP.
        """
        
        if not hasattr(self,"models_out") or "cheops" not in self.models_out or (hasattr(self,'trace') and "cheops_lindetrend_+1sig" not in self.models_out["cheops"]):
            self.make_cheops_timeseries(**kwargs)

        if self.phi_model_type=="individual":
            ixs=[self.models_out["cheops"]["filekey"]==fk for fk in self.cheops_filekeys]
        elif "split" in self.phi_model_type:
            ixs=[self.models_out["cheops"]["n_phi_model"]==n for n in np.unique(self.models_out["cheops"]["n_phi_model"].values)]
        elif self.phi_model_type=="common":
            ixs=[np.tile(True,len(self.models_out["cheops"]["time"]))]

        modname="gp" if self.fit_phi_gp else "spline"
        assert "cheops_pred_"+modname+"_med" in self.models_out["cheops"], "Must have "+modname+" in saved timeseries."

        plt.figure()
        for n_ix in range(len(ixs)):
            plt.subplot(1,len(ixs),n_ix+1)
            yoffset=5*np.std(self.models_out["cheops"]['flux'].values[ixs[n_ix]]-(self.models_out["cheops"]['cheops_lindetrend_med'].values[ixs[n_ix]]+self.models_out["cheops"]['cheops_allplmodel_med'].values[ixs[n_ix]]))
            unq_fks=np.unique(self.models_out["cheops"]["filekey"].values[ixs[n_ix]])
            for ifk,fk in enumerate(unq_fks):
                fk_ix=ixs[n_ix]&(self.models_out["cheops"]['filekey']==fk)
                phi=self.models_out['cheops'].loc[fk_ix,'phi']
                plt.plot(phi, yoffset*ifk+self.models_out['cheops'].loc[fk_ix,'flux']-self.models_out['cheops'].loc[fk_ix,'cheops_lindetrend_med']-self.models_out["cheops"].loc[fk_ix,'cheops_allplmodel_med'],
                        ".k",markersize=1.33,alpha=0.4)
                if "cheops_pred_"+modname+"_+1sig" in self.models_out['cheops']:
                    plt.fill_between(np.sort(phi),yoffset*ifk+self.models_out['cheops'].loc[fk_ix,"cheops_pred_"+modname+"_-2sig"].values[np.argsort(phi)],
                                    yoffset*ifk+self.models_out['cheops'].loc[fk_ix,"cheops_pred_"+modname+"_+2sig"].values[np.argsort(phi)],alpha=0.15,color='C'+str(int(ifk)))
                    plt.fill_between(np.sort(phi),yoffset*ifk+self.models_out['cheops'].loc[fk_ix,"cheops_pred_"+modname+"_-1sig"].values[np.argsort(phi)],
                                    yoffset*ifk+self.models_out['cheops'].loc[fk_ix,"cheops_pred_"+modname+"_+1sig"].values[np.argsort(phi)],alpha=0.15,color='C'+str(int(ifk)))
                plt.plot(np.sort(phi),yoffset*ifk+self.models_out['cheops'].loc[fk_ix,"cheops_pred_"+modname+"_med"].values[np.argsort(phi)],'-',alpha=0.45,linewidth=4,color='C'+str(int(ifk)))
            plt.xlabel("roll angle [deg]")
            if n_ix==0:
                plt.ylabel("Flux [ppt]")
            plt.ylim(-1*yoffset,(len(unq_fks))*yoffset)
        if save:
            save_suffix="" if save_suffix is None else save_suffix
            plt.savefig(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_rollangle"+modname+"_plots"+save_suffix+"."+savetype))
            

    def plot_cheops(self, save=True, savetype='png', tracename=None, fk='all', show_detrend=False,
                    ylim=None, dynamic_plot_resizing=True, save_suffix="", transparent=False, **kwargs):
        """Plot cheops lightcurves with model

        Args:
            save (bool, optional): Save the figure? Defaults to True.
            savetype (str, optional): What suffix for the plot to be saved to? Defaults to 'png'.
            input_trace (pymc trace, optional): Specify a trace. Defaults to None, which uses the saved full initialised model trace.
            fk (str, optional): Specify a Cheops filekey to plot, otherwise all available are plotted
            show_detrend (bool, optional): Whether to show both the pre-detrending flux and detrending model and the detrended transit+flux. Default is False
            transtype (str, optional): What type of transit prior was used. Must be one of 'set', 'loose', or 'none' (Defaults to 'set', i.e. constraining prior)
            ylim (tuple, optional): Manually set the ylim across all plots
            dynamic_plot_resizing (bool, optional): Whether to resize plots based on observing duration. Default: True
            save_suffix (str, optional): Add suffix when saving. Default is blank.
            transparent (bool, optional): Whether to save pngs with a transparent background
        """
        self.logger.debug(tracename)
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
        import seaborn as sns
        sns.set_palette("Paired")
        #sns.set_palette("rocket")#, as_cmap=True)

        #plt.plot(cheops_x, ,'.')
        plt.figure(figsize=(6+len(self.cheops_filekeys)*4/3,4))

        if tracename is not None:
            assert fk!="all", "to plot only loose or no transit models, need to only plot individual filekeys (i.e. set \'fk=PR...\')"
            save_suffix+="_"+np.array([f for f in ['fixtrans','loosetrans','notrans'] if f in tracename])[0]
            if not hasattr(self,'models_out') or tracename not in self.models_out:
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
        maxmins={'detrend_max':[],'detrend_min':[],'resid_max':[],'resid_min':[]}
        for n,fk in enumerate(fkloop):
            fk_ix=self.models_out[tracename]['filekey']==fk
            if hasattr(self.models_out,tracename+"_gap_models_out"):
                fk_gap_ix=self.models_out[tracename+"_gap_models_out"]['filekey']==fk

            yoffset=3*np.std(self.models_out[tracename].loc[fk_ix,'flux']-(self.models_out[tracename].loc[fk_ix,'cheops_alldetrend_med']+self.models_out[tracename].loc[fk_ix,'cheops_allplmodel_med']))
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
                         self.models_out[tracename].loc[fk_ix,"flux"], '.k',markersize=2.5,alpha=raw_alpha,zorder=1)
                binlc = bin_lc_segment(np.column_stack((self.models_out[tracename].loc[fk_ix,"time"],
                                                        self.models_out[tracename].loc[fk_ix,"flux"],
                                                        self.models_out[tracename].loc[fk_ix,"flux_err"])),1/120)
                plt.errorbar(binlc[:,0], binlc[:,1], yerr=binlc[:,2], fmt='.',color='C3',markersize=5,zorder=2,alpha=0.75)

                plt.plot(self.models_out[tracename].loc[fk_ix,"time"], yoffset+self.models_out[tracename].loc[fk_ix,"cheops_alldetrend_med"],'.',
                         markersize=2.5,c='C1',alpha=raw_alpha,zorder=5)
                if "cheops_alldetrend_+1sig" in self.models_out[tracename]:
                    plt.fill_between(self.models_out[tracename].loc[fk_ix,"time"], yoffset+self.models_out[tracename].loc[fk_ix,"cheops_alldetrend_-2sig"],
                                     yoffset+self.models_out[tracename].loc[fk_ix,"cheops_alldetrend_+2sig"],color='C0',alpha=0.15,zorder=3)
                    plt.fill_between(self.models_out[tracename].loc[fk_ix,"time"], yoffset+self.models_out[tracename].loc[fk_ix,"cheops_alldetrend_-1sig"],
                                     yoffset+self.models_out[tracename].loc[fk_ix,"cheops_alldetrend_+1sig"],color='C0',alpha=0.15,zorder=4)
                lims=np.nanpercentile(binlc[:,1],[1,99])
                maxmins['detrend_min']+=[lims[0]-0.66*yoffset]
                maxmins['detrend_max']+=[lims[1]+1.5*yoffset]

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
                     self.models_out[tracename].loc[fk_ix,"flux"]-self.models_out[tracename].loc[fk_ix,"cheops_alldetrend_med"],
                    '.k',alpha=raw_alpha,markersize=2.5,zorder=1)
            binlc = bin_lc_segment(np.column_stack((self.models_out[tracename].loc[fk_ix,'time'], 
                                                    self.models_out[tracename].loc[fk_ix,'flux']-self.models_out[tracename].loc[fk_ix,"cheops_alldetrend_med"],
                                                    self.models_out[tracename].loc[fk_ix,'flux_err'])),1/120)
            plt.errorbar(binlc[:,0], binlc[:,1], yerr=binlc[:,2], c='C3', fmt='.',markersize=5, zorder=2, alpha=0.8)
            if n==0:
                plt.ylabel("flux [ppt]")
            modtimes={}
            modfluxs={}
            modflux1sigs={}
            modflux2sigs={}
            npl=0#In case there are no planets
            for npl,pl in enumerate(self.planets):
                if np.any(self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_med"]<-1e-5):
                    #print(np.shape(pl_time),self.chplmod[fk][pl][2].shape,as_pl_time.shape)
                    if hasattr(self.models_out,tracename+"_gap_models_out") and "cheops_"+pl+"model_med" in self.models_out[tracename+"_gap_models_out"]:
                        modtimes[pl]=np.hstack([self.models_out[tracename].loc[fk_ix,"time"].values,self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"time"].values])
                        modfluxs[pl]=np.hstack([self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_med"].values,self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"cheops_"+pl+"model_med"].values])
                        modfluxs[pl]=modfluxs[pl][np.argsort(modtimes[pl])]
                    else:
                        modtimes[pl]=self.models_out[tracename].loc[fk_ix,"time"].values
                        modfluxs[pl]=self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_med"].values
                    #print(modtimes[pl],type(modtimes[pl]),len(modtimes[pl]),modfluxs[pl],type(modfluxs[pl]),len(modfluxs[pl]))
                    plt.plot(np.sort(modtimes[pl]),modfluxs[pl],'--', c='C'+str(5+2*npl), linewidth=3, alpha=0.6, zorder=10)
                    if "cheops_"+pl+"model_+1sig" in self.models_out[tracename]:
                        if hasattr(self.models_out,tracename+"_gap_models_out") and "cheops_"+pl+"model_-2sig" in self.models_out[tracename+"_gap_models_out"]:
                            modflux2sigs[pl]=[np.hstack([self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_-2sig"].values,self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"cheops_"+pl+"model_-2sig"].values])[np.argsort(modtimes[pl])],
                                         np.hstack([self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_+2sig"].values,self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"cheops_"+pl+"model_+2sig"].values])[np.argsort(modtimes[pl])]]
                            modflux1sigs[pl]=[np.hstack([self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_-1sig"].values,self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"cheops_"+pl+"model_-1sig"].values])[np.argsort(modtimes[pl])],
                                         np.hstack([self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_+1sig"].values,self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"cheops_"+pl+"model_+1sig"].values])[np.argsort(modtimes[pl])]]
                        else:
                            modflux2sigs[pl]=[self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_-2sig"].values,self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_+2sig"].values]
                            modflux1sigs[pl]=[self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_-1sig"].values,self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_+1sig"].values]
    
                        plt.fill_between(modtimes[pl],modflux2sigs[pl][0],modflux2sigs[pl][1],color='C'+str(4+2*npl), alpha=0.15, zorder=6)
                        plt.fill_between(modtimes[pl],modflux1sigs[pl][0],modflux1sigs[pl][1],color='C'+str(4+2*npl), alpha=0.15, zorder=7)
            if np.all(self.models_out[tracename].loc[fk_ix,"cheops_allplmodel_med"]>-1e-5):
                #Plotting a flat line only if none of the planets have any transits for this fk:
                plt.plot(self.models_out[tracename].loc[fk_ix,'time'], np.zeros(np.sum(fk_ix)), '--',c='C'+str(4+2*npl),linewidth=3,alpha=0.6,zorder=10)

            if np.sum([np.any(self.models_out[tracename].loc[fk_ix,"cheops_"+pl+"model_med"]<-1e-5) for pl in self.planets])>1:
                #Multiple transits together - we need a summed model
                if hasattr(self.models_out,tracename+"_gap_models_out"):
                    modtimes['all']=np.hstack([self.models_out[tracename].loc[fk_ix,"time"].values,self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"time"].values])
                    modfluxs['all']=np.hstack([self.models_out[tracename].loc[fk_ix,"cheops_allplmodel_med"].values,self.models_out[tracename+"_gap_models_out"].loc[fk_gap_ix,"cheops_allplmodel_med"].values])
                else:
                    modtimes['all']=self.models_out[tracename].loc[fk_ix,"time"].values
                    modfluxs['all']=self.models_out[tracename].loc[fk_ix,"cheops_allplmodel_med"].values
                pl='all'
                #print(modtimes[pl],type(modtimes[pl]),len(modtimes[pl]),modfluxs[pl],type(modfluxs[pl]),len(modfluxs[pl]))
                modfluxs['all']=np.array(modfluxs['all'])[np.argsort(modtimes['all'])]
                modtimes['all']=np.sort(modtimes['all'])

                plt.plot(modtimes['all'], modfluxs['all'], '--', linewidth=1.4, alpha=1, zorder=10, color='C9')
            maxmins['resid_min']+=[np.nanmin(self.models_out[tracename].loc[fk_ix,"cheops_allplmodel_med"])-yoffset*0.33]
            maxmins['resid_max']+=[yoffset*0.33]

            plt.xlabel("Time [BJD]")
            if n>0:
                plt.gca().set_yticklabels([])
            else:
                plt.gca().set_ylabel("Flux [ppt]")
            
            plt.gca().set_xticks(np.arange(np.ceil(np.nanmin(self.models_out[tracename].loc[fk_ix,"time"])*spacing)/spacing,np.max(self.models_out[tracename].loc[fk_ix,"time"]),1/spacing))
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        for n,fk in enumerate(fkloop):
            if show_detrend:
                
                if ylim is None:
                    plt.subplot(2, len(fkloop), 1+n)
                    plt.ylim(np.max(maxmins['detrend_min']),np.min(maxmins['detrend_max']))
                    plt.subplot(2, len(fkloop), 1+len(fkloop)+n)
                    plt.ylim(np.min(maxmins['resid_min']),
                    np.max(maxmins['resid_max']))
                else:
                    plt.subplot(2, len(fkloop), 1+n)
                    plt.ylim(ylim)
                    plt.subplot(2, len(fkloop), 1+len(fkloop)+n)
                    plt.ylim(ylim)
            else:
                plt.subplot(1,len(fkloop),1+n)
                if ylim is None:
                    plt.ylim(np.min(maxmins['resid_min']),
                            np.max(maxmins['resid_max']))
                else:
                    plt.ylim(ylim)

        plt.subplots_adjust(wspace=0.05,hspace=0.05)
        if save:
            save_suffix="" if save_suffix is None else save_suffix
            plt.savefig(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+save_suffix+"_cheops_plots"+save_suffix+"."+savetype),transparent=transparent)
    
    def plot_cheops_absphot(self,save=True,savetype='png',save_suffix=None,split_by_sects=True,ylim=None):
        plt.clf()
        #Plotting absolute photometry
        self.lcs['cheops']['raw_flux_medium_offset'] = np.zeros(len(self.lcs['cheops']))
        for fk in self.cheops_filekeys:
            self.lcs['cheops'].loc[self.lcs['cheops']['filekey']==fk,'raw_flux_medium_offset']=1000*(np.nanmedian(self.lcs['cheops'].loc[self.lcs['cheops']['filekey']==fk,'raw_flux']/np.nanmedian(self.lcs['cheops']['raw_flux'])-1))
        
        #Looking for centroid jumps and making correction vector using this
        centroid_jump_ix = find_jumps(np.column_stack((self.lcs['cheops']['centroidx'], self.lcs['cheops']['centroidy'])))

        n_centroid_jumps = np.nanmax(centroid_jump_ix)
        self.lcs['cheops']['raw_flux_medium_offset_centroidcorr'] = np.zeros(len(self.lcs['cheops']))
        centroid_jump_locs=[]
        for n_cent in range(1+np.nanmax(centroid_jump_ix)):
            if n_cent>0:
                prev_jump_end   = np.max(self.lcs['cheops'].loc[centroid_jump_ix==n_cent-1,'time'])
                this_jump_start = np.min(self.lcs['cheops'].loc[centroid_jump_ix==n_cent,'time'])
                centroid_jump_locs += [0.5*(prev_jump_end+this_jump_start)]
            self.lcs['cheops'].loc[centroid_jump_ix==n_cent, 'raw_flux_medium_offset_centroidcorr'] = np.nanmedian(self.lcs['cheops'].loc[centroid_jump_ix==n_cent, 'raw_flux_medium_offset'])
        from matplotlib.colors import LinearSegmentedColormap,to_rgb
        cm = LinearSegmentedColormap.from_list('dark_red_purple_blue', [to_rgb("#1c718e"),to_rgb("#173a87"),to_rgb("#4c0b5b"),to_rgb("#881353"),to_rgb("#8a2b10")], N=10)

        self.make_cheops_timeseries(overwrite=True)
        assert 'raw_flux_medium_offset' and 'raw_flux_medium_offset_centroidcorr' in self.models_out['cheops'].columns, "Make Cheops Timeseries did not place raw_flux_medium_offset into out DF..."
        if split_by_sects:
            sectinfo = self.init_phot_plot_sects_noprior('cheops', n_gaps=2, typic_dist=100, min_gap_thresh=0.6)
        else:
            sectinfo={'all':{'start':np.min(self.lcs['cheops']['time'])-0.1,'end':0.1+np.max(self.lcs['cheops']['time'])}}
        for ns,sectname in enumerate(sectinfo):
            plt.subplot(len(sectinfo),1,ns+1)
            sectinfo[sectname]['ix'] = (self.models_out['cheops']['time'].values>=sectinfo[sectname]['start'])&(self.models_out['cheops']['time'].values<=sectinfo[sectname]['end'])
            
            #Plotting flux
            detflux=self.models_out['cheops'].loc[sectinfo[sectname]['ix'], 'flux'] - \
                    self.models_out['cheops'].loc[sectinfo[sectname]['ix'], 'cheops_alldetrend_med'] + \
                    self.models_out['cheops'].loc[sectinfo[sectname]['ix'], 'raw_flux_medium_offset'] - \
                    self.models_out['cheops'].loc[sectinfo[sectname]['ix'], 'raw_flux_medium_offset_centroidcorr']

            plt.scatter(self.models_out['cheops'].loc[sectinfo[sectname]['ix'],'time'], detflux, 
                        s=1.0, c=self.models_out['cheops'].loc[sectinfo[sectname]['ix'], 'raw_flux_medium_offset_centroidcorr'],
                        alpha=0.4,zorder=1,cmap=cm)
            binsect = bin_lc_segment(np.column_stack((self.models_out['cheops'].loc[sectinfo[sectname]['ix'],'time'], detflux,
                                                      self.models_out['cheops'].loc[sectinfo[sectname]['ix'],'flux_err'])), 1/48)
            plt.errorbar(binsect[:,0],binsect[:,1],yerr=binsect[:,2],fmt='.',color="C1",ecolor="C0",alpha=0.6,zorder=2)

            plt.xlim(sectinfo[sectname]['start']-1,sectinfo[sectname]['end']+1)
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
            if ylim is None:
                ylim=(np.min(binsect[:,1])-0.25*np.nanstd(binsect[:,1]),
                        np.max(binsect[:,1])+0.25*np.nanstd(binsect[:,1]))
            plt.ylim(ylim)

            if ns==len(sectinfo)-1:
                plt.xlabel("BJD")

            for j in centroid_jump_locs:
                if (j>sectinfo[sectname]['start']-1)&(j<sectinfo[sectname]['end']+1):
                    plt.plot((j,j), ylim, "--b", label="centroid jump", alpha=0.5)
            plt.ylabel("Relative Flux [ppt]")
        if len(centroid_jump_locs)>0:
            plt.legend()
        if save:
            save_suffix="" if save_suffix is None else save_suffix
            plt.savefig(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_absphotcheops_plot"+save_suffix+"."+savetype))


    def init_phot_plot_sects_noprior(self, src, n_gaps=None, typic_dist=None, min_gap_thresh=0.02):
        """
        Initialising photometric plot without any prior knowledge of the timing of gaps (i.e. not using pre-loaded TESS sector times).
        Args:
        - src (str) - 
        - n_gaps -
        - typic_dist - typical distance between gaps. Using TESS
        - min_gap_thresh - Threshold in fractional gap length above which to cut (e.g. 0.01 = only cut gaps bigger than 1% of time)
        """
        
        if typic_dist is None:
            typic_dist=75 if src.lower()=='k2' else 15 #Assuming TESS
        if n_gaps is None:
            n_gaps=1 if src.lower()=='k2' else 5 #Assuming TESS
        sort_time=np.sort(self.lcs[src].loc[np.isfinite(self.lcs[src]['time'].values),'time'].values)
        diffs=np.diff(sort_time)#Normalising to length of time
        lower_bounds=[np.nanmin(self.lcs[src]['time'])]
        upper_bounds=[np.nanmax(self.lcs[src]['time'])]
        n_jump=0
        diffs_mask=np.ones(len(diffs))
        
        while n_jump<n_gaps:
            if np.max(diffs/np.sum(diffs[diffs_mask==1]))<min_gap_thresh:
                #The largest gap, when normalised to total gap lengths (ignoring already-removed gaps) is below threshold - stopping
                break

            #Adding a weighting such that gaps far from the known gaps are weighted higher
            diffs_prior=np.clip(1-1/np.min(sort_time[:-1][None,:]-np.array(lower_bounds)[:,None],axis=0)**(8/typic_dist),0,1)*np.clip(1-1/np.min(np.array(upper_bounds)[:,None]-sort_time[1:][None,:],axis=0)**(8/typic_dist),0,1)
            #(typic_dist - 1/(np.min(sort_time[:-1][None,:]-np.array(lower_bounds)[:,None],axis=0)+1/typic_dist))*(typic_dist - 1/(np.min(np.array(upper_bounds)[:,None]-sort_time[1:][None,:],axis=0)+1/typic_dist))/(typic_dist**2)
            new_jump_ix=np.argmax(diffs*diffs_prior*diffs_mask)
            lower_bounds+=[sort_time[1+new_jump_ix]]
            upper_bounds+=[sort_time[new_jump_ix]]
            diffs_mask[new_jump_ix]=0
                
            n_jump+=1
        sectinfo={}
        for nb in range(len(lower_bounds)):
            sectinfo[nb]={'start':np.sort(lower_bounds)[nb]-0.005,
                          'end':np.sort(upper_bounds)[nb]+0.005}
            sectinfo[nb]['dur']=sectinfo[nb]['end']-sectinfo[nb]['start']
            sectinfo[nb]['data_ix']=(self.lcs[src]['time'].values>=sectinfo[nb]['start'])&(self.lcs[src]['time'].values<=sectinfo[nb]['end'])

        self.logger.debug(sectinfo)
        return sectinfo

    
    def init_phot_plot_sects(self,src,**kwargs):
        """
        Initialise the plotting of photometry

        Args:
        src (str) - Source of photometry, e.g. 'tess' or 'k2'. 
        """
        #Getting rough timings for all fields
        from . import tools
        field_df=tools.get_field_times()
        field_df['jd_mid']=0.5*(field_df['jd_start']+field_df['jd_end'])
        field_df['dur']=field_df['jd_end']-field_df['jd_start']
        #Cutting all observations long before.after initial data
        #field_df=field_df.loc[(field_df['jd_end']>(np.min(self.lcs[src]['time'])-5))&(field_df['jd_start']<(np.max(self.lcs[src]['time'])+5))]
        self.logger.debug([field_df.shape,np.sum(np.min(abs(field_df['jd_mid'][None,:]-self.lcs[src]['time'].values[10::20][:,None]),axis=0)<(0.55*field_df['dur']))])
        field_df=field_df.loc[np.min(abs(field_df['jd_mid'][None,:]-self.lcs[src]['time'].values[10::20][:,None]),axis=0)<(0.55*field_df['dur'])]
        
        #getting true timings for jumps in photometric data:
        sort_time=np.sort(self.lcs[src].loc[np.isfinite(self.lcs[src]['time'].values),'time'].values)
        diffs=np.diff(sort_time)
        start_bool=np.hstack((True,diffs>0.33))
        end_bool=np.hstack((diffs>0.33,True))
        ends=sort_time[end_bool]
        starts=sort_time[start_bool]

        #correlating jumps to start/stops in field times
        sectinfo={}
        self.logger.debug(field_df)
        for frow in field_df.iterrows():
            #Counts as matching if 1) there is data between start & stop, and 2) there are jumps within ~1d of start/stop
            has_data = np.sum((self.lcs[src]['time']>frow[1]['jd_start'])&(self.lcs[src]['time']<frow[1]['jd_end']))>500
            end_search=abs(frow[1]['jd_end']-ends)
            end_match = np.min(end_search)<6
            start_search=abs(frow[1]['jd_start']-starts)
            start_match = np.min(start_search)<6
            self.logger.debug([frow[1]['jd_start'],frow[1]['jd_end'],ends,starts,np.min(end_search),np.min(start_search)])
            if has_data&end_match&start_match:
                #SECTOR MATCH
                sectinfo[frow[1]['field_string']]={'start':starts[np.argmin(start_search)]-0.05,
                                                   'end':ends[np.argmin(end_search)]+0.05,
                                                   'dur':ends[np.argmin(end_search)]+0.05}
                
                sectinfo[frow[1]['field_string']]['data_ix']=(self.lcs[src]['time'].values>=sectinfo[frow[1]['field_string']]['start'])&(self.lcs[src]['time'].values<=sectinfo[frow[1]['field_string']]['end'])
        #sectinfo=self.init_phot_plot_sects_noprior(src,n_gaps=len(sectinfo))
        return sectinfo

    def init_phot_plot(self, src, overwrite=False, **kwargs):
        """
        #Initialising photometric plot info - i.e. flux and planet models from either 
        #We want per-sector arrays which match time[mask]
        """
        #print(self.models_out.keys(),not hasattr(self,"models_out"), not (src in self.models_out), hasattr(self,'trace'), src+"_allplmodel_+1sig" not in self.models_out[src].columns, overwrite)
        if not hasattr(self,"models_out") or not (src in self.models_out) or (hasattr(self,'trace') and src+"_allplmodel_+1sig" not in self.models_out[src].columns) or overwrite:
            #Either no saved timeseries at all, or no specific timeseries for this source, or we now have a trace but the saved timeseries have not been updated (no +/-1 sigma regions)
            self.make_lcs_timeseries(src,**kwargs)

        if not hasattr(self,"phot_plot_info"):
            self.phot_plot_info={}
        if src not in self.phot_plot_info or overwrite:
            self.phot_plot_info[src]={}
            if src=='tess':
                self.phot_plot_info[src]['sectinfo'] = self.init_phot_plot_sects(src)
            else:
                self.phot_plot_info[src]['sectinfo'] = self.init_phot_plot_sects_noprior(src)
        for ns in self.phot_plot_info[src]['sectinfo']:
            self.phot_plot_info[src]['sectinfo'][ns]['ix'] = (self.models_out[src]['time'].values>=self.phot_plot_info[src]['sectinfo'][ns]['start'])&(self.models_out[src]['time']<=self.phot_plot_info[src]['sectinfo'][ns]['end'])
        logging.debug(self.phot_plot_info[src])
        self.phot_plot_info[src]['transmin']=np.min(np.hstack([self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_med'].values for ns in self.phot_plot_info[src]['sectinfo']]))
        self.phot_plot_info[src]['stdev']=np.nanmedian([np.nanstd(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'flux'].values) for ns in self.phot_plot_info[src]['sectinfo']])
        self.phot_plot_info[src]['flat_stdev']=np.nanmedian([np.nanstd(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],'flux'].values - \
                                                                       (self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_allplmodel_med'].values + \
                                                                        self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['ix'],src+'_gpmodel_med'].values)) for ns in self.phot_plot_info[src]['sectinfo']]
                                                            )

    def plot_phot(self, src='tess', save=True, savetype='png', plot_flat=False, plot_both=False,save_suffix=None,**kwargs):
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
        for ns,sectname in enumerate(self.phot_plot_info[src]['sectinfo']):
            if not plot_flat or plot_both:
                if plot_both:
                    plt.subplot(2,len(self.phot_plot_info[src]['sectinfo']),2*ns)
                else:
                    plt.subplot(len(self.phot_plot_info[src]['sectinfo']),1,ns+1)
                #Plotting flux
                plt.plot(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                         self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'flux'],
                         '.k',markersize=1.0,alpha=0.4,zorder=1)
                binsect=bin_lc_segment(np.column_stack((self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                                                        self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'flux'],
                                                        self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'flux_err'])),1/48)
                plt.errorbar(binsect[:,0],binsect[:,1],yerr=binsect[:,2],fmt='.',color="C1",ecolor="C0",alpha=0.6,zorder=2)
                if src+'_gpmodel_+1sig' in self.models_out[src]: #_allplmodel_med,_gpmodel_med
                    #flux model regions
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_gpmodel_-2sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_gpmodel_+2sig'],
                                     color='C4',alpha=0.15,zorder=3)
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_gpmodel_-1sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_gpmodel_+1sig'],
                                     color='C4',alpha=0.15,zorder=4)
                if src+'_allplmodel_+1sig' in self.models_out[src]: #_allplmodel_med,_gpmodel_med
                    #planet + flux model regions
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_allplmodel_-2sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_allplmodel_+2sig'],
                                     color='C2',alpha=0.15,zorder=6)
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_allplmodel_-1sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_allplmodel_+1sig'],
                                     color='C2',alpha=0.15,zorder=7)
                plt.plot(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                         self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_gpmodel_med'],
                         linewidth=2,color='C5',alpha=0.75,zorder=5)
                plt.plot(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                         self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_gpmodel_med']+self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_allplmodel_med'],
                         linewidth=2,color='C3',alpha=0.75,zorder=8)

            if plot_flat:
                if plot_both:
                    plt.subplot(2,len(self.phot_plot_info[src]['sectinfo']),2*ns+1)
                else:
                    plt.subplot(len(self.phot_plot_info[src]['sectinfo']),1,ns+1)
                #Plotting flux (minus variability model)
                plt.plot(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                         self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'flux']-self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_gpmodel_med'],
                         '.k',markersize=1.0,alpha=0.4,zorder=1)
                binsect=bin_lc_segment(np.column_stack((self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                                                        self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'flux']-self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_gpmodel_med'],
                                                        self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'flux_err'])),1/48)
                plt.errorbar(binsect[:,0],binsect[:,1],yerr=binsect[:,2],fmt='.',color="C1",ecolor="C0",alpha=0.6,zorder=2)
                if src+'_allplmodel_+1sig' in self.models_out[src]: #_allplmodel_med,_gpmodel_med
                    #planet + flux model regions
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_allplmodel_-2sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_allplmodel_+2sig'],
                                     color='C2',alpha=0.15,zorder=6)
                    plt.fill_between(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_allplmodel_-1sig'],
                                     self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_allplmodel_+1sig'],
                                     color='C2',alpha=0.15,zorder=7)
                plt.plot(self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],'time'],
                         self.models_out[src].loc[self.phot_plot_info[src]['sectinfo'][sectname]['ix'],src+'_allplmodel_med'],
                         linewidth=2,color='C3',alpha=0.75,zorder=8)
            
            plt.xlim(self.phot_plot_info[src]['sectinfo'][sectname]['start']-1,self.phot_plot_info[src]['sectinfo'][sectname]['end']+1)
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
            #                                  np.hstack((0,np.percentile(self.trace.posterior['photgp_model_x'][:,self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']],self.percentiles[p],axis=0),0)))
            #                 bf_gp+=[interpp(self.lcs[src].loc[sect_ix,'time'].values)]
            #             #print("bin_oot",bf_gp[0].shape)
            #         elif not self.cut_oot:
            #             bf_gp = np.nanpercentile(self.trace.posterior['photgp_model_x'][:,self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']], list(self.percentiles.values()), axis=0)
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
            #         pl_mod[:,self.lcs[src].loc[sect_ix&self.lcs[src]['mask'],'near_trans']]=np.nanpercentile(self.trace.posterior[src+'_summodel_x'][:,self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']], list(self.percentiles.values()), axis=0)
            #     elif self.bin_oot:
            #         pl_mod=[]
            #         for p in self.percentiles:
            #             interpp=interp1d(np.hstack((self.lcs[src].loc[sect_ix,'time'].values[0]-0.1,self.lc_fit[src].loc[self.phot_plot_info[src]['sectinfo'][ns]['fit_ix'],'time'].values,self.lcs[src].loc[sect_ix,'time'].values[-1]+0.1)),
            #                             np.hstack((0,np.percentile(self.trace.posterior[src+'_summodel_x'][:,self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']],self.percentiles[p],axis=0),0)))
            #             pl_mod+=[interpp(self.lcs[src].loc[sect_ix,'time'].values)]

            #     else:
            #         pl_mod = np.nanpercentile(self.trace.posterior[src+'_summodel_x'][:,self.phot_plot_info[src]['sectinfo'][ns]['fit_ix']], list(self.percentiles.values()), axis=0)
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
            save_suffix="" if save_suffix is None else save_suffix
            plt.savefig(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_"+src+"_plot"+save_suffix+"."+savetype))
    
    def compute_ttvs(self, input_trace, pls_with_ttvs):
        exp_times={pl:[] for pl in pls_with_ttvs}
        ttvs={pl:[] for pl in pls_with_ttvs}
        times={pl:[] for pl in pls_with_ttvs}
        for npl,pl in enumerate(pls_with_ttvs):
            if type(input_trace)==pm.backends.base.MultiTrace:
                new_p=np.nanmedian(input_trace['P_'+pl])
                new_t0=np.nanmedian(input_trace['t0_'+pl])
            elif type(input_trace)==dict:
                new_p=input_trace['P_'+pl]
                new_t0=input_trace['t0_'+pl]
            elif type(input_trace)==az.InferenceData:
                new_p=input_trace.posterior['P_'+pl].values
                new_t0=input_trace.posterior['t0_'+pl].values
            for n in range(self.planets[pl]['n_trans']):
                if type(input_trace)==pm.backends.base.MultiTrace:
                    exp_times[pl]+=[new_t0+new_p*self.planets[pl]['init_transit_inds'][n]]
                    times[pl]+=[np.nanmedian(input_trace['transit_times_'+pl+'_'+str(n)])]
                    ttvs[pl]+=[np.nanpercentile(1440*(input_trace['transit_times_'+pl+'_'+str(n)] - exp_times[pl][-1]),list(self.percentiles.values())[1:-1])]
                elif type(input_trace)==dict:
                    exp_times[pl]=[new_t0+new_p*self.planets[pl]['init_transit_inds'][n]]
                    times[pl]=[input_trace['transit_times_'+pl+'_'+str(n)]]
                    ttvs[pl]=[1440*input_trace['transit_times_'+pl+'_'+str(n)] - (new_t0+new_p*self.planets[pl]['init_transit_inds'])]
                elif type(input_trace)==az.InferenceData:
                    exp_times[pl]=[new_t0+new_p*self.planets[pl]['init_transit_inds'][n]]
                    times[pl]=[input_trace.posterior['transit_times_'+pl+'_'+str(n)].values]
                    ttvs[pl]=[1440*input_trace.posterior['transit_times_'+pl+'_'+str(n)].values - (new_t0+new_p*self.planets[pl]['init_transit_inds'])]

                #print(out[1,i],"±",0.5*(out[2,i]-out[0,i]))
            ttvs[pl]=np.vstack((ttvs[pl]))
        return ttvs,times

    def plot_ttvs(self, save=True, nplots=2, savetype='png', trace="ttv_trace", ylim=None, save_suffix=None):
        """Plot TTVs
        Args:
            save (bool, optional): Whether to save the plot. Defaults to True.
            nplots (int, optional): Number of seperate plots. Defaults to the length of the planets.
            savetype (str, optional): Type of image to save. Defaults to 'png'.
            trace (str, optional): Which trace to use. Either "trace", "ttv_trace" or "both
        """
        assert self.fit_ttvs or hasattr(self,'ttv_trace'), "Must have fitted TTVs timing values using `fit_ttvs` flag"
        assert not (trace=='ttv' and not hasattr(self,'ttv_trace')), "Must have called `run_slim_ttv_model` if plotting with `use_ttv_trace`"

        plt.figure(figsize=(5.5,9))
        pls_with_ttvs=[pl for pl in self.planets if self.planets[pl]['n_trans']>2]
        #nplots=len(pls_with_ttvs) if nplots is None else nplots
        self.ttvs_to_plot={pl:[] for pl in pls_with_ttvs}
        self.times_to_plot={pl:[] for pl in pls_with_ttvs}
        if (trace=='both')&hasattr(self,'ttv_trace')&hasattr(self,'trace'):
            trace_loop=[self.ttv_trace,self.trace] 
        elif (trace=='ttv_trace')&hasattr(self,'ttv_trace'):
            trace_loop=[self.ttv_trace] 
        elif (trace=='trace')&hasattr(self,'trace'):
            trace_loop=[self.trace] 
        else:
            trace_loop=[self.init_soln] 
        markers='o*'
        for ntr,itrace in enumerate(trace_loop):
            ttvs, times=self.compute_ttvs(itrace,pls_with_ttvs)
            for n_plt in range(nplots):
                plt.subplot(nplots,1,n_plt+1)
                for ipl,pl in enumerate(pls_with_ttvs):
                    if hasattr(self,'trace'):
                        plt.plot(times[pl], ttvs[pl][:,1],alpha=0.6,color='C'+str(2+ipl),zorder=1)
                        plt.errorbar(times[pl], ttvs[pl][:,1],yerr=[ttvs[pl][:,1]-ttvs[pl][:,0],ttvs[pl][:,2]-ttvs[pl][:,1]],
                                    fmt=markers[ntr],label=str(pl),alpha=0.6,markersize=8,color='C'+str(2+ipl),zorder=2)
                        
                    else:
                        #print(ipl,pl,np.array(ttvs[pl]))
                        plt.plot(times[pl],np.array(ttvs[pl]),'.-',label=str(pl),alpha=0.6,markersize=15,markerstyle=markers[ntr])
            
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
            if ylim is not None:
                plt.ylim(ylim)
            if i==nplots-1:
                plt.legend()
        if save:
            save_suffix="" if save_suffix is None else save_suffix
            plt.savefig(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_ttv_plot"+save_suffix+"."+savetype))

    def plot_rvs(self,save=True,savetype='png',save_suffix=None):
        assert hasattr(self,'rvs'), "No RVs found..."
        plt.figure(figsize=(8,6))
        plt.subplot(2,1,1)
        if hasattr(self,'trace'):
            rvt_mods=np.nanpercentile(self.trace.posterior['rv_model_t'].values,list(self.percentiles.values()),axis=0)
            plt.fill_between(self.rv_t,rvt_mods[0],rvt_mods[4],color='C4',alpha=0.15)
            plt.fill_between(self.rv_t,rvt_mods[1],rvt_mods[3],color='C4',alpha=0.15)
            plt.plot(self.rv_t,rvt_mods[2],c='C4',alpha=0.66)
            if self.npoly_rv>1:
                plt.plot(self.rv_t,np.nanmedian(self.trace.posterior['bkg_t'].values,axis=0),c='C2',alpha=0.3,linewidth=3)
            
        else:
            plt.plot(self.rv_t,self.init_soln['rv_model_t'],c='C4',alpha=0.66)
            if self.npoly_rv>1:
                plt.plot(self.rv_t,self.init_soln['bkg_t'],'--',c='C2',alpha=0.3,linewidth=3)
        #plt.plot(rv_t,np.nanmedian(trace_2['vrad_t'][:,:,0],axis=0),':')
        #plt.plot(rv_t,np.nanmedian(trace_2['vrad_t'][:,:,1],axis=0),':')
        
        #plt.fill_between(rv_t,rv_prcnt[:,0],rv_prcnt[:,2],color='C1',alpha=0.05)
        
        for isc,sc in enumerate(pd.unique(self.rvs['scope'])):
            plt.errorbar(self.rvs.loc[self.rvs['scope']==sc,'time'],self.rvs.loc[self.rvs['scope']==sc,'y']-self.init_soln['rv_offsets'][isc],
                         yerr=self.rvs.loc[self.rvs['scope']==sc,'yerr'],
                         fmt='.',ecolor='#aaaaaa',zorder=10,label=sc)
        plt.legend()
        plt.ylabel("RV [ms]")
        #plt.xlim(1850,1870)

        minmax=[0,0]
        
        for n,pl in enumerate(self.planets):
            plt.subplot(2,len(self.planets),len(self.planets)+1+n)
            t0 = self.init_soln['t0_'+pl] if not hasattr(self,'trace') else np.nanmedian(self.trace.posterior['t0_'+pl].values)
            p = self.init_soln['P_'+pl] if not hasattr(self,'trace') else np.nanmedian(self.trace.posterior['P_'+pl].values)
            rv_phase_x = (self.rvs['time']-t0-0.5*p)%p-0.5*p
            rv_phase_t = (self.rv_t-t0-0.5*p)%p-0.5*p
            if not hasattr(self,'trace'):
                other_pls_bg=self.init_soln['bkg_x']+np.sum([self.init_soln['vrad_x_'+inpl] for inpl in self.planets if inpl!=pl],axis=0)
                for isc,sc in enumerate(pd.unique(self.rvs['scope'])):
                    plt.errorbar(rv_phase_x[self.rvs['scope'].values==sc],
                                self.rvs.loc[self.rvs['scope']==sc,'y'] - other_pls_bg[self.rvs['scope']==sc],
                                yerr=self.rvs.loc[self.rvs['scope']==sc,'yerr'],
                                fmt='.',ecolor='#aaaaaa',zorder=10,label=sc)
                plt.plot(np.sort(rv_phase_t),self.init_soln['vrad_t_'+pl][np.argsort(rv_phase_t)],c='C1')
                minmax[0]=np.min([minmax[0],np.min(self.init_soln['vrad_t_'+pl])*1.05,np.min(self.rvs.loc[:,'y'] - other_pls_bg)*1.05])
                minmax[1]=np.max([minmax[1],np.max(self.init_soln['vrad_t_'+pl])*1.05,np.max(self.rvs.loc[:,'y'] - other_pls_bg)*1.05])
            else:
                self.logger.debug(np.dstack([self.trace.posterior['vrad_x_'+inpl].values for inpl in self.planets if inpl!=pl]).shape)
                self.logger.debug(np.sum(np.dstack([self.trace.posterior['vrad_x_'+inpl].values for inpl in self.planets if inpl!=pl]),axis=2).shape)
                other_pls_bg=np.nanmedian(self.trace.posterior['bkg_x'].values+np.sum(np.dstack([self.trace.posterior['vrad_x_'+inpl].values for inpl in self.planets if inpl!=pl]),axis=2),axis=0)
                for isc,sc in enumerate(pd.unique(self.rvs['scope'])):
                    plt.errorbar(rv_phase_x[self.rvs['scope'].values==sc],
                                self.rvs.loc[self.rvs['scope']==sc,'y'] - other_pls_bg[self.rvs['scope']==sc],
                                yerr=self.rvs.loc[self.rvs['scope']==sc,'yerr'],
                                fmt='.',ecolor='#aaaaaa',zorder=10,label=sc)
                rvt_mods=np.nanpercentile(self.trace.posterior['vrad_t_'+pl].values[:,np.argsort(rv_phase_t)],list(self.percentiles.values()),axis=0)
                plt.fill_between(np.sort(rv_phase_t),rvt_mods[0],rvt_mods[4],color='C1',alpha=0.15)
                plt.fill_between(np.sort(rv_phase_t),rvt_mods[1],rvt_mods[3],color='C1',alpha=0.15)
                plt.plot(np.sort(rv_phase_t),rvt_mods[2],c='C1',alpha=0.65)
                minmax[0]=np.min([minmax[0],np.min(rvt_mods[0])*1.05,np.min(self.rvs.loc[:,'y'] - other_pls_bg)*1.05])
                minmax[1]=np.max([minmax[1],np.max(rvt_mods[4])*1.05,np.max(self.rvs.loc[:,'y'] - other_pls_bg)*1.05])

            if n==0:
                plt.ylabel("RV [ms]")
            else:
                plt.gca().set_yticklabels([])
            plt.xlabel("Time from t0 [d]")
        for n,pl in enumerate(self.planets):
            self.logger.debug(minmax)
            plt.subplot(2,len(self.planets),len(self.planets)+1+n)
            plt.ylim(minmax[0],minmax[1])
        if save:
            save_suffix="" if save_suffix is None else save_suffix
            plt.savefig(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_ttvs"+save_suffix+"."+savetype))

    def plot_transits_fold(self,save=True,savetype='png',xlim=None,show_legend=True,sigma_fill=2,yoffsets=None,overwrite=False,save_suffix=None,**kwargs):

        if not hasattr(self,"models_out") or overwrite:
            self.make_timeseries(overwrite=overwrite,**kwargs)

        import seaborn as sns
        sns.set_palette("Paired")
        plt.figure(figsize=(5,3+2*len(self.planets)**0.66))
        if xlim is None:
            self.logger.debug([np.nanmax([self.planets[pl]['tdur'] for pl in self.planets]),-1*2/(len(self.planets)**0.5)*np.nanmax([self.planets[pl]['tdur'] for pl in self.planets]),2/(len(self.planets)**0.5)*np.nanmax([self.planets[pl]['tdur'] for pl in self.planets])])
            xlim=(-1*2/(len(self.planets)**0.5)*np.nanmax([self.planets[pl]['tdur'] for pl in self.planets]),
                  2/(len(self.planets)**0.5)*np.nanmax([self.planets[pl]['tdur'] for pl in self.planets]))
        for npl,pl in enumerate(self.planets):
            plt.subplot(len(self.planets),1,1+npl)
            yoffset=0
            t0 = self.init_soln['t0_'+pl] if not hasattr(self,'trace') else np.nanmedian(self.trace.posterior['t0_'+pl].values)
            p = self.init_soln['P_'+pl] if not hasattr(self,'trace') else np.nanmedian(self.trace.posterior['P_'+pl].values)
            dep = 1e3*self.init_soln['ror_'+pl]**2 if not hasattr(self,'trace') else np.nanmedian(self.trace.posterior['ror_'+pl].values**2)
            nscope=0
            for scope in self.lc_fit:
                if self.fit_ttvs or self.split_periods is not None and self.planets[pl]['n_trans']>2:
                    #subtract nearest fitted transit time for each time value
                    trans_times= np.array([self.init_soln['transit_times_'+pl+'_'+str(n)] for n in range(self.planets[pl]['n_trans'])]) if not hasattr(self,'trace') else np.array([np.nanmedian(self.trace.posterior['transit_times_'+pl+'_'+str(n)], axis=0) for n in range(self.planets[pl]['n_trans'])])
                    nearest_times=np.argmin(abs(self.models_out[scope]['time'].values[:,None]-trans_times[None,:]),axis=1)
                    phase = self.models_out[scope]['time'].values - trans_times[nearest_times]
                    #(self.lc_fit[src].loc[self.lc_fit[scope]['near_trans'],'time']-t0-0.5*p)%p-0.5*p
                else:
                    phase = (self.models_out[scope]['time'].values-t0-0.5*p)%p-0.5*p
                    #(self.lc_fit[scope].loc[self.lc_fit[scope]['near_trans'],'time']-t0-0.5*p)%p-0.5*p

                ix = abs(phase)<(3*self.planets[pl]['tdur'])
                if np.sum(ix)>0:
                    phase=phase[ix]
                    n_pts=np.sum((phase<xlim[1])&(phase>xlim[0]))
                    raw_alpha=np.clip(6*(n_pts)**(-0.4),0.02,0.99)
                    #pl2mask=[list(info.keys())[n2] in range(3) if n2!=n]

                    #Need to also remove the influence of other planets here:
                    transmin = np.nanmin(self.models_out[scope].loc[ix,scope+'_'+pl+'model_med'])
                    if scope=='cheops':
                        plflux=self.models_out[scope].loc[ix,'flux'].values-self.models_out[scope].loc[ix,scope+"_alldetrend_med"].values-self.models_out[scope].loc[ix,scope+"_allplmodel_med"].values+self.models_out[scope].loc[ix,scope+"_"+pl+"model_med"].values
                    else:
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
                else:
                    #No detected transit, but we can use the derived depth/other flux STD to get us nice sized "hole"
                    transmin=dep
                    std=np.nanmedian(abs(np.diff(self.models_out[scope].loc[:,'flux'].values-self.models_out[scope].loc[:,scope+"_alldetrend_med"].values)))
                if yoffsets is None:
                    yoffset+= abs(transmin) + 3*std
                else:
                    yoffset+= yoffsets
                nscope+=1
            
            # if len(self.cheops_filekeys)>0:
            #     if yoffsets is None:
            #         yoffset+=abs(transmin)
            #     else:
            #         yoffset+= yoffsets
            #     if not self.fit_ttvs or self.planets[pl]['n_trans']<=2:
            #         chphase = (self.models_out['cheops']['time'].values-t0-0.5*p)%p-0.5*p
            #         #(self.lc_fit[scope].loc[self.lc_fit[scope]['near_trans'],'time']-t0-0.5*p)%p-0.5*p
            #     else:
            #         #subtract nearest fitted transit time for each time value
            #         trans_times= np.array([self.init_soln['transit_times_'+pl+'_'+str(n)] for n in range(self.planets[pl]['n_trans'])]) if not hasattr(self,'trace') else np.array([np.nanmedian(self.trace.posterior['transit_times_'+pl+'_'+str(n)], axis=0) for n in range(self.planets[pl]['n_trans'])])
            #         nearest_times=np.argmin(abs(self.models_out['cheops']['time'].values[:,None]-trans_times[None,:]),axis=1)
            #         chphase = self.models_out['cheops']['time'].values - trans_times[nearest_times]
            #         #(self.lc_fit[src].loc[self.lc_fit[scope]['near_trans'],'time']-t0-0.5*p)%p-0.5*p
            #     chix = abs(chphase)<(3*self.planets[pl]['tdur'])
            #     chphase=chphase[chix]
            #     ch_n_pts=np.sum((chphase<xlim[1])&(chphase>xlim[0]))
            #     ch_raw_alpha=np.clip(9*(ch_n_pts)**(-0.4),0.02,0.99)
            #     #print("alphas=",raw_alpha,ch_raw_alpha)
            #     cheopsally = self.models_out['cheops'].loc[chix,'flux'].values-self.models_out['cheops'].loc[chix,'cheops_alldetrend_med'].values-self.models_out['cheops'].loc[chix,'cheops_allplmodel_med'].values+self.models_out['cheops'].loc[chix,'cheops_'+pl+'model_med'].values
            #     plt.plot(chphase, yoffset+cheopsally,'.k',markersize=5,c='C'+str(nscope*2),alpha=ch_raw_alpha,zorder=5)
            #     binchelc=bin_lc_segment(np.column_stack((np.sort(chphase), cheopsally[np.argsort(chphase)],
            #                                                 self.models_out['cheops'].loc[chix,'flux_err'])),
            #                             self.planets[pl]['tdur']/8)
            #     plt.errorbar(binchelc[:,0],yoffset+binchelc[:,1],yerr=binchelc[:,2],fmt='.',markersize=8,alpha=0.8,ecolor='#ccc',zorder=6,color='C'+str(1+nscope*2),label="CHEOPS")
                
            #     chgapphase = (np.hstack([self.models_out['cheops']['time'],self.models_out['cheops_gap_models_out']['time']])-t0-0.5*p)%p-0.5*p
            #     chgapix    = abs(chgapphase)<(3*self.planets[pl]['tdur'])
            #     if "cheops_"+pl+"model_+1sig" in self.models_out[scope] and sigma_fill>0:
            #         if int(sigma_fill)>=2:
            #             modflux2sig=[np.hstack([self.models_out['cheops']['cheops_'+pl+'model_-2sig'],self.models_out['cheops_gap_models_out']['cheops_'+pl+'model_-2sig']]),
            #                          np.hstack([self.models_out['cheops']['cheops_'+pl+'model_+2sig'],self.models_out['cheops_gap_models_out']['cheops_'+pl+'model_+2sig']])]
            #             plt.fill_between(np.sort(chgapphase[chgapix]), yoffset+modflux2sig[0][chgapix][np.argsort(chgapphase[chgapix])],
            #                              yoffset+modflux2sig[1][chgapix][np.argsort(chgapphase[chgapix])],
            #                              zorder=6,alpha=0.15,color='C'+str(4+2*npl))
            #         modflux1sig=[np.hstack([self.models_out['cheops']['cheops_'+pl+'model_-1sig'],self.models_out['cheops_gap_models_out']['cheops_'+pl+'model_-1sig']]),
            #                      np.hstack([self.models_out['cheops']['cheops_'+pl+'model_+1sig'],self.models_out['cheops_gap_models_out']['cheops_'+pl+'model_+1sig']])]
            #         plt.fill_between(np.sort(chgapphase[chgapix]), yoffset+modflux1sig[0][chgapix][np.argsort(chgapphase[chgapix])],
            #                              yoffset+modflux1sig[1][chgapix][np.argsort(chgapphase[chgapix])],
            #                          zorder=7,alpha=0.15,color='C'+str(4+2*npl))
            #     modflux=np.hstack([self.models_out['cheops']['cheops_'+pl+'model_med'],self.models_out['cheops_gap_models_out']['cheops_'+pl+'model_med']])
            #     plt.plot(np.sort(chgapphase[chgapix]), yoffset+modflux[chgapix][np.argsort(chgapphase[chgapix])],
            #             '--',zorder=8,alpha=0.6,color='C'+str(5+2*npl),linewidth=2.5)
                
            #     yoffset+=5*np.nanmedian(abs(np.diff(self.models_out['cheops']['flux']-self.models_out['cheops']['cheops_alldetrend_med']-self.models_out['cheops']['cheops_allplmodel_med'])))
            
            plt.ylabel("Flux [ppt]")
            #print(npl,len(self.planets))
            if npl==len(self.planets)-1:
                plt.xlabel("Time from transit [d]")
            else:
                plt.gca().set_xticklabels([])
            self.logger.debug([transmin,std])
            if yoffsets is None:
                plt.ylim(transmin-3*std,yoffset-abs(transmin))
            else:
                plt.ylim(transmin-3*std,yoffset)
            plt.xlim(xlim)
        if show_legend:
            plt.legend(loc=4) 
        plt.subplots_adjust(hspace=0.05)

        if save:
            save_suffix="" if save_suffix is None else save_suffix
            plt.savefig(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_folded_trans"+save_suffix+"."+savetype))

    def MakeExoFopFiles(self, toi_names, desc=None, tsnotes='',plnotes='',table=None,username="osborn",
                        initials="ho",files=["cheops_plots","folded_trans"],
                        upload_loc="/Users/hosborn/Postdoc/Cheops/ChATeAUX/",check_toi_list=True, **kwargs):
        """Make ExoFop Files for upload. There are three things to upload - the lightcurve plot, the timseries table entry, and potentially a meta-description.
        Once this is complete, the outputs must be uploaded to exofop

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
                self.logger.warning("Without using `check_toi_list`, we have to assume the TOIs provided here match the planets in the model, i.e. with periods:",",".join([str(npl)+":"+str(self.planets[npl]['period']) for npl in self.planets]))
            for t in range(len(toi_names)):
                if toi_names[t].find(".")==-1:
                    toi_names[t]+='.01'
        
        for filetype in ["ExoFopTimeSeries","ExoFopFiles","ExoFopComments","ExoFopPlanetParameters"]:
            if not os.path.exists(os.path.join(upload_loc,filetype)):
                os.mkdir(os.path.join(upload_loc,filetype))

        if table is None and not hasattr(self,'trace_summary'):
            self.save_trace_summary(returndf=True)
        if table is None:
            table=self.trace_summary
                
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
                ph=((self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'time'].values-t0-0.5*p)%p-0.5*p)/dur
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
        
        self.logger.info([tsnotes,type(tsnotes),base_toi_name,type(base_toi_name),desc,type(desc)])
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
        
            obs_starts[fk]=Time(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'time'].values[0],format='jd')
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
                        self.logger.debug([os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_"+fk+"_"+f+"*"),
                                glob.glob(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_"+fk+"_"+f+"*"))])
                        filename=glob.glob(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_"+fk+"_"+f+"*"))[0]
                        self.logger.debug([maxcovtoiname,str(maxcovtoiname[fk])+"L-"+initials.lower()+date+"cheops-gto-chateaux."+filename[-3:]])
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
                    filename=glob.glob(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_"+f+".*"))[0]
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
            'ObsDur':str(np.round(np.ptp(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'time'].values)*1440,1)),
            'ObsNum':str(len(self.lcs["cheops"].loc[self.cheops_fk_mask[fk],'time'].values)),
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

    def MakePlanetPropertiesTable(self,DR2ID=None,data_ref="",table=None):
        """
         Make a table of planet properties for each planet.
         
         Args:
         	 DR2ID: Gaia DR2 ID
         	 data_ref: Data reference to use for data analysis
         	 table: If None ( default ) the trace summary table is used (save_trace_summary)
        """
        assert hasattr(self,'trace'), "Must have already sampled the model"

        if table is None and not hasattr(self,'trace_summary'):
            self.save_trace_summary(returndf=True)
        if table is None:
            table=self.trace_summary
        
        if DR2ID is None:
            if hasattr(self,'monotools_lc') and hasattr(self.monotools_lc,'all_ids') and 'tess' in self.monotools_lc.all_ids and 'data' in self.monotools_lc.all_ids['tess'] and 'GAIA' in self.monotools_lc.all_ids['tess']['data']:
                DR2ID=int(self.monotools_lc.all_ids['tess']['data']['GAIA'])
            else:
                assert hasattr(self, "radec"), "If you do not specify the Gaia DR2 ID then there must be a `radec` quantity initialised in the model"
                from astroquery.gaia import Gaia
                from astropy import units as u
                tab=Gaia.conesearch(sdelf.radec,radius=5*u.arcsec).results[0].to_pandas()
                tab=tab.loc[tab['phot_g_mean']<12.5]
                DR2ID=tab.loc[np.argmin(tab['distance']),'ID']
                if type(DR2ID)==pd.Series or type(DR2ID)==pd.DataFrame: 
                    DR2ID=DR2ID.iloc[0]

        allpl_dats=[]
        for pl in self.planets:
            pldic={"obj_id_catname":self.name,"obj_id_gaiadr2":DR2ID,"obj_id_planet_catname":self.name+" "+pl,
                   "obj_trans_t0_bjd":table.loc["t0_"+pl,"mean"],"obj_trans_t0_bjd_err":table.loc["t0_"+pl,"sd"],
                   "obj_trans_period_days":table.loc["P_"+pl,"mean"],"obj_trans_period_days_err":table.loc["P_"+pl,"sd"]}
            if self.assume_circ:
                pldic.update({"obj_trans_ecosw":0,"obj_trans_ecosw_err":0,"obj_trans_esinw":0,"obj_trans_esinw_err":0})
            else:
                pldic.update({"obj_trans_ecosw":np.nanmedian(self.trace.posterior["ecc_"+pl]*np.cos(self.trace.posterior["omega_"+pl])),
                            "obj_trans_ecosw_err":np.nanstd(self.trace.posterior["ecc_"+pl]*np.cos(self.trace.posterior["omega_"+pl])),
                            "obj_trans_esinw":np.nanmedian(self.trace.posterior["ecc_"+pl]*np.sin(self.trace.posterior["omega_"+pl])),
                            "obj_trans_esinw_err":np.nanstd(self.trace.posterior["ecc_"+pl]*np.sin(self.trace.posterior["omega_"+pl]))})
            pldic.update({"obj_trans_depth_ppm":1e6*table.loc["ror_"+pl,"mean"]**2,"obj_trans_depth_ppm_err":table.loc["ror_"+pl,"mean"]**2*2*table.loc["ror_"+pl,"sd"]/table.loc["ror_"+pl,"mean"],
                          "obj_trans_duration_days":table.loc["tdur_"+pl,"mean"],"obj_trans_duration_days_err":table.loc["tdur_"+pl,"sd"]})
            if hasattr(self, 'rvs'):
                pldic.update({'obj_rv_k_mps':table.loc['K_'+pl,'mean'],'obj_rv_k_mps_err':table.loc['K_'+pl,'sd']})
            else:
                pldic.update({'obj_rv_k_mps':0,'obj_rv_k_mps_err':0})
            pldic.update({'db_info_reference':data_ref,
                          'db_info_remarks':'chexoplanet-generated planet properties table.'})
            allpl_dats+=[pldic]
        colstr="obj_id_catname	obj_id_gaiadr2	obj_id_planet_catname	obj_trans_t0_bjd	obj_trans_t0_bjd_err	obj_trans_period_days	obj_trans_period_days_err	obj_trans_ecosw	obj_trans_ecosw_err	obj_trans_esinw	obj_trans_esinw_err	obj_trans_depth_ppm	obj_trans_depth_ppm_err	obj_trans_duration_days	obj_trans_duration_days_err	obj_rv_k_mps	obj_rv_k_mps_err	db_info_reference	db_info_remarks".split("\t")

        with open(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_PlanetProperties.tsv"),"w") as plandatfile:
            plandatfile.write("|".join(colstr)+"\n")
            print("|".join(colstr))
            for dicdat in allpl_dats:
                outstr="|".join([str(dicdat[col]) for col in colstr])
                print(outstr)
                assert(len(outstr.split("|"))==len(colstr))
                plandatfile.write(outstr+"\n")
        #for mod in self.models_out:
        #    self.models_out[mod].to_csv(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_"+mod+"_timeseries.csv"))

    def plot_corner(self):
        """
        Plotting corner for each planet.
        """
        import corner
        for pl in self.planets:
            ivars=[]
            if self.fit_ttvs:
                ivars+=[var for var in self.trace.varnames if 'transit_time_'+pl in var and '__' not in var]
            elif self.split_periods is not None and self.planets[pl]['n_trans']>2 and pl in self.split_periods:
                #taking split_P
                ivars+=[var for var in self.trace.varnames if 'split_P_'+pl in var and '__' not in var]
                ivars+=[var for var in self.trace.varnames if 'split_t0_'+pl in var and '__' not in var]
            else:
                ivars+=['P_'+pl,'t0_'+pl]
            ivars+=['logror_'+pl,'b_'+pl,'Rs','Teff']+['u_star_'+scope for scope in self.lcs]
            ivars+=['Ms'] if self.spar_param=='Mstar' else self.spar_param
            if not self.assume_circ:
                ivars+=['ecc_'+pl,'omega_'+pl]
            fig=corner.corner(self.trace,var_names=ivars)
            fig.savefig(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_corner_"+pl+".pdf"))
        #Now doing TESS/CHEOPS stuff
        if self.fit_gp:
            fig=corner.corner(self.trace,var_names=['phot_sigma','phot_w0','tess_logs'])
            #fig=corner.corner(self.trace,var_names=['phot_S0','phot_w0','tess_logs'])
            fig.savefig(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_corner_tessgp.pdf"))
        if "cheops" in self.lcs:
            ivars=list(self.cheops_quad_decorrs.keys())+list(self.cheops_linear_decorrs.keys())+['cheops_logs']+["cheops_mean_"+str(fk) for fk in self.cheops_filekeys]
            fig=corner.corner(self.trace,var_names=ivars)
            fig.savefig(os.path.join(self.save_file_loc,self.name.replace(" ","_"),self.unq_name+"_corner_cheops.pdf"))
    
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
                descr+="We then sampled this {{\it CHEOPS}}-only model using \\texttt{pymc} and calculated which detrending parameters improved the fit taking all parameters with a Bayes Factors less than "+str(np.round(self.signif_thresh,1))+". "
            elif self.use_signif:
                descr+="We then sampled this {{\it CHEOPS}}-only model using \\texttt{pymc} and calculated which detrending parameters improved the fit taking all parameters with significant non-zero correlation coefficients (i.e. $>"+str(np.round(self.signif_thresh,1))+"\\sigma$ from 0). "
            descr+="\nWe perform two models - one a comparison with and without a transit model to the {{\it CHEOPS}} data to assess the presence of a transit in the photometry, and a second including available TESS data in order to assess the improvement in radius and ephemeris precision."
            descr+="Both models used \\texttt{pymc} and \\texttt{exoplanet} and included the TOI parameters (e.g. ephemeris) as priors. "
            if self.fit_phi_gp:
                descr+="We also included a \\texttt{celerite} Gaussian Process model \\citep{celerite} to fit variations in flux as a function of roll angle not removed by the decorrelations. "
            descr+="The resulting best-fit model for the {{\it CHEOPS}} data is shown in Figure \ref{fig:cheops}.\n\n"
            descr+="\begin{figure}\n\includegraphics[width=\columnwidth]{"+self.unq_name+"_cheops_plots.png}\n\caption{"
            descr+="Cheops photometry. Upper panel shows raw and binned {{\it CHEOPS}} photometry, and the modelled flux variations due to decorrelation "
            if self.fit_phi_gp: descr+="and Gaussian process "
            descr+="models offset above, with a best-fit line as well as 1- \& 2-$\\sigma$ regions. Lower panel shows the detrended {{\it CHEOPS}} photometry as well as the best-fit line as well as 1- \& 2-$\\sigma$ regions of a combined {\\it TESS} and {\\it CHEOPS} transit model."
            descr+="}\n\label{fig:CheopsOther}\n\end{figure}\n\n"
            descr+=""
        return descr

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