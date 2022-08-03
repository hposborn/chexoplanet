import exoplanet as xo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iteround import saferound
from scipy.signal import savgol_filter


from astropy.io import fits
from astropy.io import ascii
import astropy.units as u
from astropy.units import cds
from astropy import constants as c
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body, Angle

import pickle
import os.path
from datetime import date
import os
import glob

import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("theano").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)

floattype=np.float64

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

    def __init__(self, targetname, overwrite=True, radec=None, **kwargs):
        self.name=targetname
        self.overwrite=overwrite
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
                       'use_bayes_fact':False,  # Determine the detrending factors to use with a Bayes Factor
                       'use_signif':True,       # Determine the detrending factors to use by simply selecting those with significant non-zero coefficients
                       'signif_thresh':1.25,    # Threshold for detrending parameters in sigma
                       'use_multinest':False,   # use_multinest - bool - currently not supported
                       'use_pymc3':True,        # use_pymc3 - bool
                       'assume_circ':False,     # assume_circ - bool - Assume circular orbits (no ecc & omega)?
                       'fit_ttvs':False,        # Fit a TTVorbit exoplanet model which searches for TTVs
                       'fit_phi_gp':True,       # fit_phi_gp - bool - co-fit a GP to the roll angle.
                       'ecc_prior':'auto',      # ecc_prior - string - 'uniform', 'kipping' or 'vaneylen'. If 'auto' we decide based on multiplicity
                       'npoly_rv':2,            # npoly_rv - int - order of polynomial fit to RVs
                       'use_mstar':True,        # use_mstar - bool - Whether to model using the stellar Mass (otherwise set use_logg)
                       'use_logg':False,        # use_logg - bool - Whether to model using the stellar logg (otherwise Mass)
                       'constrain_lds':True,    # constrain_lds - bool - Use constrained LDs from model or unconstrained?
                       'ld_mult':3.,            # ld_mult - float - How much to multiply theoretical LD param uncertainties
                       'fit_contam':False}      # fit_contam - bool - Fit for "second light" (i.e. a binary or planet+blend)

        if radec is not None:
            self.radec=radec

        self.planets={}

        for param in self.defaults:
            if not hasattr(self,param) or self.overwrite:
                setattr(self,param,self.defaults[param])
        self.update(**kwargs)                    

        #Initalising save locations
        if self.load_from_file and not self.overwrite:
            #Catching the case where the file doesnt exist:
            success = self.LoadModelFromFile(loadfile=self.save_file_loc)
            self.load_from_file = success

        self.percentiles={'-2sig':2.2750132, '-1sig':15.8655254, 'med':50., '+1sig':84.1344746, '+2sig':97.7249868}

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
               'use_multinest','use_pymc3','assume_circ','fit_ttvs','fit_phi_gp','use_mstar','use_logg','constrain_lds','fit_contam']
        boolstr=''
        for i in bools:
            boolstr+=str(int(getattr(self,i)))
            
        nonbools=['flat_knotdist','cut_distance','mask_distance','bin_size','signif_thresh','ecc_prior','npoly_rv','ld_mult']
        nonboolstrs=[]
        for i in nonbools:
            nonboolstrs+=[str(getattr(self,i)) if len(str(getattr(self,i)))<5 else str(getattr(self,i))[:5]]
        self.unq_name=self.name+"_"+date.today().strftime("%Y%m%d")+"_"+str(int(boolstr, 2))+"_"+"_".join(nonboolstrs)


    def add_lc(self, time, flux, flux_err, source='tess'):
        if not hasattr(self,'lcs'):
            self.lcs={}
        self.lcs[source]=pd.DataFrame({'time':time,'flux':flux,'flux_err':flux_err})

    def add_cheops_lc(self, filekey, fileloc, DRP=True, PIPE=False, PIPE_binary=False, **kwargs):
        """AI is creating summary for add_cheops_lc

        Args:
            filekey (str): Unique filekey for this Cheops lightcurve
            fileloc (str): Location of lightcurve fits file
            DRP (bool, optional): Is this a DRP file? Defaults to True.
            PIPE (bool, optional): Is this a PIPE file? Defaults to False.
            PIPE_binary (bool, optional): Is this a PIPE file with two stars (ie binary model)? Defaults to False.
        """

        self.update(**kwargs)

        assert DRP^PIPE, "Must have either DRP or PIPE flagged"
        
        if not hasattr(self,"cheops_lc"):
            self.cheops_lc = pd.DataFrame()
        
        if PIPE:
            binchar='1' if PIPE_binary else ''
            sources={'time':'BJD_TIME', 'flux':'FLUX'+binchar, 'flux_err':'FLUXERR'+binchar, 
                     'phi':'ROLL'+binchar, 'bg':'BG'+binchar, 'centroid_x':'XC'+binchar,
                     'centroid_y':'YC'+binchar, 'deltaT':'thermFront_2','smear':None}
        elif DRP:
            sources={'time':'BJD_TIME', 'flux':'FLUX', 'flux_err':'FLUXERR', 
                     'phi':'ROLL_ANGLE', 'bg':'BACKGROUND', 'centroid_x':'CENTROID_X', 
                     'centroid_y':'CENTROID_Y', 'deltaT':None, 'smear':'SMEARING_LC'}
        
        f=fits.open(fileloc)

        iche=pd.DataFrame()
        for s in sources:
            if sources[s] is not None:
                iche[s]=f[1].data[sources[s]]
            if s=='flux':
                iche[s]=(iche[s].values/np.nanmedian(f[1].data[sources['flux']])-1)*1000
            if s=='flux_err':
                iche[s]=(iche[s].values/np.nanmedian(f[1].data[sources['flux']]))*1000
            if s=='phi':
                iche[s]=roll_rollangles(iche[s].values)
        iche['xoff']=iche['centroid_x']-np.nanmedian(iche['centroid_x'])
        iche['yoff']=iche['centroid_y']-np.nanmedian(iche['centroid_y'])
        iche['phi_sorting']=np.argsort(iche['phi'].values)
        iche['time_sorting']=np.argsort(iche['time'].values[iche['phi_sorting'].values])

        #Getting moon-object angle:

        if hasattr(self,'radec') and self.radec is not None:
            moon_coo = get_body('moon', Time(iche['time'],format='jd',scale='tdb'))
            v_moon = np.arccos(
                    np.cos(moon_coo.ra.radian)*np.cos(moon_coo.dec.radian)*np.cos(self.radec.ra.radian)*np.cos(self.radec.dec.radian) +
                    np.sin(moon_coo.ra.radian)*np.cos(moon_coo.dec.radian)*np.sin(self.radec.ra.radian)*np.cos(self.radec.dec.radian) +
                    np.sin(moon_coo.dec.radian)*np.sin(self.radec.dec.radian))
            dv_rot = np.degrees(np.arcsin(np.sin(moon_coo.ra.radian-self.radec.ra.radian)*np.cos(moon_coo.dec.radian)/np.sin(v_moon)))
            iche['cheops_moon_angle']=iche['phi'].values-dv_rot

        iche['filekey']=np.tile(filekey,len(f[1].data[sources['time']]))
        
        #Performing simple anomaly masking using background limit, nans, and flux outliers:
        bgthresh=np.percentile(iche['bg'].values,95)*1.5
        iche['mask']=(~np.isnan(iche['flux']))&(~np.isnan(iche['flux_err']))&cut_anom_diff(iche['flux'].values)&(iche['bg']<bgthresh)
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
        self.rv_t = np.arange(np.nanmin(self.rvs['time']-10),np.nanmax(self.rvs['time']+10),0.0666*np.min([self.planets[pl]['period'] for pl in self.planets]))

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
                self.Mstar=rhostar[0]*self.Rstar[0]**3
        else:
            self.rhostar=np.array(rhostar).astype(float)
            if Mstar is None:
                self.Mstar=rhostar[0]*self.Rstar[0]**3

    def add_planet(self, name,tcen,period,tdur,depth,tcen_err=None,
                   period_err=None,b=None,rprs=None,K=None,overwrite=False):
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
        
        if period_err is not None:
            period_err = 0.25*tdur/period

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

    def init_lc(self,**kwargs):
        """Initialise survey (i.e. TESS) lightcurve. 
        This will create a lightcurve as the fit_lc object.

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
        assert self.fit_flat^self.fit_gp, "Cannot both flatten data and fit GP. Choose one"        
        
        #masking, binning, flattening light curve

        if not hasattr(self,'binlc'):
            self.binlc={}
        for src in self.lcs:
            self.lcs[src]['mask']=~np.isnan(self.lcs[src]['flux'].values)&~np.isnan(self.lcs[src]['flux_err'].values)
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
            self.binlc[src]=pd.DataFrame({'time':ibinlc[:,0],'flux':ibinlc[:,1],'flux_err':ibinlc[:,2]})
            if self.fit_flat:
                ibinlc2=bin_lc_segment(np.column_stack((self.lcs[src]['time'].values[self.lcs[src]['mask']],
                                                        self.lcs[src]['flux'].values[self.lcs[src]['mask']],
                                                        self.lcs[src]['flux_err'].values[self.lcs[src]['mask']])),
                                           self.bin_size)
                self.binlc[src]['flux_flat']=ibinlc2[:,1]
            self.binlc[src]['in_trans'] = np.tile(False,len(self.binlc[src]['time']))
            self.binlc[src]['near_trans'] = np.tile(False,len(self.binlc[src]['time']))
            for pl in self.planets:
                self.binlc[src]['in_trans']+=abs((self.binlc[src]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.mask_distance*self.planets[pl]['tdur']
                self.binlc[src]['near_trans']+=abs((self.binlc[src]['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.cut_distance*self.planets[pl]['tdur']
            
        if not hasattr(self,'lc_fit') or self.overwrite:
            self.lc_fit=pd.DataFrame()
        for src in self.lcs:
            vals=['time','flux','flux_err','in_trans','near_trans']
            if self.fit_flat: vals+=['spline']
            for val in vals:
                srcval='flux_flat' if val=='flux' and self.fit_flat else val
                if self.cut_oot:
                    #Cutting far-from-transit values:
                    newvals=self.lcs[src].loc[self.lcs[src]['mask']&self.lcs[src]['near_trans'],srcval]
                elif self.bin_oot:
                    #Merging the binned and raw timeseries so that near-transit data is raw and far-from-transit is binned:
                    newvals=np.hstack((self.lcs[src][srcval][self.lcs[src]['mask']&self.lcs[src]['near_trans']],self.binlc[src][srcval][~self.binlc[src]['near_trans']]))
                if srcval not in self.lc_fit.columns:
                    self.lc_fit[val]=newvals
                else:
                    self.lc_fit[val]=np.hstack((self.lc_fit[val],newvals))

            #Adding source to the array:
            if 'src' not in self.lc_fit.columns:
                self.lc_fit['src']=np.tile(src,len(newvals))
            else:
                self.lc_fit['src']=np.hstack((self.lc_fit['src'],np.tile(src,len(newvals))))
            self.lc_fit=self.lc_fit.sort_values('time')                

        if self.train_gp and self.fit_gp:
            self.init_gp()

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

            logs_tess = pm.Normal("logs_tess", mu=np.log(np.std(self.lc_fit['flux'])), sd=1)
            
            #Initialising the SHO frequency
            max_cad=self.bin_size
            lcrange=27
            av_dur = np.average([self.planets[key]['tdur'] for key in self.planets])
            success=False;target=0.05
            while not success and target<0.21:
                try:
                    low=(2*np.pi)/(0.25*lcrange/(target/0.05))
                    up=(2*np.pi)/(25*av_dur*(target/0.05))
                    w0 = pm.InverseGamma("w0",testval=(2*np.pi)/10,**pmx.estimate_inverse_gamma_parameters(lower=low,upper=up))
                    success=True
                except:
                    low=(2*np.pi)/(10)
                    up=(2*np.pi)/(6*max_cad)
                    target*=1.15
                    success=False
            print("w0",success,target,low,up,(2*np.pi)/low,(2*np.pi)/up)
            
            #Initialising the power:
            success=False;target=0.01
            maxpower=1.0*np.nanstd(self.lc_fit.loc[~self.lc_fit['in_trans'],'flux'].values)
            minpower=0.02*np.nanmedian(abs(np.diff(self.lc_fit.loc[~self.lc_fit['in_trans'],'flux'].values)))
            
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
            print("power",success,target)
            S0 = pm.Deterministic("S0", power/(w0**4))

            # GP model for the light curve
            kernel = theano_terms.SHOTerm(S0=S0, w0=w0, Q=1/np.sqrt(2))

            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            gp = celerite2.theano.GaussianProcess(kernel, mean=mean)
            gp.compute(self.lc_fit.loc[~self.lc_fit['in_trans'],'time'].values,
                       diag=self.lc_fit.loc[~self.lc_fit['in_trans'],'flux_err'].values ** 2 + tt.exp(logs_tess)**2, quiet=True)
            gp.marginal("obs", observed=self.lc_fit.loc[~self.lc_fit['in_trans'],'flux'].values)
            
            #photgp_model_x = pm.Deterministic("photgp_model_x", gp.predict(self.lc_fit['flux'][~self.lc_fit['in_trans']], t=self.lc_fit['time'][~self.lc_fit['in_trans']], return_var=False))

            #optimizing:
            start = ootmodel.test_point
            print(ootmodel.test_point)

            oot_soln = pmx.optimize(start=start)

        #Sampling:
        with ootmodel: 
            self.oot_gp_trace = pm.sample(tune=500, draws=1200, start=oot_soln, 
                                          compute_convergence_checks=False)

    def init_cheops(self, **kwargs):
        """Initialising the Cheops data.
        This includes running an initial PyMC model on the Cheops data alone to see which detrending parameters to use.

        Optional Args:
            use_signif (bool, optional):     Determine the detrending factors to use by simply selecting those with significant non-zero coefficients. Defaults to True
            use_bayes_fact (bool, optional): Determine the detrending factors to use with a Bayes Factor (Default is False)
            signif_thresh (float, optional): #hreshold for detrending parameters in sigma (default: 1.25)
        """
        
        self.update(**kwargs)

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

        #Initialising initial lists of decorrelation parameters:
        all_linear_decorr_pars=['sinphi','cosphi','bg','centroid_x','centroid_y','time']
        all_quad_decorr_pars=['bg','centroid_x','centroid_y','sinphi','cosphi','time']
        if 'smear' in self.cheops_lc.columns: all_linear_decorr_pars+=['smear']
        if 'smear' in self.cheops_lc.columns: all_quad_decorr_pars+=['smear']
        if 'deltaT' in self.cheops_lc.columns: all_linear_decorr_pars+=['deltaT']
        if 'deltaT' in self.cheops_lc.columns: all_quad_decorr_pars+=['deltaT'] 

        #Checking which transits are in which dataset:
        cheops_in_trans=np.tile(False,(len(self.cheops_lc['time']),len(self.planets)))
        for ipl,pl in enumerate(self.planets):
            cheops_in_trans[:,ipl]=abs((self.cheops_lc['time'].values-self.planets[pl]['tcen']-0.5*self.planets[pl]['period'])%self.planets[pl]['period']-0.5*self.planets[pl]['period'])<self.mask_distance*self.planets[pl]['tdur']

        self.cheops_filekeys = pd.unique(self.cheops_lc['filekey'])
        self.ichlc_models={}
        self.cheops_mads={}
        self.norm_cheops_dat={fk:{} for fk in self.cheops_filekeys}
        self.init_chefit_summaries={fk:{} for fk in self.cheops_filekeys}
        self.linear_assess={fk:{} for fk in self.cheops_filekeys}
        self.quad_assess={fk:{} for fk in self.cheops_filekeys}

        #Looping over all Cheops datasets and building individual models which we can then extract stats for each detrending parameter
        for fk in self.cheops_filekeys:
            print("Performing Cheops-only minimisation with all detrending params for filekey ",fk)
            #Initialising the data specific to each Cheops visit:
            x=self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'].values
            y=self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values
            #Using a robust average (logged) of the point-to-point error & the std as a prior for the decorrelation parameters
            self.cheops_mads[fk]=np.exp(0.5*(np.log(np.std(y))+np.log(np.nanmedian(abs(np.diff(y)))*1.06)))
            yerr=self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux_err'].values
            for var in all_linear_decorr_pars+all_quad_decorr_pars:
                if var in self.cheops_lc.columns:
                    self.norm_cheops_dat[fk][var]=(self.cheops_lc.loc[self.cheops_fk_mask[fk],var].values-np.nanmedian(self.cheops_lc.loc[self.cheops_fk_mask[fk],var].values))/np.nanstd(self.cheops_lc.loc[self.cheops_fk_mask[fk],var].values)
                elif var[:3]=='sin':
                    self.norm_cheops_dat[fk][var]=np.sin(self.cheops_lc.loc[self.cheops_fk_mask[fk],var[3:]].values*np.pi/180)
                elif var[:3]=='cos':
                    self.norm_cheops_dat[fk][var]=np.sin(self.cheops_lc.loc[self.cheops_fk_mask[fk],var[3:]].values*np.pi/180)
            #Launching a PyMC3 model
            with pm.Model() as self.ichlc_models[fk]:
                #Adding planet model info if there's any transit in the lightcurve
                if np.any(cheops_in_trans[self.cheops_lc['filekey']==fk,:]):
                    Rs = pm.Bound(pm.Normal, lower=0)("Rs",mu=self.Rstar[0], sd=self.Rstar[1])
                    Ms = pm.Bound(pm.Normal, lower=0)("Ms",mu=self.Mstar[0], sd=self.Mstar[1])
                    u_star_cheops = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star_cheops", 
                                                    mu=np.clip(np.nanmedian(self.ld_dists['cheops'],axis=0),0,1),
                                                    sd=np.clip(np.nanstd(self.ld_dists['cheops'],axis=0),0.1,1.0), 
                                                    shape=2, testval=np.clip(np.nanmedian(self.ld_dists['cheops'],axis=0),0,1))

                    
                    logrors={};bs={};t0s={};pers={};cheops_planets_x={};orbits={}
                    for ipl,pl in enumerate(self.planets):
                        if np.any(cheops_in_trans[self.cheops_lc['filekey']==fk,ipl]):
                            
                            #If this timeseries specifically has this planet in, we need to fit for it
                            logrors[pl] = pm.Uniform("logror_"+pl, lower=np.log(0.001), upper=np.log(0.1), 
                                                testval=np.log(np.sqrt(self.planets[pl]['depth'])))
                            #r_pl = pm.Deterministic("r_pl",109.1*tt.exp(logror)/Rs)
                            bs[pl] = xo.distributions.ImpactParameter("b_"+pl, ror=tt.exp(logrors[pl]),
                                                                testval=self.planets[pl]['b'])
                            t0s[pl] = pm.Normal("t0_"+pl, mu=self.planets[pl]['tcen'],sd=self.planets[pl]['tcen_err'],testval=self.planets[pl]['tcen'])
                            pers[pl] = pm.Normal("per_"+pl, mu=self.planets[pl]['period'],sd=self.planets[pl]['period_err'],testval=self.planets[pl]['period'])

                            orbits[pl] = xo.orbits.KeplerianOrbit(r_star=Rs, m_star=Ms, period=pers[pl], t0=t0s[pl], b=bs[pl])
                            cheops_planets_x[pl] = pm.Deterministic("cheops_planets_x_"+str(pl), xo.LimbDarkLightCurve(u_star_cheops).get_light_curve(orbit=orbits[pl], r=tt.exp(logrors[pl])/Rs,t=x)*1000)[:,0]

                logs_cheops = pm.Normal("logs_cheops", mu=np.log(np.nanmedian(abs(np.diff(y))))-3, sd=3)

                #Initialising linear (and quadratic) parameters:
                linear_decorr_dict={};quad_decorr_dict={}
                
                for decorr_1 in all_linear_decorr_pars:
                    if decorr_1=='time':
                        linear_decorr_dict[decorr_1]=pm.Normal("dfd"+decorr_1,mu=0,sd=np.ptp(self.norm_cheops_dat[fk][decorr_1])/self.cheops_mads[fk],testval=np.random.normal(0,0.05))
                    else:
                        linear_decorr_dict[decorr_1]=pm.Normal("dfd"+decorr_1,mu=0,sd=self.cheops_mads[fk],testval=np.random.normal(0,0.05))
                for decorr_2 in all_quad_decorr_pars:
                    quad_decorr_dict[decorr_2]=pm.Normal("d2fd"+decorr_2+"2",mu=0,sd=self.cheops_mads[fk],testval=np.random.normal(0,0.05))
                cheops_obs_mean=pm.Normal("cheops_mean",mu=0.0,sd=0.5*np.nanstd(y))
                cheops_flux_cor = pm.Deterministic("cheops_flux_cor",cheops_obs_mean + tt.sum([linear_decorr_dict[param]*self.norm_cheops_dat[fk][param] for param in all_linear_decorr_pars], axis=0) + \
                                                    tt.sum([quad_decorr_dict[param]*self.norm_cheops_dat[fk][param]**2 for param in all_quad_decorr_pars], axis=0))
                #A
                if np.any(cheops_in_trans[self.cheops_lc['filekey']==fk,:]):
                    #for pl in cheops_planets_x:
                    #    tt.printing.Print("cheops_planets_x"+str(pl))(cheops_planets_x[pl])
                    if len(cheops_planets_x)>1:
                        cheops_summodel_x = pm.Deterministic("cheops_summodel_x", tt.sum(tt.stack([cheops_planets_x[pl] for pl in cheops_planets_x]),axis=0) + cheops_flux_cor)
                    else:
                        cheops_summodel_x = pm.Deterministic("cheops_summodel_x", tt.sum([cheops_planets_x[pl] for pl in cheops_planets_x],axis=0) + cheops_flux_cor)
                else:
                    cheops_summodel_x = pm.Deterministic("cheops_summodel_x", cheops_flux_cor)
                cheops_sigma2 = yerr ** 2 + tt.exp(logs_cheops)**2
                llk_cheops = pm.Potential("llk_cheops", -0.5 * tt.sum((y - cheops_summodel_x) ** 2 / cheops_sigma2 + np.log(cheops_sigma2)))
                pm.Deterministic("out_llk_cheops",llk_cheops)
                
                #Minimizing:
                if np.any(cheops_in_trans[self.cheops_lc['filekey']==fk,:]):
                    comb_soln = pmx.optimize(vars=[Rs,Ms,u_star_cheops]+[t0s[pl] for pl in t0s]+[pers[pl] for pl in t0s]+[logrors[pl] for pl in t0s]+[logs_cheops])
                    comb_soln = pmx.optimize(start=comb_soln,
                                             vars=[linear_decorr_dict[par] for par in linear_decorr_dict] + \
                                             [cheops_obs_mean,logs_cheops] + \
                                             [quad_decorr_dict[par] for par in quad_decorr_dict])
                else:
                    comb_soln = pmx.optimize(vars=[linear_decorr_dict[par] for par in linear_decorr_dict] + \
                                             [cheops_obs_mean,logs_cheops] + \
                                             [quad_decorr_dict[par] for par in quad_decorr_dict])
                comb_soln = pmx.optimize(start=comb_soln)
                trace = pmx.sample(tune=300, draws=400, chains=3, cores=3, start=comb_soln)
            var_names=[var for var in trace.varnames if '__' not in var and np.product(trace[var].shape)<6*np.product(trace['Rs'].shape)]
            self.init_chefit_summaries[fk]=pm.summary(trace,var_names=var_names,round_to=7)

            for par in all_linear_decorr_pars:
                dfd_fitvalue=self.init_chefit_summaries[fk].loc["dfd"+par,'mean']
                dfd_fitsd=self.init_chefit_summaries[fk].loc["dfd"+par,'sd']
                dfd_priorsd=1
                if self.use_bayes_fact:
                    self.linear_assess[fk][par] = np.exp(-0.5*((dfd_fitvalue)/dfd_fitsd)**2) * dfd_priorsd/dfd_fitsd
                elif self.use_signif:
                    self.linear_assess[fk][par] = abs(dfd_fitvalue)/dfd_fitsd
            for par in all_quad_decorr_pars:
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
            if self.use_bayes_fact:
                #Bayes factor is <1sigma = significant trend = use this in the decorrelation
                self.cheops_linear_decorrs[fk]={par for par in all_linear_decorr_pars if self.linear_assess[fk][par]<1}
                self.cheops_quad_decorrs[fk]={par for par in all_quad_decorr_pars if self.quad_assess[fk][par]<1}
            elif self.use_signif:
                #detrend mean is >1sigma = significant trend = use this in the decorrelation
                self.cheops_linear_decorrs[fk]={par for par in all_linear_decorr_pars if self.linear_assess[fk][par]>self.signif_thresh}
                self.cheops_quad_decorrs[fk]={par for par in all_quad_decorr_pars if self.quad_assess[fk][par]>self.signif_thresh}

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

    def init_model(self, **kwargs):
        """AI is creating summary for init_model
        """

        self.update(**kwargs)

        assert not self.use_multinest, "Multinest is not currently possible"
        assert not (self.fit_flat&self.fit_gp), "Cannot both flatten data and fit GP. Choose one"
        assert self.use_mstar^self.use_logg, "Must be either use_mstar or use_logg, not both/neither"

        if self.fit_ttvs:
            self.init_transit_times={}
            self.init_transit_inds={}
            for pl in self.planets:
                tcens=self.planets[pl]['tcen']+np.arange(-50,250)*self.planets[pl]['period']
                ix=np.min(abs(tcens[:,None]-np.hstack((self.lc_fit['time'],self.cheops_lc['time']))[None,:]),axis=1)<self.planets[pl]['tdur']*0.5
                self.init_transit_times[pl]=tcens[ix]
                self.init_transit_inds[pl]=np.arange(-50,250)[ix]
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
                self.model_params['deltaVmag_contam'] = pm.Uniform("deltaVmag_contam", upper=12, lower=2.5)
                self.model_params['cheops_mult'] = pm.Deterministic("cheops_mult",(1+tt.power(2.511,-1*self.model_params['deltaVmag_contam']))) #Factor to multiply normalised lightcurve by
            else:
                self.model_params['tess_mult']=1.0
                self.model_params['cheops_mult']=1.0
            # -------------------------------------------
            #                  Orbits
            # -------------------------------------------
            if not self.fit_ttvs:
                self.model_params['t0'] = pm.Normal("t0", mu=np.array([self.planets[key]['tcen'] for key in self.planets]), 
                            sd=np.array([2*self.planets[key]['tcen_err'] for key in self.planets]),
                            shape=len(self.planets))
                min_ps={pl:self.planets[pl]['period']*(1-0.5*self.planets[pl]['tdur']/(np.ptp(self.lc_fit['time']))) for pl in self.planets}
                max_ps={pl:self.planets[pl]['period']*(1+0.5*self.planets[pl]['tdur']/(np.ptp(self.lc_fit['time']))) for pl in self.planets}
                print(min_ps,max_ps,[(max_ps[key]-self.planets[key]['period']) for key in self.planets])
                self.model_params['P'] = pm.Bound(pm.Normal, lower=np.array([min_ps[key] for key in self.planets]),
                            upper=np.array([max_ps[key] for key in self.planets]))("P",
                            mu=np.array([self.planets[key]['period'] for key in self.planets]), 
                            sd=np.array([np.clip(self.planets[key]['period_err'],0,(max_ps[key]-self.planets[key]['period'])) for key in self.planets]),
                            shape=len(self.planets))
            else:
                #Initialising transit times:
                self.model_params['transit_times']=[]
                for pl in self.planets:
                    self.model_params['transit_times'].append(pm.Normal("transit_times_"+pl, mu=self.init_transit_times[pl], sd=self.planets[pl]['tdur']*0.05,
                                            shape=len(self.init_transit_times[pl]), testval=self.init_transit_times[pl]))

            # Wide log-normal prior for semi-amplitude
            if hasattr(self,'rvs'):
                self.model_params['logK'] = pm.Normal("logK", mu=np.log(np.tile(2,len(self.planets))), sd=np.tile(10,len(self.planets)), 
                                                      shape=len(self.planets), testval=np.tile(2.0,len(self.planets)))
            
            # Eccentricity & argument of periasteron
            if not self.assume_circ:
                BoundedBeta = pm.Bound(pm.Beta, lower=1e-5, upper=1-1e-5)
                self.model_params['ecc'] = BoundedBeta("ecc", alpha=0.867 ,beta=3.03, testval=0.05, shape=len(self.planets))
                self.model_params['omega'] = pmx.Angle("omega", shape=len(self.planets))
            '''
            #This was to model a non-transiting companion:
            P_nontran = pm.Normal("P_nontran", mu=27.386209624, sd=2*0.04947295)
            logK_nontran = pm.Normal("logK_nontran", mu=2,sd=10, testval=2)
            Mpsini_nontran = pm.Deterministic("Mp_nontran", tt.exp(logK_nontran) * 28.439**-1 * Ms**(2/3) * (P_nontran/365.25)**(1/3) * 317.8)
            t0_nontran = pm.Uniform("t0_nontran", lower=np.nanmedian(rv_x)-27.386209624*0.55, upper=np.nanmedian(rv_x)+27.386209624*0.55)
            '''
            self.model_params['logror'] = pm.Uniform("logror", lower=np.log(np.tile(0.001,len(self.planets))), upper=np.log(np.tile(0.1,len(self.planets))), 
                                testval=np.log(np.sqrt([self.planets[key]['depth'] for key in self.planets])), shape=len(self.planets))
            self.model_params['ror'] = pm.Deterministic("ror",tt.exp(self.model_params['logror']))
            self.model_params['r_pl'] = pm.Deterministic("r_pl",109.1*self.model_params['ror']/self.model_params['Rs'])
            self.model_params['b'] = xo.distributions.ImpactParameter("b", ror=self.model_params['ror'], shape=len(self.planets), 
                                                testval=np.array([self.planets[key]['b'] for key in self.planets]))
            
            self.model_params['u_stars']={}
            for scope in self.ld_dists:
                if self.constrain_lds:
                    self.model_params['u_stars'][scope] = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star"+scope, 
                                                mu=np.clip(np.nanmedian(self.ld_dists[scope],axis=0),0,1),
                                                sd=np.clip(np.nanstd(self.ld_dists[scope],axis=0),0.1,1.0), 
                                                shape=2, testval=np.clip(np.nanmedian(self.ld_dists[scope],axis=0),0,1))
                else:
                    self.model_params['u_stars'][scope] = xo.distributions.QuadLimbDark("u_star_"+scope, testval=np.array([0.3, 0.2]))
            
            if self.fit_ttvs:
                if self.assume_circ:
                    orbit = xo.orbits.TTVOrbit(b=self.model_params['b'],
                                    transit_times=self.model_params['transit_times'],
                                    transit_inds=[self.init_transit_inds[pl] for pl in self.planets],
                                    r_star=self.model_params['Rs'], m_star=self.model_params['Ms'])
                else:
                    orbit = xo.orbits.TTVOrbit(b=self.model_params['b'],
                                    transit_times=self.model_params['transit_times'],
                                    transit_inds=[self.init_transit_inds[pl] for pl in self.planets],
                                    r_star=self.model_params['Rs'], m_star=self.model_params['Ms'],
                                    ecc=self.model_params['ecc'],omega=self.model_params['omega'])
                self.model_params['t0'] = pm.Deterministic("t0", orbit.t0)
                self.model_params['P'] = pm.Deterministic("P", orbit.period)

            else:
                # Then we define the orbit
                if self.assume_circ:
                    orbit = xo.orbits.KeplerianOrbit(r_star=self.model_params['Rs'], m_star=self.model_params['Ms'], 
                                                     period=self.model_params['P'], t0=self.model_params['t0'], b=self.model_params['b'])
                else:
                    orbit = xo.orbits.KeplerianOrbit(r_star=self.model_params['Rs'], m_star=self.model_params['Ms'], period=self.model_params['P'], 
                                                     t0=self.model_params['t0'], b=self.model_params['b'],ecc=self.model_params['ecc'],omega=self.model_params['omega'])
        
            # -------------------------------------------
            #           Derived planet params:
            # -------------------------------------------
            if hasattr(self,'rvs'):
                if not self.assume_circ:
                    self.model_params['Mp'] = pm.Deterministic("Mp", tt.exp(self.model_params['logK']) * 28.439**-1 * (1-self.model_params['ecc']**2)**(0.5) * self.model_params['Ms']**(2/3) * (self.model_params['P']/365.25)**(1/3) * 317.8)
                else:
                    #Determining 
                    self.model_params['Mp'] = pm.Deterministic("Mp", tt.exp(self.model_params['logK']) * 28.439**-1 * self.model_params['Ms']**(2/3) * (self.model_params['P']/365.25)**(1/3) * 317.8)
                self.model_params['rho_p'] = pm.Deterministic("rho_p_gcm3",5.513*self.model_params['Mp']/self.model_params['r_pl']**3)
            
            self.model_params['a_Rs']=pm.Deterministic("a_Rs",orbit.a)
            self.model_params['sma']=pm.Deterministic("sma",self.model_params['a_Rs']*self.model_params['Rs']*0.00465)
            self.model_params['S_in']=pm.Deterministic("S_in",((695700000*self.model_params['Rs'])**2.*5.67e-8*self.model_params['Teff']**4)/(1.496e11*self.model_params['sma'])**2.)
            self.model_params['Tsurf_p']=pm.Deterministic("Tsurf_p",(((695700000*self.model_params['Rs'])**2.*self.model_params['Teff']**4.*(1.-0.2))/(4*(1.496e11*self.model_params['sma'])**2.))**(1./4.))
            
            #Getting the transit duration:
            self.model_params['vels'] = tt.diagonal(orbit.get_relative_velocity(self.model_params['t0']))
            tdur=pm.Deterministic("tdur",(2*self.model_params['Rs']*tt.sqrt((1+self.model_params['ror'])**2-self.model_params['b']**2))/tt.sqrt(self.model_params['vels'][0]**2 + self.model_params['vels'][1]**2))


            # -------------------------------------------
            #                    RVs:
            # -------------------------------------------
            if hasattr(self,'rvs'):
                self.model_params['vrad_x'] = pm.Deterministic("vrad_x",orbit.get_radial_velocity(self.rvs['time'], K=tt.exp(self.model_params['logK'])))
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
                self.model_params['rv_model_x'] = pm.Deterministic("rv_model_x", self.model_params['bkg_x'] + tt.sum(self.model_params['vrad_x'], axis=1))

                # Also define the model on a fine grid as computed above (for plotting)
                self.model_params['vrad_t'] = pm.Deterministic("vrad_t",orbit.get_radial_velocity(self.rv_t, K=tt.exp(self.model_params['logK'])))
                '''vrad_t = pm.Deterministic("vrad_t",tt.stack([orbit.get_radial_velocity(rv_t, K=tt.exp(logK))[:,0],
                                                                orbit.get_radial_velocity(rv_t, K=tt.exp(logK))[:,1],
                                                                orbit_nontran.get_radial_velocity(rv_t, K=tt.exp(logK_nontran))],axis=1))
                '''
                #orbit.get_radial_velocity(rv_t, K=tt.exp(logK)))
                if self.npoly_rv>1:
                    self.model_params['bkg_t'] = pm.Deterministic("bkg_t", tt.dot(np.vander(self.rv_t - self.rv_x_ref, self.npoly_rv),self.model_params['rv_trend']))
                    self.model_params['rv_model_t'] = pm.Deterministic("rv_model_t", self.model_params['bkg_t'] + tt.sum(self.model_params['vrad_t'], axis=1))
                else:
                    self.model_params['rv_model_t'] = pm.Deterministic("rv_model_t", tt.sum(self.model_params['vrad_t'], axis=1))
            
            # -------------------------------------------
            #                 TESS GP:
            # -------------------------------------------
            if self.fit_gp:
                minmax={}
                # Here we interpolate the histograms of the pre-trained GP samples as the input prior for each:
                minmax['logs_tess']=np.percentile(self.oot_gp_trace["logs_tess"],[0.5,99.5])
                self.model_params['logs_tess']=pm.Interpolated("logs_tess",x_points=np.linspace(minmax['logs_tess'][0],minmax['logs_tess'][1],201)[1::2],
                                        pdf_points=np.histogram(self.oot_gp_trace["logs_tess"],np.linspace(minmax['logs_tess'][0],minmax['logs_tess'][1],101))[0]
                                        )    
                minmax['S0']=np.percentile(self.oot_gp_trace["S0"],[0.5,99.5])
                self.model_params['tess_S0']=pm.Interpolated("tess_S0",x_points=np.linspace(minmax['S0'][0],minmax['S0'][1],201)[1::2],
                                        pdf_points=np.histogram(self.oot_gp_trace["S0"],np.linspace(minmax['S0'][0],minmax['S0'][1],101))[0]
                                        )
                minmax["w0"]=np.percentile(self.oot_gp_trace["w0"],[0.5,99.5])
                self.model_params['tess_w0']=pm.Interpolated("tess_w0",x_points=np.linspace(minmax["w0"][0],minmax["w0"][1],201)[1::2],
                                            pdf_points=np.histogram(self.oot_gp_trace["w0"],np.linspace(minmax["w0"][0],minmax["w0"][1],101))[0]
                                            )
                minmax["mean"]=np.percentile(self.oot_gp_trace["mean"],[0.5,99.5])
                self.model_params['tess_mean']=pm.Interpolated("tess_mean",
                                            x_points=np.linspace(minmax['mean'][0],minmax['mean'][1],201)[1::2],
                                            pdf_points=np.histogram(self.oot_gp_trace['mean'],np.linspace(minmax['mean'][0],minmax['mean'][1],101))[0]
                                            )
                self.model_params['tess_kernel'] = theano_terms.SHOTerm(S0=self.model_params['tess_S0'], 
                                                                        w0=self.model_params['tess_w0'], Q=1/np.sqrt(2))#, mean = phot_mean)

                self.model_params['gp_tess'] = celerite2.theano.GaussianProcess(self.model_params['tess_kernel'], self.lc_fit['time'].values, mean=self.model_params['tess_mean'])
                self.model_params['gp_tess'].compute(self.lc_fit['time'].values, diag=self.lc_fit['flux_err'].values ** 2 + tt.exp(self.model_params['logs_tess'])**2, quiet=True)
            else:
                self.model_params['logs_tess']=pm.Normal("logs_tess", mu=np.log(np.std(self.lc_fit['flux'].values)), sd=1)
            # -------------------------------------------
            #         Cheops detrending (linear)
            # -------------------------------------------

            self.model_params['logs_cheops'] = pm.Normal("logs_cheops", mu=np.log(np.nanmedian(abs(np.diff(self.cheops_lc.loc[self.cheops_lc['mask'],'flux'].values)))), sd=3)

            #Initialising linear (and quadratic) parameters:
            self.model_params['linear_decorr_dict']={i:{} for i in self.cheops_filekeys};self.model_params['quad_decorr_dict']={i:{} for i in self.cheops_filekeys}
            
            for fk in self.cheops_filekeys:
                for decorr_1 in self.cheops_linear_decorrs[fk]:
                    if decorr_1=='time':
                        self.model_params['linear_decorr_dict'][fk][decorr_1]=pm.Normal("dfd"+decorr_1+"_"+str(fk),mu=0,sd=np.ptp(self.norm_cheops_dat[fk][decorr_1])/self.cheops_mads[fk],
                                                                                    testval=self.init_chefit_summaries[fk].loc["dfd"+decorr_1,'mean'])
                    else:
                        self.model_params['linear_decorr_dict'][fk][decorr_1]=pm.Normal("dfd"+decorr_1+"_"+str(fk),mu=0,sd=self.cheops_mads[fk],
                                                                                    testval=self.init_chefit_summaries[fk].loc["dfd"+decorr_1,'mean'])
                for decorr_2 in self.cheops_quad_decorrs[fk]:
                    self.model_params['quad_decorr_dict'][fk][decorr_2]=pm.Normal("d2fd"+decorr_2+"2_"+str(fk),mu=0, sd=self.cheops_mads[fk],
                                                                                  testval=self.init_chefit_summaries[fk].loc["d2fd"+decorr_1+"2",'mean'])

            #Creating the flux correction vectors:
            self.model_params['cheops_obs_means']={};self.model_params['cheops_flux_cor']={}
            
            for fk in self.cheops_filekeys:
                self.model_params['cheops_obs_means'][fk]=pm.Normal("cheops_mean_"+str(fk),mu=0.0,sd=0.5*np.nanstd(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values))
                if len(self.cheops_quad_decorrs[fk])>0:
                    #Linear and quadratic detrending
                    self.model_params['cheops_flux_cor'][fk] = pm.Deterministic("cheops_flux_cor_"+str(fk),self.model_params['cheops_obs_means'][fk] + \
                                                                                    tt.sum([self.model_params['linear_decorr_dict'][fk][param]*self.norm_cheops_dat[fk][param] for param in self.cheops_linear_decorrs[fk]], axis=0) + \
                                                                                    tt.sum([self.model_params['quad_decorr_dict'][fk][param]*self.norm_cheops_dat[fk][param]**2 for param in self.cheops_quad_decorrs[fk]], axis=0))
                elif len(self.cheops_linear_decorrs[fk])>0:
                    #Linear detrending only
                    print(fk,{param:self.norm_cheops_dat[fk][param].shape for param in self.cheops_linear_decorrs[fk]})
                    self.model_params['cheops_flux_cor'][fk] = pm.Deterministic("cheops_flux_cor_"+str(fk),self.model_params['cheops_obs_means'][fk] + \
                                                                                    tt.sum([self.model_params['linear_decorr_dict'][fk][param]*self.norm_cheops_dat[fk][param] for param in self.cheops_linear_decorrs[fk]], axis=0))
                else:
                    #No detrending at all
                    self.model_params['cheops_flux_cor'][fk] = pm.Deterministic("cheops_flux_cor_"+str(fk),tt.tile(self.model_params['cheops_obs_means'][fk],np.sum(self.cheops_fk_mask[fk])))
            # -------------------------------------------
            #      Cheops detrending (roll angle GP)
            # -------------------------------------------
            self.model_params['cheops_planets_x']={}
            self.model_params['cheops_summodel_x']={}
            self.model_params['llk_cheops']={}
            if self.fit_phi_gp:
                self.model_params['rollangle_power'] = pm.InverseGamma("rollangle_power",testval=np.nanmedian(abs(np.diff(self.cheops_lc['flux']))), 
                                                  **pmx.estimate_inverse_gamma_parameters(
                                                            lower=0.2*np.sqrt(np.nanmedian(abs(np.diff(self.cheops_lc['flux'])))),
                                                            upper=2*np.sqrt(np.nanstd(self.cheops_lc['flux'])))
                                                 )
                self.model_params['rollangle_lengthscale'] = pm.InverseGamma("rollangle_lengthscale", testval=25, 
                                                        **pmx.estimate_inverse_gamma_parameters(lower=30,upper=80))
                self.model_params['rollangle_w0'] = pm.Deterministic("rollangle_w0",(2*np.pi)/self.model_params['rollangle_lengthscale'])
                #pm.InverseGamma("rollangle_w0",testval=(2*np.pi)/30, **pmx.estimate_inverse_gamma_parameters(lower=(2*np.pi)/80,upper=(2*np.pi)/20))
                self.model_params['rollangle_S0'] = pm.Deterministic("rollangle_S0", self.model_params['rollangle_power']/(self.model_params['rollangle_w0']**4))

                self.model_params['rollangle_kernels']={}
                self.model_params['gp_rollangles']={}
                self.model_params['gp_rollangle_model_phi']={}
            else:
                cheops_sigma2s={}
            
            for fk in self.cheops_filekeys:
                # Roll angle vs flux GP
                if self.fit_phi_gp:
                    self.model_params['rollangle_kernels'][fk] = theano_terms.SHOTerm(S0=self.model_params['rollangle_S0'], w0=self.model_params['rollangle_w0'], Q=1/np.sqrt(2))#, mean = phot_mean)
                    self.model_params['gp_rollangles'][fk] = celerite2.theano.GaussianProcess(self.model_params['rollangle_kernels'][fk], np.sort(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values).astype(np.float32), mean=0)
                #Creating the lightcurve:
                self.model_params['cheops_planets_x'][fk] = pm.Deterministic("cheops_planets_x_"+str(fk), xo.LimbDarkLightCurve(self.model_params['u_stars']["cheops"]).get_light_curve(orbit=orbit, r=self.model_params['r_pl']/109.2,
                                                                                                        t=self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'].values)*1000/self.model_params['cheops_mult'])
                #Adding the correlation model:
                self.model_params['cheops_summodel_x'][fk] = pm.Deterministic("cheops_summodel_x_"+str(fk), tt.sum(self.model_params['cheops_planets_x'][fk],axis=1) + self.model_params['cheops_flux_cor'][fk])
                if self.fit_phi_gp:
                    self.model_params['gp_rollangles'][fk].compute(np.sort(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values),
                                             diag=self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux_err'].values[self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values] ** 2 + \
                                                  tt.exp(self.model_params['logs_cheops'])**2, quiet=True)
                    self.model_params['gp_rollangle_model_phi'][fk] = pm.Deterministic("gp_rollangle_model_phi_"+str(fk), 
                                                        self.model_params['gp_rollangles'][fk].predict((self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values[self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values] - \
                                                                                  self.model_params['cheops_summodel_x'][fk][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values]), 
                                                                            t=np.sort(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values), return_var=False))
            # -------------------------------------------
            #      Evaluating log likelihoods
            # -------------------------------------------
            for fk in self.cheops_filekeys:
                if self.fit_phi_gp:
                    self.model_params['llk_cheops'][fk] = self.model_params['gp_rollangles'][fk].marginal("llk_cheops_"+str(fk), observed = self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values - self.model_params['cheops_summodel_x'][fk])
                else:
                    cheops_sigma2s[fk] = self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux_yerr'].values ** 2 + tt.exp(self.model_params['logs_cheops'])**2
                    self.model_params['llk_cheops'][fk] = pm.Potential("llk_cheops_"+str(fk), 
                                                 -0.5 * tt.sum((self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'].values - \
                                                                self.model_params['cheops_summodel_x'][fk]) ** 2 / \
                                                               cheops_sigma2s[fk] + np.log(cheops_sigma2s[fk]))
                                                )
            self.model_params['tess_model_x'] = pm.Deterministic("tess_model_x", xo.LimbDarkLightCurve(self.model_params['u_stars']['tess']).get_light_curve(orbit=orbit, r=self.model_params['r_pl']/109.2,
                                                                                                    t=self.lc_fit['time'].values)*1000/self.model_params['tess_mult'])
            self.model_params['tess_summodel_x'] = pm.Deterministic("tess_summodel_x", tt.sum(self.model_params['tess_model_x'],axis=1))
            if self.fit_gp:
                self.model_params['llk_tess'] = self.model_params['gp_tess'].marginal("llk_tess", observed=self.lc_fit['flux'].values - self.model_params['tess_summodel_x'])
                self.model_params['photgp_model_x'] = pm.Deterministic("photgp_model_x", self.model_params['gp_tess'].predict(self.lc_fit['flux'].values - self.model_params['tess_summodel_x'], t=self.lc_fit['time'].values, return_var=False))
            else:
                tess_sigma2s = self.lc_fit['flux_err'].values ** 2 + tt.exp(self.model_params['logs_tess'])**2
                self.model_params['llk_tess'] = pm.Potential("llk_tess", -0.5 * tt.sum((self.lc_fit['flux'].values - self.model_params['tess_summodel_x']) ** 2 / tess_sigma2s + np.log(tess_sigma2s)))
            
            if hasattr(self,"rvs"):
                rv_logjitter = pm.Normal("rv_logjitter",mu=np.nanmin(self.rvs['yerr'].values)-3,sd=3)
                rv_sigma2 = self.rvs['yerr'].values ** 2 + tt.exp(rv_logjitter)**2
                llk_rv = pm.Potential("llk_rv", -0.5 * tt.sum((self.rvs['y'].values - self.model_params['rv_model_x']) ** 2 / rv_sigma2 + np.log(rv_sigma2)))

            #print(self.model.check_test_point())

            #First try to find best-fit transit stuff:
            if not self.fit_ttvs:
                comb_soln = pmx.optimize(vars=[self.model_params[par] for par in ['t0','P','logror','logs_tess','logs_cheops']])
            else:
                comb_soln = pmx.optimize(vars=[i for i in self.model_params['transit_times']] + \
                                                [self.model_params['logror'],self.model_params['logs_tess'],self.model_params['logs_cheops']] + \
                                                [self.model_params['linear_decorr_dict'][fk][par] for fk in self.cheops_filekeys for par in self.model_params['linear_decorr_dict'][fk]])

            #Now let's do decorrelation seperately:
            comb_soln = pmx.optimize(start=comb_soln,
                                    vars=[self.model_params['linear_decorr_dict'][fk][par] for fk in self.cheops_filekeys for par in self.model_params['linear_decorr_dict'][fk]] + \
                                    [self.model_params['cheops_obs_means'][fk] for fk in self.cheops_filekeys] + [self.model_params['logs_cheops']] + \
                                    [self.model_params['quad_decorr_dict'][fk][par] for fk in self.cheops_filekeys for par in self.model_params['quad_decorr_dict'][fk]])
            
            #More complex transit fit. Also RVs:
            ivars=[self.model_params['b'],self.model_params['logror'],self.model_params['logs_tess'],self.model_params['logs_cheops']]
            if self.fit_ttvs:
                ivars+=[i for i in self.model_params['transit_times']]+[self.model_params['u_stars'][u] for u in self.model_params['u_stars']]
            else:
                ivars+=[self.model_params['t0'],self.model_params['P']]+[self.model_params['u_stars'][u] for u in self.model_params['u_stars']]
            if not self.assume_circ:
                ivars+=[self.model_params['ecc'],self.model_params['omega']]
            if hasattr(self,'rvs'):
                ivars+=[self.model_params['logK'],self.model_params['rv_offsets']]
                if self.npoly_rv>1:
                    ivars+=[self.model_params['rv_trend']]
            comb_soln = pmx.optimize(vars=ivars)

            #Doing everything:
            self.init_soln = pmx.optimize(start=comb_soln)
        
    def sample_model(self,n_tune_steps=1200,n_draws=998,n_cores=3,n_chains=2,**kwargs):
        """Sample model

        Args:
            n_tune_steps (int, optional): Number of steps during tuning. Defaults to 1200.
            n_draws (int, optional): Number of model draws per chain. Defaults to 998.
            n_cores (int, optional): Number of cores. Defaults to 3.
            n_chains (int, optional): Number of chains per core. Defaults to 2.
        """
        with self.model:
            #As we have detrending which is indepdent from each other, we can vastly improve sampling speed by splitting up these as `parameter_groups` in pmx.sample
            #`regularization_steps` also, apparently, helps for big models with lots of correlation
            
            #+[combined_model['d2fd'+par+'2_'+i+'_interval__'] for par in quad_decorr_dict[i]]
            groups=[[self.model_params['cheops_obs_means'][fk]]+[self.model['dfd'+par+'_'+str(fk)] for par in self.model_params['linear_decorr_dict'][fk]] for fk in self.cheops_filekeys]
            if hasattr(self,'rvs'):
                rvgroup=[self.model_params['rv_offsets'],self.model_params['logK']]
                if self.npoly_rv>1:
                    rvgroup+=[self.model_params['rv_trend']]
                groups+=[rvgroup]
            self.trace = pmx.sample(tune=n_tune_steps, draws=n_draws, 
                                    chains=n_chains*n_cores, cores=n_cores, 
                                    start=self.init_soln, target_accept=0.8,
                                    parameter_groups=groups,**kwargs)
        self.save_trace_summary()

    def save_trace_summary(self,returndf=True):
        var_names=[var for var in self.trace.varnames if 'rv_' not in var and 'gp_' not in var and '__' not in var and (np.product(self.trace[var].shape)<6*np.product(self.trace['Rs'].shape) or 'transit_times' in var)]
        df=pm.summary(self.trace,var_names=var_names,round_to=8,
                      stat_funcs={"5%": lambda x: np.percentile(x, 5),"-$1\sigma$": lambda x: np.percentile(x, 15.87),
                                  "median": lambda x: np.percentile(x, 50),"+$1\sigma$": lambda x: np.percentile(x, 84.13),
                                  "95%": lambda x: np.percentile(x, 95)})
        df.to_csv(os.path.join(self.save_file_loc,self.name,self.unq_name+"_model_summary.csv"))
        if returndf:
            return df

    def save_timeseries(self,returnobj=False):
        assert hasattr(self,'trace'), "Must have run an MCMC"
        
        models_out={}
        for src in self.lcs:
            models_out[src]=self.lcs[src].loc[self.lcs[src]['mask']]
            if self.fit_gp:
                if self.bin_oot:
                    #Need to interpolate to the smaller values
                    from scipy.interpolate import interp1d
                    for p in self.percentiles:
                        interpp=interp1d(np.hstack((self.lcs[src]['time'].values[0]-0.1,self.lc_fit['time'],self.lcs[src]['time'].values[0]+0.1)),
                                         np.hstack((0,np.percentile(self.trace['photgp_model_x'],self.percentiles[p],axis=0),0)))
                        models_out[src][src+"_gpmodel_"+p]=interpp(self.lcs[src].loc[self.lcs[src]['mask'],'time'].values)
                elif not self.cut_oot:
                    for p in self.percentiles:
                        models_out[src][src+"_gpmodel_"+p]=np.percentile(self.trace['photgp_model_x'],self.percentiles[p],axis=0)
                elif self.cut_oot:
                    for p in self.percentiles:
                        models_out[src][src+"_gpmodel_"+p] = np.tile(np.nan,len(models_out[src]['time']))
                        models_out[src][src+"_gpmodel_"+p][self.lcs[src]['near_trans']&self.lcs[src]['mask']] = np.percentile(self.trace['photgp_model_x'],self.percentiles[p],axis=0)
            
            for npl,pl in enumerate(self.planets):
                for p in self.percentiles:
                    models_out[src][src+'_'+pl+"model_"+p]=np.zeros(np.sum(self.lcs[src]['mask']))
                    models_out[src].loc[self.lcs[src].loc[self.lcs[src]['mask'],'near_trans'],src+'_'+pl+"model_"+p]=np.nanpercentile(self.trace[src+'_model_x'][:,self.lc_fit['near_trans'],npl],self.percentiles[p],axis=0)

        models_out['cheops']=pd.DataFrame()
        for col in ['time','flux','flux_err','phi','bg','centroid_x','centroid_y','deltaT','xoff','yoff','filekey']:
            models_out['cheops'][col]=np.hstack([self.cheops_lc.loc[self.cheops_fk_mask[fk],col] for fk in self.cheops_filekeys])
        if self.fit_phi_gp:
            for p in self.percentiles:
                models_out['cheops']['che_pred_gp_'+p]=np.hstack([np.nanpercentile(self.trace['gp_rollangle_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])
                models_out['cheops']['che_alldetrend_'+p]=np.hstack([np.nanpercentile(self.trace['gp_rollangle_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])

        for p in self.percentiles:
            models_out['cheops']['che_lindetrend_'+p]=np.hstack([np.nanpercentile(self.trace['cheops_flux_cor_'+fk]+self.trace['gp_rollangle_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])
        for npl,pl in enumerate(self.planets):
            for p in self.percentiles:
                models_out['cheops']['che_'+pl+"model_"+p]=np.hstack([np.nanpercentile(self.trace['cheops_planets_x_'+str(fk)][:,:,npl],self.percentiles[p],axis=0) for fk in self.cheops_filekeys])
        for p in self.percentiles:
            models_out['cheops']['che_allplmodel_'+p]=np.hstack([np.nanpercentile(np.sum(self.trace['cheops_planets_x_'+str(fk)][:,:,:],axis=2),self.percentiles[p],axis=0) for fk in self.cheops_filekeys])

        if hasattr(self,'rvs'):
            models_out['rv']=self.rvs
            models_out['rv_t']=pd.DataFrame()
            for p in self.percentiles:
                models_out['rv_t']["rvt_bothmodel_"+p]=np.nanpercentile(self.trace['rv_model_t'], self.percentiles[p], axis=0)
                
                for npl,pl in enumerate(self.planets):
                    models_out['rv_t']["rvt_"+pl+"model_"+p]=np.nanpercentile(self.trace['vrad_t'][:,:,npl], self.percentiles[p], axis=0)
                    models_out['rv']["rv_"+pl+"model_"+p]=np.nanpercentile(self.trace['vrad_x'][:,:,npl], self.percentiles[p], axis=0)

                if self.npoly_rv>1:
                    models_out['rv_t']["rvt_bkgmodel_"+p]=np.nanpercentile(self.trace['bkg_t'][:,:], self.percentiles[p], axis=0)
                models_out['rv']["rv_bkgmodel_"+p]=np.nanpercentile(self.trace['bkg_x'][:,:], self.percentiles[p], axis=0)
        
        for mod in models_out:
            models_out[mod].to_csv(os.path.join(self.save_file_loc,self.name,self.unq_name+"_"+mod+"_timeseries.csv"))
        if returnobj:
            return models_out

    def print_settings(self):
        settings=""
        for key in self.defaults:
            settings+=key+"\t\t"+str(getattr(self,key))+"\n"
        print(settings)

    def save_model(self):
        if not os.path.exists(os.path.join(self.save_file_loc,self.name)):
            os.mkdir(os.path.join(self.save_file_loc,self.name))
        pickle.dump(self.trace,open(os.path.join(self.save_file_loc,self.name,self.unq_name+"_mcmctrace.pkl"),"rb"))

    def plot_rollangle_gps(self,save=True):
        if not hasattr(self,'trace'):
            yoffset=np.nanmedian([5*np.std(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux']-self.init_soln['cheops_summodel_x_'+fk]) for fk in self.cheops_filekeys])
            for ifk,fk in enumerate(self.cheops_filekeys):
                plt.plot(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'],
                        yoffset*ifk+self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux']-self.init_soln['cheops_summodel_x_'+fk],
                        ".k",markersize=1.33,alpha=0.4)
                plt.plot(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values[self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values.astype(int)],
                         yoffset*ifk+self.init_soln['gp_rollangle_model_phi_'+fk],':')
                         #np.sort(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi']),
                         #yoffset*ifk+self.init_soln['gp_rollangle_model_phi_'+fk][np.argsort(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'])],':') #[self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)]
        else:
            yoffset=np.nanmedian([5*np.std(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux']-np.nanmedian(self.trace['cheops_summodel_x_'+fk],axis=0)) for fk in self.cheops_filekeys])
            for ifk,fk in enumerate(self.cheops_filekeys):

                plt.plot(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'],
                            yoffset*ifk+self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'] - \
                            np.nanmedian(self.trace['cheops_summodel_x_'+fk],axis=0),
                        ".k",markersize=1.33,alpha=0.4)
                plt.fill_between(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values[self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values.astype(int)],
                                 yoffset*ifk+np.nanpercentile(self.trace['gp_rollangle_model_phi_'+fk],5,axis=0),
                                 yoffset*ifk+np.nanpercentile(self.trace['gp_rollangle_model_phi_'+fk],95,axis=0),alpha=0.15,color='C2')
                plt.fill_between(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values[self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values.astype(int)],
                                 yoffset*ifk+np.nanpercentile(self.trace['gp_rollangle_model_phi_'+fk],16,axis=0),
                                 yoffset*ifk+np.nanpercentile(self.trace['gp_rollangle_model_phi_'+fk],84,axis=0),alpha=0.15,color='C2')
                plt.plot(self.cheops_lc.loc[self.cheops_fk_mask[fk],'phi'].values[self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_phi_sorting'].values.astype(int)],
                         yoffset*ifk+np.nanmedian(self.trace['gp_rollangle_model_phi_'+fk],axis=0),'-',alpha=0.6,c='C2') #[self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)]
        plt.xlabel("roll angle [deg]")
        plt.ylabel("Flux [ppt]")
        plt.ylim(-1*yoffset,(len(self.cheops_filekeys))*yoffset)
        if save:
            plt.savefig(os.path.join(self.save_file_loc,self.name,self.unq_name+"_rollangle_plots.png"))

    def plot_cheops(self,save=True):
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

        #plt.plot(cheops_x, ,'.')
        plt.figure(figsize=(6+len(self.cheops_filekeys)*4/3,4))
        self.chmod={}
        self.chplmod={}
        if not hasattr(self,'trace'):
            yoffset=np.nanmedian([5*np.std(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux']-self.init_soln['cheops_summodel_x_'+fk]) for fk in self.cheops_filekeys])
        else:
            yoffset=np.nanmedian([5*np.std(self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux']-np.nanmedian(self.trace['cheops_summodel_x_'+fk],axis=0)) for fk in self.cheops_filekeys])

        for n,fk in enumerate(self.cheops_filekeys):
            plt.subplot(2,len(self.cheops_filekeys),1+n)
            plt.plot(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'], 
                     self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'], '.k',markersize=1.2,alpha=0.4,zorder=1)
            binlc = bin_lc_segment(np.column_stack((self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'], 
                                                    self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux'],
                                                    self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux_err'])),1/96)
            plt.errorbar(binlc[:,0], binlc[:,1], yerr=binlc[:,2], fmt='.',markersize=8,zorder=2,alpha=0.75)
            if not hasattr(self,'trace'):
                self.chmod[fk]=[None,None,None,None,None]
                self.chmod[fk][2]=self.init_soln['cheops_flux_cor_'+str(fk)]
                if self.fit_phi_gp:
                    self.chmod[fk][2]+=self.init_soln['gp_rollangle_model_phi_'+str(fk)][self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)]
            else:
                if self.fit_phi_gp:
                    self.chmod[fk]=np.nanpercentile(self.trace['cheops_flux_cor_'+str(fk)]+self.trace['gp_rollangle_model_phi_'+str(fk)][:,self.cheops_lc.loc[self.cheops_fk_mask[fk],'mask_time_sorting'].values.astype(int)],list(self.percentiles.values()),axis=0)
                else:
                    self.chmod[fk]=np.nanpercentile(self.trace['cheops_flux_cor_'+str(fk)],list(self.percentiles.values()),axis=0)
            plt.plot(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'], yoffset+self.chmod[fk][2],'.',markersize=1.4,c='C2',alpha=0.5,zorder=5)
            if hasattr(self,'trace'):
                plt.fill_between(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'], 
                                 yoffset+self.chmod[fk][0], yoffset+self.chmod[fk][4],color='C2',alpha=0.15,zorder=3)
                plt.fill_between(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'], 
                                 yoffset+self.chmod[fk][1], yoffset+self.chmod[fk][3],color='C2',alpha=0.15,zorder=4)
            lims=np.nanpercentile(binlc[:,1],[1,99])
            plt.ylim(lims[0]-0.66*yoffset,lims[1]+1.5*yoffset)

            if n>0:
                plt.gca().set_yticklabels([])
            plt.gca().set_xticklabels([])

            plt.subplot(2,len(self.cheops_filekeys),1+len(self.cheops_filekeys)+n)
            plt.plot(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'], 
                     self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux']-self.chmod[fk][2],
                    '.k',alpha=0.4,markersize=1.2,zorder=1)
            binlc = bin_lc_segment(np.column_stack((self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'], 
                                                    self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux']-self.chmod[fk][2],
                                                    self.cheops_lc.loc[self.cheops_fk_mask[fk],'flux_err'])),1/96)
            plt.errorbar(binlc[:,0], binlc[:,1], yerr=binlc[:,2], fmt='.',markersize=8,zorder=2,alpha=0.75)
            
            self.chplmod[fk]={}
            for npl,pl in enumerate(self.planets):
                if not hasattr(self,'trace'):
                    self.chplmod[fk][pl]=[None,None,self.init_soln['cheops_planets_x_'+str(fk)][:,npl],None,None]
                else:
                    self.chplmod[fk][pl]=np.nanpercentile(self.trace['cheops_planets_x_'+str(fk)][:,:,npl],list(self.percentiles.values()),axis=0)
                if np.any(self.chplmod[fk][pl][2]<0):
                    plt.plot(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'],self.chplmod[fk][pl][2],
                                '--',c='C'+str(3+npl),linewidth=3,alpha=0.6,zorder=10)
                    if hasattr(self,'trace'):
                        plt.fill_between(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'], 
                                        self.chplmod[fk][pl][0],self.chplmod[fk][pl][4],color='C'+str(3+npl),alpha=0.15,zorder=6)
                        plt.fill_between(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'], 
                                        self.chplmod[fk][pl][1],self.chplmod[fk][pl][3],color='C'+str(3+npl),alpha=0.15,zorder=7)
            if np.sum([np.any(self.chplmod[fk][pl][2]) for pl in self.planets])>1:
                #Multiple transits together - we need a summed model
                plt.plot(self.cheops_lc.loc[self.cheops_fk_mask[fk],'time'],np.sum([self.chplmod[fk][pl][2] for pl in self.planets],axis=0),
                        '--k',linewidth=3,alpha=0.4)
            plt.ylim(np.min([np.min(self.chplmod[fk][pl][2]) for pl in self.planets])-yoffset*0.8,yoffset*0.8)
            plt.xlabel("time [BJD]")
            if n>0:
                plt.gca().set_yticklabels([])
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.subplot(2,len(self.cheops_filekeys),1)
        plt.ylabel("flux [ppt]")
        plt.subplot(2,len(self.cheops_filekeys),len(self.cheops_filekeys)+1)
        plt.ylabel("flux [ppt]")

        plt.subplots_adjust(wspace=0.05,hspace=0.05)
        if save:
            plt.savefig(os.path.join(self.save_file_loc,self.name,self.unq_name+"_cheops_plots.png"),dpi=350)

    def plot_tess(self,save=True):        
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter    
        src='tess'
        #Finding sector gaps
        diffs=np.diff(self.lcs[src].loc[self.lcs[src]['mask'],'time'])
        jumps=diffs>0.66
        total_obs_time = np.sum(diffs[diffs<0.25])
        likely_sects=np.round(total_obs_time/24.5)
        gap_starts=np.hstack([self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[0]-0.1,self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[:-1][jumps],self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[-1]])
        gap_ends=np.hstack([self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[0],self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[1:][jumps],self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[-1]+0.1])
        sectinfo={1:{'start':self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[0]}}
        nsect=1
        while nsect<(likely_sects+1):
            new_gap_ix=np.argmin(abs(gap_starts-(sectinfo[nsect]['start']+26)))
            sectinfo[nsect]['end']=gap_starts[new_gap_ix]
            sectinfo[nsect]['dur']=sectinfo[nsect]['end']-sectinfo[nsect]['start']
            if gap_ends[new_gap_ix]<self.lcs[src].loc[self.lcs[src]['mask'],'time'].values[-1]:
                sectinfo[nsect+1]={'start':gap_ends[new_gap_ix]}
            else:
                break
            nsect+=1
        #print(likely_sects+1,nsect,sectinfo)

        plt.figure(figsize=(11,9))
        for ns in sectinfo:
            sect_ix = (self.lcs[src]['time']>=sectinfo[ns]['start'])&(self.lcs[src]['time']<=sectinfo[ns]['end'])&self.lcs[src]['mask']
            sect_fit_ix = (self.lc_fit['time']>=sectinfo[ns]['start'])&(self.lc_fit['time']<=sectinfo[ns]['end'])

            plt.subplot(len(sectinfo),1,ns)
            plt.plot(self.lcs[src].loc[sect_ix,'time'],self.lcs[src].loc[sect_ix,'flux'],'.k',markersize=1.0,alpha=0.4,zorder=1)
            binsect=bin_lc_segment(np.column_stack((self.lcs[src].loc[sect_ix,'time'],
                                                    self.lcs[src].loc[sect_ix,'flux'],
                                                    self.lcs[src].loc[sect_ix,'flux_err'])),1/48)
            plt.errorbar(binsect[:,0],binsect[:,1],yerr=binsect[:,2],fmt='.',ecolor="#aaa",alpha=0.6,zorder=2)
            
            if self.fit_gp:
                #Plotting GP
                if hasattr(self,'trace'):
                    #Using MCMC
                    bf_gp = np.nanpercentile(self.trace['photgp_model_x'][:,sect_fit_ix], list(self.percentiles.values()), axis=0)
                    plt.fill_between(self.lc_fit.loc[sect_fit_ix,'time'],bf_gp[0],bf_gp[4],color='C5',alpha=0.15,zorder=3)
                    plt.fill_between(self.lc_fit.loc[sect_fit_ix,'time'],bf_gp[1],bf_gp[3],color='C5',alpha=0.15,zorder=4)
                    plt.plot(self.lc_fit.loc[sect_fit_ix,'time'],bf_gp[2],linewidth=2,color='C5',alpha=0.75,zorder=5)
                    fluxmod=bf_gp[2]
                else:
                    #Using initial soln
                    fluxmod=self.init_soln['photgp_model_x'][sect_fit_ix]
                    plt.plot(self.lc_fit.loc[sect_fit_ix,'time'],fluxmod,linewidth=2,color='C5',alpha=0.75,zorder=5)

            elif not self.fit_gp and self.fit_flat:
                #Plotting kepler spline
                fluxmod=self.lcs[src].loc[sect_ix,'spline'].values
                fitfluxmod=self.lc_fit.loc[sect_fit_ix,'spline'].values
                plt.plot(self.lcs[src].loc[sect_ix,'time'], fluxmod, 
                         linewidth=2,color='C5',alpha=0.75)
            else:
                fluxmod = np.zeros(np.sum(sect_ix))
            
            if hasattr(self,'trace'):
                if self.cut_oot and self.fit_flat:
                    pl_mod=np.zeros((5,np.sum(self.lcs[src].loc[sect_ix,'mask'])))
                    pl_mod[:,self.lcs[src].loc[sect_ix&self.lcs[src]['mask'],'near_trans']]=np.nanpercentile(self.trace['tess_summodel_x'][:,sect_fit_ix], list(self.percentiles.values()), axis=0)
                else:
                    pl_mod = np.nanpercentile(self.trace['tess_summodel_x'][:,sect_fit_ix], list(self.percentiles.values()), axis=0)
                plt.fill_between(self.lcs[src].loc[sect_ix,'time'],fluxmod+pl_mod[0],fluxmod+pl_mod[4],color='C2',alpha=0.15,zorder=6)
                plt.fill_between(self.lcs[src].loc[sect_ix,'time'],fluxmod+pl_mod[1],fluxmod+pl_mod[3],color='C2',alpha=0.15,zorder=7)
                plt.plot(self.lcs[src].loc[sect_ix,'time'],fluxmod+pl_mod[2],linewidth=2,color='C2',alpha=0.75,zorder=8)
                transmin=np.min(pl_mod[0])

            else:
                if self.cut_oot and self.fit_flat:
                    pl_mod=np.tile(np.nan,np.sum(self.lcs[src].loc[sect_ix,'mask']))
                    pl_mod[self.lcs[src].loc[sect_ix&self.lcs[src]['mask'],'near_trans']]=fitfluxmod+self.init_soln['tess_summodel_x'][sect_fit_ix]
                    pl_mod[~self.lcs[src].loc[sect_ix&self.lcs[src]['mask'],'near_trans']]=fluxmod[[~self.lcs[src].loc[sect_ix&self.lcs[src]['mask'],'near_trans']]]
                    plt.plot(self.lcs[src].loc[sect_ix,'time'],pl_mod,linewidth=2,color='C2',alpha=0.75,zorder=8)
                else:
                    plt.plot(self.lc_fit.loc[sect_fit_ix,'time'],fluxmod,linewidth=2,color='C2',alpha=0.75,zorder=8)

                transmin=np.min(self.init_soln['tess_summodel_x'][sect_fit_ix])


            plt.xlim(sectinfo[ns]['start']-1,sectinfo[ns]['end']+1)
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
            stdev=np.std(self.lcs[src].loc[sect_ix,'flux'])
            plt.ylim(transmin-1.66*stdev,1.66*stdev)

            if ns==len(sectinfo):
                plt.xlabel("BJD")
            plt.ylabel("Flux [ppt]")
        
        if save:
            plt.savefig(os.path.join(self.save_file_loc,self.name,self.unq_name+"_tess_plot.png"),dpi=350)


    def plot_rvs(self,save=True):
        plt.figure(figsize=(8,6))
        plt.subplot(2,1,1)
        if hasattr(self,'trace'):
            rvt_mods=np.nanpercentile(self.trace['rv_model_t'],list(self.percentiles.values()),axis=0)
            plt.fill_between(self.rv_t,rvt_mods[0],rvt_mods[4],color='C4',alpha=0.15)
            plt.fill_between(self.rv_t,rvt_mods[1],rvt_mods[3],color='C4',alpha=0.15)
            plt.plot(self.rv_t,rvt_mods[2],c='C4',alpha=0.66)
            if self.npoly_rv>1:
                plt.plot(self.rv_t,np.nanmedian(self.trace['bkg_t'],axis=1),c='C2',alpha=0.3,linewidth=3)
            
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
                plt.plot(np.sort(rv_phase_t),self.init_soln['vrad_t'][np.argsort(rv_phase_t)][:,n],c='C1')

            else:
                other_pls_bg=np.nanmedian(self.trace['bkg_x']+np.sum([self.trace['vrad_x'][:,:,inpl] for inpl in range(len(self.planets)) if inpl!=n],axis=0),axis=0)
                for isc,sc in enumerate(pd.unique(self.rvs['scope'])):
                    plt.errorbar(rv_phase_x[self.rvs['scope'].values==sc],
                                self.rvs.loc[self.rvs['scope']==sc,'y'] - other_pls_bg[self.rvs['scope']==sc],
                                yerr=self.rvs.loc[self.rvs['scope']==sc,'yerr'],
                                fmt='.',ecolor='#aaaaaa',zorder=10,label=sc)
                rvt_mods=np.nanpercentile(self.trace['vrad_t'][:,np.argsort(rv_phase_t),n],list(self.percentiles.values()),axis=0)
                plt.fill_between(np.sort(rv_phase_t),rvt_mods[0],rvt_mods[4],color='C1',alpha=0.15)
                plt.fill_between(np.sort(rv_phase_t),rvt_mods[1],rvt_mods[3],color='C1',alpha=0.15)
                plt.plot(np.sort(rv_phase_t),rvt_mods[2],c='C1',alpha=0.65)
                        
            if n==0:
                plt.ylabel("RV [ms]")
            else:
                plt.gca().set_yticklabels([])
            plt.xlabel("Time from t0 [d]")
            
        if save:
            plt.savefig(os.path.join(self.save_file_loc,self.name,self.unq_name+"_rv_plots.png"),dpi=350)
