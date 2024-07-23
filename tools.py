import numpy as np
import pandas as pd
import os
import warnings
import httplib2
import importlib

warnings.filterwarnings("ignore")

chexo_tablepath = os.path.join(os.path.dirname(__file__),'tables')
if os.environ.get('CHEXOPATH') is None:
    chexo_savepath = os.path.join(os.path.dirname( __file__ ),'data')
else:
    chexo_savepath = os.environ.get('CHEXOPATH')

id_dic={'TESS':'TIC','tess':'TIC','Kepler':'KIC','kepler':'KIC','KEPLER':'KIC',
        'K2':'EPIC','k2':'EPIC','CoRoT':'CID','corot':'CID'}
lc_dic={'tess':'ts','kepler':'k1','k2':'k2','corot':'co','cheops':'ch'}


def vals_to_latex(vals):
    #Function to turn -1,0, and +1 sigma values into round latex strings for a table
    try:
        roundval=int(np.min([-1*np.floor(np.log10(abs(vals[1]-vals[0])))+1,-1*np.floor(np.log10(abs(vals[2]-vals[1])))+1]))
        errs=[vals[2]-vals[1],vals[1]-vals[0]]
        if np.round(errs[0],roundval-1)==np.round(errs[1],roundval-1):
            #Errors effectively the same...
            if roundval<0:
                return " $ "+str(int(np.round(vals[1],roundval)))+" \pm "+str(int(np.round(np.average(errs),roundval)))+" $ "
            else:
                return " $ "+str(np.round(vals[1],roundval))+" \pm "+str(np.round(np.average(errs),roundval))+" $ "
        else:
            if roundval<0:
                return " $ "+str(int(np.round(vals[1],roundval)))+"^{+"+str(int(np.round(errs[0],roundval)))+"}_{-"+str(int(np.round(errs[1],roundval)))+"} $ "
            else:
                return " $ "+str(np.round(vals[1],roundval))+"^{+"+str(np.round(errs[0],roundval))+"}_{-"+str(np.round(errs[1],roundval))+"} $ "
    except:
        return " - "

def vals_to_short(vals,roundval=None):
    #Function to turn -1,0, and +1 sigma values into round latex strings for a table
    try:
        if roundval is None:
            roundval=int(np.min([-1*np.floor(np.log10(abs(vals[1]-vals[0])))+1,-1*np.floor(np.log10(abs(vals[2]-vals[1])))+1]))-1
        return " $ "+str(np.round(vals[1],roundval))+" $ "
    except:
        return " - "
    
    
def vals_to_overleaf(name,vals,include_short=True):
    if len(vals)==2 and vals[1]<0.5*vals[0]:
        vals=[vals[0]-vals[1],vals[0],vals[0]+vals[1]]
    
    replace_vals = {'_':'','[':'',']':'','/':'div','-':'minus','0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}
    for symbol, text in replace_vals.items():
        name = name.replace(symbol, text)
    st = "\\newcommand{\\T"+name+"}{"+vals_to_latex(vals)+"}\n"
    if include_short:
        st+="\\newcommand{\\T"+name+"short}{"+vals_to_short(vals)+"}\n"
    return st

def cut_anom_diff(flux,thresh=4.2):
    #Uses differences between points to establish anomalies.
    #Only removes single points with differences to both neighbouring points greater than threshold above median difference (ie ~rms)
    #Fast: 0.05s for 1 million-point array.
    #Must be nan-cut first
    diffarr=np.vstack((np.diff(flux[1:]),np.diff(flux[:-1])))
    diffarr/=np.median(abs(diffarr[0,:]))
    #Adding a test for the first and last points if they are >3*thresh from median RMS wrt next two points.
    anoms=np.hstack((abs(flux[0]-np.median(flux[1:3]))<(np.median(abs(diffarr[0,:]))*thresh*5),
                     ((diffarr[0,:]*diffarr[1,:])>0)+(abs(diffarr[0,:])<thresh)+(abs(diffarr[1,:])<thresh),
                     abs(flux[-1]-np.median(flux[-3:-1]))<(np.median(abs(diffarr[0,:]))*thresh*5)))
    return anoms

def get_field_times():
    
    from astropy.time import Time
    yr_now=int(np.ceil(Time.now().jyear-2017.25))
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tables","all_campaign_data_yr"+str(yr_now)+".csv")):
        from datetime import datetime

        kep_times=pd.read_table(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tables",'k1_qs.txt'),delimiter="\t")
        all_fields=pd.DataFrame({'field':kep_times['Q'].values,
                                'field_string':np.array(['Q'+str(q) for q in kep_times['Q'].values]),
                                'mission':np.tile("kepler",len(kep_times)),
                                'jd_start':np.array([Time(datetime.strptime(s, '%Y %b %d')).jd for s in kep_times['Start']]),
                                'jd_end':np.array([Time(datetime.strptime(s, '%Y %b %d')).jd for s in kep_times['Stop']]),
                                })
        
        k2_times=pd.read_table(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tables",'k2_camps.txt'),delimiter="\t")
        k2_times['mission']=np.tile("k2",len(k2_times))
        k2_times['jd_start']=np.array([Time(datetime.strptime(s, '%Y %b %d')).jd for s in k2_times['Observation Start Date']])
        k2_times['jd_end']=np.array([Time(datetime.strptime(s, '%Y %b %d')).jd for s in k2_times['Observation Stop Date']])
        k2_times['field_string']=k2_times['Campaign']
        k2_times['field']=np.array([float(s[1:].replace('a','.1').replace('b','.2')) for s in k2_times['Campaign']])
        
        all_fields=all_fields.append(k2_times.loc[:,['field','field_string','mission','jd_start','jd_end']])

        
        for yr in np.arange(1,yr_now):
            sect_times=pd.read_html("https://tess.mit.edu/tess-year-"+str(yr)+"-observations/")[0]
            sect_times['mission']=np.tile("tess",len(sect_times))
            sect_times['jd_start']=np.array([Time(datetime.strptime(s.split('-')[0], '%m/%d/%y')).jd for s in sect_times['Dates']])
            sect_times['jd_end']=np.array([Time(datetime.strptime(s.split('-')[1], '%m/%d/%y')).jd for s in sect_times['Dates']])
            sect_times['field']=sect_times['Sector']
            sect_times['field_string']=np.array(["S"+str(f) for f in sect_times['Sector']])
            all_fields=all_fields.append(sect_times.loc[:,['field','field_string','mission','jd_start','jd_end']])

        all_fields=all_fields.set_index(np.arange(len(all_fields)))
        all_fields.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tables","all_campaign_data_yr"+str(yr_now)+".csv"))
        return all_fields
    else:
        return pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tables","all_campaign_data_yr"+str(yr_now)+".csv"),index_col=0)

def weighted_avg_and_std(values, errs, masknans=True, axis=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if len(values)>1:
        average = np.average(values, weights=1/errs**2,axis=axis)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, weights=1/errs**2,axis=axis)
        binsize_adj = np.sqrt(len(values)) if axis is None else np.sqrt(values.shape[axis])
        return [average, np.sqrt(variance)/binsize_adj]
    elif len(values)==1:
        return [values[0], errs[0]]
    else:
        return [np.nan, np.nan]

def bin_lc_segment(lc_segment, binsize, return_digi=False):
    if len(lc_segment)>0:
        binnedx = np.arange(np.min(lc_segment[:,0])-0.5*binsize,np.max(lc_segment[:,0])+0.5*binsize,binsize)
        return binlc_given_x(lc_segment,binnedx,return_digi)
    else:
        return lc_segment
    
def binlc_given_x(lc_segment,binnedx, return_digi=False):
    digi=np.digitize(lc_segment[:,0],binnedx)
    binlc=np.vstack([[[np.nanmedian(lc_segment[digi==d,0])]+\
                                weighted_avg_and_std(lc_segment[digi==d,1],lc_segment[digi==d,2])] for d in np.unique(digi)])
    if return_digi:
        return binlc, digi
    else:
        return binlc
    

def check_past_PIPE_params(folder):
    #Checking to see what PIPE optimised parameters we can find
    import glob
    import os
    
    #Getting all filekey folders for a given object:
    past_fks=glob.glob(os.path.join(folder,"PR*_*_V0?00"))
    print(past_fks)
    past_fk_params={}
    allkeys={'im':[],'sa':[]}
    #Looping over all filekeys:
    for fk in past_fks:
        ifk=os.path.basename(os.path.normpath(fk))
        pipefold=os.path.join(folder,fk,"Outdata","00000")
        if os.path.exists(os.path.join(pipefold,"logfile.txt")):
            past_fk_params[ifk]={'im':{},'sa':{}}
            #Opening the logfile for each:
            with open(os.path.join(pipefold,"logfile.txt")) as flog:
                #Searching for and saving lines with format: Fri Apr 19 13:20:48 2024 [4.45 min] k: 7, r: 25, bg: 1, d: 1, s: 1, mad=311.78 [im]
                paramline=[line for line in flog if np.array(['k: ' in line, 'r: ' in line, 'bg: ' in line, 'd: ' in line, 's: ' in line]).sum()>3]
                #Adding params to a dict. Spliting into "imagette" and "subarray"
                for niter, pl in enumerate(paramline):
                    #This natively overwrites the previous iteration...
                    strip_pl=pl.replace(' ','').replace('\n','').replace('mad=','mad:')
                    #print(strip_pl,strip_pl.split(']')[1][:-4].split(','))
                    #Wed Jul 17 21:12:47 2024 [50.42 min] k: 2, r: 40, bg: 1, d: 0, s: 1, mad=1122.07 [sa]
                    if strip_pl[-4:]=="[sa]":
                        parse_dict={i.split(':')[0]:i.split(':')[1] for i in strip_pl[strip_pl.find('k:'):-4].split(',')}
                        parse_dict.update({'niter':niter,'filekey':ifk})
                        past_fk_params[ifk]['sa'][niter]=parse_dict
                        print(strip_pl,parse_dict)
                        allkeys['sa']+=list(parse_dict.keys())
                    elif strip_pl[-4:]=="[im]":
                        parse_dict={i.split(':')[0]:i.split(':')[1] for i in strip_pl[strip_pl.find('k:'):-4].split(',')}
                        parse_dict.update({'niter':niter,'filekey':ifk})
                        past_fk_params[ifk]['im'][niter]=parse_dict
                        allkeys['im']+=list(parse_dict.keys())
        if past_fk_params[ifk]['sa']!={}:
            past_fk_params[ifk]['sa']=pd.DataFrame(past_fk_params[ifk]['sa']).T
            for col in past_fk_params[ifk]['sa'].columns:
                if col!='filekey':
                    past_fk_params[ifk]['sa'][col]=pd.to_numeric(past_fk_params[ifk]['sa'][col])
            past_fk_params[ifk]['sa']=past_fk_params[ifk]['sa'].sort_values('mad')
            sa_params=pd.concat([past_fk_params[ifk]['sa'] for ifk in past_fk_params])
            #Creating mad normalised to best MAD for a given filekey:
            # sa_params['norm_mad']=sa_params['mad']
            # for ifk in past_fk_params:
            #     sa_params.loc[sa_params['filekey']==ifk,'norm_mad']=sa_params.loc[sa_params['filekey']==ifk,'mad']/np.nanmin(sa_params.loc[sa_params['filekey']==ifk,'mad'])
        else:
             past_fk_params[ifk]['sa']=None
        if past_fk_params[ifk]['im']!={}:
            past_fk_params[ifk]['im']=pd.DataFrame(past_fk_params[ifk]['im']).T
            for col in past_fk_params[ifk]['im'].columns:
                if col!='filekey':
                    past_fk_params[ifk]['im'][col]=pd.to_numeric(past_fk_params[ifk]['im'][col])
            past_fk_params[ifk]['im']=past_fk_params[ifk]['im'].sort_values(['mad','niter'],ascending=[True,False])
            im_params=pd.concat([past_fk_params[ifk]['im'] for ifk in past_fk_params])
            #Creating mad normalised to best MAD for a given filekey:
            # im_params['norm_mad']=im_params['mad']
            # for ifk in past_fk_params:
            #     im_params.loc[im_params['filekey']==ifk,'norm_mad']=im_params.loc[im_params['filekey']==ifk,'mad']/np.nanmin(im_params.loc[im_params['filekey']==ifk,'mad'])
        else:
            past_fk_params[ifk]['im']=None
    
    #Now we need to pick the most common/best optimisation approach
    pkey_modes={'im':{},'sa':{}}
    
    print(past_fk_params)
    if np.sum([past_fk_params[fk]['sa'] is not None for fk in past_fk_params])>0:
        pkey_modes['sa']['exists']=True
        #Simply taking the parameters for the minimum MAD found across all filekeys:
        for col in ['k', 'r', 'bg', 'd', 's']:
            pkey_modes['sa'][col]=sa_params.iloc[np.argmin(sa_params['mad'].values)][col]
    else:
        pkey_modes['sa']['exists']=False

    if np.sum([past_fk_params[fk]['im'] is not None for fk in past_fk_params])>0:
        pkey_modes['im']['exists']=True
        #Simply taking the parameters for the minimum MAD found across all filekeys:
        for col in ['k', 'r', 'bg', 'd', 's']:
            pkey_modes['im'][col]=im_params.iloc[np.argmin(im_params['mad'].values)][col]
    else:
        pkey_modes['im']['exists']=False

        # #Step1 find unique values for each of the 5 key parameters across all filekeys (k, r, bg, d, s)
        # unique_vals={}
        # mad_trends={}
        # for col in ['k', 'r', 'bg', 'd', 's']:
        #     unique_vals[col]=np.unique([np.unique(past_fk_params[fk]['sa'][col].values) for fk in past_fk_params])
        #     #Finding the  mad vs this parameter
        #     mad_trends[col]=scipy.stats.linregree(sa_params.loc[col],sa_params.loc['norm_mad'])
        # sorted_by_rv=list(mad_trends.leys())[np.argsort([mad_trends[col].rvalue for col in mad_trends])]

        # ikeys, ivalues = zip(*unique_vals.items())

        ##### THIS DOESNT WORK BECAUSE THERE ARE NO PERMUTATIONS IN COMMON BETWEEN PIPE RUNS #####
        # import itertools
        # permutations_dicts = [dict(zip(ikeys, v)) for v in itertools.product(*ivalues)]
        # mads={} #Storing median absolute deviations
        # for n_permute,i_permute in enumerate(permutations_dicts):
        #     ixs={fk:np.column_stack([past_fk_params[fk]['sa'][i_key].values==i_permute[i_key] for i_key in i_permute]) for fk in past_fk_params}
        #     print(ixs)
        #     ix_vals=list(ixs.values())
        #     print(i_permute,np.sum(ix_vals))
        #     if np.all(ix_vals):
        #         #All filekeys have this permutation. Checking the minimum
        #         mads[n_permute]=np.nanmedian([np.min(past_fk_params[fk]['sa'].loc[ixs[fk],'mad']) for fk in past_fk_params])
        #     elif np.sum(ix_vals)>2 and np.sum(~ix_vals)<2:
        #         #Most of the optimisations have this permutation - checking anyway
        #         mads[n_permute]=np.nanmedian([np.min(past_fk_params[fk]['sa'].loc[ixs[fk],'mad']) for fk in past_fk_params if np.sum(ixs[fk])>0])
        # #Now taking the lowest average MAD to be the "default" PIPE optimisation:
        # min_mad = min(mads.values())
        # pkey_modes['sa'] = permutations_dicts[[k for k in mads if mads[k] == min_mad][0]]

        ##### NEW TECHNIQUE: FIND MOST CLEARLY USEFUL PARAMETER UNIVERSALLY

        # print(allkeys['sa'])
        # for pkey in np.unique(allkeys['sa']):
        #     all_pars=[past_fk_params[fk]['sa'][pkey] for fk in past_fk_params if pkey in past_fk_params[fk]['sa']]
        #     pkey_modes['sa'][pkey]=max(set(all_pars), key=all_pars.count)
    # else:
    #     pkey_modes['sa']['exists']=False
    
    # if np.sum([len(past_fk_params[fk]['im']) for fk in past_fk_params])>0:
    #     pkey_modes['im']['exists']=True
    #     for pkey in np.unique(allkeys['im']):
    #         all_pars=[past_fk_params[fk]['im'][pkey] for fk in past_fk_params if pkey in past_fk_params[fk]['im']]
    #         pkey_modes['im'][pkey]=max(set(all_pars), key=all_pars.count)
    # else:
    #     pkey_modes['im']['exists']=False
    
    return pkey_modes

#Copied from Andrew Vanderburg:
    
def robust_mean(y, cut):
    """Computes a robust mean estimate in the presence of outliers.
    Args:
        y: 1D numpy array. Assumed to be normally distributed with outliers.
        cut: Points more than this number of standard deviations from the median are
                ignored.
    Returns:
        mean: A robust estimate of the mean of y.
        mean_stddev: The standard deviation of the mean.
        mask: Boolean array with the same length as y. Values corresponding to
                outliers in y are False. All other values are True.
    """
    # First, make a robust estimate of the standard deviation of y, assuming y is
    # normally distributed. The conversion factor of 1.4826 takes the median
    # absolute deviation to the standard deviation of a normal distribution.
    # See, e.g. https://www.mathworks.com/help/stats/mad.html.
    absdev = np.abs(y - np.median(y))
    sigma = 1.4826 * np.median(absdev)

    # If the previous estimate of the standard deviation using the median absolute
    # deviation is zero, fall back to a robust estimate using the mean absolute
    # deviation. This estimator has a different conversion factor of 1.253.
    # See, e.g. https://www.mathworks.com/help/stats/mad.html.
    if sigma < 1.0e-24:
        sigma = 1.253 * np.mean(absdev)

    # Identify outliers using our estimate of the standard deviation of y.
    mask = absdev <= cut * sigma

    # Now, recompute the standard deviation, using the sample standard deviation
    # of non-outlier points.
    sigma = np.std(y[mask])

    # Compensate the estimate of sigma due to trimming away outliers. The
    # following formula is an approximation, see
    # http://w.astro.berkeley.edu/~johnjohn/idlprocs/robust_mean.pro.
    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

    # Identify outliers using our second estimate of the standard deviation of y.
    mask = absdev <= cut * sigma

    # Now, recompute the standard deviation, using the sample standard deviation
    # with non-outlier points.
    sigma = np.std(y[mask])

    # Compensate the estimate of sigma due to trimming away outliers.
    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

    # Final estimate is the sample mean with outliers removed.
    mean = np.mean(y[mask])
    mean_stddev = sigma / np.sqrt(len(y) - 1.0)

    return mean, mean_stddev, mask

def kepler_spline(time, flux, flux_mask = None, transit_mask = None, bk_space=1.25, maxiter=5, outlier_cut=3, polydegree=3, reflect=False):
    """Computes a best-fit spline curve for a light curve segment.
    The spline is fit using an iterative process to remove outliers that may cause
    the spline to be "pulled" by discrepent points. In each iteration the spline
    is fit, and if there are any points where the absolute deviation from the
    median residual is at least 3*sigma (where sigma is a robust estimate of the
    standard deviation of the residuals), those points are removed and the spline
    is re-fit.
    Args:
        time: Numpy array; the time values of the light curve.
        flux: Numpy array; the flux (brightness) values of the light curve.
        flux_mask (np.ndarray of booleans, optional): Numpy array where False values refer to anomalies. Defaults to None
        transit_mask (np.ndarray of booleans, optional): Numpy array where False values refer to in-transit points
        bk_space: Spline break point spacing in time units.
        maxiter: Maximum number of attempts to fit the spline after removing badly
                fit points.
        outlier_cut: The maximum number of standard deviations from the median
                spline residual before a point is considered an outlier.
        polydegree: Polynomial degre. Defaults to 3
        reflect: Whether to perform spline fit using reflection of final time...
    Returns:
        spline: The values of the fitted spline corresponding to the input time
                values.
        mask: Boolean mask indicating the points used to fit the final spline.
    Raises:
        InsufficientPointsError: If there were insufficient points (after removing
                outliers) for spline fitting.
        SplineError: If the spline could not be fit, for example if the breakpoint
                spacing is too small.
    """
    from scipy.signal import savgol_filter
    from scipy.interpolate import LSQUnivariateSpline, BSpline,splev,splrep

    region_starts=np.sort(time)[1+np.hstack((-1,np.where(np.diff(np.sort(time))>bk_space)[0]))]
    region_ends  =np.sort(time)[np.hstack((np.where(np.diff(np.sort(time))>bk_space)[0],len(time)-1))]
    if flux_mask is None:
        flux_mask=~(np.isnan(flux)|np.isnan(time))
    spline = []
    mask = []
    for n in range(len(region_starts)):
        ix=(time>=region_starts[n])*(time<=region_ends[n])
        
        # Mask indicating the points used to fit the spline.
        imask = flux_mask[ix]
        imask = imask*transit_mask[ix] if transit_mask is not None else imask

        if np.sum(imask)>4:
            # Rescale time into [0, 1].
            #t_min = np.min(time[ix])
            #t_max = np.max(time[ix])
            #n_interior_knots = int(np.round((t_max-t_min)/bk_space))
            #qs = np.linspace(0, 1, n_interior_knots+2)[1:-1]
            qs = np.arange(np.min(time[ix]), np.max(time[ix]), bk_space)[1:-1]
            qs = (qs-np.min(time[ix]))/(np.max(time[ix])-np.min(time[ix]))

            # Values of the best fitting spline evaluated at the time points.
            ispline = None


            if reflect and (region_ends[n]-region_starts[n])>1.8*bk_space:
                incad=np.nanmedian(np.diff(time[ix]))
                xx=[np.arange(region_starts[n]-bk_space*0.9,region_starts[n]-incad,incad),
                    np.arange(region_ends[n]+incad,region_ends[n]+bk_space*0.9,incad)]
                # Adding the lc, plus a reflected region either side of each part.
                # Also adding a boolean array to show where the reflected parts are
                # Also including zeros to make sure flux spline does not wander far from lightcurve
                itime=np.hstack((time[ix][0]-1.35*bk_space, time[ix][0]-1.3*bk_space, xx[0], time[ix], xx[1], time[ix][-1]+1.3*bk_space,time[ix][-1]+1.35*bk_space))
                imask=np.hstack((True, True, imask[:len(xx[0])][::-1], imask, imask[-1*len(xx[1]):][::-1], True, True))
                iflux=np.hstack((0.0,0.0,flux[ix][:len(xx[0])][::-1], flux[ix], flux[ix][-1*len(xx[1]):][::-1], 0.0, 0.0 ))
                ibool=np.hstack((np.zeros(len(xx[0])+2),np.ones(np.sum(ix)),np.zeros(len(xx[1])+2))).astype(bool)

            else:
                itime=time[ix]
                iflux=flux[ix]
                ibool=np.tile(True,len(itime))

            for ni in range(maxiter):
                if ispline is not None:
                    # Choose points where the absolute deviation from the median residual is
                    # less than outlier_cut*sigma, where sigma is a robust estimate of the
                    # standard deviation of the residuals from the previous spline.
                    residuals = iflux - ispline

                    new_imask = robust_mean(residuals[imask], cut=outlier_cut)[2]
                    # in ",ni,"th run")
                    if np.all(new_imask):
                        break    # Spline converged.
                    #Otherwise we're adding the updated mask to the mask
                    imask[imask] = new_imask

                if np.sum(imask) > 4:
                    # Fewer than 4 points after removing outliers. We could plausibly return
                    # the spline from the previous iteration because it was fit with at least
                    # 4 points. However, since the outliers were such a significant fraction
                    # of the curve, the spline from the previous iteration is probably junk,
                    # and we consider this a fatal error.
                    try:
                        with warnings.catch_warnings():
                            # Suppress warning messages printed by pydlutils.bspline. Instead we
                            # catch any exception and raise a more informative error.
                            warnings.simplefilter("ignore")

                            # Fit the spline on non-outlier points.
                            #curve = BSpline.iterfit(time[mask], flux[mask], bkspace=bk_space)[0]
                            knots = np.quantile(itime[imask], qs)
                            #print(np.all(np.isfinite(flux[mask])),np.average(flux[mask]))
                            tck = splrep(itime[imask], iflux[imask], t=knots, k=polydegree)
                        ispline = splev(itime, tck)

                        # Evaluate spline at the time points.
                        #spline = curve.value(time)[0]

                        #spline = np.copy(flux)
                    except (IndexError, TypeError, ValueError) as e:
                        raise ValueError(
                                "Fitting spline failed with error: '%s'. This might be caused by the "
                                "breakpoint spacing being too small, and/or there being insufficient "
                                "points to fit the spline in one of the intervals." % e)
                else:
                    ispline=np.tile(np.nanmedian(iflux[imask]),len(iflux))
                    break
            spline+=[ispline[ibool]]
            mask+=[imask[ibool]]
        else:
            spline+=[np.tile(np.nanmedian(flux[ix]),np.sum(ix))]
            mask+=[imask]
        

    return np.hstack(spline), np.hstack(mask)

def starpars_from_MonoTools_lc(lc):
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
    return [Rstar, Teff, logg]


def get_lds(n_samples,Teff,logg,FeH=0.0,xi_def=1.0, how='tess'):
    from scipy.interpolate import CloughTocher2DInterpolator as ct2d
    import pandas as pd
    if how.lower()=='tess':
        #Best-performing models according to https://arxiv.org/abs/2203.05661 is Phoenix 17 r-method:
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tables","J_A+A_600_A30_tableab.csv")):
            from astroquery.vizier import Vizier
            setattr(Vizier,'ROW_LIMIT',999999999)
            lds=Vizier.get_catalogs("J/A+A/600/A30/tableab")[0].to_pandas()
            lds.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tables","J_A+A_600_A30_tableab.csv"))
        else:
            lds=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tables","J_A+A_600_A30_tableab.csv"),index_col=0)
        lds=lds.loc[(lds['Type']=='r')&((lds['Mod']=="PD")^(lds['Teff']>3000))]
        if 'xi' not in lds.columns:
            lds['xi']=np.tile(1,len(lds))
    elif how.lower()=='cheops':
        lds=pd.read_fwf(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tables","Cheops_Quad_LDs_AllFeHs.txt"),header=None,widths=[5,7,5,5,9])
        lds=pd.DataFrame({'logg':lds.iloc[3::3,0].values.astype(float),'Teff':lds.iloc[3::3,1].values.astype(float),
                          'Z':lds.iloc[3::3,2].values.astype(float),'xi':lds.iloc[3::3,3].values.astype(float),
                          'aLSM':lds.iloc[3::3,4].values.astype(float),'bLSM':lds.iloc[4::3,4].values.astype(float),
                          'CHI2':lds.iloc[5::3,4].values.astype(float)})
    elif how.lower() in ['k2','kepler']:
        arr = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tables","KeplerLDlaws.txt"),skip_header=2)
        #Selecting FeH manually:
        lds=pd.DataFrame({'Teff':arr[:,0],'logg':arr[:,1],'Z':arr[:,2],
                          'aLSM':arr[:,4],'bLSM':arr[:,5],'xi':np.tile(1,len(arr[:,0]))})
        
    Z = np.unique(lds['Z'])[np.argmin(abs(np.unique(lds['Z'])-FeH))]
    
    xi = np.unique(lds['xi'])[np.argmin(abs(np.unique(lds['xi'])-xi_def))]
    
    tab_ix = (lds['Z']==Z)*(lds['xi']==xi)
    print(np.sum(tab_ix))
    a_interp=ct2d(np.column_stack((lds.loc[tab_ix,'Teff'].values.astype(float),
                                   lds.loc[tab_ix,'logg'].values.astype(float))),
                  lds.loc[tab_ix,'aLSM'].values.astype(float))
    b_interp=ct2d(np.column_stack((lds.loc[tab_ix,'Teff'].values.astype(float),
                                   lds.loc[tab_ix,'logg'].values.astype(float))),
                  lds.loc[tab_ix,'bLSM'].values.astype(float))
    
    Teff_samples = np.random.normal(Teff[0],np.clip(np.average(abs(np.array(Teff[1:]))),50,1000),n_samples)
    logg_samples = np.random.normal(logg[0],np.clip(np.average(abs(np.array(logg[1:]))),0.1,1000),n_samples)

    outarr=np.column_stack((a_interp(np.clip(Teff_samples,2300,12000),np.clip(logg_samples,0,5)),
                            b_interp(np.clip(Teff_samples,2300,12000),np.clip(logg_samples,0,5))))
    return outarr

def ExoFopUpload(username=None,password=None,saveloc='.'):
    #Logging in and grabbing the requisite cookie
    username=input("Input ExoFOP username") if username is None else username
    password=input("Input password") if password is None else password
    os.system("wget --keep-session-cookies --save-cookies "+saveloc+"/mycookie.txt --post-data \"username="+username+"&password="+password+"&ref=login_user&ref_page=/tess/\" \"https://exofop.ipac.caltech.edu/tess/password_check.php\"")

def cut_high_rollangle_scatter(mask,roll_angles,flux,flux_err,sd_thresh=3.0,roll_ang_binsize=25, **kwargs):
    """Find regions which have high scatter as a function of rollangle.
    Do this iteratively by comparing in-bin scatter to out-of-bin scatter, masking points each time"""
    newmask=np.tile(True,len(mask))
    #Using 30-deg bins
    max_sigma_off=sd_thresh+10
    n_loop=0
    while max_sigma_off>sd_thresh and n_loop<10:
        sorted_roll_angles=np.sort(roll_angles[newmask&mask])
        sorted_flux=flux[newmask&mask][np.argsort(roll_angles[newmask&mask])]
        sorted_flux_err=flux_err[newmask&mask][np.argsort(roll_angles[newmask&mask])]
        rolllc=np.column_stack((sorted_roll_angles,sorted_flux,sorted_flux_err))
        av_1s,sd_1s,av_2s,sd_2s,av_3s,sd_3s=[],[],[],[],[],[]
        range1=np.arange(np.min(sorted_roll_angles)-5/6*roll_ang_binsize,np.max(sorted_roll_angles)+1.01/6*roll_ang_binsize,roll_ang_binsize)
        bins1=binlc_given_x(rolllc,range1)
        for n in range(len(bins1)):
            ix=np.arange(len(bins1))!=n
            av,sd=weighted_avg_and_std(bins1[ix,1], bins1[ix,2])
            av_1s+=[av];sd_1s+=[sd]
        snr1s=(bins1[:,2]-np.array(sd_1s))/np.nanmedian(bins1[:,2])
        range2=np.arange(np.min(sorted_roll_angles)-3/6*roll_ang_binsize,np.max(sorted_roll_angles)+3.01/6*roll_ang_binsize,roll_ang_binsize)
        bins2=binlc_given_x(rolllc,range2)
        for n in range(len(bins2)):
            ix=np.arange(len(bins2))!=n
            av,sd=weighted_avg_and_std(bins2[ix,1], bins2[ix,2])
            av_2s+=[av];sd_2s+=[sd]
        snr2s=(bins2[:,2]-np.array(sd_2s))/np.nanmedian(bins2[:,2])
        range3=np.arange(np.min(sorted_roll_angles)-1/6*roll_ang_binsize,np.max(sorted_roll_angles)+5.01/6*roll_ang_binsize,roll_ang_binsize)
        bins3=binlc_given_x(rolllc,range3)
        for n in range(len(bins3)):
            ix=np.arange(len(bins3))!=n
            av,sd=weighted_avg_and_std(bins3[ix,1], bins3[ix,2])
            av_3s+=[av];sd_3s+=[sd]
        snr3s=(bins3[:,2]-np.array(sd_3s))/np.nanmedian(bins3[:,2])
        maxes=[np.nanmax(snr1s), np.nanmax(snr2s), np.nanmax(snr3s)]
        max_sigma_off=np.max(maxes)
        #Masking all points within the range
        if np.argmax(maxes)==0:
            ix=list(snr1s).index(max_sigma_off)
            newmask[(roll_angles>range1[ix])&(roll_angles<range1[ix+1])]=False
        elif np.argmax(maxes)==1:
            ix=list(snr2s).index(max_sigma_off)
            newmask[(roll_angles>range2[ix])&(roll_angles<range2[ix+1])]=False
        else:
            ix=list(snr3s).index(max_sigma_off)
            newmask[(roll_angles>range3[ix])&(roll_angles<range3[ix+1])]=False
        n_loop+=1
    return newmask

def roll_rollangles(roll_angles,mask=None):
    """Shifting the roll angles in such a way that the largest jump in the data does not occur during the time series of sorted rollangles.
    For example, continuous roll angles with a jump from 150-210 degrees would have the subsequent values "rolled" to the start to be continous from -160 to 150.
    This assists with e.g. roll angle plotting

    Args:
        roll_angles (np.ndarray): Array of Cheops roll angles

    Returns:
        rolled_roll_angles: [description]
    """
    mask=np.tile(True,len(roll_angles)) if mask is None else mask

    sorted_roll_angles=np.sort(roll_angles)
    if np.max(np.diff(sorted_roll_angles[mask]))>10:
        phi_jump=0.5*(sorted_roll_angles[mask][np.argmax(np.diff(sorted_roll_angles[mask]))+1]+sorted_roll_angles[mask][np.argmax(np.diff(sorted_roll_angles[mask]))])
        phi_jump+=np.random.normal(0,1e-5) #Doing this in case median rolls these onto zero offsets which messes up stuff in the future
        #print(np.max(np.diff(sorted_roll_angles)),sorted_roll_angles[np.argmax(np.diff(sorted_roll_angles))],sorted_roll_angles[np.argmax(np.diff(sorted_roll_angles))+1])
    else:
        phi_jump=0
    return (roll_angles-phi_jump)%360+phi_jump

def roll_all_rollangles(roll_angles, mask=None):
    """Shifting the roll angles across a number of fluxkeys in such a way that the largest jump in the data does not occur during the time series of sorted rollangles.
    For example, continuous roll angles with a jump from 150-210 degrees would have the subsequent values "rolled" to the start to be continous from -160 to 150.
    This assists with e.g. roll angle plotting

    Args:
        roll_angles (np.ndarray): Array of Cheops roll angles

    Returns:
        rolled_roll_angles: [description]
    """
    mask=np.tile(True,len(roll_angles)) if mask is None else mask

    digis=np.digitize(roll_angles[mask]%360,np.arange(0,360.01,5))
    digi_counts=[digis[d]==d for d in range(72)]
    min_counts=np.argmin(digi_counts)
    if np.sum((roll_angles[mask]>(5*min_counts))&(roll_angles[mask]<(5*(min_counts+1))))>0:
        rem_rollangs=np.sort(roll_angles[mask][(roll_angles[mask]>(5*min_counts))&(roll_angles[mask]<(5*(min_counts+1)))])
        jump_loc=np.argmax(np.diff(rem_rollangs))
        new_phi_jump=0.5*(rem_rollangs[jump_loc]+rem_rollangs[jump_loc+1])
    else:
        new_phi_jump=5*min_counts+2.5
    return (roll_angles-new_phi_jump)%360+new_phi_jump

def create_angle_spline_dmatrix(phis,bkpt_cad=10):
    from patsy import dmatrix
    if np.max(np.diff(np.sort(phis)))<(2*bkpt_cad) and np.min(phis)<2*bkpt_cad and np.max(phis)>(360-2*bkpt_cad):
        #No clear gap
        av_cad=np.nanmedian(np.diff(np.sort(phis)))
        n_knots=int(np.floor(len(phis)*av_cad/bkpt_cad))
        knots=np.quantile(phis,np.linspace(0,1,2*n_knots)[1::2])
        dmat_norm = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": phis, "knots": knots},
        )[:,2:-2]
        dmat_shifted = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": (phis-180)%360, "knots": np.sort((knots-180)%360)},
        )[:,2:-2]
        knot_shift=np.sum(knots>180)
        knot_quads=np.searchsorted(knots,np.array([0,90,180,270,360]))
        knot_quad_shift=np.searchsorted(np.sort((knots-180)%360),np.array([0,90,180,270,360]))
        arr=np.hstack((dmat_shifted[:,knot_quad_shift[2]:knot_quad_shift[3]],
                    dmat_norm[:,knot_quads[1]:knot_quads[3]],
                    dmat_shifted[:,knot_quad_shift[1]:knot_quad_shift[2]]))
        return arr,knots
    else:
        #Clear gap
        av_cad=np.nanmedian(np.diff(np.sort(phis)))
        n_knots=int(np.ceil(len(phis)*av_cad/bkpt_cad))
        
        #Finding gap 
        folddoublephi=np.hstack([np.sort(phis),360+np.sort(phis)])
        imaxgap=np.argmax(np.diff(folddoublephi))
        gap_pos=0.5*(folddoublephi[imaxgap]+folddoublephi[imaxgap+1])
        gap_pos=gap_pos-360 if gap_pos>360 else gap_pos
        
        knots=np.quantile(phis-gap_pos,np.linspace(0,1,2*n_knots)[1::2])+gap_pos
        
        dmat = dmatrix(
            "bs(x, knots=knots, degree=3, include_intercept=True) - 1",
            {"x": (phis-gap_pos)%360, "knots": np.sort((knots-gap_pos)%360)},
        )[:,2:-2]
        #Shifting back around so knots align with 0->360deg
        knot_shift=np.sum(knots<gap_pos)
        arr=np.hstack((dmat[:,-1*knot_shift:],dmat[:,:-1*knot_shift]))
        print(knots,knot_shift,gap_pos,(len(knots)-knot_shift))
        return arr,knots


def vals_to_latex(vals):
    #Function to turn -1,0, and +1 sigma values into round latex strings for a table
    try:
        roundval=int(np.min([-1*np.floor(np.log10(abs(vals[1]-vals[0])))+1,-1*np.floor(np.log10(abs(vals[2]-vals[1])))+1]))
        errs=[vals[2]-vals[1],vals[1]-vals[0]]
        if np.round(errs[0],roundval-1)==np.round(errs[1],roundval-1):
            #Errors effectively the same...
            if roundval<0:
                return " $ "+str(int(np.round(vals[1],roundval)))+" \pm "+str(int(np.round(np.average(errs),roundval)))+" $ "
            else:
                return " $ "+str(np.round(vals[1],roundval))+" \pm "+str(np.round(np.average(errs),roundval))+" $ "
        else:
            if roundval<0:
                return " $ "+str(int(np.round(vals[1],roundval)))+"^{+"+str(int(np.round(errs[0],roundval)))+"}_{-"+str(int(np.round(errs[1],roundval)))+"} $ "
            else:
                return " $ "+str(np.round(vals[1],roundval))+"^{+"+str(np.round(errs[0],roundval))+"}_{-"+str(np.round(errs[1],roundval))+"} $ "
    except:
        return " - "

def vals_to_short(vals,roundval=None):
    #Function to turn -1,0, and +1 sigma values into round latex strings for a table
    try:
        if roundval is None:
            roundval=int(np.min([-1*np.floor(np.log10(abs(vals[1]-vals[0])))+1,-1*np.floor(np.log10(abs(vals[2]-vals[1])))+1]))-1
        return " $ "+str(np.round(vals[1],roundval))+" $ "
    except:
        return " - "
    
    
def vals_to_overleaf(name,vals,include_short=True):
    if len(vals)==2 and vals[1]<0.5*vals[0]:
        vals=[vals[0]-vals[1],vals[0],vals[0]+vals[1]]
    
    replace_vals = {'_':'','[':'',']':'','/':'div','-':'minus','0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}
    for symbol, text in replace_vals.items():
        name = name.replace(symbol, text)
    st = "\\newcommand{\\T"+name+"}{"+vals_to_latex(vals)+"}\n"
    if include_short:
        st+="\\newcommand{\\T"+name+"short}{"+vals_to_short(vals)+"}\n"
    return st


def get_all_tois():
    #Reading TOI data from web or file (updates every 10 days)
    from astropy.time import Time
    round_date=np.round(Time.now().jd,-1)
    if not os.path.exists("TOI_tab_jd_"+str(int(round_date))+".csv"):
        info=pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
        info.to_csv("TOI_tab_jd_"+str(int(round_date))+".csv")
    else:
        info=pd.read_csv("TOI_tab_jd_"+str(int(round_date))+".csv",index_col=0)
    info['star_TOI']=info['TOI'].values.astype(int)
    return info

def get_all_cheops_obs(toi_cat,overwrite=False):
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    round_date=np.round(Time.now().jd,-1)
    if not os.path.exists("/home/hosborn/home1/python/chexoplanet/Chateaux/All_Obs"+str(int(round_date))+".csv") or overwrite:
        #,"pi_name":{"equals":"Hugh OSBORN"},'data_arch_rev':{'equals':3}
        #,"prog_id":{"contains":"CHEOPS-56"}
        from dace_query.cheops import Cheops
        all_cheops_obs=pd.DataFrame(Cheops.query_database(limit=500,filters={"prog_id":{"contains":["CHEOPS-56","CHEOPS-9000"]}}))
        #ext_cheops_obs=pd.DataFrame(Cheops.query_database(limit=500,filters={"prog_id":{"equals":[900051]}}))
        #all_cheops_obs=pd.concat([pri_cheops_obs,ext_cheops_obs],axis=0)
        all_cheops_obs=all_cheops_obs.loc[(all_cheops_obs['data_arch_rev']==3)&(all_cheops_obs['pi_name']=="Hugh Osborn")]
        print(all_cheops_obs)

        #print(all_cheops_obs['obj_id_catname'].values)
        #Getting RaDec from string:
        obs_radecs=SkyCoord([rd.split(" / ")[0] for rd in all_cheops_obs['obj_pos_coordinates_hms_dms'].values], 
                            [rd.split(" / ")[1] for rd in all_cheops_obs['obj_pos_coordinates_hms_dms'].values],
                            unit=(u.hourangle,u.deg))
        all_cheops_obs['ra']  = obs_radecs.ra.deg
        all_cheops_obs['dec'] = obs_radecs.dec.deg
        
        #Using Observed RA/Dec and TOI RA/Dec to match these:
        
        toi_radecs=SkyCoord(toi_cat['RA'].values,toi_cat['Dec'].values,
                            unit=(u.hourangle,u.deg),
                            pm_ra_cosdec=np.array([pmra if np.isfinite(pmra) else 0.0 for pmra in toi_cat['PM RA (mas/yr)'].values])*u.mas/u.year,
                            pm_dec=np.array([pmdec if np.isfinite(pmdec) else 0.0 for pmdec in toi_cat['PM Dec (mas/yr)'].values])*u.mas/u.year,
                            obstime=Time(2015.5,format='jyear'))
        toi_radecs_epoch2000 = toi_radecs.apply_space_motion(Time(2000.0,format='jyear'))
        idx, d2d, _ = obs_radecs.match_to_catalog_sky(toi_radecs_epoch2000)
        #all_cheops_obs.iloc[np.argmin(d2d.arcsec)],all_cheops_obs.iloc[np.argmax(d2d.arcsec)])
        assert np.all(d2d < 3*u.arcsec), "All the Observed TOIs should match RA/Dec to the TOI catalogue - "+str(np.sum(d2d > 3*u.arcsec))+"are not found ("+",".join(list(all_cheops_obs.loc[d2d > 3*u.arcsec,'obj_id_catname'].values))+")"
        all_cheops_obs['star_TOI']=toi_cat["star_TOI"].values[idx]
        all_cheops_obs['TIC ID']=toi_cat["TIC ID"].values[idx]
        all_cheops_obs.to_csv("/home/hosborn/home1/python/chexoplanet/Chateaux/All_Obs"+str(int(round_date))+".csv")
    else:
        all_cheops_obs=pd.read_csv("/home/hosborn/home1/python/chexoplanet/Chateaux/All_Obs"+str(int(round_date))+".csv")

    return all_cheops_obs

def find_jumps(vects):
    """
    Finding clear jumps for an MxN matrix (where M is number of observations and N is number of independent dimensions/measurements per observation) using scipy kmeans
    """

    N=1
    distortion_norm=1e3
    from scipy.cluster import vq
    from scipy.spatial import distance
    nan_mask=np.isfinite(np.sum(vects,1))
    while N<5 and distortion_norm>0.1:
        clusters, distortion = vq.kmeans(vects[nan_mask],N)
        distortion_norm=distortion/np.ptp(clusters)
        N+=1
    #Finds first distortion value under 0.1
    split_ix=np.argmin(distance.cdist(vects,clusters),axis=1)
    if np.sum(~nan_mask)>0:
        #print(split_ix,nan_mask,~nan_mask,split_ix[~nan_mask],vects[~nan_mask])
        split_ix[~nan_mask]=np.nan #Making these nan rather than 0
        split_ix[~nan_mask]=split_ix[1+np.arange(len(nan_mask))[~nan_mask]]#Assuming neighbours have the correct index and are not nanned
    return split_ix

def get_cheops_ORs():
    #Get the CHEOPS ORs which contain basic info about planet ephemeris
    
    #Reading input data from PHT2
    input_ors_pri=pd.read_csv("/home/hosborn/home1/python/chexoplanet/Chateaux/Chateaux_ORs_pri.csv")
    input_ors_ext=pd.read_csv("/home/hosborn/home1/python/chexoplanet/Chateaux/Chateaux_ORs_ext.csv")
    input_ors=pd.concat([input_ors_pri,input_ors_ext])
    input_ors=input_ors.set_index(np.arange(len(input_ors)))
    #Changing non-obvious names to the TOI:
    #input_ors['TOI_name'] = [change_dic[o] if o in change_dic else o.replace("-","").upper() for o in input_ors['Target Name'].values]
    input_ors['file_key']=["CH_PR"+str(int(input_ors.iloc[i]['Programme Type']))+str(int(input_ors.iloc[i]['Programme Id'])).zfill(4)+"_TG"+str(int(input_ors.iloc[i]['Observation Request Id'])).zfill(4)+"01_V0300" for i in range(len(input_ors))]
    newsers=[]
    nior=len(input_ors)
    for ior,orrow in input_ors.loc[input_ors['Number Of Visits']>1].iterrows():
        ser=input_ors.loc[ior]
        ser['file_key']=ser['file_key'].replace("01_V0300","02_V0300")
        ser=ser.rename(nior)
        newsers+=[ser]
        nior+=1
    input_ors=pd.concat([input_ors,pd.concat(newsers,axis=1).T],axis=0)
    #input_ors=input_ors.set_index(np.arange(len(input_ors)))

    return input_ors

def update_period_w_tls(time,flux,per,N_search_pers=250,oversampling=5):
    from transitleastsquares import transitleastsquares as tls
    tlsmodel=tls(t=np.sort(time), y=1+flux[np.argsort(time)]*1e-3)
    #Hacked TLS a bit to get this estimate of the sampling frequency in Period - needed in case N_searchfreqs<100, at which point all periods are searched
    span=np.max(time)-np.min(time)
    deltaP=per**(4/3)/(13.23*oversampling*span)
    #print("TLS period update. period_min=",per-0.5*N_search_pers*deltaP,"period_max=",per+0.5*N_search_pers*deltaP)
    outtls=tlsmodel.power(period_min=per-0.5*N_search_pers*deltaP,period_max=per+0.5*N_search_pers*deltaP,
                          use_threads=4, duration_grid_step=1.2,
                          oversampling_factor=oversampling, transit_depth_min=100e-6)
    return outtls.period

def ProcessTOI(toi,toi_cat,all_cheops_obs,cheops_ors,commentstring="V0300_horus",**kwargs):
    import glob
    #step 1 - get observations of a given TOI
    if len(glob.glob("/home/hosborn/home1/data/Cheops_data/TOI"+str(toi)+"/*"+commentstring+"_model.pkl"))>0:
        print("Skipping TOI="+str(toi),"("+str(len(glob.glob("/home/hosborn/home1/data/Cheops_data/TOI"+str(toi)+"/*"+commentstring+"_model.pkl")))+"files found)")
        return None

    #Getting the data from TOI list, OR list and Observation list for this TOI:
    these_obs=all_cheops_obs.loc[all_cheops_obs['star_TOI']==toi]
    these_tois=toi_cat.loc[toi_cat['star_TOI']==toi].sort_values('Period (days)')
    these_ors=cheops_ors.loc[cheops_ors['Target Name']==these_obs.iloc[0]['obj_id_catname']]
    fks=[f.replace("CH_","") for f in these_obs.loc[:,'file_key'].values]
    print(toi, fks)

    #Figuring out which TOI links to which observation (if any)
    these_obs['TOI']=np.zeros(len(these_obs))
    print(these_ors['file_key'])
    for i,iob in these_obs.iterrows():
        #print(fk,"CH_"+fk in these_obs['file_key'].values,"CH_"+fk in these_ors['file_key'].values)
        fk=iob['file_key']
        ior=these_ors.loc[these_ors['file_key']==fk]#.iloc[0]
        print(fk,type(ior),ior.shape)
        ior=ior.iloc[0] if type(ior)==pd.DataFrame else ior
        itoi=np.argmin((these_tois['Period (days)'].values.astype(float)-float(ior["Transit Period [day]"]))**2)
        these_obs.loc[i,'TOI']=these_tois['TOI'].values[itoi]
        these_ors.loc[these_ors['file_key']=="CH_"+fk,'TOI']=these_tois['TOI'].values[itoi]    
    
    #Loading lightcurve
    from MonoTools.MonoTools import lightcurve
    lc=lightcurve.multilc(these_tois.iloc[0]['TIC ID'],'tess',load=False,update_tess_file=False)
                          #radec=SkyCoord(these_obs.iloc[0]['ra']*u.deg,these_obs.iloc[0]['dec']*u.deg),load=False)
    from . import newfit
    #Initialising model
    mod = newfit.chexo_model("TOI"+str(toi), radec=lc.radec, overwrite=True, 
                             comment=commentstring, save_file_loc="/home/hosborn/home1/data/Cheops_data")
    mod.get_tess()
    for i,iob in these_obs.iterrows():
        print(type(iob['file_key']),iob['file_key'])
        mod.add_cheops_lc(filekey=iob['file_key'].replace("CH_",""), PIPE=True, DRP=False, download=True,
                          mag=iob['obj_mag_v'], Teff=lc.all_ids['tess']['data']['Teff'])
    
    #Rstar, Teff, logg = tools.starpars_from_MonoTools_lc(lc)
    #mod.init_starpars(Rstar=Rstar,Teff=Teff,logg=logg)
    #mod.add_lc(lc.time+2457000,lc.flux,lc.flux_err)
    #for fk in fks:
    #    mod.add_cheops_lc(filekey=fk,fileloc=None, download=True, PIPE=True, DRP=False,
    #                      mag=lc.all_ids['tess']['data']['Tmag'])
    
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
                        period_err=2*float(these_tois.iloc[nix]['Period (days) err']),**kwargs)
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
    #try:
    #Initialising lightcurves:
    mod.init_lc(fit_gp=False, fit_flat=True, cut_oot=True, bin_oot=False)
    mod.init_cheops(use_bayes_fact=True, use_signif=False, overwrite=False)

    #Initialising full model:
    mod.init_model(use_mstar=False, use_logg=True,fit_phi_spline=True,fit_phi_gp=False,phi_model_type="common", constrain_lds=True, fit_ttvs=False, assume_circ=True)

    mod.sample_model()
    mod.model_comparison_cheops()

    #mod.plot_rollangle_gps(save=True)
    mod.plot_rollangle_model(save=True)
    mod.plot_cheops(save=True,show_detrend=True)
    mod.plot_phot('tess',save=True)
    mod.plot_transits_fold(save=True)
    
    df=mod.save_trace_summary()
    mod.save_model_to_file()
    
    mod.MakeExoFopFiles(list(["TOI"+t for t in these_tois['TOI'].values.astype(str)]),
                        upload_loc="/home/hosborn/home1/data/Cheops_data/Ext_ChATeAUX/")
    
################################################
#         Copied from MonoTools.tools          #
################################################

def CutHighRegions(flux, mask, std_thresh=3.2,n_pts=25,n_loops=2):
    # Masking anomalous high region using a running 25-point median and std comparison
    # This is best used for e.g. Corot data which has SAA crossing events.

    digi=np.vstack([np.arange(f,len(flux)-(n_pts-f)) for f in range(n_pts)])
    stacked_fluxes=np.vstack([flux[digi[n]] for n in range(n_pts)])

    std_threshs=np.linspace(std_thresh-1.5,std_thresh,n_loops)

    for n in range(n_loops):
        stacked_masks=np.vstack([mask[digi[n]] for n in range(n_pts)])
        stacked_masks=stacked_masks.astype(int).astype(float)
        stacked_masks[stacked_masks==0.0]=np.nan

        meds=np.nanmedian(stacked_fluxes*stacked_masks,axis=0)
        stds=np.nanstd(stacked_fluxes*stacked_masks,axis=0)
        #Adding to the mask any points identified in 80% of these passes:
        #print(np.vstack([np.hstack((np.tile(False,1+n2),stacked_fluxes[n2]*stacked_masks[n2]>(meds+std_threshs[n]*stds),
        #                 np.tile(False,n_pts-n2+1))) for n2 in np.arange(n_pts)]))
        #print(np.nansum(np.vstack([np.hstack((np.tile(False,1+n2),stacked_fluxes[n2]*stacked_masks[n2]>(meds+std_threshs[n]*stds),
        #                 np.tile(False,n_pts-n2+1))) for n2 in np.arange(n_pts)]),axis=0))
        #print(np.nansum(np.vstack([np.hstack((np.tile(False,1+n2),stacked_fluxes[n2]*stacked_masks[n2]>(meds+std_threshs[n]*stds),
        #                 np.tile(False,n_pts-n2+1))) for n2 in np.arange(n_pts)]),axis=0).shape)
        #print(np.nansum(np.vstack([np.hstack((np.tile(False,1+n2),stacked_fluxes[n2]*stacked_masks[n2]>(meds+std_threshs[n]*stds),
        #                 np.tile(False,n_pts-n2+1))) for n2 in np.arange(n_pts)]),axis=0)[1:-1].shape)
        #print(mask.shape)

        mask*=np.nansum(np.vstack([np.hstack((np.tile(False,1+n2),
                                              stacked_fluxes[n2]*stacked_masks[n2]>(meds+std_threshs[n]*stds),
                                              np.tile(False,n_pts-n2+1))) for n2 in np.arange(n_pts)])
                           ,axis=0)[1:-1]<20
    return mask

def CutAnomDiff(flux,thresh=4.2):
    #Uses differences between points to establish anomalies.
    #Only removes single points with differences to both neighbouring points greater than threshold above median difference (ie ~rms)
    #Fast: 0.05s for 1 million-point array.
    #Must be nan-cut first
    diffarr=np.vstack((np.diff(flux[1:]),np.diff(flux[:-1])))
    diffarr/=np.median(abs(diffarr[0,:]))
    #Adding a test for the first and last points if they are >3*thresh from median RMS wrt next two points.
    anoms=np.hstack((abs(flux[0]-np.median(flux[1:3]))<(np.median(abs(diffarr[0,:]))*thresh*5),
                     ((diffarr[0,:]*diffarr[1,:])>0)+(abs(diffarr[0,:])<thresh)+(abs(diffarr[1,:])<thresh),
                     abs(flux[-1]-np.median(flux[-3:-1]))<(np.median(abs(diffarr[0,:]))*thresh*5)))
    return anoms

def find_time_regions(time,split_gap_size=1.5,**kwargs):
    if np.nanmax(np.diff(np.sort(time)))>split_gap_size:
        #We have gaps in the lightcurve, so we'll find the bins by looping through those gaps
        time_starts = np.hstack((np.nanmin(time),np.sort(time)[1+np.where(np.diff(np.sort(time))>split_gap_size)[0]]))
        time_ends   = np.hstack((time[np.where(np.diff(np.sort(time))>split_gap_size)[0]],np.nanmax(time)))
        return [(time_starts[i],time_ends[i]) for i in range(len(time_starts))]
    else:
        return [(np.nanmin(time),np.nanmax(time))]

def formwindow(dat,cent,size,boxsize,gapthresh=1.0):

    win = (dat[:,0]>cent-size/2.)&(dat[:,0]<cent+size/2.)
    box = (dat[:,0]>cent-boxsize/2.)&(dat[:,0]<cent+boxsize/2.)
    if np.sum(win)>0:
        high=dat[win,0][-1]
        low=dat[win,0][0]
        highgap = high < (cent+size/2.)-gapthresh
        lowgap = low > (cent-size/2.)+gapthresh

        if highgap and not lowgap:
            win = (dat[:,0] > high-size)&(dat[:,0] <= high)
        elif lowgap and not highgap:
            win = (dat[:,0] < low+size)&(dat[:,0] >= low)

        win = win&(~box)
    return win, box

def dopolyfit(win,mask=None,stepcent=0.0,d=3,ni=10,sigclip=3):
    mask=np.tile(True,len(win)) if mask is None else mask
    maskedwin=win[mask]

    #initial fit and llk:
    best_base = np.polyfit(maskedwin[:,0]-stepcent,maskedwin[:,1],w=1.0/maskedwin[:,2]**2,deg=d)
    best_offset = (maskedwin[:,1]-np.polyval(best_base,maskedwin[:,0]))**2/maskedwin[:,2]**2
    best_llk=-0.5 * np.sum(best_offset)

    #initialising this "random mask"
    randmask=np.tile(True,len(maskedwin))

    for iter in range(ni):
        # If a point's offset to the best model is great than a normally-distributed RV, it gets masked
        # This should have the effect of cutting most "bad" points,
        #   but also potentially creating a better fit through bootstrapping:
        randmask = abs(np.random.normal(0.0,1.0,len(maskedwin)))<best_offset
        randmask = np.tile(True,len(maskedwin)) if np.sum(randmask)==0 else randmask

        new_base = np.polyfit(maskedwin[randmask,0]-stepcent,maskedwin[randmask,1],
                              w=1.0/np.power(maskedwin[randmask,2],2),deg=d)
        #winsigma = np.std(win[:,1]-np.polyval(base,win[:,0]))
        new_offset = (maskedwin[:,1]-np.polyval(new_base,maskedwin[:,0]))**2/maskedwin[:,2]**2
        new_llk=-0.5 * np.sum(new_offset)
        if new_llk>best_llk:
            #If that fit is better than the last one, we update the offsets and the llk:
            best_llk=new_llk
            best_offset=new_offset[:]
            best_base=new_base[:]
    return best_base

def med_and_std(values):
    return [np.nanmedian(values),np.nanstd(values)]

def update_lc_locs(epoch,most_recent_sect):
    #Updating the table of lightcurve locations using the scripts on the MAST/TESS "Bulk Downloads" page.
    all_sects=np.arange(np.max(epoch.index.values),most_recent_sect).astype(int)+1
    for sect in all_sects:
        fitsloc="https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_"+str(sect).zfill(2)+"_lc.sh"
        h = httplib2.Http()
        resp, content = h.request(fitsloc)
        if int(resp['status']) < 400:
            filename=content.split(b'\n')[1].decode().split(' ')[-2].split('-')
            epoch.loc[sect]=pd.Series({'date':int(filename[0][4:]),'runid':int(filename[3])})
        else:
            print("Sector "+str(sect)+" not (yet) found on MAST | RESPONCE:"+resp['status'])
    epoch.to_csv(chexo_tablepath+"/tess_lc_locations.csv")
    return epoch

def observed(tic,radec=None,maxsect=83):
    # Using either "webtess" page or Chris Burke's tesspoint to check if TESS object was observed:
    # Returns dictionary of each sector and whether it was observed or not
    
    tesspoint = importlib.import_module("tess-point.tess_stars2px")
    #from tesspoint import tess_stars2px_function_entry as tess_stars2px
    if radec is None:  
        ticStringList = ['{0:d}'.format(x) for x in [np.int64(tic)]]    
        # Setup mast query
        request = {'service':'Mast.Catalogs.Filtered.Tic', \
            'params':{'columns':'*', 'filters':[{ \
                    'paramName':'ID', 'values':ticStringList}]}, \
            'format':'json', 'removenullcolumns':True}
        headers, outString = tesspoint.mastQuery(request)
        outObject = json.loads(outString)
        radec=SkyCoord(np.array([x['ra'] for x in outObject['data']])[0]*u.deg,
                       np.array([x['dec'] for x in outObject['data']])[0]*u.deg)
    #Now doing tic + radec search:
    result = tesspoint.tess_stars2px_function_entry(tic, radec.ra.deg, radec.dec.deg)
    sectors = result[3]
    out_dic={s:True if s in sectors else False for s in np.arange(maxsect)}
    #print(out_dic)
    return out_dic

def partition_list(a, k):
    """AI is creating summary for partition_list

    Args:
        a (list): Ordered list of lengths that we wish to evenly split into k pieces
        k (int): Number of parts along which to split a

    Returns:
        list: Ordered index of which of `k` bins the value in `a` belongs
    """
    if k <= 1: return np.tile(0,len(a))
    if k == len(a): return np.arange(k)
    assert k<len(a) #Cannot have more plot rows that data sectors...
    partition_between = [(i+1)*len(a) // k for i in range(k-1)]
    average_height = float(sum(a))/k
    best_score = None
    best_partitions = None
    count = 0

    while True:
        starts = [0] + partition_between
        ends = partition_between + [len(a)]
        partitions = [a[starts[i]:ends[i]] for i in range(k)]
        heights = list(map(sum, partitions))
        abs_height_diffs = list(map(lambda x: abs(average_height - x), heights))
        worst_partition_index = abs_height_diffs.index(max(abs_height_diffs))
        worst_height_diff = average_height - heights[worst_partition_index]

        if best_score is None or abs(worst_height_diff) < best_score:
            best_score = abs(worst_height_diff)
            best_partitions = partitions
            no_improvements_count = 0
        else:
            no_improvements_count += 1

        if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
            return np.hstack([np.tile(i,ends[i]-starts[i]) for i in range(len(starts))])
            #best_partitions
        count += 1

        move = -1 if worst_height_diff < 0 else 1
        bound_to_move = 0 if worst_partition_index == 0\
                        else k-2 if worst_partition_index == k-1\
                        else worst_partition_index-1 if (worst_height_diff < 0) ^ (heights[worst_partition_index-1] > heights[worst_partition_index+1])\
                        else worst_partition_index
        direction = -1 if bound_to_move < worst_partition_index else 1
        partition_between[bound_to_move] += move * direction

def MakeBokehTable(df, dftype='toi', cols2use=None, cols2avoid=None, errtype=' err', width=300, height=350):
    """Form Bokeh table from an input pandas dataframe

    Args:
        df ([type]): [description]
        dftype (str, optional): [description]. Defaults to 'toi'.
        cols2use ([type], optional): [description]. Defaults to None.
        cols2avoid ([type], optional): [description]. Defaults to None.
        width (int, optional): [description]. Defaults to 300.
        height (int, optional): [description]. Defaults to 350.

    Returns:
        [type]: [description]
    """
    from bokeh.models import ColumnDataSource
    from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

    if type(df)==pd.Series:
        df=pd.DataFrame(df).T
    if cols2use is None:
        if dftype is None or dftype=='':
            cols2use=df.columns
        elif dftype=='toi':
            cols2use=['TIC ID','TESS Disposition', 'TFOPWG Disposition', 'TESS Mag','RA', 'Dec',
                'Epoch (BJD)','Period (days)','Duration (hours)', 'Depth (mmag)','Planet Radius (R_Earth)','SNR_per_transit',
                'Stellar Eff Temp (K)', 'Stellar Radius (R_Sun)','Comments','Cheops_Observability','Cheops_Max_Efficiency',
                'Cheops_Obs_dates','Year2_obs_times', 'Year3_obs_times','Year4_obs_times', 'TESS_data', 'TESS_dvr']
        elif dftype=='tic':
            cols2use='ID, ra, dec, Tmag, plx, eclong, eclat, Bmag, Vmag, Jmag,  Kmag, GAIAmag, Teff, logg, MH, rad, mass, rho, d'.split(', ')
    if cols2avoid is None:
        if dftype is None or dftype=='':
            cols2avoid=[]
        elif dftype=='toi':
            cols2avoid=['SG1A','SG1B','SG2','SG3','SG4','SG5','ACWG ESM','ACWG TSM',
                                'Time Series Observations','Spectroscopy Observations','Imaging Observations',
                                'TESS Disposition','Master','Planet Insolation (Earth Flux)','Depth (mmag)',
                                'Planet Equil Temp (K)','Previous CTOI','PM RA (mas/yr)','PM Dec (mas/yr)']
        elif dftype=='tic':
            cols2avoid='pmRA, pmDEC, objType, typeSrc, version, HIP, TYC, UCAC, TWOMASS, SDSS, ALLWISE, GAIA, APASS, KIC, POSflag, PMflag, lumclass, lum, ebv, numcont, contratio, disposition, duplicate_id, priority, EBVflagTeffFlag, gaiabp, gaiarp, gaiaqflag, starchareFlag, VmagFlag, BmagFlag, splists, RA_orig, Dec_orig, raddflag, wdflag, dstArcSec'.split(', ')
    #Making Datatable inset:
    err_cols=[]
    nonerr_cols=[]
    #errless_cols=[]

    cols2use=[c for c in cols2use if not err_string_parse(c)[0]]
   
    df=df.rename(columns={col:col[:-1] for col in df.columns if col[-1]==' '}) #Removing trailing spaces

    #Creating error arrays
    for col in df.columns:
        if col in cols2use and col not in cols2avoid and 'e_'+col in df.columns:
            nonerr_cols+=[col]
            err_cols+=['e_'+col]
            #If we have multiple errors, we'll do a median to make sure we only end up with one:
            df['e_'+col]=np.nanmedian(np.vstack([abs(df[ecol].values.astype(float)) for ecol in ['e_'+col,'epos_'+col,'eneg_'+col,col+' err',col+'_err1',col+'_err2',col+' Error'] if ecol in df.columns]),axis=0)

        elif col in cols2use and col not in cols2avoid and type(df[col].values[0]) in [int,float,np.float64,np.int64,str] and 'e_'+col not in df.columns:
            nonerr_cols+=[col]
            err_cols+=['e_'+col]
            df['e_'+col]=np.tile(np.nan,df.shape[0])

    nonerr_cols=np.nan_to_num(np.array(nonerr_cols),0.0)
    err_cols=np.nan_to_num(np.array(err_cols),0.0)
    #errless_cols=np.nan_to_num(np.array(errless_cols),0.0)
    newdf=pd.DataFrame()
    newdf['col']=nonerr_cols
    columns=[TableColumn(field='col', title='Column')]
    for pl in range(df.shape[0]):
        if 'TOI' in df.columns:
            name = str(df.iloc[pl]['TOI'])+' '
        elif 'CTOI' in df.columns:
            name = str(df.iloc[pl]['CTOI'])+' '
        elif 'id' in df.columns:
            name = str(df.iloc[pl]['id'])+' '
        else:
            name = str(df.iloc[pl].name)+' '
        newdf[name+'Value']=[0.0 if df.iloc[pl][val] in [None,np.nan,-np.inf,np.inf,''] else df.iloc[pl][val] for val in nonerr_cols]
        newdf[name+'Errs']=[0.0 if df.iloc[pl][val] in [None,np.nan,-np.inf,np.inf,''] else df.iloc[pl][val] for val in err_cols]
        #newdf=newdf.fillna(0.0)
        columns+=[TableColumn(field=name+'Value', title=name+'Value')]
        columns+=[TableColumn(field=name+'Errs', title=name+'Errs')]
    #print(newdf)
    data_table = DataTable(source=ColumnDataSource(newdf), columns=columns, width=width, height=height)    
    return data_table

def GetExoFop(icid, mission='tess',file=''):
    cols={'Telescope':'telescope','Instrument':'instrument','Teff (K)':'teff','Teff (K) Error':'teffe',
          'Teff':'teff','Teff Error':'teffe','log(g)':'logg',
          'log(g) Error':'logge','Radius (R_Sun)':'rad','Radius':'rad','Radius Error':'rade',
          'Radius (R_Sun) Error':'rade','logR\'HK':'logrhk',
          'logR\'HK Error':'logrhke','S-index':'sindex','S-index Error':'sindexe','H-alpha':'haplha','H-alpha Error':'halphae',
          'Vsini':'vsini','Vsini Error':'vsinie','Rot Per':'rot_per','Rot Per Error':'rot_pere','Metallicity':'feh',
          'Metallicity Error':'fehe','Mass (M_Sun)':'mass','Mass':'mass','Mass Error':'masse',
          'Mass (M_Sun) Error':'masse','Density (g/cm^3)':'rho_gcm3',
          'Density':'rho_gcm3',
          'Density (g/cm^3) Error':'rho_gcm3e','Luminosity':'lum','Luminosity Error':'lume',
          'Observation Time (BJD)':'obs_time_bjd','Distance':'dis','Distance Error':'dise',
          'RV (m/s)':'rv_ms','RV Error':'rv_mse','Distance (pc)':'dis','Distance (pc) Error':'dise',
          '# of Contamination sources':'n_contams', 'B':'bmag', 'B Error':'bmage', 'Dec':'dec', 'Ecliptic Lat':'lat_ecl',
          'Ecliptic Long':'long_ecl', 'Gaia':'gmag', 'Gaia Error':'gmage', 'Galactic Lat':'lat_gal', 'Galactic Long':'long_gal',
          'H':'hmag', 'H Error':'hmage', 'In CTL':'in_ctl', 'J':'jmag', 'J Error':'jmage', 'K':'kmag', 'K Error':'kmage',
          'Planet Name(s)':'planet_names', 'Proper Motion Dec (mas/yr)':'pm_dec',
          'Proper Motion RA (mas/yr)':'pm_ra', 'RA':'ra','RA (J2015.5)':'ra', 'Dec (J2015.5)':'dec',
          'Star Name & Aliases':'star_name', 'TESS':'tmag','Kep':'kepmag',
          'TESS Error':'tmage', 'TIC Contamination Ratio':'ratio_contams', 'TOI':'toi', 'V':'vmag', 'V Error':'vmage',
          'WISE 12 micron':'w3mag', 'WISE 12 micron Error':'w3mage', 'WISE 22 micron':'w4mag',
          'WISE 22 micron Error':'w4mage', 'WISE 3.4 micron':'w1mag', 'WISE 3.4 micron Error':'w1mage',
          'WISE 4.6 micron':'w2mag', 'WISE 4.6 micron Error':'w2mag', 'n_TOIs':'n_tois','spec':'spec',
          'Campaign':'campaign','Object Type':'objtype'}
    '''
    Index(['mission', 'ra', 'dec', 'GalLong', 'GalLat', 'Aliases', 'campaign',
           'Proposals', 'objtype', 'bmag', 'bmag_err', 'g', 'g_err', 'vmag',
           'vmag_err', 'r', 'r_err', 'kepmag', 'kepmag_err', 'i', 'i_err', 'jmag',
           'jmag_err', 'hmag', 'hmag_err', 'kmag', 'kmag_err', 'w1mag',
           'w1mag_err', 'w2mag', 'w2mag_err', 'w3mag', 'w3mag_err', 'w4mag',
           'w4mag_err', 'Teff', 'Teff_err', 'logg', 'logg_err', 'Radius',
           'Radius_err', 'FeH', 'FeH_err', 'Distance', 'Distance_err', 'Mass',
           'Mass_err', 'Density', 'Density_err', 'spec', 'bmagem', 'bmagep', 'gem',
           'gep', 'vmagem', 'vmagep', 'rem', 'rep', 'kepmagem', 'kepmagep'],
          dtype='object')
    Index(['iem', 'iep', 'jmagem', 'jmagep', 'hmagem', 'hmagep', 'kmagem',
           'kmagep', 'w1magem', 'w1magep', 'w2magem', 'w2magep', 'w3magem',
           'w3magep', 'w4magem', 'w4magep', 'Teffem', 'Teffep', 'loggem', 'loggep',
           'Radiusem', 'Radiusep', 'FeHem', 'FeHep', 'Distanceem', 'Distanceep',
           'Massem', 'Massep', 'Densityem', 'Densityep', 'bmage', 'ge', 'vmage',
           're', 'ie', 'jmage', 'hmage', 'kmage', 'w1mage', 'w2mage', 'w3mage',
           'Teffe', 'logge', 'Radiuse', 'FeHe', 'Distancee', 'Masse', 'Densitye'],
          dtype='object')
    '''

    #Strips online file for a given epic/tic
    if mission.lower() in ['kep','kepler']:
        kicinfo=GetKICinfo(icid)
        #Checking if the object is also in the TIC:
        ticout=Catalogs.query_criteria(catalog="Tic",coordinates=str(kicinfo['ra'])+','+str(kicinfo['dec']),
                                       radius=20*u.arcsecond,objType="STAR",columns=['ID','KIC','Tmag','Vmag']).to_pandas()
        if len(ticout.shape)>1:
            ticout=ticout.loc[np.argmin(ticout['Tmag'])]
            icid=ticout['ID']
            mission='tess'
        elif ticout.shape[0]>0:
            #Not in TIC
            return kicinfo
    else:
        kicinfo = None
    assert mission.lower() in ['tess','k2','corot']
    outdat={}
    outdat['mission']=mission.lower()
    #Searching TESS and K2 ExoFop for info (and TIC-8 info):
    req=requests.get("https://exofop.ipac.caltech.edu/"+mission.lower()+"/download_target.php?id="+str(icid), timeout=120)
    if req.status_code==200:
        #Splitting into each 'paragraph'
        sections=req.text.split('\n\n')
        for sect in sections:
            #Processing each section:
            if sect[:2]=='RA':
                #This is just general info - saving
                for line in sect.split('\n'):
                    if mission.lower()=='tess':
                        if line[:28].strip() in cols:
                            outdat[cols[line[:28].strip()]]=line[28:45].split('  ')[0].strip()
                        else:
                            outdat[re.sub('\ |\^|\/|\{|\}|\(|\)|\[|\]', '',line[:28])]=line[28:45].split('  ')[0].strip()
                    elif mission.lower()=='k2':
                        if line[:13].strip() in cols:
                            outdat[cols[line[:13].strip()]]=line[13:].strip()
                        else:
                            outdat[re.sub('\ |\^|\/|\{|\}|\(|\)|\[|\]', '',line[:13])]=line[13:].strip()
            elif sect[:24]=='TESS Objects of Interest':
                #Only taking number of TOIs and TOI number:
                outdat['n_TOIs']=len(sect.split('\n'))-2
                outdat['TOI']=sect.split('\n')[2][:15].strip()
            elif sect[:7]=='STELLAR':
                #Stellar parameters
                labrow=sect.split('\n')[1]
                boolarr=np.array([s==' ' for s in labrow])
                splits=[0]+list(2+np.where(boolarr[:-3]*boolarr[1:-2]*~boolarr[2:-1]*~boolarr[3:])[0])+[len(labrow)]
                labs = [re.sub('\ |\^|\/|\{|\}|\(|\)|\[|\]', '',labrow[splits[i]:splits[i+1]]) for i in range(len(splits)-1)]
                spec=[]
                if mission.lower()=='tess':
                    #Going through all sources of Stellar params:
                    for row in sect.split('\n')[2:]:
                        stpars=np.array([row[splits[i]:splits[i+1]].strip() for i in range(len(splits)-1)])
                        for nl in range(len(labs)):
                            if labs[nl].strip() not in cols:
                                label=re.sub('\ |\/|\{|\}|\(|\)|\[|\]', '', labs[nl]).replace('Error','_err')
                            else:
                                label=cols[labs[nl].strip()]
                            if not label in outdat.keys() and stpars[1]=='' and stpars[nl].strip()!='':
                                #Stellar info just comes from TIC, so saving simply:
                                outdat[label] = stpars[nl]
                            elif stpars[1]!='' and stpars[nl].strip()!='':
                                #Stellar info comes from follow-up, so saving with _INSTRUMENT:
                                spec+=['_'+row[splits[3]:splits[4]].strip()]
                                outdat[labs[nl]+'_'+stpars[1]] = stpars[nl]
                elif mission.lower()=='k2':
                    for row in sect.split('\n')[1:]:
                        if row[splits[0]:splits[1]].strip() not in cols:
                            label=re.sub('\ |\/|\{|\}|\(|\)|\[|\]', '', row[splits[0]:splits[1]]).replace('Error','_err')
                        else:
                            label=cols[row[splits[0]:splits[1]].strip()]

                        if not label in outdat.keys() and row[splits[3]:splits[4]].strip()=='huber':
                            outdat[label] = row[splits[1]:splits[2]].strip()
                            outdat[label+'_err'] = row[splits[2]:splits[3]].strip()
                        elif label in outdat.keys() and row[splits[3]:splits[4]].strip()!='huber':
                            if row[splits[3]:splits[4]].strip()!='macdougall':
                                spec+=['_'+row[splits[3]:splits[4]].strip()]
                                #Adding extra stellar params with _user (no way to tell the source, e.g. spectra)
                                outdat[label+'_'+row[splits[3]:splits[4]].strip()] = row[splits[1]:splits[2]].strip()
                                outdat[label+'_err'+'_'+row[splits[3]:splits[4]].strip()] = row[splits[2]:splits[3]].strip()
                outdat['spec']=None if len(spec)==0 else ','.join(list(np.unique(spec)))
            elif sect[:9]=='MAGNITUDE':
                labrow=sect.split('\n')[1]
                boolarr=np.array([s==' ' for s in labrow])
                splits=[0]+list(2+np.where(boolarr[:-3]*boolarr[1:-2]*~boolarr[2:-1]*~boolarr[3:])[0])+[len(labrow)]
                for row in sect.split('\n')[2:]:
                    if row[splits[0]:splits[1]].strip() not in cols:
                        label=re.sub('\ |\/|\{|\}|\(|\)|\[|\]', '', row[splits[0]:splits[1]]).replace('Error','_err')
                    else:
                        label=cols[row[splits[0]:splits[1]].strip()]
                    outdat[label] = row[splits[1]:splits[2]].strip()
                    outdat[label+'_err'] = row[splits[2]:splits[3]].strip()

        outdat=pd.Series(outdat,name=icid)

        #Replacing err and err1/2 with em and ep
        for col in outdat.index:
            try:
                outdat[col]=float(outdat[col])
            except:
                pass
            if col.find('_err1')!=-1:
                outdat=outdat.rename(index={col:'epos_'+col.replace('_err1','')})
            elif col.find('_err2')!=-1:
                outdat=outdat.rename(index={col:'eneg_'+col.replace('_err2','')})
            elif col.find('_err')!=-1:
                outdat['epos_'+col.replace('_err','')]=outdat[col]
                outdat['eneg_'+col.replace('_err','')]=outdat[col]
                outdat=outdat.rename(index={col:col.replace('_err','e')})
        for col in outdat.index:
            if 'radius' in col:
                outdat=outdat.rename(index={col:col.replace('radius','rad')})
            if col[-2:]=='em' and col[:-1] not in outdat.index and type(outdat[col])!=str:
                #average of em and ep -> e
                outdat[col[:-1]]=0.5*(abs(outdat[col])+abs(outdat[col[:-1]+'p']))
        return outdat, kicinfo
    elif kicinfo is not None:
        return None, kicinfo
    else:
        return None, None
