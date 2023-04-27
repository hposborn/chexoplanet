import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

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
    from astroquery.vizier import Vizier
    setattr(Vizier,'ROW_LIMIT',999999999)
    if how.lower()=='tess':
        #Best-performing models according to https://arxiv.org/abs/2203.05661 is Phoenix 17 r-method:
        lds=Vizier.get_catalogs("J/A+A/600/A30/tableab")[0].to_pandas()
        lds=lds.loc[(lds['Type']=='r')&((lds['Mod']=="PD")^(lds['Teff']>3000))]
        if 'xi' not in lds.columns:
            lds['xi']=np.tile(1,len(lds))
    elif how.lower()=='cheops':
        lds=pd.read_fwf(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Cheops_Quad_LDs_AllFeHs.txt"),header=None,widths=[5,7,5,5,9])
        lds=pd.DataFrame({'logg':lds.iloc[3::3,0].values.astype(float),'Teff':lds.iloc[3::3,1].values.astype(float),
                          'Z':lds.iloc[3::3,2].values.astype(float),'xi':lds.iloc[3::3,3].values.astype(float),
                          'aLSM':lds.iloc[3::3,4].values.astype(float),'bLSM':lds.iloc[4::3,4].values.astype(float),
                          'CHI2':lds.iloc[5::3,4].values.astype(float)})
    elif how.lower() in ['k2','kepler']:
        arr = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"KeplerLDlaws.txt"),skip_header=2)
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

def roll_rollangles(roll_angles):
    """Shifting the roll angles in such a way that the largest jump in the data does not occur during the time series of sorted rollangles.
    For example, continuous roll angles with a jump from 150-210 degrees would have the subsequent values "rolled" to the start to be continous from -160 to 150.
    This assists with e.g. roll angle plotting

    Args:
        roll_angles (np.ndarray): Array of Cheops roll angles

    Returns:
        rolled_roll_angles: [description]
    """
    sorted_roll_angles=np.sort(roll_angles)
    if np.max(np.diff(sorted_roll_angles))>10:
        phi_jump=0.5*(sorted_roll_angles[np.argmax(np.diff(sorted_roll_angles))+1]+sorted_roll_angles[np.argmax(np.diff(sorted_roll_angles))])
        phi_jump+=np.random.normal(0,1e-5) #Doing this in case median rolls these onto zero offsets which messes up stuff in the future
        #print(np.max(np.diff(sorted_roll_angles)),sorted_roll_angles[np.argmax(np.diff(sorted_roll_angles))],sorted_roll_angles[np.argmax(np.diff(sorted_roll_angles))+1])
    else:
        phi_jump=0
    print(phi_jump)
    return (roll_angles-phi_jump)%360+phi_jump

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