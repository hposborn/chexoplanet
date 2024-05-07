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
    #Checking to s
    return None

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