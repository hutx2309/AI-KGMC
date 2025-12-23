import numpy as np
import geopandas as gpd
import os
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import savgol_filter
import warnings
warnings.simplefilter("ignore", UserWarning)

import configparser
config = configparser.ConfigParser()
config.read('config.fig')

yr_start = config['general'].getint('experiment_start_year')
yr_end = config['general'].getint('experiment_end_year')
###  project dir
prj_dir = config['general']['prj_dir']

###
calib_out = config['Calibration_Dir']['Output_Dir']

year_days = 365
 
sel_fields = gpd.read_file('sample0.2field_4calib.geojson')
# load organized observations
obs_GPP = np.load(calib_out + "obs_SLOPE_GPP.npy")
obs_crop_type = np.load(calib_out + "crop_window.npy")

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

def dbl_logistic(x, a3, a4, a6, a7, a8):
    #sigma = (a3*a4+a6*a7)/(a3+a6)
    ST = 1/(1+np.exp(-a3*(x-a4)))     
    FT = 1/(1+np.exp(-a6*(a7-x)))    
    
    a8 = 0
    bias = 1000
    iis = np.arange(365)
    STi = 1/(1+np.exp(-a3*(iis-a4)))  
    FTi = 1/(1+np.exp(-a6*(a7-iis)))
    delta = np.abs(STi - FTi)
    if np.min(delta) < bias:
        a8 = np.argmin(delta)
    
    
    #a8 = (a3*a4+a6*a7)/(a3+a6)
    ratio = x.copy()*0 + 1
    ratio[x>a8] = 0
    y = ST*ratio + FT*(1-ratio)
    #y[int(a8)-1:int(a8)+2] = a2
    return y

def initial_paras(GPP_TS):
    # determine a2, peak of a time series
    a2 = 1
  
    # determine a8, date of peak of a time series
    days = []
    for i in range(120, 300):
        if GPP_TS[i]>a2-0.05:
            days.append(i)
    
    if len(days) == 0:
        GPP_TS_new = GPP_TS.copy()
        x = np.where(GPP_TS_new>0.1)[0]
        y = GPP_TS_new[x]
        xx = np.where(GPP_TS_new<=0.1)[0]
        f=interpolate.interp1d(x,y,kind="slinear", fill_value="extrapolate")
        GPP_TS_new[xx] = f(xx)
        for i in range(120, 300):
            if GPP_TS_new[i]>a2-0.05:
                days.append(i)
                
    a8 = days[int(len(days)/2)]
    
 
    a1 = 0
    
    # determine a3, slope of spring time series
    GPP_TS_S = GPP_TS[0:a8+15]
    GPP_TS_new = GPP_TS_S.copy()
    x = np.where(GPP_TS_new>0.0)[0]
    y = GPP_TS_new[x]
    xx = np.arange(np.min(x),np.max(x),5)
    f=interpolate.interp1d(x,y,kind="slinear", fill_value="extrapolate")
    GPP_TS_new[xx] = f(xx)
    x = []
    y = []
    for i in range(len(GPP_TS_new)):
        if GPP_TS_new[i]>a1 and GPP_TS_new[i]<a2+0.03:
            x.append(i)
            y.append(GPP_TS_new[i])
    
    if len(x)<3:
        GPP_TS_new = GPP_TS_S.copy()
        x = np.where(GPP_TS_new>0.0)[0]
        y = GPP_TS_new[x]
        xx = np.arange(np.min(x),np.max(x),1)
        f=interpolate.interp1d(x,y,kind="slinear", fill_value="extrapolate")
        GPP_TS_new[xx] = f(xx)
        x = []
        y = []
        for i in range(len(GPP_TS_new)):
            if GPP_TS_new[i]>a1 and GPP_TS_new[i]<a2:
                x.append(i)
                y.append(GPP_TS_new[i])
    if len(x)<2:
        coff = [0, 0]
    else:
        coff = np.polyfit(x,y,1)  

    for i in range(len(x)-4):
        if x[i<30]:
            continue
        coff1 = np.polyfit(x[i:i+4],y[i:i+4],1)
        if (coff1[0])>(coff[0]):
            coff = coff1
    a3 = coff[0]
    b3 = coff[1]
    
    # determine a4, middele point of spring time series
    a4 = (a1/2+a2/2-b3)/a3
    if a1>0.4:
        a4 = (0.6-b3)/a3
    a3 = 4*a3/(a2-a1)
    #a3 = max(a3, 0.2)
    
    
    # determine a5, baseline of fall time series
    GPP_TS_F = GPP_TS[a8:]
    GPP_TS_F = GPP_TS_F[::-1]
    
    a5 = 0
    
    # determine a6, slope of fall time series
    
    GPP_TS_new = GPP_TS_F.copy()
    x = np.where(GPP_TS_new>0.1)[0]
    y = GPP_TS_new[x]
    xx = np.arange(np.min(x),np.max(x),10)
    f=interpolate.interp1d(x,y,kind="slinear", fill_value="extrapolate")
    GPP_TS_new[xx] = f(xx)
    x = []
    y = []
    for i in range(len(GPP_TS_new)):
        if GPP_TS_new[i]>a5 and GPP_TS_new[i]<a2:
            x.append(i)
            y.append(GPP_TS_new[i])
    
    if len(x)<3:
        GPP_TS_new = GPP_TS_F.copy()
        x = np.where(GPP_TS_new>0.1)[0]
        y = GPP_TS_new[x]
        xx = np.arange(np.min(x),np.max(x),1)
        f=interpolate.interp1d(x,y,kind="slinear", fill_value="extrapolate")
        GPP_TS_new[xx] = f(xx)
        x = []
        y = []
        for i in range(len(GPP_TS_new)):
            if GPP_TS_new[i]>a5 and GPP_TS_new[i]<a2:
                x.append(i)
                y.append(GPP_TS_new[i])
    if len(x)<2:
        coff = [0, 0]
    else:
        coff = np.polyfit(x,y,1)    
    for i in range(len(x)-2):
        coff1 = np.polyfit(x[i:i+3],y[i:i+3],1)
        if (coff1[0])>(coff[0]):
            coff = coff1
    a6 = coff[0]
    b6 = coff[1]
    
    # determine a7, middele point of spring time series
    a7 = len(GPP_TS)-(a5/2+a2/2-b6)/a6
    if a5>0.4:
        a7 = len(GPP_TS)-(0.6-b6)/a6
    #a7 = len(NDVI_TS)-(0.6-b6)/a6
    a6 = 4*a6/(a2-a5)
    #a6 = max(a6, 0.2)

    
    return a3, a4, a6, a7, a8

def find50(GPP_TS, paras):
    up50_date = 0
    up50_value = 0
    
    down50_date = 0
    down50_value = 0
    
    a1 = 0
    a2 = 1
    a5 = 0
    a8 = int(paras[-1])
    
    spring_ts = GPP_TS[0:a8]
    for i in range(len(spring_ts)-1):
        if spring_ts[i]<=0.5*(a1+a2) and spring_ts[i+1]>=0.5*(a1+a2):
            up50_date = i+1
            up50_value = spring_ts[i]
            break
    
    fall_ts = GPP_TS[a8:]
    for i in range(len(fall_ts)-1):
        if fall_ts[i]>=0.5*(a5+a2) and fall_ts[i+1]<=0.5*(a5+a2):
            down50_date = a8+i+1
            down50_value = fall_ts[i]
            break
    
    return up50_date, up50_value, down50_date, down50_value

def cal_GPP_width(GPP_ts, crop_window):
    fitted_GPP = np.zeros(GPP_ts.shape)*np.nan
    points_50 = np.zeros(GPP_ts.shape)*np.nan
 
    for yr in range(0,GPP_ts.shape[0]):
        try:
            o_tmp_GPP_TS = GPP_ts[yr,:]
            # constrained by crop window
            o_tmp_cw_TS = crop_window[yr, :]
            o_tmp_GPP_TS[np.isnan(o_tmp_GPP_TS)] = 0
            o_tmp_GPP_TS[o_tmp_cw_TS == 0] = 0
            tmp_GPP_TS = o_tmp_GPP_TS.copy()
            
            if len(tmp_GPP_TS[tmp_GPP_TS > 0])> 30: 
                ## Max filter 
                for k in range(3, len(tmp_GPP_TS)-3):
                    tmp_GPP_TS[k] = np.max(o_tmp_GPP_TS[k-3:k+3])
                    
                ## Determine the envelopes of GPP time series
                lmin, lmax = hl_envelopes_idx(tmp_GPP_TS)
                time = np.arange(len(tmp_GPP_TS))+1
                
                ## Do double logistic fitting 
                xx = time[lmax]
                yy = tmp_GPP_TS[lmax]

                f=interpolate.interp1d(xx,yy,kind="linear", fill_value="extrapolate")
                new_GPP_TS = f(time)
                new_GPP_TS[np.max(lmax):] = new_GPP_TS[np.max(lmax)]
                new_GPP_TS[:np.min(lmax)] = new_GPP_TS[np.min(lmax)]
                
                R = 7
                new_GPP_TS = savgol_filter(new_GPP_TS,R*2+1,1) 

                GPP_max = np.max(new_GPP_TS)
                new_GPP_TS = new_GPP_TS/GPP_max
                xx = time[lmax]
                yy = new_GPP_TS[lmax]
                
                paras = np.array(initial_paras(new_GPP_TS))
                
                bounds=((0.0, 30, 0.0, 180, 120), (0.3, 240,  0.3, 360, 330))

                try:
                    c1, cov1 = curve_fit(dbl_logistic, xx, yy, paras, bounds=bounds, maxfev=10000)
                except:
                    c1 = paras
                    
                ## Find 50% points
                fitted_GPP_TS = dbl_logistic(time, *c1)
                up50_date, up50_value, down50_date, up50_value = find50(fitted_GPP_TS, c1)
                fitted_GPP_TS = fitted_GPP_TS*GPP_max
                
                xx = np.array([up50_date, down50_date])
                yy = np.array([up50_value, up50_value])*GPP_max

                # combine time series
                fitted_GPP[yr, :] = fitted_GPP_TS
                points_50[yr, xx[0]] = yy[0]
                points_50[yr, xx[1]] = yy[1]
        except:
            print("cal GPP width failed for year %d"%(yr))
    return fitted_GPP, points_50 

# calculate groupx lenght at 50% points
fit_GPP = np.zeros(obs_GPP.shape)*np.nan
point_50 = np.zeros(obs_GPP.shape)*np.nan

for i in range(0, len(sel_fields)):
    # ----------- filter non croplands --------------------------
    tmp_cw = obs_crop_type[i,:,:].reshape(-1)
    if max(tmp_cw) > 10.0 or min(tmp_cw) < 0.0 or max(tmp_cw) <=0:
        continue

    ct_id = sel_fields['cty_id'][i]
    uid =sel_fields["unique_fid"][i]
    print(i, '...', ct_id, '...', uid)
    # -------------get the data for each county-------------------
    cw_i = obs_crop_type[i, :, :year_days]
    obs_GPP_i = obs_GPP[i,:,:year_days]
    fitted_GPP, points_50  = cal_GPP_width(obs_GPP_i, cw_i)
    
    # combine time series
    fit_GPP[i, :, :] = fitted_GPP
    point_50[i,:, :] = points_50 
    
output_dir = calib_out + 'SLOPE_GPP_width/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir,exist_ok=True)  

np.save(output_dir + "/fit_GPP.npy", fit_GPP)
np.save(output_dir + "/point_50.npy", point_50)