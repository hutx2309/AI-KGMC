# pymoo 0.6.0
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination.ftol import MultiObjectiveSpaceTermination

from pymoo.config import Config
Config.show_compile_hint = False

from ecosystools.task_pool import MPITaskPool

from ecosystools.utils import *
from ecosystools.deep_learning import *
from ecosystools import Soil,Climate,Plant
from osgeo import ogr
# from ecosystools.task_pool import MPITaskPool
import geopandas as gpd
import numpy as np
import os, time

from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import savgol_filter
 
import configparser
import ast
config = configparser.ConfigParser()
config.read('config.fig')

###
model_start_year = config['general'].getint('model_start_year')
experiment_start_year = config['general'].getint('experiment_start_year')
experiment_end_year = config['general'].getint('experiment_end_year')
###
NLDAS_database = config['general']['NLDAS_database']
gSSURGO_Dir = config['general']['gSSURGO_Dir']
Global_DEM_file = config['general']['Global_DEM_file']
Global_Ta_file = config['general']['Global_Ta_file']

##
PFT_dir = config['general']['PFT_dir']

## copy PFT files to memory
tmp_PFT_dir = '/dev/shm/ecosys_PFT_%s/'% getpass.getuser()
if not os.path.exists(tmp_PFT_dir):
    os.makedirs(tmp_PFT_dir,exist_ok=True)  
os.system('cp -rf %s %s'% (PFT_dir, tmp_PFT_dir))

###  project dir
prj_dir = config['general']['prj_dir']
SurroModel_Dir = config['Surrogate_model_Dir']['SurrogateModel_Dir']

calib_out = config['Calibration_Dir']['Output_Dir']

#
Crop_Type_Names = {}
for key in config['CropID']:
    Crop_Type_Names[key]=config['CropID'][key]

#
plant_density = {}
for key in config['plant_density']:
    plant_density[key]=config['plant_density'].getfloat(key)
    
#
yield_convert_factor = {}
for key in config['yield_convert_factor']:
    yield_convert_factor[key]=config['yield_convert_factor'].getfloat(key)

# 
year_days = 365
days = year_days*(experiment_end_year-experiment_start_year+1)
year_range = np.arange(experiment_start_year,experiment_end_year+1)

# the dir to save the model outputs
model_dir = SurroModel_Dir + 'model/'
param_dir = SurroModel_Dir + 'seqMDF_r22_SLOPE.2fields_MN/'
if not os.path.exists(param_dir):
    os.makedirs(param_dir,exist_ok=True)

sel_fields = gpd.read_file('sample0.2field_4calib.geojson')
uids=np.array(sel_fields['unique_fid'].values)#np.array(shp.index)
ct_ids =np.array(sel_fields['cty_id'].values)
field_Lon = sel_fields.centroid.x.values
field_Lat = sel_fields.centroid.y.values

# load organized observations
obs_GPP = np.load(calib_out + "/obs_SLOPE_GPP.npy")
obs_Yield = np.load(calib_out + "/obs_Yield.npy")
obs_crop_type = np.load(calib_out + "/crop_window.npy")
N_fertilizer = np.load(calib_out + "/N_fertilizer.npy")
P_fertilizer = np.load(calib_out + "/P_fertilizer.npy")
till_dep = np.load(calib_out + "/Till_Dep.npy")
till_mix = np.load(calib_out + "/Till_Mix.npy")

# load 50% points of observations
point_50 =  np.load(calib_out + "SLOPE_GPP_width/point_50.npy")

# 
parameters = []

problem_combine = {}

for k in config['Calibrated_parameters'].keys():
    if k == 'nsamples':
        continue
    tmp = ast.literal_eval(config['Calibrated_parameters'][k])
    problem_={
        'num_vars':len(tmp.keys()),
        'names':list(tmp.keys()),
        'bounds': [tmp[k] for k in tmp.keys()]
    }
    problem_combine[k]=problem_
    
    
    exec('problem_%s=problem_'% k)
    parameters.extend(list(tmp.keys()))
    
    #param_values_ = np.load(work_output_dir+'param_values_%s.npy' % k)
    #exec('param_values_%s=param_values_'% k)
#del param_values_,problem_
del problem_

# 
parameters=np.array(parameters)
parameters=np.sort(np.unique(parameters))

# group parameters
problem_maiz31_1 = {'num_vars': 1,
 'names': ['GROUPX'],
 'bounds': [[8, 22]]}

problem_soyb31_1 = {'num_vars': 1,
  'names': ['GROUPX'],
  'bounds': [[8, 22]]}


problem_maiz31_2 = {'num_vars': 1,
 'names': ['CHL'],
 'bounds': [[0.01, 0.06]]}

problem_soyb31_2 = {'num_vars': 1,
  'names': ['CHL'],
  'bounds': [[0.01, 0.06]]}

problem_maiz31_3 = {'num_vars': 2,
  'names': ['STMX','SDMX'],
  'bounds': [[0.5, 5],[1,10]]}

problem_soyb31_3 = {'num_vars': 2,
  'names': ['STMX','SDMX'],
  'bounds': [[0.5, 5],[1,8]]}
 
problem_combine_1 = {'maiz31': {'num_vars': 1,
  'names': ['GROUPX'],
  'bounds': [[8, 22]]},
 'soyb31': {'num_vars': 1,
  'names': ['GROUPX'],
  'bounds': [[8, 22]]}}

problem_combine_2 = {'maiz31': {'num_vars': 1,
  'names': ['CHL'],
  'bounds': [[0.01, 0.06]]},
 'soyb31': {'num_vars': 1,
  'names': ['CHL'],
  'bounds': [[0.01, 0.06]]}}

problem_combine_3 = {'maiz31': {'num_vars': 2,
  'names': ['STMX','SDMX'],
  'bounds': [[0.5,5],[1,10]]},
 
 'soyb31': {'num_vars': 2,
  'names': ['STMX','SDMX'],
  'bounds': [[0.5,5],[1,8]]}}

# 
input_names = ['RADN','TMAX_AIR','TMIN_AIR','HMAX_AIR','HMIN_AIR','WIND','PRECN', # climate
             'BKDS_0_5','SOC_0_5','CSAND_0_5','CSILT_0_5','FC_0_5','WP_0_5','SCNV_0_5','PH_0_5','CEC_0_5', # soil layer 1
             'BKDS_5_30','SOC_5_30','CSAND_5_30','CSILT_5_30','FC_5_30','WP_5_30','SCNV_5_30','PH_5_30','CEC_5_30', # soil layer 2
             'BKDS_30_100','SOC_30_100','CSAND_30_100','CSILT_30_100','FC_30_100','WP_30_100','SCNV_30_100','PH_30_100','CEC_30_100', # soil layer 3
             'DOY','N_fertilizer','P_fertilizer', 'Crop_Type','Till_Dep', 'Till_Mix' ]
input_names.extend(parameters)

output_names=np.array(['GPP','ET','Yield'])#,'Yield'])
n_feature = len(input_names)
n_output = len(output_names)

# 
v_max = ast.literal_eval(config['scales']['v_max'])
v_min = ast.literal_eval(config['scales']['v_min'])

# 
def Get_X(fi,PFT_params,climate,soil, step, temp_x2, temp_x3):
 
    for k in PFT_params.keys():
        exec("param_values_%s=PFT_params['%s']" % (k,k))
        n_samples=len(PFT_params[k])
 
    X = np.zeros((n_samples,days,len(input_names)))
    for year in range(experiment_start_year,experiment_end_year+1): # climate data
        start_i = year_days*(year-experiment_start_year)
        end_i = year_days*(year-experiment_start_year+1)
 
        X[:,start_i:end_i,0] = maxmin_norm_with_scaler(climate[0,year-experiment_start_year,:(end_i-start_i)],v_max['RADN'],v_min['RADN']) # add 7 climate data
        X[:,start_i:end_i,1] = maxmin_norm_with_scaler(climate[1,year-experiment_start_year,:(end_i-start_i)],v_max['TMAX_AIR'],v_min['TMAX_AIR']) # add 7 climate data
        X[:,start_i:end_i,2] = maxmin_norm_with_scaler(climate[2,year-experiment_start_year,:(end_i-start_i)],v_max['TMIN_AIR'],v_min['TMIN_AIR']) # add 7 climate data
        X[:,start_i:end_i,3] = maxmin_norm_with_scaler(climate[3,year-experiment_start_year,:(end_i-start_i)],v_max['HMAX_AIR'],v_min['HMAX_AIR']) # add 7 climate data
        X[:,start_i:end_i,4] = maxmin_norm_with_scaler(climate[4,year-experiment_start_year,:(end_i-start_i)],v_max['HMIN_AIR'],v_min['HMIN_AIR']) # add 7 climate data
        X[:,start_i:end_i,5] = maxmin_norm_with_scaler(climate[5,year-experiment_start_year,:(end_i-start_i)],v_max['WIND'],v_min['WIND']) # add 7 climate data
        X[:,start_i:end_i,6] = maxmin_norm_with_scaler(climate[6,year-experiment_start_year,:(end_i-start_i)],v_max['PRECN'],v_min['PRECN']) # add 7 climate data

        #DOY   
        X[:,start_i:end_i,34] = maxmin_norm_with_scaler(np.arange(1,end_i-start_i+1).reshape(1,-1),v_max['DOY'],v_min['DOY'])
        X[:,start_i:end_i,35] = maxmin_norm_with_scaler(N_fertilizer[fi,year-experiment_start_year,:(end_i-start_i)].reshape(1,-1),v_max['N_fertilizer'],v_min['N_fertilizer'])#N_fertilizer
        X[:,start_i:end_i,36] = maxmin_norm_with_scaler(P_fertilizer[fi,year-experiment_start_year,:(end_i-start_i)].reshape(1,-1),v_max['P_fertilizer'],v_min['P_fertilizer'])#N_fertilizer

        X[:,start_i:end_i,37] = obs_crop_type[fi,year-experiment_start_year,:(end_i-start_i)].reshape(1,-1)
        
        crop_type = X[:,start_i:end_i,37]
        
     
        X[:,start_i:end_i,38] = till_dep[fi,year-experiment_start_year,:(end_i-start_i)].reshape(1,-1)

 
        X[:,start_i:end_i,39] = till_mix[fi,year-experiment_start_year,:(end_i-start_i)].reshape(1,-1)
        #parameters
        unique_crop_type = np.unique(crop_type).astype('int')
        for vi in range(len(parameters)):
            var_=parameters[vi]
            param_ = np.zeros((n_samples,end_i-start_i))
            for crp_ in unique_crop_type:
                if str(crp_) not in Crop_Type_Names.keys():
                    continue
                crop_name = Crop_Type_Names[str(crp_)]

                if (crop_name in config['Calibrated_parameters'].keys()):
                    sampled_parameters = np.array(eval('problem_%s_%d' % (crop_name, step))['names'])

                    if step == 1:
                        if (var_ in sampled_parameters):
                            var_values_ = eval('param_values_%s'% crop_name)[:,var_==sampled_parameters]
                            var_values_ = var_values_.reshape(-1,1) + np.zeros((end_i-start_i)).reshape(1,-1)
                            #var_values_ = torch.squeeze(var_values_)
                            param_[crop_type==crp_] = maxmin_norm_with_scaler(var_values_[crop_type==crp_],v_max[var_],v_min[var_])
                        else:
                            p_ = Plant.Plant_Species(tmp_PFT_dir+'PFT_2021/'+crop_name)
                            param_[crop_type==crp_] = maxmin_norm_with_scaler(float(eval('p_.%s' % var_)),v_max[var_],v_min[var_]) 

                    elif step == 2:
                        if (var_ in sampled_parameters):
                            var_values_ = eval('param_values_%s'% crop_name)[:,var_==sampled_parameters]
                            var_values_ = var_values_.reshape(-1,1) + np.zeros((end_i-start_i)).reshape(1,-1)
                            #var_values_ = torch.squeeze(var_values_)
                            param_[crop_type==crp_] = maxmin_norm_with_scaler(var_values_[crop_type==crp_],v_max[var_],v_min[var_])
                        else:
                            keys = temp_x2.get(crop_name)
                            if var_ in keys:
                                temp_X = temp_x2[crop_name][var_]   # use values already calibrated for GPP in step 1
                                param_[crop_type==crp_] = maxmin_norm_with_scaler(temp_X,v_max[var_],v_min[var_])   
                            else:
                                p_ = Plant.Plant_Species(tmp_PFT_dir+'PFT_2021/'+crop_name)
                                param_[crop_type==crp_] = maxmin_norm_with_scaler(float(eval('p_.%s' % var_)),v_max[var_],v_min[var_])   

                    elif step == 3:

                        if (var_ in sampled_parameters):
                            var_values_ = eval('param_values_%s'% crop_name)[:,var_==sampled_parameters]
                            var_values_ = var_values_.reshape(-1,1) + np.zeros((end_i-start_i)).reshape(1,-1)
                            #var_values_ = torch.squeeze(var_values_)
                            param_[crop_type==crp_] = maxmin_norm_with_scaler(var_values_[crop_type==crp_],v_max[var_],v_min[var_])
                        else:
                            keys = temp_x3.get(crop_name)
                            if var_ in keys:
                                temp_X = temp_x3[crop_name][var_]   # use values already calibrated for GPP in step 1 and step 2
                                param_[crop_type==crp_] = maxmin_norm_with_scaler(temp_X,v_max[var_],v_min[var_])   
                            else:
                                p_ = Plant.Plant_Species(tmp_PFT_dir+'PFT_2021/'+crop_name)
                                param_[crop_type==crp_] = maxmin_norm_with_scaler(float(eval('p_.%s' % var_)),v_max[var_],v_min[var_])   
                            
                else:
                    p_ = Plant.Plant_Species(tmp_PFT_dir+'PFT_2021/'+crop_name)
                    param_[crop_type==crp_] = maxmin_norm_with_scaler(float(eval('p_.%s' % var_)),v_max[var_],v_min[var_])
                X[:,start_i:end_i,40+vi]=param_
       
    #soil data
    X[:,:,7] = maxmin_norm_with_scaler(soil['Soil_BKDS'][0].reshape(-1,1),v_max['BKDS'],v_min['BKDS'])
    X[:,:,8] = maxmin_norm_with_scaler(soil['Soil_SOC'][0].reshape(-1,1),v_max['SOC'],v_min['SOC'])
    X[:,:,9] = maxmin_norm_with_scaler(soil['Soil_SAND'][0].reshape(-1,1),v_max['CSAND'],v_min['CSAND'])
    X[:,:,10] = maxmin_norm_with_scaler(soil['Soil_SILT'][0].reshape(-1,1),v_max['CSILT'],v_min['CSILT'])
    X[:,:,11] = maxmin_norm_with_scaler(soil['Soil_FC'][0].reshape(-1,1),v_max['FC'],v_min['FC'])
    X[:,:,12] = maxmin_norm_with_scaler(soil['Soil_WP'][0].reshape(-1,1),v_max['WP'],v_min['WP'])
    X[:,:,13] = maxmin_norm_with_scaler(soil['Soil_KSat'][0].reshape(-1,1),v_max['SCNV'],v_min['SCNV'])
    X[:,:,14] = maxmin_norm_with_scaler(soil['Soil_PH'][0].reshape(-1,1),v_max['PH'],v_min['PH'])
    X[:,:,15] = maxmin_norm_with_scaler(soil['Soil_CEC'][0].reshape(-1,1),v_max['CEC'],v_min['CEC'])

    X[:,:,16] = maxmin_norm_with_scaler(soil['Soil_BKDS'][1].reshape(-1,1),v_max['BKDS'],v_min['BKDS'])
    X[:,:,17] = maxmin_norm_with_scaler(soil['Soil_SOC'][1].reshape(-1,1),v_max['SOC'],v_min['SOC'])
    X[:,:,18] = maxmin_norm_with_scaler(soil['Soil_SAND'][1].reshape(-1,1),v_max['CSAND'],v_min['CSAND'])
    X[:,:,19] = maxmin_norm_with_scaler(soil['Soil_SILT'][1].reshape(-1,1),v_max['CSILT'],v_min['CSILT'])
    X[:,:,20] = maxmin_norm_with_scaler(soil['Soil_FC'][1].reshape(-1,1),v_max['FC'],v_min['FC'])
    X[:,:,21] = maxmin_norm_with_scaler(soil['Soil_WP'][1].reshape(-1,1),v_max['WP'],v_min['WP'])
    X[:,:,22] = maxmin_norm_with_scaler(soil['Soil_KSat'][1].reshape(-1,1),v_max['SCNV'],v_min['SCNV'])
    X[:,:,23] = maxmin_norm_with_scaler(soil['Soil_PH'][1].reshape(-1,1),v_max['PH'],v_min['PH'])
    X[:,:,24] = maxmin_norm_with_scaler(soil['Soil_CEC'][1].reshape(-1,1),v_max['CEC'],v_min['CEC'])

    X[:,:,25] = maxmin_norm_with_scaler(soil['Soil_BKDS'][2].reshape(-1,1),v_max['BKDS'],v_min['BKDS'])
    X[:,:,26] = maxmin_norm_with_scaler(soil['Soil_SOC'][2].reshape(-1,1),v_max['SOC'],v_min['SOC'])
    X[:,:,27] = maxmin_norm_with_scaler(soil['Soil_SAND'][2].reshape(-1,1),v_max['CSAND'],v_min['CSAND'])
    X[:,:,28] = maxmin_norm_with_scaler(soil['Soil_SILT'][2].reshape(-1,1),v_max['CSILT'],v_min['CSILT'])
    X[:,:,29] = maxmin_norm_with_scaler(soil['Soil_FC'][2].reshape(-1,1),v_max['FC'],v_min['FC'])
    X[:,:,30] = maxmin_norm_with_scaler(soil['Soil_WP'][2].reshape(-1,1),v_max['WP'],v_min['WP'])
    X[:,:,31] = maxmin_norm_with_scaler(soil['Soil_KSat'][2].reshape(-1,1),v_max['SCNV'],v_min['SCNV'])
    X[:,:,32] = maxmin_norm_with_scaler(soil['Soil_PH'][2].reshape(-1,1),v_max['PH'],v_min['PH'])
    X[:,:,33] = maxmin_norm_with_scaler(soil['Soil_CEC'][2].reshape(-1,1),v_max['CEC'],v_min['CEC'])

    return X

# functions for calculating the length of two 50% points in GPP curves
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

n_hidden=64#256#64#128#64#64 #hidden state number
n_layers=2#2#4 #layer of lstm
model_version='LSTM_%dl_%d_U_case_%s.sav' % (n_layers,n_hidden,'_'.join(output_names))  #####save file !!!!!!!!!!!!!!!!!!!! change this before new training
path_save = model_dir + model_version + 'best'
#build model and move to GPU
model=LSTM(n_feature,n_hidden,n_layers,n_output,0,0.2)
model,train_losses,val_losses = load_model(model,path_save)

# Step 1: define the problem of calibrating GPP using GROUPX
class Problem_GPP_1(Problem):
    def __init__(self,fi,climate,soil):
        xl=[]
        xu=[]

        for c in problem_combine_1.keys():
            for vi in range(len(problem_combine_1[c]['names'])):
                xl.append(problem_combine_1[c]['bounds'][vi][0])
                xu.append(problem_combine_1[c]['bounds'][vi][1])

        super().__init__(n_var=2,n_obj=1,xl=xl,xu=xu)

        self.fi = fi
        self.climate = climate
        self.soil = soil
        #self.croptypes = croptypes

    def _evaluate(self, X, out, *args, **kwargs):
        PFT_params={}
        ii = 0
        for c in problem_combine_1.keys():
            nc = len(problem_combine_1[c]['names'])
            PFT_params[c] = torch.from_numpy(X[:,ii:(ii+nc)])
            ii = ii + nc

        fi = int(self.fi)
        
        #
        # t=time.time()
        X_inputs = Get_X(fi,PFT_params,self.climate,self.soil,1,0,0)
        # print(1,time.time()-t)
        #
        Y1_pred,Y2_pred = model_prediction(model,X_inputs,slide_window=365,n_output1=n_output)
        Y_pred_GPP_daily = maxmin_norm_reverse(Y1_pred[:,:,0],v_max['GPP'],v_min['GPP']) #GPP
        # print(2,time.time()-t)
       
        sim_GPP = np.zeros((X_inputs.shape[0],(experiment_end_year-experiment_start_year+1),year_days))*np.nan
 
        for year in range(experiment_start_year,experiment_end_year+1):
            start_i = year_days*(year-experiment_start_year)
            end_i = year_days*(year-experiment_start_year+1)
            sim_GPP[:,year-experiment_start_year,:year_days] = Y_pred_GPP_daily[:,start_i:end_i]
 
        # print(3,time.time()-t)

        GPP_width_diff = []

        # here X_inputs have a population size of algorithm
        for i in range(0,X_inputs.shape[0]):
            sim_GPP_site = sim_GPP[i,:,:]
            Ndiff_GPP = 0
           
            for yr in range(0,sim_GPP_site.shape[0]):
                # simulated GPP curves
                tmp_simGPP_TS0 = sim_GPP_site[yr,:]
                tmp_simGPP_TS0[np.isnan(tmp_simGPP_TS0)] = 0
                tmp_simGPP_TS = tmp_simGPP_TS0.copy()
                
                # at least 30 days of gpp are not zeros 
                if len(tmp_simGPP_TS[tmp_simGPP_TS > 0])>30: 
                    ## Max filter 
                    for k in range(3, len(tmp_simGPP_TS)-3):
                        tmp_simGPP_TS[k] = np.max(tmp_simGPP_TS0[k-3:k+3])
                
                    ## Determine the envelopes of GPP time series
                    lmin_sim, lmax_sim = hl_envelopes_idx(tmp_simGPP_TS)
                    time_sim = np.arange(len(tmp_simGPP_TS))+1
                    ## Do double logistic fitting 
                    xx_sim = time_sim[lmax_sim]
                    yy_sim = tmp_simGPP_TS[lmax_sim]

                    f_sim=interpolate.interp1d(xx_sim,yy_sim,kind="linear", fill_value="extrapolate")
                    new_simGPP_TS = f_sim(time_sim)
                    new_simGPP_TS[np.max(lmax_sim):] = new_simGPP_TS[np.max(lmax_sim)]
                    new_simGPP_TS[:np.min(lmax_sim)] = new_simGPP_TS[np.min(lmax_sim)]

                    R = 7
                    new_simGPP_TS = savgol_filter(new_simGPP_TS,R*2+1,1) 
 
                    simGPP_max = np.max(new_simGPP_TS)
                    new_simGPP_TS = new_simGPP_TS/simGPP_max
                    xx_sim = time_sim[lmax_sim]
                    yy_sim = new_simGPP_TS[lmax_sim]
                    try:
                        paras_sim = np.array(initial_paras(new_simGPP_TS))
                    except:
                        continue
                    bounds=((0.0, 30, 0.0, 180, 120), (0.3, 240,  0.3, 360, 330))
                    try:
                        c2, cov2 = curve_fit(dbl_logistic, xx_sim, yy_sim, paras_sim, bounds=bounds, maxfev=10000)
                    except:
                        c2 = paras_sim   
                    ## Find 50% points
                    point_50_obs = point_50[fi,yr,:]
                    obs_dates = np.where(point_50_obs >0)[0]
                    if len(obs_dates) < 2 or np.isnan(obs_dates).all():
                        obs_dates = [120, 250]
                        # print("no observation dates")
                    else:
                        fitted_simGPP_TS = dbl_logistic(time_sim, *c2)
                        up50_date_sim, up50_value_sim, down50_date_sim, down50_value_sim = find50(fitted_simGPP_TS, c2)
                        Ndiff_GPP = Ndiff_GPP + abs(down50_date_sim - obs_dates[1]) + abs(up50_date_sim-obs_dates[0])
 
                else:
                    continue
 
            GPP_width_diff.append(Ndiff_GPP)
 
        # print('RMSE_GPP',RMSE_GPP_)
        # print(4,time.time()-t)
        out["F"] = np.column_stack([GPP_width_diff])

# Step 2: define the problem of calibrating GPP using VCMX
class Problem_GPP_2(Problem):
    def __init__(self,fi,climate,soil,temp_x2):
        xl=[]
        xu=[]

        for c in problem_combine_2.keys():
            for vi in range(len(problem_combine_2[c]['names'])):
                xl.append(problem_combine_2[c]['bounds'][vi][0])
                xu.append(problem_combine_2[c]['bounds'][vi][1])

        super().__init__(n_var=2,n_obj=1,xl=xl,xu=xu)

        self.fi = fi
        self.climate = climate
        self.soil = soil
        self.temp_x2 = temp_x2
        #self.croptypes = croptypes

    def _evaluate(self, X, out, *args, **kwargs):
        PFT_params={}
        ii = 0
        for c in problem_combine_2.keys():
            nc = len(problem_combine_2[c]['names'])
            PFT_params[c] = torch.from_numpy(X[:,ii:(ii+nc)])
            ii = ii + nc

        fi = int(self.fi)
        
        #
        # t=time.time()
        X_inputs = Get_X(fi,PFT_params,self.climate,self.soil,2,self.temp_x2,0)
        # print(1,time.time()-t)
        #
        Y1_pred,Y2_pred = model_prediction(model,X_inputs,slide_window=365,n_output1=n_output)
        Y_pred_GPP_daily = maxmin_norm_reverse(Y1_pred[:,:,0],v_max['GPP'],v_min['GPP']) #GPP
        # print(2,time.time()-t)
       
        sim_GPP = np.zeros((X_inputs.shape[0],(experiment_end_year-experiment_start_year+1),year_days))*np.nan
 
        for year in range(experiment_start_year,experiment_end_year+1):
            start_i = year_days*(year-experiment_start_year)
            end_i = year_days*(year-experiment_start_year+1)
            sim_GPP[:,year-experiment_start_year,:year_days] = Y_pred_GPP_daily[:,start_i:end_i]
 
        # print(3,time.time()-t)

        RMSE_GPP_ = []

        crop_types = obs_crop_type[fi,:experiment_end_year-experiment_start_year+1,:year_days]
        
        obs_GPP_site = obs_GPP[fi,:experiment_end_year-experiment_start_year+1,:year_days]
        for i in range(0,X_inputs.shape[0]):
            sim_GPP_tmp = sim_GPP[i,:,:]
            NRMSE_GPP = 0
            for ci in Crop_Type_Names.keys():
                if ci == 'covercrop':
                    continue
                ci=int(ci)
                if ci in crop_types:
                    obs_GPP_ = obs_GPP_site[crop_types==ci]
                    sim_GPP_ = sim_GPP_tmp[crop_types==ci]
                    if len(obs_GPP_[~np.isnan(obs_GPP_)]) > 0:
                        NRMSE_GPP = NRMSE_GPP + RMSE(obs_GPP_,sim_GPP_)/np.nanmean(obs_GPP_)
 
            RMSE_GPP_.append(NRMSE_GPP)
 
        # print('RMSE_GPP',RMSE_GPP_)
        # print(4,time.time()-t)
        out["F"] = np.column_stack([RMSE_GPP_])

# Step 3: define the problem of calibrating Yield
class Problem_Yield(Problem):
    def __init__(self,fi,climate,soil, temp_x3):
        xl=[]
        xu=[]

        for c in problem_combine_3.keys():
            for vi in range(len(problem_combine_3[c]['names'])):
                xl.append(problem_combine_3[c]['bounds'][vi][0])
                xu.append(problem_combine_3[c]['bounds'][vi][1])

        super().__init__(n_var=4,n_obj=1,xl=xl,xu=xu)

        self.fi = fi
        self.climate = climate
        self.soil = soil
        self.temp_x3 = temp_x3
        #self.croptypes = croptypes

    def _evaluate(self, X, out, *args, **kwargs):
        PFT_params={}
        ii = 0
        for c in problem_combine_3.keys():
            nc = len(problem_combine_3[c]['names'])
            PFT_params[c] = torch.from_numpy(X[:,ii:(ii+nc)])
            ii = ii + nc

        fi = int(self.fi)
        
        #
        # t=time.time()
        X_inputs = Get_X(fi,PFT_params,self.climate,self.soil, 3, 0, self.temp_x3)
        # print(1,time.time()-t)
        #
        Y1_pred,Y2_pred = model_prediction(model,X_inputs,slide_window=365,n_output1=n_output)
 
        Y_pred_Yield_daily = maxmin_norm_reverse(Y1_pred[:,:,2],v_max['Yield'],v_min['Yield']) #Yield
        # print(2,time.time()-t)
        
        sim_Yield = np.zeros((X_inputs.shape[0],(experiment_end_year-experiment_start_year+1),year_days))*np.nan
        for year in range(experiment_start_year,experiment_end_year+1):
            start_i = year_days*(year-experiment_start_year)
            end_i = year_days*(year-experiment_start_year+1)
 
            sim_Yield[:,year-experiment_start_year,:year_days] = Y_pred_Yield_daily[:,start_i:end_i]
            
        # print(3,time.time()-t)

        RMSE_Yield_ = []
        
        crop_types = obs_crop_type[fi,:experiment_end_year-experiment_start_year+1,:year_days]
        
        obs_Yield_site = obs_Yield[fi,:experiment_end_year-experiment_start_year+1,:year_days]
        for i in range(0,X_inputs.shape[0]):
           
            sim_Yield_tmp = sim_Yield[i,:,:]
            NRMSE_Yield = 0
            for ci in Crop_Type_Names.keys():
                if ci == 'covercrop':
                    continue
                ci=int(ci)
                if ci in crop_types:
                    obs_Yield_ = obs_Yield_site[crop_types==ci]
                    sim_Yield_ = sim_Yield_tmp[crop_types==ci]
                    if len(obs_Yield_[~np.isnan(obs_Yield_)]) > 0:
                        NRMSE_Yield = NRMSE_Yield + RMSE(obs_Yield_,sim_Yield_)/np.nanmean(obs_Yield_)
                        
            RMSE_Yield_.append(NRMSE_Yield)
  
        # print('RMSE_Yield',RMSE_Yield_)
        # print(4,time.time()-t)

        out["F"] = np.column_stack([RMSE_Yield_])

#get gSSURGO soil dataset
def write_mesoil(work_dir, u_idx, c_idx):
    driver = ogr.GetDriverByName("OpenFileGDB")
    conn = driver.Open(gSSURGO_Dir + 'gSSURGO_CONUS.gdb', 0)
    gSSURGO_MapunitRaster_file = gSSURGO_Dir + 'MapunitRaster_30m.tif'
    mapunits = Soil.get_gSSURGO_mapunit(gSSURGO_MapunitRaster_file,sel_fields.loc[(sel_fields["unique_fid"]==u_idx) & 
                                                                                  (sel_fields["cty_id"] == c_idx)])
    if len(mapunits)>0:
        cokeys, cokeys_pct = Soil.get_mapunit_components(conn,mapunits[0])
        if len(cokeys) > 0:
            output_file = work_dir + 'mesoil'
            try:
                for ci in range(len(cokeys)):
                    status = Soil.write_gSSURGO_soil(conn,cokeys[ci],output_file)
                    if not status:
                        print(u_idx,ci,'soil file incorrect')
                        if os.path.exists(output_file):
                            os.system('rm %s' % output_file)
                        continue
                    else:
                        print(u_idx,ci)
                        break

            except Exception as e:
                print(u_idx,e)
    else:
        print(u_idx,'no soil file corrected')   
    conn = None
    if not os.path.exists(output_file):
        return False
    else:
        return True

def resample_mesoil(mesoil_file):
    soil_layers = 3
    Soil_BKDS = np.zeros((soil_layers))*np.nan 
    Soil_SAND = np.zeros((soil_layers))*np.nan 
    Soil_SILT = np.zeros((soil_layers))*np.nan 
    Soil_FC = np.zeros((soil_layers))*np.nan 
    Soil_WP = np.zeros((soil_layers))*np.nan 
    Soil_KSat = np.zeros((soil_layers))*np.nan 
    Soil_SOC = np.zeros((soil_layers))*np.nan 
    Soil_PH = np.zeros((soil_layers))*np.nan 
    Soil_CEC = np.zeros((soil_layers))*np.nan 
    
    s = Soil.Soil()
    s.read_soil_file(mesoil_file)
    
    Soil_Depth =s.CDPTH.astype('float64')
    Soil_BKDS_ =s.BKDS.astype('float64')
    Soil_SAND_ =s.CSAND.astype('float64')    
    Soil_SILT_ =s.CSILT.astype('float64') 
    Soil_FC_ =s.FC.astype('float64')
    Soil_WP_ =s.WP.astype('float64')    
    Soil_KSat_ =s.SCNV.astype('float64')    
    Soil_SOC_ =s.CORGC.astype('float64')   
    Soil_PH_ =s.PH.astype('float64')    
    Soil_CEC_ =s.CEC.astype('float64')    
    
    Soil_BKDS[0],Soil_SOC[0],Soil_SAND[0],Soil_SILT[0],Soil_FC[0],Soil_WP[0],Soil_KSat[0],Soil_PH[0],Soil_CEC[0] = Soil.resample_soil_properties(
        Soil_Depth,0,0.05,Soil_BKDS_,Soil_SOC_,Soil_SAND_,Soil_SILT_,Soil_FC_,Soil_WP_,Soil_KSat_,Soil_PH_,Soil_CEC_)
    Soil_BKDS[1],Soil_SOC[1],Soil_SAND[1],Soil_SILT[1],Soil_FC[1],Soil_WP[1],Soil_KSat[1],Soil_PH[1],Soil_CEC[1] = Soil.resample_soil_properties(
        Soil_Depth,0.05,0.30,Soil_BKDS_,Soil_SOC_,Soil_SAND_,Soil_SILT_,Soil_FC_,Soil_WP_,Soil_KSat_,Soil_PH_,Soil_CEC_)
    Soil_BKDS[2],Soil_SOC[2],Soil_SAND[2],Soil_SILT[2],Soil_FC[2],Soil_WP[2],Soil_KSat[2],Soil_PH[2],Soil_CEC[2] = Soil.resample_soil_properties(
        Soil_Depth,0.30,1.00,Soil_BKDS_,Soil_SOC_,Soil_SAND_,Soil_SILT_,Soil_FC_,Soil_WP_,Soil_KSat_,Soil_PH_,Soil_CEC_)
    
    soil = {}
    soil['Soil_BKDS']=Soil_BKDS
    soil['Soil_SAND']=Soil_SAND
    soil['Soil_SILT']=Soil_SILT
    soil['Soil_FC']=Soil_FC
    soil['Soil_WP']=Soil_WP
    soil['Soil_KSat']=Soil_KSat
    soil['Soil_SOC']=Soil_SOC
    soil['Soil_PH']=Soil_PH
    soil['Soil_CEC']=Soil_CEC
    
    return soil


#for fi in range(0,len(sites)):
tmp_output_dir = '/dev/shm/Ecosys_calib_%s/' % getpass.getuser()
if not os.path.exists(tmp_output_dir):
    os.makedirs(tmp_output_dir,exist_ok=True)  

termination = MultiObjectiveSpaceTermination(tol=0.0025,  only_feas = True)
algorithm = NSGA2(pop_size=200)
# 

def work(task):
    t=time.time()
    u_id =uids[task]
    c_id = ct_ids[task]
    if os.path.exists(param_dir + '%d_%s_%s_NSGA2_STMX.npy' % (task, c_id, u_id)):
        return
 
    print(c_id, u_id)

    lon = field_Lon[np.logical_and(sel_fields["unique_fid"]==u_id, sel_fields["cty_id"] == c_id)][0]
    lat = field_Lat[np.logical_and(sel_fields["unique_fid"]==u_id, sel_fields["cty_id"] == c_id)][0]        
    site_DEM = get_elevation(Global_DEM_file,lon,lat)

    #read soil data
    work_dir = tmp_output_dir +'%s_%s/' % (c_id,u_id)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir,exist_ok=True)

    status = write_mesoil(work_dir,u_id, c_id)
    if not status:
        print("no soil data")  
    mesoil_file = work_dir+'mesoil'
    soil = resample_mesoil(mesoil_file)
    os.system('rm -r %s' % work_dir)    
    ##
    #read climate data
    climate =  np.zeros((7,experiment_end_year-experiment_start_year+1,366))*np.nan
    for year in range(experiment_start_year,experiment_end_year+1):
        #climate data
        RADN, TMAX_AIR, TMIN_AIR, HMAX_AIR, HMIN_AIR, WIND, PRECN = Climate.convert_from_NLDAS(NLDAS_database,lon,lat,year,elevation=site_DEM)
        climate[0,year-experiment_start_year,:]=RADN[:]
        climate[1,year-experiment_start_year,:]=TMAX_AIR[:]
        climate[2,year-experiment_start_year,:]=TMIN_AIR[:]
        climate[3,year-experiment_start_year,:]=HMAX_AIR[:]
        climate[4,year-experiment_start_year,:]=HMIN_AIR[:]
        climate[5,year-experiment_start_year,:]=WIND[:]
        climate[6,year-experiment_start_year,:]=PRECN[:]      

       # step 1: calibrate GPP
    vectorized_problem_gpp1 = Problem_GPP_1(task,climate,soil)

    res1_gpp = minimize(vectorized_problem_gpp1,
                algorithm,
                termination,
                seed=1,
                verbose=False)
    
    print(len(res1_gpp.X.shape))

    if len(res1_gpp.X.shape) > 1:
        np.save(param_dir + '%d_%s_%s_NSGA2_GROUPX.npy' % (task, c_id, u_id),res1_gpp.X[0])
        np.save(param_dir + '%d_%s_%s_NSGA2_target_values_GROUPX.npy' % (task, c_id, u_id),res1_gpp.F[0])  
        # get the parameters with minimized gpp error
        temp_x2 = {"maiz31":{"GROUPX": res1_gpp.X[0][0]}, "soyb31":{"GROUPX": res1_gpp.X[0][1]}}
    elif len(res1_gpp.X.shape) == 1:
        np.save(param_dir + '%d_%s_%s_NSGA2_GROUPX.npy' % (task, c_id, u_id),res1_gpp.X)
        np.save(param_dir + '%d_%s_%s_NSGA2_target_values_GROUPX.npy' % (task, c_id, u_id),res1_gpp.F)  
        temp_x2 = {"maiz31":{"GROUPX": res1_gpp.X[0]}, "soyb31":{"GROUPX": res1_gpp.X[1]}}
    else:
        print("calibration failed for gpp")
        
    # step 2: calibrate GPP with vcmax
    vectorized_problem_gpp2 = Problem_GPP_2(task, climate, soil, temp_x2)

    res2_gpp = minimize(vectorized_problem_gpp2,
                algorithm,
                termination,
                seed=1,
                verbose=False)


    print(len(res2_gpp.X.shape))
 
    if len(res2_gpp.X.shape) > 1:
        np.save(param_dir + '%d_%s_%s_NSGA2_CHL.npy' % (task, c_id, u_id),res2_gpp.X[0])
        np.save(param_dir + '%d_%s_%s_NSGA2_target_values_CHL.npy' % (task, c_id, u_id),res2_gpp.F[0])
        
        # get the parameters with minimized gpp error
        temp_x3 = {"maiz31":{"GROUPX": temp_x2["maiz31"]["GROUPX"],"CHL": res2_gpp.X[0][0]},                 
                    "soyb31":{"GROUPX": temp_x2["soyb31"]["GROUPX"],"CHL": res2_gpp.X[0][1]}}
        
    elif len(res2_gpp.X.shape) == 1:
        np.save(param_dir + '%d_%s_%s_NSGA2_CHL.npy' % (task, c_id, u_id),res2_gpp.X)
        np.save(param_dir + '%d_%s_%s_NSGA2_target_values_CHL.npy' % (task, c_id, u_id),res2_gpp.F)  
        
        temp_x3 = {"maiz31":{"GROUPX": temp_x2["maiz31"]["GROUPX"],"CHL": res2_gpp.X[0]},
                    "soyb31":{"GROUPX": temp_x2["soyb31"]["GROUPX"],"CHL": res2_gpp.X[1]}}
    else:
        print("calibration failed for gpp")
 
        
    # step 3: calibrate Yield
    vectorized_problem_Yield = Problem_Yield(task,climate,soil, temp_x3)

    res_yld = minimize(vectorized_problem_Yield,
                algorithm,
                termination,
                seed=1,
                verbose=False)

    np.save(param_dir +  '%d_%s_%s_NSGA2_STMX.npy' % (task, c_id, u_id),res_yld.X)
    np.save(param_dir + '%d_%s_%s_NSGA2_target_values_STMX.npy' % (task, c_id, u_id),res_yld.F)  
    print(task, c_id, u_id,time.time()-t,res_yld.X.shape)

tasks = []

for i in range(0, len(sel_fields)):
    # ----------------filter points not sampled ----------------
    tmp_points = point_50[i,:,:].reshape(-1)
    if np.isnan(tmp_points).all() or np.nanmax(tmp_points) <=0:
        continue
    tasks.append(i)

exe = MPITaskPool()
exe.run(tasks, work, log_freq=1)