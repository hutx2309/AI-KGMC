from ecosystools.utils import *
from ecosystools.deep_learning import *
from ecosystools import Soil, Plant
import os
import geopandas as gpd
import pickle

#read variables from config.config
import configparser
import ast
config = configparser.ConfigParser()
config.read('config.fig')

###
model_start_year = config['general'].getint('model_start_year')
experiment_start_year = config['general'].getint('experiment_start_year')
experiment_end_year = config['general'].getint('experiment_end_year')

PFT_dir = config['general']['PFT_dir']
###
###  project dir
prj_dir = config['general']['prj_dir']

###
SurroModel_Dir = config['Surrogate_model_Dir']['SurrogateModel_Dir']

#
Crop_Type_Names = {}
for key in config['CropID']:
    Crop_Type_Names[key]=config['CropID'][key]
    
#
plant_density = {}
for key in config['plant_density']:
    plant_density[key]=config['plant_density'].getfloat(key)
    
# 
nsamples = config['Calibrated_parameters'].getint('nsamples')

model_dir = SurroModel_Dir + 'model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir,exist_ok=True)

with open(SurroModel_Dir+'results.pkl', 'rb') as f:
    simulation_results = pickle.load(f) 

sel_shp = gpd.read_file('sample0.1field_4surrogate.geojson')
 
state_name = {"17": "IL", "18": "IN", "19":"IA","20":"KS", "26":"MI", "27":"MN","29": "MO","38":"ND",
              "31" :"NE","39": "OH","46": "SD","55": "WI"}
dir0 = '/scratch/bbkc/yizhi/carbon_input_database/'
obs_crop_type = np.load(prj_dir +"orgSampleObs_4surrogate/crop_window.npy")
# obs_till_dep = np.load(prj_dir +"orgSampleObs_4surrogate/till_dep.npy")
# obs_till_mix = np.load(prj_dir +"orgSampleObs_4surrogate/till_mix.npy")
# 
year_days = 365
days = year_days*(experiment_end_year-experiment_start_year+1)
year_range = np.arange(experiment_start_year,experiment_end_year+1)

# 
sim_crop_type = np.zeros((nsamples,experiment_end_year-experiment_start_year+1,year_days)) 
# sim_till_dep = np.zeros((nsamples,experiment_end_year-experiment_start_year+1,year_days)) 
# sim_till_mix = np.zeros((nsamples,experiment_end_year-experiment_start_year+1,year_days)) 
for SR in range(nsamples):
    if 'c_id' not in simulation_results[SR]:
        print('Simulations of %s not exists' % SR)
        continue
    c_idx = simulation_results[SR]['c_id']
    u_idx = simulation_results[SR]['u_id']
 
    # load crop window
    row_index = sel_shp.loc[(sel_shp.cty_id == c_idx) & (sel_shp.unique_fid == u_idx)].index.to_list()
    if len(row_index) == 1:
        sim_crop_type[SR, :, :] =  obs_crop_type[row_index[0],:,:]
        # sim_till_dep[SR,:,:] = obs_till_dep[row_index[0],:,:]
        # sim_till_mix[SR,:,:] = obs_till_mix[row_index[0],:,:]
    else:
        print("duplicated sites")

#
climate = np.zeros((nsamples,7,experiment_end_year-experiment_start_year+1,year_days))*np.nan
sim_GPP = np.zeros((nsamples,experiment_end_year-experiment_start_year+1,year_days))*np.nan
sim_ET = np.zeros((nsamples,experiment_end_year-experiment_start_year+1,year_days))*np.nan
sim_Yield = np.zeros((nsamples,experiment_end_year-experiment_start_year+1,year_days))*np.nan
N_fertilizer = np.zeros((nsamples,experiment_end_year-experiment_start_year+1,year_days))*np.nan
P_fertilizer = np.zeros((nsamples,experiment_end_year-experiment_start_year+1,year_days))*np.nan
 
for SR in range(nsamples):
    if 'c_id' not in simulation_results[SR]:
        print('Simulations of %s not exists' % SR)
        continue
    climate[SR,:,:,:] = simulation_results[SR]['climate_data'][:,:,:year_days]
    sim_GPP[SR,:,:] = simulation_results[SR]['sim_GPP'][:,:year_days]
    sim_ET[SR,:,:] = simulation_results[SR]['sim_ET'][:,:year_days]
    sim_Yield[SR,:,:] = simulation_results[SR]['sim_Yield'][:,:year_days]
    
    N_fertilizer[SR,:,:] = simulation_results[SR]['sim_N_fertilizer'][:,:year_days]
    P_fertilizer[SR,:,:] = simulation_results[SR]['sim_P_fertilizer'][:,:year_days]

sim_ET[sim_ET<-50] = np.nan

# resample the soil depth to 0-5,5-30, and 30-100
soil_layers = 3
Soil_BKDS = np.zeros((nsamples,soil_layers))*np.nan 
Soil_SAND = np.zeros((nsamples,soil_layers))*np.nan 
Soil_SILT = np.zeros((nsamples,soil_layers))*np.nan 
Soil_FC = np.zeros((nsamples,soil_layers))*np.nan 
Soil_WP = np.zeros((nsamples,soil_layers))*np.nan 
Soil_KSat = np.zeros((nsamples,soil_layers))*np.nan 
Soil_SOC = np.zeros((nsamples,soil_layers))*np.nan 
Soil_PH = np.zeros((nsamples,soil_layers))*np.nan 
Soil_CEC = np.zeros((nsamples,soil_layers))*np.nan 

for SR in range(nsamples):
    if 'c_id' not in simulation_results[SR]:
        print('Simulations of %s not exists' % SR)
        continue
        
    Soil_Depth = simulation_results[SR]['Soil_Depth']
    Soil_BKDS_ = simulation_results[SR]['Soil_BKDS']
    Soil_SOC_ = simulation_results[SR]['Soil_SOC']
    Soil_SAND_ = simulation_results[SR]['Soil_SAND']
    Soil_SILT_ = simulation_results[SR]['Soil_SILT']
    Soil_FC_ = simulation_results[SR]['Soil_FC']
    Soil_WP_ = simulation_results[SR]['Soil_WP']
    Soil_KSat_ = simulation_results[SR]['Soil_KSat']
    Soil_PH_ = simulation_results[SR]['Soil_PH']
    Soil_CEC_ = simulation_results[SR]['Soil_CEC']
    
    Soil_BKDS[SR,0],Soil_SOC[SR,0],Soil_SAND[SR,0],Soil_SILT[SR,0],Soil_FC[SR,0],Soil_WP[SR,0],Soil_KSat[SR,0],Soil_PH[SR,0],Soil_CEC[SR,0] = Soil.resample_soil_properties(
        Soil_Depth,0,0.05,Soil_BKDS_,Soil_SOC_,Soil_SAND_,Soil_SILT_,Soil_FC_,Soil_WP_,Soil_KSat_,Soil_PH_,Soil_CEC_)
    Soil_BKDS[SR,1],Soil_SOC[SR,1],Soil_SAND[SR,1],Soil_SILT[SR,1],Soil_FC[SR,1],Soil_WP[SR,1],Soil_KSat[SR,1],Soil_PH[SR,1],Soil_CEC[SR,1] = Soil.resample_soil_properties(
        Soil_Depth,0.05,0.30,Soil_BKDS_,Soil_SOC_,Soil_SAND_,Soil_SILT_,Soil_FC_,Soil_WP_,Soil_KSat_,Soil_PH_,Soil_CEC_)
    Soil_BKDS[SR,2],Soil_SOC[SR,2],Soil_SAND[SR,2],Soil_SILT[SR,2],Soil_FC[SR,2],Soil_WP[SR,2],Soil_KSat[SR,2],Soil_PH[SR,2],Soil_CEC[SR,2] = Soil.resample_soil_properties(
        Soil_Depth,0.30,1.00,Soil_BKDS_,Soil_SOC_,Soil_SAND_,Soil_SILT_,Soil_FC_,Soil_WP_,Soil_KSat_,Soil_PH_,Soil_CEC_)


# 
del simulation_results
nan_index1= np.array([(np.sum(np.isnan(sim_GPP[ii,:,:365])) > 0) for ii in range(sim_GPP.shape[0])])
nan_index2 = np.array([(np.sum(np.isnan(N_fertilizer[ii,:,:365])) > 0)  for ii in range(N_fertilizer.shape[0])])
nan_index3= np.array([(np.sum(np.isnan(sim_ET[ii,:,:365]))> 0)  for ii in range(sim_ET.shape[0])])

nan_index = np.logical_or(np.logical_or(nan_index1,nan_index2),nan_index3)

# 
parameters = []

for k in config['Calibrated_parameters'].keys():
    if k == 'nsamples':
        continue
    tmp = ast.literal_eval(config['Calibrated_parameters'][k])
    problem_={
        'num_vars':len(tmp.keys()),
        'names':list(tmp.keys()),
        'bounds': [tmp[k] for k in tmp.keys()]
    }
    exec('problem_%s=problem_'% k)
    parameters.extend(list(tmp.keys()))
    
    param_values_ = np.load(SurroModel_Dir+'param_values_%s.npy' % k)
    exec('param_values_%s=param_values_'% k)
del param_values_,problem_

# 
parameters=np.array(parameters)
parameters=np.sort(np.unique(parameters))

# 
input_names = ['RADN','TMAX_AIR','TMIN_AIR','HMAX_AIR','HMIN_AIR','WIND','PRECN', # climate
            'BKDS_0_5','SOC_0_5','CSAND_0_5','CSILT_0_5','FC_0_5','WP_0_5','SCNV_0_5','PH_0_5','CEC_0_5', # soil layer 1
            'BKDS_5_30','SOC_5_30','CSAND_5_30','CSILT_5_30','FC_5_30','WP_5_30','SCNV_5_30','PH_5_30','CEC_5_30', # soil layer 2
            'BKDS_30_100','SOC_30_100','CSAND_30_100','CSILT_30_100','FC_30_100','WP_30_100','SCNV_30_100','PH_30_100','CEC_30_100', # soil layer 3
            'DOY','N_fertilizer','P_fertilizer', 'Crop_Type']# 'Till_Dep', 'Till_Mix' ]  # manage
input_names.extend(parameters)

output_names=np.array(['GPP','ET','Yield'])#,'Yield'])

# 
v_max = ast.literal_eval(config['scales']['v_max'])
v_min = ast.literal_eval(config['scales']['v_min'])

# 
X = np.zeros((len(nan_index[~nan_index]),days,len(input_names)))*np.nan
Y = np.zeros((len(nan_index[~nan_index]),days,len(output_names)))*np.nan

# 
for year in range(experiment_start_year,experiment_end_year+1): # climate data
    print(year)
    start_i = year_days*(year-experiment_start_year)
    end_i = year_days*(year-experiment_start_year+1)

    X[:,start_i:end_i,0] = maxmin_norm_with_scaler(climate[~nan_index,0,year-experiment_start_year,:(end_i-start_i)],v_max['RADN'],v_min['RADN']) # add 7 climate data
    X[:,start_i:end_i,1] = maxmin_norm_with_scaler(climate[~nan_index,1,year-experiment_start_year,:(end_i-start_i)],v_max['TMAX_AIR'],v_min['TMAX_AIR']) # add 7 climate data
    X[:,start_i:end_i,2] = maxmin_norm_with_scaler(climate[~nan_index,2,year-experiment_start_year,:(end_i-start_i)],v_max['TMIN_AIR'],v_min['TMIN_AIR']) # add 7 climate data
    X[:,start_i:end_i,3] = maxmin_norm_with_scaler(climate[~nan_index,3,year-experiment_start_year,:(end_i-start_i)],v_max['HMAX_AIR'],v_min['HMAX_AIR']) # add 7 climate data
    X[:,start_i:end_i,4] = maxmin_norm_with_scaler(climate[~nan_index,4,year-experiment_start_year,:(end_i-start_i)],v_max['HMIN_AIR'],v_min['HMIN_AIR']) # add 7 climate data
    X[:,start_i:end_i,5] = maxmin_norm_with_scaler(climate[~nan_index,5,year-experiment_start_year,:(end_i-start_i)],v_max['WIND'],v_min['WIND']) # add 7 climate data
    X[:,start_i:end_i,6] = maxmin_norm_with_scaler(climate[~nan_index,6,year-experiment_start_year,:(end_i-start_i)],v_max['PRECN'],v_min['PRECN']) # add 7 climate data

    #DOY   
    X[:,start_i:end_i,34] = maxmin_norm_with_scaler(np.arange(1,end_i-start_i+1).reshape(1,-1),v_max['DOY'],v_min['DOY'])
    X[:,start_i:end_i,35] = maxmin_norm_with_scaler(N_fertilizer[~nan_index,year-experiment_start_year,:(end_i-start_i)],v_max['N_fertilizer'],v_min['N_fertilizer'])#N_fertilizer
    X[:,start_i:end_i,36] = maxmin_norm_with_scaler(P_fertilizer[~nan_index,year-experiment_start_year,:(end_i-start_i)],v_max['P_fertilizer'],v_min['P_fertilizer'])#N_fertilizer

    crop_type = sim_crop_type[~nan_index,year-experiment_start_year,:(end_i-start_i)]
    X[:,start_i:end_i,37] = crop_type

    # till_dep = sim_till_dep[~nan_index,year-experiment_start_year,:(end_i-start_i)]
    # X[:,start_i:end_i,38] = till_dep

    # till_mix = sim_till_mix[~nan_index,year-experiment_start_year,:(end_i-start_i)]
    # X[:,start_i:end_i,39] = till_mix

    # X[:,start_i:end_i,38] = sim_irrigation[~nan_index,year-experiment_start_year,:(end_i-start_i)]
    
    #parameters
    unique_crop_type = np.unique(crop_type).astype('int')
    for vi in range(len(parameters)):
        var_=parameters[vi]
        param_ = np.zeros((len(nan_index[~nan_index]),end_i-start_i))
        for crp_ in unique_crop_type:
            if str(crp_) not in Crop_Type_Names.keys():
                continue
            crop_name = Crop_Type_Names[str(crp_)]
            
            if (crop_name in config['Calibrated_parameters'].keys()):
                sampled_parameters = np.array(eval('problem_%s' % crop_name)['names'])
                if (var_ in sampled_parameters) :
                    var_values_ = eval('param_values_%s'% crop_name)[:,var_==sampled_parameters][~nan_index]
                    var_values_ = var_values_.reshape(-1,1) + np.zeros((end_i-start_i)).reshape(1,-1)
                    param_[crop_type==crp_] = maxmin_norm_with_scaler(var_values_[crop_type==crp_],v_max[var_],v_min[var_])
                else:
                    p_ = Plant.Plant_Species(PFT_dir+crop_name)
                    param_[crop_type==crp_] = maxmin_norm_with_scaler(float(eval('p_.%s' % var_)),v_max[var_],v_min[var_])                    
            else:
                p_ = Plant.Plant_Species(PFT_dir+crop_name)
                param_[crop_type==crp_] = maxmin_norm_with_scaler(float(eval('p_.%s' % var_)),v_max[var_],v_min[var_])
            X[:,start_i:end_i,38+vi]=param_

    Y[:,start_i:end_i,0] = maxmin_norm_with_scaler(sim_GPP[~nan_index,year-experiment_start_year,:(end_i-start_i)],v_max['GPP'],v_min['GPP']) #GPP
    Y[:,start_i:end_i,1] = maxmin_norm_with_scaler(sim_ET[~nan_index,year-experiment_start_year,:(end_i-start_i)],v_max['ET'],v_min['ET']) #ET
    Y[:,start_i:end_i,2] = maxmin_norm_with_scaler(sim_Yield[~nan_index,year-experiment_start_year,:(end_i-start_i)],v_max['Yield'],v_min['Yield']) #Yield

#soil data
X[:,:,7] = maxmin_norm_with_scaler(Soil_BKDS[~nan_index,0].reshape(-1,1),v_max['BKDS'],v_min['BKDS'])
X[:,:,8] = maxmin_norm_with_scaler(Soil_SOC[~nan_index,0].reshape(-1,1),v_max['SOC'],v_min['SOC'])
X[:,:,9] = maxmin_norm_with_scaler(Soil_SAND[~nan_index,0].reshape(-1,1),v_max['CSAND'],v_min['CSAND'])
X[:,:,10] = maxmin_norm_with_scaler(Soil_SILT[~nan_index,0].reshape(-1,1),v_max['CSILT'],v_min['CSILT'])
X[:,:,11] = maxmin_norm_with_scaler(Soil_FC[~nan_index,0].reshape(-1,1),v_max['FC'],v_min['FC'])
X[:,:,12] = maxmin_norm_with_scaler(Soil_WP[~nan_index,0].reshape(-1,1),v_max['WP'],v_min['WP'])
X[:,:,13] = maxmin_norm_with_scaler(Soil_KSat[~nan_index,0].reshape(-1,1),v_max['SCNV'],v_min['SCNV'])
X[:,:,14] = maxmin_norm_with_scaler(Soil_PH[~nan_index,0].reshape(-1,1),v_max['PH'],v_min['PH'])
X[:,:,15] = maxmin_norm_with_scaler(Soil_CEC[~nan_index,0].reshape(-1,1),v_max['CEC'],v_min['CEC'])

X[:,:,16] = maxmin_norm_with_scaler(Soil_BKDS[~nan_index,1].reshape(-1,1),v_max['BKDS'],v_min['BKDS'])
X[:,:,17] = maxmin_norm_with_scaler(Soil_SOC[~nan_index,1].reshape(-1,1),v_max['SOC'],v_min['SOC'])
X[:,:,18] = maxmin_norm_with_scaler(Soil_SAND[~nan_index,1].reshape(-1,1),v_max['CSAND'],v_min['CSAND'])
X[:,:,19] = maxmin_norm_with_scaler(Soil_SILT[~nan_index,1].reshape(-1,1),v_max['CSILT'],v_min['CSILT'])
X[:,:,20] = maxmin_norm_with_scaler(Soil_FC[~nan_index,1].reshape(-1,1),v_max['FC'],v_min['FC'])
X[:,:,21] = maxmin_norm_with_scaler(Soil_WP[~nan_index,1].reshape(-1,1),v_max['WP'],v_min['WP'])
X[:,:,22] = maxmin_norm_with_scaler(Soil_KSat[~nan_index,1].reshape(-1,1),v_max['SCNV'],v_min['SCNV'])
X[:,:,23] = maxmin_norm_with_scaler(Soil_PH[~nan_index,1].reshape(-1,1),v_max['PH'],v_min['PH'])
X[:,:,24] = maxmin_norm_with_scaler(Soil_CEC[~nan_index,1].reshape(-1,1),v_max['CEC'],v_min['CEC'])

X[:,:,25] = maxmin_norm_with_scaler(Soil_BKDS[~nan_index,2].reshape(-1,1),v_max['BKDS'],v_min['BKDS'])
X[:,:,26] = maxmin_norm_with_scaler(Soil_SOC[~nan_index,2].reshape(-1,1),v_max['SOC'],v_min['SOC'])
X[:,:,27] = maxmin_norm_with_scaler(Soil_SAND[~nan_index,2].reshape(-1,1),v_max['CSAND'],v_min['CSAND'])
X[:,:,28] = maxmin_norm_with_scaler(Soil_SILT[~nan_index,2].reshape(-1,1),v_max['CSILT'],v_min['CSILT'])
X[:,:,29] = maxmin_norm_with_scaler(Soil_FC[~nan_index,2].reshape(-1,1),v_max['FC'],v_min['FC'])
X[:,:,30] = maxmin_norm_with_scaler(Soil_WP[~nan_index,2].reshape(-1,1),v_max['WP'],v_min['WP'])
X[:,:,31] = maxmin_norm_with_scaler(Soil_KSat[~nan_index,2].reshape(-1,1),v_max['SCNV'],v_min['SCNV'])
X[:,:,32] = maxmin_norm_with_scaler(Soil_PH[~nan_index,2].reshape(-1,1),v_max['PH'],v_min['PH'])
X[:,:,33] = maxmin_norm_with_scaler(Soil_CEC[~nan_index,2].reshape(-1,1),v_max['CEC'],v_min['CEC'])

#
del climate,N_fertilizer,P_fertilizer
del sim_GPP, sim_ET, sim_Yield
del Soil_BKDS, Soil_SOC, Soil_SAND, Soil_SILT, Soil_FC, Soil_WP, Soil_KSat, Soil_PH, Soil_CEC
del sim_crop_type #, sim_till_mix, sim_till_dep

# 
n_feature = len(input_names) #number of input features
n_output=len(output_names) #number of output features

# 
import  random
_index = random.sample(range(X.shape[0]),X.shape[0])

train_n=int(0.6*X.shape[0])  # split the data into training dataset (60%), validation dataset (30%), and test set(10%)
val_n=int(0.3*X.shape[0])
test_n=X.shape[0] - train_n - val_n 

train_index = np.array(_index[:train_n])
val_index = np.array(_index[train_n:(train_n+val_n)])
test_index = np.array(_index[(train_n+val_n):])

if not os.path.exists (SurroModel_Dir + 'train_vali_split_dataset.sav' ):
    torch.save({'train_index': train_index,
                'val_index':val_index,
                'test_index': test_index
                }, SurroModel_Dir + 'train_vali_split_dataset.sav' ) 
else:
    tmp=torch.load(SurroModel_Dir + 'train_vali_split_dataset.sav')
    train_index = tmp['train_index']
    val_index = tmp['val_index']
    test_index = tmp['test_index']

# 
train_n,val_n,test_n

# 
n_hidden=64#256#64#128#64#64 #hidden state number
n_layers=2#2#4 #layer of lstm
model_version='LSTM_%dl_%d_U_case_%s.sav' % (n_layers,n_hidden,'_'.join(output_names))  #####save file !!!!!!!!!!!!!!!!!!!! change this before new training
path_save = model_dir + model_version
#build model and move to GPU
model=LSTM(n_feature,n_hidden,n_layers,n_output,0,0.2)

# 
#training model
#watch -n 1 nvidia-smi 
#torch.cuda.set_device(0)# could used to specfic what GPU used (0-3) when in jupyter notebook, while in sbatch, it will change the default GPU automatically
epochs = 1500
model_training(model,epochs,path_save,X_input=X, train_index=train_index,val_index=val_index,slide_window=365, 
               Y_output1=Y,batch_size=1000)
