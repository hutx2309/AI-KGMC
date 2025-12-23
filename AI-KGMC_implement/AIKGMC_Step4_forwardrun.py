import numpy as np
import geopandas as gpd
import os
from ecosystools import ecosys_input,Site, Climate, Plant,Site,Soil 
from ecosystools.utils import *
from osgeo import ogr
import pickle
import ast
import glob
 
from ecosystools.task_pool import MPITaskPool
exe = MPITaskPool()

import configparser
config = configparser.ConfigParser()
config.read('config.fig')

##
exe_path = config['general']['ecosys_exe']
PFT_dir = config['general']['PFT_dir']
output_define_dir = config['general']['ecosys_output']
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
###  project dir
prj_dir = config['general']['prj_dir']

###
SurroModel_Dir = config['Surrogate_model_Dir']['SurrogateModel_Dir']
param_dir = SurroModel_Dir + 'seqMDF_r22_SLOPE.2fields/'

Calib_Output_Dir = config['Calibration_Dir']['Output_Dir']

forward_input_dir = config['Forward_Simulation']['Input_Dir']
forward_output_dir = config['Forward_Simulation']['Output_Dir']

if not os.path.exists(forward_output_dir):
    os.makedirs(forward_output_dir,exist_ok=True)

#
Crop_Type_Names = {}
for key in config['CropID']:
    Crop_Type_Names[key]=config['CropID'][key]

#
plant_density = {}
for key in config['plant_density']:
    plant_density[key]=config['plant_density'].getfloat(key)

Harvested = {}
for key in config['Harvested']:
    Harvested[key]=config['Harvested'].get(key)
#
yield_convert_factor = {}
for key in config['yield_convert_factor']:
    yield_convert_factor[key]=config['yield_convert_factor'].getfloat(key)


sel_shp = gpd.read_file('sample0.2field_4calib.geojson')
uids=np.array(sel_shp['unique_fid'].values)#np.array(shp.index)
ct_ids =np.array(sel_shp['cty_id'].values)
field_Lon = sel_shp.centroid.x.values
field_Lat = sel_shp.centroid.y.values

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

del problem_

#define temporary outputs    
 
tmp_Run_Dir = '/dev/shm/run_%s/' % (getpass.getuser())
if not os.path.exists(tmp_Run_Dir):
    os.makedirs(tmp_Run_Dir,exist_ok=True)

#get gSSURGO soil dataset
def write_mesoil(work_dir, u_idx, c_idx):
    driver = ogr.GetDriverByName("OpenFileGDB")
    conn = driver.Open(gSSURGO_Dir + 'gSSURGO_CONUS.gdb', 0)
    gSSURGO_MapunitRaster_file = gSSURGO_Dir + 'MapunitRaster_30m.tif'
    mapunits = Soil.get_gSSURGO_mapunit(gSSURGO_MapunitRaster_file,sel_shp.loc[(sel_shp["unique_fid"]==u_idx) & 
                                                                               (sel_shp["cty_id"] == c_idx)])
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
#
def get_fertilizer(fertilizer_event):
    f_default = {'N_form': 'default',
                 'P_form': 'default',
                 'fertilizer_method': 'banded',
                 'F_depth': 0.02,
                 'F_inhibitors': 1,
                 'Manure_N_amount': 0,
                 'Manure_C_amount': 0}
    
    for k in f_default.keys():
        if k not in fertilizer_event.keys():
            fertilizer_event[k]=f_default[k]
    return fertilizer_event

# conventional tillage
# tillage_event ={'days_before_planting':2, "mixing_coefficient":18, "till_depth":0.30} 

def work(task):
    tsk, state = task
    
    u_id =uids[tsk]
    c_id = ct_ids[tsk]

    #crop_windows
    with open(forward_input_dir + "%s/data/final_data_cropland/crop_window/%s.pkl"%(state, c_id), 'rb') as f:
        crop_windows = pickle.load(f)
    #fertilizer
    with open(forward_input_dir + "%s/data/final_data_cropland/fertilizer/%s.pkl"%(state, c_id), 'rb') as f:
        fertilizer = pickle.load(f)
 
    # tillage
    with open(forward_input_dir + "%s/data/final_data_cropland/tillage/%s.pkl"%(state,c_id), 'rb') as f:
        till= pickle.load(f)

    lon = field_Lon[np.logical_and(sel_shp["unique_fid"]==u_id, sel_shp["cty_id"] == c_id)][0]
    lat = field_Lat[np.logical_and(sel_shp["unique_fid"]==u_id, sel_shp["cty_id"] == c_id)][0]     
     
    basename =  '%s_%d' % (c_id, u_id)
    work_dir = tmp_Run_Dir + str(basename) + os.sep
    
    if os.path.exists(work_dir):
        os.system('rm -r "%s"' % work_dir)    
    os.makedirs(work_dir,exist_ok=True)

    #extract climate data from NLDAS database
    for year in range(model_start_year,experiment_end_year+1):
        if os.path.exists(work_dir+'me%0.4dw.csv' % year):
            continue
        s=Climate.Climate(TTYPE='H')
        s.read_NLDAS(NLDAS_database,lon,lat,year)
        s.write_climate(work_dir+'me%0.4dw.csv' % year)

    #write gSSURGO-based soil file
    status = write_mesoil(work_dir, u_id, c_id)
    if not status:
        return

    #extract DEM and multiyear-averaged Ta for mesite setup
    site_DEM = get_elevation(Global_DEM_file,lon,lat)
    site_Ta = get_Ta(Global_Ta_file,lon,lat)

    #define site information
    tmp_site = Site.Site(ALATG=lat,ALTIG=site_DEM,ATCAG=site_Ta,IERSNG=3)

    #initialize the input information
    s = ecosys_input.ecosys_input(start_year=model_start_year,end_year=experiment_end_year,n_cycles=1,
                                    mesite = tmp_site,
                                    output_start_year=experiment_start_year,model_start_year=model_start_year)
    #update crop, fertilizer, tillage, and irrigation information
    if u_id in crop_windows['data'].keys():
        #
        crop_window = crop_windows['data'][u_id]
        for k in crop_window.keys():
            plant_DOY, harvest_DOY = [int(float(x)) for x in k.split('-')]
            crop_type = Crop_Type_Names[str(int(float(crop_window[k])))]
            p_year,p_month,p_mday = julian2calendar(crop_windows['start_year'],plant_DOY)
            h_year,h_month,h_mday = julian2calendar(crop_windows['start_year'],harvest_DOY)
            p_density = plant_density[crop_type]   


            if crop_type not in Harvested.keys() or Harvested[crop_type]=='NO':
                if p_year%2==0:
                    s.Planting.add_covercrop(planting_year=p_year, planting_month=p_month, planting_mday=p_mday, crop_type=crop_type+'0',
                                            PPI=p_density, terminate_year=h_year, terminate_month=h_month, terminate_mday=h_mday)    
                else:
                    s.Planting.add_covercrop(planting_year=p_year, planting_month=p_month, planting_mday=p_mday, crop_type=crop_type+'1',
                                            PPI=p_density, terminate_year=h_year, terminate_month=h_month, terminate_mday=h_mday)                   
            else:
                if p_year%2==0:
                    s.Planting.add_cashcrop(planting_year=p_year, planting_month=p_month, planting_mday=p_mday, crop_type=crop_type+'0',
                                            PPI=p_density, terminate_year=h_year, terminate_month=h_month, terminate_mday=h_mday,
                                            ECUT21=0.0, ECUT22=1.0, ECUT23=0.0, ECUT24=0.0) 
                else:
                    s.Planting.add_cashcrop(planting_year=p_year, planting_month=p_month, planting_mday=p_mday, crop_type=crop_type+'1',
                                            PPI=p_density, terminate_year=h_year, terminate_month=h_month, terminate_mday=h_mday,
                                            ECUT21=0.0, ECUT22=1.0, ECUT23=0.0, ECUT24=0.0)                         

    if u_id in fertilizer['data'].keys():
        fertilizer_events = fertilizer['data'][u_id]
        for k in fertilizer_events.keys():
            f_DOY = int(k)
            f_year,f_month,f_mday = julian2calendar(fertilizer['start_year'],f_DOY)

            fertilizer_event = get_fertilizer(fertilizer_events[k]) # complete the fertilizer information with default values
            s.Fertilizer.add_by_type(year = f_year, month = f_month, mday = f_mday, 
                                        N_rate = fertilizer_event['N_rate']*0.112085116, P_rate = fertilizer_event['P_rate']*0.112085116, 
                                        N_form = fertilizer_event['N_form'], P_form = fertilizer_event['P_form'], 
                                        fertilizer_method = fertilizer_event['fertilizer_method'], F_depth = fertilizer_event['F_depth'],  
                                        F_inhibitors = fertilizer_event['F_inhibitors'], Manure_N_amount = fertilizer_event['Manure_N_amount'],
                                        Manure_C_amount = fertilizer_event['Manure_C_amount'])
    # satellite tillage 
    if u_id in till['data'].keys():
        till_events = till['data'][u_id]
        for k in till_events.keys():
            t_DOY = int(k)
            t_year,t_DOY = julian2julian(till['start_year'],t_DOY)
            if t_year < experiment_start_year or t_year > experiment_end_year:
                continue
            
            if t_DOY == 366:
                t_DOY=365            
                
            till_info = till_events[k] # 

            t_year,t_month,t_mday = julian2calendar(t_year, t_DOY)
        
            s.Disturbance.add(year=t_year,month=t_month,mday=t_mday,
                                    IDIST=till_info['mixing_ratio'],DDIST=till_info['tillage_depth'])

    # converntional tillage

    # for year in range(experiment_start_year,experiment_end_year+1):
    #     tmp_plant_df = s.Planting.df.loc[s.Planting.df.planting_year==year] 
    #     tmp_plant_df.index= range(len(tmp_plant_df))
    #     for pi in tmp_plant_df.index:
    #         p_year,p_doy =  calendar2julian(tmp_plant_df.planting_year[pi],
    #                                         tmp_plant_df.planting_month[pi],tmp_plant_df.planting_mday[pi])
            
    #         t_year,t_month,t_mday = julian2calendar(p_year,p_doy-tillage_event['days_before_planting'])
        
    #         s.Disturbance.add(year=t_year,month=t_month,mday=t_mday,
    #                                 IDIST=tillage_event['mixing_coefficient'],DDIST=tillage_event['till_depth'])
    
    s.write_runfile(run_dir=work_dir)

    # output settings 
    os.system('cp  "{}"/sc*1 "{}"'.format(output_define_dir, work_dir))   # define outputfiles

    #revise the crop parameters
    for v in Crop_Type_Names.values():
        if v not in config['Calibrated_parameters'].keys():
            os.system('cp  "{}" "{}0"'.format(PFT_dir+v, work_dir+v)) 
            os.system('cp  "{}" "{}1"'.format(PFT_dir+v, work_dir+v)) 

    GROUPX_file = glob.glob(param_dir + '*%s_%d_NSGA2_GROUPX.npy' % (c_id, u_id))
    CHL_file = glob.glob(param_dir + '*%s_%d_NSGA2_CHL.npy' % (c_id, u_id))
    STMX_file = glob.glob(param_dir + '*%s_%d_NSGA2_STMX.npy' % (c_id, u_id))
    
    if len(GROUPX_file) ==0 or len(CHL_file) ==0 or len(STMX_file) ==0:
        return
    
    GROUPX= np.load(GROUPX_file[0])
    CHL = np.load(CHL_file[0])
    STMX = np.load(STMX_file[0])
    if len(STMX.shape) == 1:
        res_x = [ CHL[0], GROUPX[0], STMX[0], STMX[1], CHL[1],  GROUPX[1], STMX[2], STMX[3]]
    elif len(STMX.shape) > 1:
        res_x = [ CHL[0], GROUPX[0], STMX[0][0], STMX[0][1],CHL[1],  GROUPX[1], STMX[0][2], STMX[0][3]]
    else:
        return
    ii = 0
    for k in problem_combine.keys():
        param_names = problem_combine[k]['names']
        nc = len(param_names)
        param_values = res_x[ii:(ii+nc)]
        ii = ii + nc
        p_ = Plant.Plant_Species(PFT_dir+k)
        p_.revise_parameters(param_names, param_values, work_dir + k+'0')
        p_.revise_parameters(param_names, param_values, work_dir + k+'1')
    
    run_ecosys(exe_path,work_dir,runfile='runfile',runlog='runlog.log',rm_work_dir=True,output_db=forward_output_dir+basename+'.db')

# load 50% points of observations
point_50 =  np.load(Calib_Output_Dir + "SLOPE_GPP_width/point_50.npy")
state_name = {"17": "IL", "18": "IN", "19":"IA","20":"KS", "26":"MI", "27":"MN","29": "MO","38":"ND",
              "31" :"NE","39": "OH","46": "SD","55": "WI"}

tasks = []

for i in range(0, len(sel_shp)):
    # ----------------filter points not sampled ----------------
    tmp_points = point_50[i,:,:].reshape(-1)
    if np.isnan(tmp_points).all() or np.nanmax(tmp_points) <=0:
        continue
    state_id = sel_shp["state_id"].values[i]
    tasks.append((i, state_name[state_id]))
 

exe.run(tasks, work, log_freq=1)