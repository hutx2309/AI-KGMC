import numpy as np
import geopandas as gpd
import os
from ecosystools import ecosys_input,Site, Climate, Plant,Soil 
from ecosystools.utils import *
from osgeo import ogr
import pickle
from ecosystools.task_pool import MPITaskPool

PFT_para = ['ANGBR','ANGSH','STMX','SDMX','GRMX','GRDM','GFILL',
           'WTSTDI','RRAD1M','RRAD2M','PORT','PR','RSRR','RSRA','PTSHT','RTFQ',
           'UPMXZH','UPKMZH','UPMNZH','UPMXZO','UPKMZO','UPMNZO','UPMXPO','UPKMPO',
           'UPMNPO','OSMO','RCS','RSMX','DMLF','DMSHE','DMSTK','DMRSV','DMHSK','DMEAR',
           'DMGR','DMRT','DMND','CNLF','CNSHE','CNSTK','CNRSV','CNHSK','CNEAR','CNGR',
           'CNRT','CNND','CPLF','CPSHE','CPSTK','CPRSV','CPHSK','CPEAR','CPGR','CPRT','CPND']
Percent = [0.1, 0.5, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 2, 5, 10]

#read variables from config.config
import configparser
config = configparser.ConfigParser()
config.read('config.config')

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
exe_path = config['general']['ecosys_exe']
PFT_dir = config['general']['PFT_dir']
output_define_dir = config['general']['ecosys_output']

###  project dir
prj_dir = config['general']['prj_dir']

##
TrainFieldName=config['Surrogate_model_Dir']['TrainFieldName']

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


dir0 = '/taiga/illinois/aces/nres/kaiyug/hutx/'

sen_dir = prj_dir +"Sensity_Ana_multiPara/"
if not os.path.exists(sen_dir):
    os.makedirs(sen_dir,exist_ok=True)

#define temporary outputs   
# tmp_Run_Dir = sen_dir 
tmp_Run_Dir = '/dev/shm/Ecosys_%s/' % (TrainFieldName)
if not os.path.exists(tmp_Run_Dir):
    os.makedirs(tmp_Run_Dir,exist_ok=True)

# ===================== fields shp  ============================================
sel_shp = gpd.read_file('I3_SA_20fields.geojson')
uids=np.array(sel_shp['unique_fid'].values)#np.array(shp.index)
ct_ids =np.array(sel_shp['fips'].values)
shp_centroid = sel_shp.centroid.to_crs('epsg:4326')
field_Lon = shp_centroid.x.values
field_Lat = shp_centroid.y.values

# ============ functions to get gSSURGO soil dataset ==================================
def write_mesoil(work_dir, u_idx, c_idx):
    driver = ogr.GetDriverByName("OpenFileGDB")
    conn = driver.Open(gSSURGO_Dir + 'gSSURGO_CONUS.gdb', 0)
    gSSURGO_MapunitRaster_file = gSSURGO_Dir + 'MapunitRaster_30m.tif'
    mapunits = Soil.get_gSSURGO_mapunit(gSSURGO_MapunitRaster_file,sel_shp.loc[(sel_shp["unique_fid"]==u_idx) & (sel_shp["fips"] == c_idx)])
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

def work(task):
    tsk, i, j, state = task 
    
    u_id =uids[tsk]
    c_id = ct_ids[tsk]

    # read origional shp file and load data
    shp_I = gpd.read_file(dir0 + '%s_database/sampled_data/field_boundaries/%s.gpkg' % (state,TrainFieldName))

    #crop_windows
    with open(dir0 + '%s_database/sampled_data/crop_window/%s.pkl' % (state,TrainFieldName), 'rb') as f:
        crop_windows = pickle.load(f)

    #fertilizer
    with open(dir0 + '%s_database/sampled_data/fertilizer/%s.pkl' % (state,TrainFieldName), 'rb') as f:
        fertilizer = pickle.load(f)
   
    lon = field_Lon[np.logical_and(sel_shp["unique_fid"]==u_id, sel_shp["fips"] == c_id)][0]
    lat = field_Lat[np.logical_and(sel_shp["unique_fid"]==u_id, sel_shp["fips"] == c_id)][0]     

    sitename = shp_I.loc[(shp_I["unique_fid"]==u_id) & (shp_I["fips"] == c_id)].index[0]
    # irr = irrigation[sitename]
    print(sitename, u_id)

    basename = '%d_%s_%d_%s_%0.1f' % (tsk,c_id, u_id, PFT_para[i],Percent[j])
    
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
    status = write_mesoil(work_dir,u_id, c_id)
    if not status:
        return
    
    #extract DEM and multiyear-averaged Ta for mesite setup
    site_DEM = get_elevation(Global_DEM_file,lon,lat)
    site_Ta = get_Ta(Global_Ta_file,lon,lat)

    #define site information
    tmp_site = Site.Site(ALATG=lat,ALTIG=site_DEM,ATCAG=site_Ta,IERSNG=1)

    #initialize the input information
    s = ecosys_input.ecosys_input(start_year=model_start_year,end_year=experiment_end_year,n_cycles=1,
                                mesite = tmp_site,
                                output_start_year=experiment_start_year,model_start_year=model_start_year)
    #update crop, fertilizer, tillage, and irrigation information
    
    if sitename in crop_windows['data'].keys():
        #
        crop_window = crop_windows['data'][sitename]
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
                                            PPI=p_density, terminate_year=h_year, terminate_month=h_month, terminate_mday=h_mday) 
                else:
                    s.Planting.add_cashcrop(planting_year=p_year, planting_month=p_month, planting_mday=p_mday, crop_type=crop_type+'1',
                                            PPI=p_density, terminate_year=h_year, terminate_month=h_month, terminate_mday=h_mday)                         
                
    if sitename in fertilizer['data'].keys():
        fertilizer_events = fertilizer['data'][sitename]
        for k in fertilizer_events.keys():
            f_DOY = int(k)
            f_year,f_month,f_mday = julian2calendar(fertilizer['start_year'],f_DOY)
            
            fertilizer_event = get_fertilizer(fertilizer_events[k]) # complete the fertilizer information with default values
            s.Fertilizer.add_by_type(year = f_year, month = f_month, mday = f_mday, 
                                    N_rate = fertilizer_event['N_rate'], P_rate = fertilizer_event['P_rate'], 
                                    N_form = fertilizer_event['N_form'], P_form = fertilizer_event['P_form'], 
                                    fertilizer_method = fertilizer_event['fertilizer_method'], F_depth = fertilizer_event['F_depth'],  
                                    F_inhibitors = fertilizer_event['F_inhibitors'], Manure_N_amount = fertilizer_event['Manure_N_amount'],
                                    Manure_C_amount = fertilizer_event['Manure_C_amount'])

     
    #add plant-based auto irrigation
    # if irr==1:
    #     s.Irrigation_files = ['NO']*(experiment_start_year-model_start_year) + ['autoplant']*(experiment_end_year-experiment_start_year+1)
    #     s.Auto_Irrigation.add(trigger_name='autoplant',
    #                           start_month=1,start_mday=1,
    #                           end_month=12, end_mday=31, 
    #                           IFLGVX=1,FIRRX=-1.5,CIRRX=1.00,DIRRX=0.5,WDPTHI=0)
    # ============ for ecosys_2023 ======================
    for year in range(model_start_year,experiment_end_year+1):
        s.Weather_Option.revise_weather_option(year=year,param_names=['NPX','NPY'],param_values=[15,3])
    
    # write ecosy input fils    
    s.write_runfile(run_dir=work_dir)

    # define outputfiles
    os.system('cp  "{}"/sc*1 "{}"'.format(output_define_dir, work_dir))   

    ## ============= using default plant parameters =================
    # for v in Crop_Type_Names.values():
    #     os.system('cp  "{}" "{}0"'.format(PFT_dir+v, work_dir+v)) 
    #     os.system('cp  "{}" "{}1"'.format(PFT_dir+v, work_dir+v))  

    #revise the crop parameters
    for v in Crop_Type_Names.values():
        if v in ['maiz31', 'soyb31']:
            p = Plant.Plant_Species(PFT_dir + v)
            p_org = getattr(p,(PFT_para[i]))
            p_revise = float(p_org)*Percent[j]
            p.revise_parameters([PFT_para[i]], [p_revise], work_dir + v + '0')
            p.revise_parameters([PFT_para[i]], [p_revise], work_dir + v + '1')
        else:
            os.system('cp  "{}" "{}0"'.format(PFT_dir+v, work_dir+v)) 
            os.system('cp  "{}" "{}1"'.format(PFT_dir+v, work_dir+v))
    #run ecosys model        
    run_ecosys(exe_path,work_dir,runfile='runfile',runlog='runlog.log',rm_work_dir=True,output_db=sen_dir+basename+'.db')


state_name = {"17": "IL", "18": "IN", "19":"IA"}

tasks = []    

for fi in range(len(sel_shp)):
    fips = sel_shp["fips"].values[fi]
    state_id = fips[0:2]
    for i in range(len(PFT_para)):
        for j in range(len(Percent)):
            tasks.append((fi, i, j, state_name[state_id]))

# ============================ parallel run ============================
exe = MPITaskPool()
exe.run(tasks, work, log_freq=1)