# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""
"""
PB only used to create CatchmentsToProcess_All.txt map with all catchments in
maybe use this for later
"""

import os
import sys
import pdb
import pandas
import string
import datetime
import time
import numpy as np
import re
from configparser import ConfigParser
import array
import logging
import random
from pcrasterCommand import pcrasterCommand, getPCrasterPath

if __name__=="__main__":

    ########################################################################
    #   Read settings file
    ########################################################################
    iniFile = os.path.normpath(sys.argv[1])
    #iniFile = "P:/watmodel/CWATM/calibration/multi_calibration/calibration-master/settings1.txt"
    print ("=================== START ===================")
    print (iniFile)
    #print (">> Reading settings file ("+sys.argv[1]+")...")

    if not (os.path.isfile(iniFile)):
        print("No inifile found or error reading")
        sys.exit()


    parser = ConfigParser()
    parser.read(iniFile)

    root = parser.get('DEFAULT', 'Root')

    path_temp = os.path.join(root,parser.get('Path', 'Temp'))
    path_maps = os.path.join(root,os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps_pcraster"))
    path_result = os.path.join(root,parser.get('Path', 'Result'))

    CatchmentsToProcessFile = os.path.join(root,parser.get('MultiComputer', 'CatchmentsToProcess'))


    No_of_calibration_PCs = float(parser.get('MultiComputer','No_of_calibration_PCs'))

    Qtss_csv = os.path.join(root,parser.get('CSV', 'Qtss'))
    Qgis_csv = os.path.join(root,parser.get('CSV', 'Qgis'))
    Qgis_out = os.path.join(root,parser.get('CSV', 'QgisOut'))

    pcraster_path = parser.get('Path', 'PCRHOME')

    config = {}
    for execname in ["pcrcalc","map2asc","asc2map","col2map","map2col","mapattr","resample"]:
        config[execname] = getPCrasterPath(pcraster_path,execname)

    pcrcalc = config["pcrcalc"]
    col2map = config["col2map"]
    map2col = config["map2col"]
    resample = config["resample"]


    ########################################################################
    #   Make stationdata array from the qgis csv
    ########################################################################
    
    print (">> Reading Qgis2.csv file...")
    stationdata = pandas.read_csv(os.path.join(Qgis_csv),sep=",",index_col=0)
    stationdata_sorted = stationdata.sort_index(by=['DrainArLDD'],ascending=False)
    
    stationdata['SPREAD_WORKLOAD_ID'] = np.nan
    
    tmp_map = os.path.join(path_temp,"tmp.map")
    tmp2_map = os.path.join(path_temp,"tmp2.map")
    tmp_txt = os.path.join(path_temp,"tmp.txt")
    tmp2_txt = os.path.join(path_temp,"tmp2.txt")
    
    station_map = os.path.join(path_result,"outlet.map")
    SPREAD_WORKLOAD_ID_map = os.path.join(path_result,"SPREAD_WORKLOAD_ID.map")
    pcrasterCommand(pcrcalc + " 'F0 = scalar(cover(F1,0.0))*0.0'", {"F0": SPREAD_WORKLOAD_ID_map, "F1":station_map})
         
    # Loop through catchments, biggest catchment first, and give nested 
    # catchments that don't have an ID yet and ID
    # Additionally, make map of IDs, and add IDs to stationdata dataframe
    SPREAD_WORKLOAD_ID = 1
    for index, row in stationdata_sorted.iterrows():
        if ~np.isnan(stationdata.loc[index,'SPREAD_WORKLOAD_ID']):
            continue        
        catchment_map = os.path.join(path_temp,"catchmask%05d.map" % float(index))                    
        pcrasterCommand(pcrcalc + " 'F0 = if(defined(F1),F2)'", {"F0": tmp_map, "F1":catchment_map, "F2":station_map})
        pcrasterCommand(map2col + " F0 F1"  , {"F0":tmp_map, "F1":tmp2_txt})
        f = open(tmp2_txt,"r")
        for line in f.readlines():
            (X,Y,value) = line.split()
            stationdata.loc[int(value),'SPREAD_WORKLOAD_ID'] = SPREAD_WORKLOAD_ID            
            print ("catchment "+str(value)+" gets a SPREAD_WORKLOAD_ID of "+str(SPREAD_WORKLOAD_ID))
        f.close()
        pcrasterCommand(pcrcalc + " 'F0 = F0+cover(scalar(F1),0)*"+str(SPREAD_WORKLOAD_ID)+"'", {"F0": SPREAD_WORKLOAD_ID_map, "F1":catchment_map})        
        SPREAD_WORKLOAD_ID = SPREAD_WORKLOAD_ID+1        

    # Make dataframe with for each ID the total area in number of pixels
    df = pandas.DataFrame({'SPREAD_WORKLOAD_ID' : np.arange(1,np.max(stationdata['SPREAD_WORKLOAD_ID'])+1)})
    df = df.set_index('SPREAD_WORKLOAD_ID')
    df['Area'] = np.nan
    for index, row in df.iterrows():    
        area = np.sum(stationdata.loc[stationdata['SPREAD_WORKLOAD_ID']==index,'DrainArLDD'])
        df.loc[index,'Area'] = area
    
    # Assign IDs to different computers (PCs)
    # In stationdata dataframe, list the PC for each catchment
    # Also make map of PCs
    SPREAD_WORKLOAD_PC_map = os.path.join(path_result,"SPREAD_WORKLOAD_PC.map")
    pcrasterCommand(pcrcalc + " 'F0 = scalar(cover(F1,0.0))*0.0'", {"F0": SPREAD_WORKLOAD_PC_map, "F1":station_map})
    totalarea = np.sum(stationdata.loc[:,'DrainArLDD'])
    area_per_pc = totalarea/No_of_calibration_PCs
    df['PC'] = np.nan
    for PC in np.arange(1,float(No_of_calibration_PCs)+1):
        area = 0
        for index, row in df.iterrows():
            if np.isnan(df.loc[index,'PC']):
                if area<area_per_pc:
                    df.loc[index,'PC'] = PC
                    stationdata.loc[stationdata['SPREAD_WORKLOAD_ID']==index,'PC'] = PC
                    area = area+df.loc[index,'Area']
                    pcrasterCommand(pcrcalc + " 'F0 = F0+cover(scalar(F1=="+str(index)+"),0)*"+str(PC)+"'", {"F0": SPREAD_WORKLOAD_PC_map, "F1":SPREAD_WORKLOAD_ID_map})
    
    # For each PC, make file listing the catchments
    for PC in np.arange(1,float(No_of_calibration_PCs)+1):
        df_sub = df[df['PC']==PC]
        if len(df_sub)>0:            
            CatchmentsToProcess = []
            for index, row in df_sub.iterrows():            
                CatchmentsToProcess = CatchmentsToProcess+stationdata.loc[stationdata['SPREAD_WORKLOAD_ID']==index,'ID'].tolist()
            PC_str = "0"+str(int(PC))
            PC_str = PC_str[-2:]
            print ("Making CatchmentsToProcess_"+PC_str+".txt file")
            CatchmentsToProcess = pandas.DataFrame(CatchmentsToProcess)
            CatchmentsToProcess.to_csv("CatchmentsToProcess_"+PC_str+".txt",index=False,header=False)
    CatchmentsToProcess = stationdata['ID'].tolist()
    CatchmentsToProcess = pandas.DataFrame(CatchmentsToProcess)
    CatchmentsToProcess.to_csv(CatchmentsToProcessFile,index=False,header=False)
    
    # Write stationdata dataframe to Qgis3.csv in results directory
    stationdata.to_csv(os.path.join(Qgis_out),',')
    print ("==================== END ====================")