# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
import random
import numpy as np
import pandas
import pdb
import time
import struct
from configparser import ConfigParser

import shutil
from pcrasterCommand import pcrasterCommand, getPCrasterPath


########################################################################
#   Read settings file
########################################################################


iniFile = os.path.normpath(sys.argv[1])
#iniFile = "P:/watmodel/CWATM/calibration/multi_calibration/calibration-master/settings1.txt"
print("=================== START ===================")
print(iniFile)

if not (os.path.isfile(iniFile)):
    print("No inifile found or error reading")
    sys.exit()


CatchmentArea = "CatchmentArea"
# iniFile = os.path.normpath(settings)

parser = ConfigParser()
parser.read(iniFile)

root = parser.get('DEFAULT', 'Root')

path_temp = os.path.join(root, parser.get('Path', 'Temp'))
path_maps = os.path.join(root, os.path.join(parser.get('Path', 'CatchmentDataPath'), "maps_pcraster"))
path_result = os.path.join(root, parser.get('Path', 'Result'))

file_CatchmentsToProcess = os.path.join(root, parser.get('MultiComputer', 'CatchmentsToProcess'))

if not os.path.exists(path_temp): os.makedirs(path_temp)

CatchmentDataPath = os.path.join(root,parser.get('Path','CatchmentDataPath'))
SubCatchmentPath = os.path.join(root,parser.get('Path','SubCatchmentPath'))

pcraster_path = parser.get('Path', 'PCRHOME')
path_MeteoData = parser.get('Path', 'MeteoData')

Qtss_csv = os.path.join(root, parser.get('CSV', 'Qtss'))
Qgis_csv = os.path.join(root, parser.get('CSV', 'Qgis'))
Qgis_out = os.path.join(root, parser.get('CSV', 'QgisOut'))

#switch_SubsetMeteoData = int(parser.get('DEFAULT', 'SubsetMeteoData'))

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
stationdata_sorted = stationdata.sort_values(by=[CatchmentArea],ascending=True)
stationdata['Inflow'] = np.nan
stationdata['Inflow_ID'] = np.nan
CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)

j =0
for index, row in stationdata_sorted.iterrows():

    if index == 15:
        III =1
    Series = CatchmentsToProcess.loc[:,0]
    if len(Series[Series==str(row["ID"])]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
        continue
    print ("=================== "+row['ID']+" ====================")
    print (">> Starting map subsetting for catchment "+row['ID']+", size "+str(row[CatchmentArea])+" pixels...")

    t = time.time()

    path_subcatch = os.path.join(SubCatchmentPath,row['ID'])

    # Delete all files in catchment directory
    for root, dirs, files in os.walk(path_subcatch, topdown=False):
         
        for name in files:
            os.remove(os.path.join(root, name))
            #print "   Deleting "+name
        for name in dirs:
            #print "   Deleting "+os.path.join(root, name)
            os.rmdir(os.path.join(root, name))
    if not os.path.exists(path_subcatch):
        os.makedirs(path_subcatch)
    if not os.path.exists(os.path.join(path_subcatch,'maps')):
        os.makedirs(os.path.join(path_subcatch,'maps'))
    if not os.path.exists(os.path.join(path_subcatch,'inflow')):
        os.makedirs(os.path.join(path_subcatch,'inflow'))
    if not os.path.exists(os.path.join(path_subcatch,'out')):
        os.makedirs(os.path.join(path_subcatch,'out'))
    
    # Make mask map for subcatchment
    subcatchmask_map = os.path.join(path_subcatch,"maps","mask.map")
    interstation_regions_map = os.path.join(path_result,"interstation_regions.map")
    pcrasterCommand(pcrcalc + " 'F0 = boolean(if(scalar(F1) eq "+str(index)+",scalar(1)))'", {"F0":subcatchmask_map,"F1":interstation_regions_map})

    #PB no small mask needed
    #tmp1_map = os.path.join(path_temp,"tmp1.map")
    #smallsubcatchmask_map = os.path.join(path_subcatch,"maps","masksmall.map")
    #pcrasterCommand(pcrcalc + " 'F0 = if(F1==1,F2)'", {"F0": tmp1_map, "F1":subcatchmask_map,"F2":subcatchmask_map})
    #pcrasterCommand(resample + " -c 0 F0 F1" , {"F0":tmp1_map, "F1":smallsubcatchmask_map})
    

    
    # Ensure that there is only one outlet pixel in outlet map
    station_map = os.path.join(path_result,"outlet.map")
    subcatchstation_map = os.path.join(path_subcatch,"maps","outlet.map")    
    pcrasterCommand(pcrcalc + " 'F0 = boolean(if(scalar(F1) eq "+str(index)+",scalar(1)))'", {"F0":subcatchstation_map,"F1":station_map})
    #subcatchstation_small_map = os.path.join(path_subcatch,"maps","outletsmall.map")
    #pcrasterCommand(resample + " F0 F1 --clone F2" , {"F0": subcatchstation_map, "F1":subcatchstation_small_map, "F2":smallsubcatchmask_map})

    gaugeloc_txt = os.path.join(path_subcatch,"maps","gaugeloc.txt")
    pcrasterCommand(map2col + " F0 F1"  , {"F0": os.path.join(path_subcatch,"maps","outlet.map"), "F1":gaugeloc_txt})


    # Make inlet map
    inlets_map = os.path.join(path_result,"inlets.map")
    subcatchinlets_map = os.path.join(path_subcatch,"inflow","inflow.map")
    shutil.copyfile(inlets_map,subcatchinlets_map)
    pcrasterCommand(pcrcalc + " 'F0 = F1*scalar(F2)'", {"F0":subcatchinlets_map,"F1":subcatchinlets_map,"F2":subcatchmask_map})



    subcatchinlets_map = os.path.join(path_subcatch, "inflow", "inflow.map")
    subcatchinlets_new_map = os.path.join(path_subcatch, "inflow", "inflow_new.map")
    #subcatchinlets_new2_map = os.path.join(path_subcatch, "inflow", "inflow_new2.map")

    #pcrasterCommand(pcrcalc + " 'F0 = F1*0.0'", {"F0": subcatchinlets_new_map, "F1": subcatchinlets_map})
    pcrasterCommand(pcrcalc + " 'F1 = if(scalar(boolean(F0)) gt 0,nominal(F0))'", {"F0": subcatchinlets_map,"F1": subcatchinlets_new_map})

    inflow_txt = os.path.join(path_subcatch,"inflow","inflowloc.txt")
    pcrasterCommand(map2col + " F0 F1"  , {"F0": subcatchinlets_new_map, "F1":inflow_txt})


    inflow = 0
    inId = []
    f = open(inflow_txt, "r")
    for line in f.readlines():
        inflow += 1
        (X, Y, ID) = line.split()
        ii = "G{:04d}".format(int(ID))
        inId.append(ii)
    f.close()
    if inflow>0:
        inId.sort()
        s = inId[0]
        for ii in inId[1:]:
            s = s + " " + ii
    else:
        s=""


        
    elapsed = time.time() - t
    print ("   Time elapsed: "+"{0:.2f}".format(elapsed)+" s")

    stationdata.loc[index, 'Inflow'] = inflow
    stationdata.loc[index, 'Inflow_ID'] = s

# Write stationdata dataframe to Qgis3.csv in results directory
stationdata.to_csv(Qgis_out,',')
print ("==================== END ====================")