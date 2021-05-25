# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

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
from shutil import copyfile
from pcrasterCommand import pcrasterCommand, getPCrasterPath

if __name__=="__main__":

    ########################################################################
    #   Read settings file
    ########################################################################

    iniFile = os.path.normpath(sys.argv[1])
    #iniFile = "P:/watmodel/CWATM/calibration/multi_calibration/calibration-master/settings1.txt"
    print ("=================== START ===================")
    print (iniFile)

    if not (os.path.isfile(iniFile)):
        print("No inifile found or error reading")
        sys.exit()

    #iniFile = os.path.normpath(settings)

    parser = ConfigParser()
    parser.read(iniFile)

    root = parser.get('DEFAULT', 'Root')

    path_temp = os.path.join(root,parser.get('Path', 'Temp'))
    path_maps = os.path.join(root,os.path.join(parser.get('Path', 'CatchmentDataPath'),"maps_pcraster"))
    path_result = os.path.join(root,parser.get('Path', 'Result'))
    MaxPercArea= parser.get('DEFAULT', 'MaxPercArea')
  

    ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y")  # Start of forcing
    ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y")  # Start of forcing

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
    
    tmp_map = os.path.join(path_temp,"tmp.map")
    tmp2_map = os.path.join(path_temp,"tmp2.map")
    tmp_txt = os.path.join(path_temp,"tmp.txt")
    tmp2_txt = os.path.join(path_temp,"tmp2.txt")
    
    #pdb.set_trace()
        
    
    ########################################################################
    #   Make stationdata array from the qgis csv
    ########################################################################

    print (">> Reading Qgis2.csv file...")
    stationdata = pandas.read_csv(os.path.join(Qgis_csv),sep=",",index_col=0)

    
    ########################################################################
    #   Make map with station locations
    ########################################################################
  
    print (">> Make map with station locations (outlet.map)...")
  
    if not os.path.exists(path_temp): os.makedirs(path_temp)
    if not os.path.exists(path_result): os.makedirs(path_result)
  
    ldd_map = os.path.join(path_maps,"ldd.map")
    station_map = os.path.join(path_result,"outlet.map")
    station_txt = os.path.join(path_temp,"station.txt")
    with open(station_txt, 'w') as f:
        for index, row in stationdata.iterrows():
            s = str(row['XCorrected'])+","+str(row['YCorrected'])+","+str(index)+".0\n"
            f.write(s)
        #    print >> f,row['XCorrected'],row['YCorrected'],float(index)
    pcrasterCommand(col2map + " F0 F1 -N --clone F2 --large"  , {"F0": station_txt, "F1":station_map, "F2":ldd_map})
  
    
    ########################################################################
    #     Check for station location conflicts (stations with same coordinates)
    ########################################################################
    
    print (">> Check for station conflicts...")
    counter = 0
    for index, row in stationdata.iterrows():
        #pcrasterCommand(pcrcalc + " 'F0 = if(scalar(F1)==scalar("+str(int(index))+"),scalar(1.0))'", {"F0": tmp_map, "F1":station_map})
        #pcrasterCommand(pcrcalc + " 'F0 = if(scalar(F1)==scalar(5555),scalar(1.0))'", {"F0": tmp_map, "F1":station_map})
        pcrasterCommand(pcrcalc + " 'F0 = if(scalar(F1)=="+str(int(index))+",scalar(F1))'", {"F0": tmp_map, "F1":station_map})
        pcrasterCommand(map2col + " F0 F1"  , {"F0": tmp_map, "F1":tmp_txt})
        #f = file(tmp_txt,"r")
        f = open(tmp_txt, "r")
        counter2 = 0
        for line in f.readlines():
            (X,Y,value) = line.split()
            counter2 = counter2+1
        if counter2==0:
            print ("Station ID "+str(index)+" not found in outlet.map! Is there another station with the same location?")
            counter = counter+1
        elif counter2>1:
            print ("Station ID "+str(index)+" found multiple times in outlet.map!")
            counter = counter+1
        if counter2==1:
            if int(value)!=index:
                print ("Wrong station ID for station "+str(index)+"; instead of "+str(index)+" we found "+value)
                counter = counter+1
            else:
                print ("Station "+str(index)+" OK")
        f.close()
    if counter>0:
        print ("Number of station location conflicts: "+str(counter))
        print ("Fix these! Enter 'c' to continue for now")
        #pdb.set_trace()
    
    
    ########################################################################
    #   Compute catchment mask maps and write to temporary directory and
    #   make array with catchment upstream number of pixels
    ########################################################################
  
    sys.stdout.write(">> Compute catchment masks (catchmaskXXXXX.map)")
  
    stationdata['CatchmentArea'] = np.nan
    accuflux_map = os.path.join(path_temp,"accuflux.map")
    pcrasterCommand(pcrcalc + " 'F0 = accuflux(F1,1)'", {"F0": accuflux_map, "F1":ldd_map})
    for index, row in stationdata.iterrows():
        sys.stdout.write(".")
  
        # make map of station location
        content = "%s %s %s\n" % (row['XCorrected'],row['YCorrected'],1)
        f1 = open(tmp_txt,"w")
        f1.write(content)
        f1.close()
        pcrasterCommand(col2map + " F0 F1 -N --clone F2 --large" ,{"F0": tmp_txt, "F1":tmp_map, "F2":ldd_map})
  
        # compute catchment mask
        catchment_map = os.path.join(path_temp,"catchmask%05d.map" % float(index))
        pcrasterCommand(pcrcalc + " 'F0 = boolean(catchment(F1,F2))'", {"F0": catchment_map, "F1":ldd_map, "F2":tmp_map})
        pcrasterCommand(pcrcalc + " 'F0 = if((scalar(F0) gt (scalar(F0) *0)) then F0)' ", {"F0": catchment_map})
  
        # compute number of upstream pixels for station
        pcrasterCommand(pcrcalc + " 'F0 = if(defined(F1),F2)'", {"F0": tmp2_map, "F1":tmp_map, "F2":accuflux_map})
        pcrasterCommand(map2col + " F0 F1"  , {"F0": tmp2_map, "F1":tmp2_txt})
        f = open(tmp2_txt,"r")
        for line in f.readlines():
            (X,Y,value) = line.split()
            stationdata.loc[index,'CatchmentArea'] = float(value)
        f.close()
  
  
    ########################################################################
    #   Make map with IDs for interstation regions
    ########################################################################
  
    sys.stdout.write("\n>> Compute interstation regions (interstation_regions.map)")
  
    interstation_regions_map = os.path.join(path_result,"interstation_regions.map")
    pcrasterCommand(pcrcalc + " 'F0 = scalar(F1)*0-1'", {"F0": interstation_regions_map, "F1":ldd_map}) # initialize interstation_regions_map
    stationdata_sorted = stationdata.sort_index(by=['CatchmentArea'],ascending=False)
    for index, row in stationdata_sorted.iterrows():
        sys.stdout.write(".")
        catchment_map = os.path.join(path_temp,"catchmask%05d.map" % float(index))
        pcrasterCommand(pcrcalc + " 'F0 = F0 * (1-scalar(cover(F1,0.0)))'", {"F0": interstation_regions_map, "F1":catchment_map})
        pcrasterCommand(pcrcalc + " 'F0 = F0 + scalar(cover(F1,0.0)) * " + str(index) + "'", {"F0": interstation_regions_map, "F1":catchment_map})
  
  
    ########################################################################
    #   Make map with sampling frequency for interstation regions
    #   The map indicates how many times a particular pixel is included in a
    #   catchment
    ########################################################################
  
    sys.stdout.write("\n>> Compute sampling frequency (sampling_frequency.map)...")
  
    sampling_frequency_map = os.path.join(path_result,"sampling_frequency.map")
    pcrasterCommand(pcrcalc + " 'F0 = scalar(F1)*0'", {"F0": sampling_frequency_map, "F1":ldd_map}) # initialize interstation_regions_map
    for index, row in stationdata.iterrows():
        sys.stdout.write(".")
        catchment_map = os.path.join(path_temp,"catchmask%05d.map" % float(index))
        pcrasterCommand(pcrcalc + " 'F0 = F0 + scalar(cover(F1,0.0))'", {"F0": sampling_frequency_map, "F1":catchment_map})
  
    ########################################################################
    #   Remove calibration stations along the same river with upstream area
    #   too close to each other, as that causes problems in the calibration
    #   and increase the calibration load
    ########################################################################
  
    
  
    #pcraster_path = "/ADAPTATION/usr/anaconda2/bin/"
    os.system(pcraster_path + "/mapattr -p "+ sampling_frequency_map + " > maxtable.txt")
    f=open("maxtable.txt")
    for line in f.readlines():
      if "max_val" in line:
        max_val=line.split()[1]
    f.close()
  
    dst= os.path.join(path_result,"outlet2.map")
    src= os.path.join(path_result,"outlet.map")
    copyfile(src, dst)
    dst= os.path.join(path_result,"ldd.map")
    src= os.path.join(path_maps,"ldd.map")
    copyfile(src, dst)
    dst= os.path.join(path_result,"ups.map")
    src= os.path.join(path_maps,"ups.map")
    copyfile(src, dst)
  
  
    os.chdir(path_result)
    for SF in range(int(max_val),1,-1):
        print (SF)   #SF=16
        cmd=(pcraster_path+"/pcrcalc SFOutlets.map=if(sampling_frequency.map=="+ str(SF) +",outlet2.map)")    #Map with outlets only with a specific sampling frequency
        os.system(cmd)
  
        os.system(pcraster_path+"/pcrcalc UpsSFO.map=if(defined(SFOutlets.map),ups.map)")
        os.system(pcraster_path+"/pcrcalc UpsmaxSFO.map=UpsSFO.map*(1+" + str(MaxPercArea) +" )")
        os.system(pcraster_path+"/pcrcalc UpsmaxSFOc.map = cover(UpsmaxSFO.map,0)")
        os.system(pcraster_path+"/pcrcalc pointSpread.map = spreadlddzone(ldd.map, nominal(rounddown(UpsmaxSFOc.map)),0,1)")
        os.system(pcraster_path+"/pcrcalc pointFreearea.map = scalar(pointSpread.map) gt ups.map")
        os.system(pcraster_path+"/pcrcalc pointFreearea.map = if(defined(SFOutlets.map),0,pointFreearea.map)")
        os.system(pcraster_path+"/pcrcalc outlet2.map=if(not boolean(pointFreearea.map),outlet2.map)")
    os.system(pcraster_path+"/map2col outlet2.map outlet2.txt")
  ########################################################################
    #   Make csv listing for each catchment 1) the directly connected
    #   subcatchments and 2) the corresponding inflow locations
    ########################################################################
    OriginalStations=list()
    for index, row in stationdata.iterrows():
        OriginalStations.append(int(row['ID_numeric.1']))


    stationdata = pandas.read_csv(os.path.join(Qgis_csv),sep=",",index_col='ID_numeric')
     
    stationWithAreaBigEnough = list()
    f=open(os.path.join(path_result, "outlet2.txt"), "r")
    content=f.readlines()
    for row in content:
        stationWithAreaBigEnough.append(int(row.split(' ')[-1][:-1]))
    f.close()
    difflist=list(set(OriginalStations) - set(stationWithAreaBigEnough))
    print ('Removing stations along same river with upstream area very close to each other' ,difflist)
    for sta in difflist:
      stationdata=stationdata.drop(index=int(sta))
       
    
    stationdata.to_csv(os.path.join(Qgis_out),sep=",")
    
    
    sys.exit()
    sys.stdout.write("\n>> Compute links and inflow locations (inlets.map and direct_links.csv)")
