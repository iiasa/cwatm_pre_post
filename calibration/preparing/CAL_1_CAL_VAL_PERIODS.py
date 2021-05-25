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
#from ConfigParser import SafeConfigParser
from configparser import ConfigParser
import configparser
import array
import logging
import random
from dateutil.relativedelta import relativedelta
import calendar

if __name__=="__main__":

    ########################################################################
    #   Read settings file
    ########################################################################
    # PB change format date to write, val and cal is first/last day of month
    
    iniFile = os.path.normpath(sys.argv[1])
    #iniFile = "P:/watmodel/CWATM/calibration/multi_calibration/calibration-master/settings1.txt"
    print ("=================== START ===================")
    #print (">> Reading settings file ("+sys.argv[1]+")...")
    print (iniFile)

    if not (os.path.isfile(iniFile)):
        print("No inifile found or error reading")
        sys.exit()

    parser = ConfigParser()
    parser.read(iniFile)

    root = parser.get('DEFAULT', 'Root')
    path_result = os.path.join(root,parser.get('Path', 'Result'))
    if os.path.exists(path_result)==False:
        os.mkdir(path_result)

    MinQlength = int(parser.get('DEFAULT', 'MinQlength'))
    SpinOff = int(parser.get('DEFAULT', 'SpinoffYears'))

    ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT','ForcingStart'),"%d/%m/%Y")  # Start of forcing
    ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT','ForcingEnd'),"%d/%m/%Y")  # Start of forcing
    ForcingStart2 = ForcingStart - relativedelta(years=SpinOff)

    Qtss_csv =  os.path.join(root,parser.get('CSV', 'Qtss'))
    Qgis_csv =  os.path.join(root,parser.get('CSV', 'Qgis'))
    Qgis_out = os.path.join(root,parser.get('CSV', 'QgisOut'))

    ########################################################################
    #   Make stationdata array from the qgis csv
    ########################################################################

    (drive, path) = os.path.splitdrive(Qgis_csv)
    (path, fil)  = os.path.split(path)
    print (">> Reading Qgis file ("+fil+")...")
    stationdata = pandas.read_csv(Qgis_csv,sep=",")
    stationdata = stationdata[stationdata['Suitable']==1]
    stationdata['ID_numeric'] = ""
    numeric_only = re.compile(r'[^\d.]+')
    for index, row in stationdata.iterrows():
        stationdata.loc[index,'ID_numeric'] = str(int(numeric_only.sub('',stationdata.loc[index,'ID'])))
    stationdata.index = stationdata['ID_numeric']


    ########################################################################
    #   Compute start and end of calibration and validation periods
    ########################################################################

    sys.stdout.write(">> Compute calibration and validation periods")

    stationdata['EnoughQdata'] = np.nan
    stationdata['Val_Start'] = np.nan
    stationdata['Val_End'] = np.nan
    stationdata['Cal_Start'] = np.nan
    stationdata['Cal_End'] = np.nan

    streamflow_data = pandas.read_csv(Qtss_csv,sep=",", parse_dates=True, index_col=0,infer_datetime_format=True,dayfirst=True)
    for index, row in stationdata.iterrows():
        print (index)
        if index == '14':
            ii =1

        # Retrieve observed streamflow
        observed_streamflow = streamflow_data[row['ID']]
        observed_streamflow[observed_streamflow<0] = np.NaN

        # For which period do we have forcing and observed streamflow?
        mask = observed_streamflow.isnull()
        ForcingStart_inobs = mask.index.searchsorted(ForcingStart)
        ForcingEnd_inobs = mask.index.searchsorted(ForcingEnd)
        mask.loc[:ForcingStart_inobs] = True
        mask.loc[ForcingEnd_inobs:] = True

        # If record is twice as long as MinQlength, then split the record in equally
        # long calibration and validation periods. However, if the record is between
        # MinQlength and 2*MinQlength long, then take MinQlength for the calibration
        # period and the remainder for the validation period. If record is shorther than
        # MinQlength, reject the catchment.
        mask_cumsum = np.cumsum((mask.values-1)*-1+0) # cumulative sum counting number of days with non-missing values
        total_number_vals =  np.max(mask_cumsum)
        if total_number_vals <= (365*MinQlength):
            #sys.stdout.write(row['ID']+': streamflow record too short, setting EnoughQdata to 0')
            stationdata.loc[index,'EnoughQdata'] = 0
            sys.stdout.write("x")
            stationdata.loc[index,'Val_Start'] = 'Streamflow record too short'
            continue
        stationdata.loc[index,'EnoughQdata'] = 1
        sys.stdout.write(".")
        
        SplitFraction = 1-float(MinQlength*365)/float(total_number_vals)
        if SplitFraction>0.5:
            SplitFraction = 0.5
            
            #ADDED BY FEYERA HIRPA Feb 5, 2016
            #To limit the total number of Qobs to 16 years (8 for calib, 8 for valid)
        if total_number_vals-4>365*16:    
            Split = np.where(mask_cumsum == np.round(total_number_vals-(365*8)))[0]
            Split = np.asarray(Split)
            Split = Split[0]
            Split = mask.index[Split]
            Split = Split.replace(day=1)  #PB replace with first day of month, just for thew beautiness
            
            Start = np.where(mask_cumsum == np.round(total_number_vals-(365*16)))[0]
            Start = np.asarray(Start)
            Start = Start[0]
            Start = mask.index[Start]

            # PB set to first day of month
            Start = Start.replace(day=1)


        else:
            #print(total_number_vals,SplitFraction)
            Split = np.where(mask_cumsum == np.round(total_number_vals*SplitFraction))[0]        
            Split = np.asarray(Split)
            Split = Split[0]
            Split = mask.index[Split]
            #print(Split)
            Start = mask[mask == False].index[0]
            
        End = mask[mask == False].index[-1]
        lastday = calendar.monthrange(End.year, End.month)[1]
        End = End.replace(day=lastday)


        print(Start)
        print(End)        
        if (Split-Start).days>365:
            stationdata.loc[index,'Val_Start'] = Start.strftime("%d/%m/%Y")
            Split1 = Split - relativedelta(days = 1)
            stationdata.loc[index,'Val_End'] =Split1.strftime("%d/%m/%Y") #+" 00:00" # Round to December 31st of previous year
        else:
            stationdata.loc[index,'Val_Start'] = "Streamflow record too short for validation"
            stationdata.loc[index,'EnoughQdata'] = 0
            
        stationdata.loc[index,'Cal_Start'] = Split.strftime("%d/%m/%Y")   #+" 00:00" # Round to January 1st of year
        stationdata.loc[index,'Cal_End'] = End.strftime("%d/%m/%Y")

    # Save dataframe with new catchment area and cal val period columns
    print ("\n>> Saving Qgis2.csv with Cal_Start, Cal_End, Val_Start, and Val_End columns, includes only catchments with suitable==1 & EnoughQdata==1")
    stationdata = stationdata[stationdata['EnoughQdata']==1]
    stationdata.to_csv(os.path.join(Qgis_out).replace("\\","/"),',')
    print ("==================== END ====================")

