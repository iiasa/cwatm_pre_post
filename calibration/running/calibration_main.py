# -*- coding: utf-8 -*-
"""Please refer to quick_guide.pdf for usage instructions"""

import os
import sys
import numpy as np
import pandas
import re
import pdb
import time
from datetime import datetime
from configparser import ConfigParser
import glob
import datetime
import random
import platform as plat
from sys import platform


########################################################################
#   Read settings file
########################################################################
if __name__ == "__main__":
#if True:

    text = "Hi\nThis is the automatic calibration of CWatM\nThanks for helping us calibrating the World\n\n You already run the calibration process.\n\nYou can stop it anytime. Just cancal the process by clicking on the left top corner X (sometimes several times)\n\n"
    print (text)


    iniFile = os.path.normpath(sys.argv[1])
    #iniFile = "settings8.txt"

    if not(os.path.isfile(iniFile)):
        print(iniFile)
        print("No inifile found or error reading")
        sys.exit()

    #file_CatchmentsToProcess = os.path.normpath(sys.argv[2])
    #file_CatchmentsToProcess ="P:/watmodel/CWATM/calibration/multi_calibration/calibration-master/CatchmentsToProcess_All.txt"

    parser = ConfigParser()
    parser.read(iniFile)

    #iniFile = os.path.basename(iniFile)

    if platform == "win32":
        root = parser.get('DEFAULT','Root')
    else:
        root = parser.get('DEFAULT','RootLinux')

    path_result = os.path.join(root,parser.get('Path', 'Result'))
    SubCatchmentPath = os.path.join(root,parser.get('Path','SubCatchmentPath'))

    #forcing_start=parser.get('DEFAULT', 'ForcingStart')
    try:
        ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT', 'ForcingStart'),"%d/%m/%Y %H:%M")  # Start of forcing
        ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT', 'ForcingEnd'), "%d/%m/%Y %H:%M")  # Start of forcing
    except:
        ForcingStart = datetime.datetime.strptime(parser.get('DEFAULT', 'ForcingStart'),"%d/%m/%Y")  # Start of forcing
        ForcingEnd = datetime.datetime.strptime(parser.get('DEFAULT', 'ForcingEnd'), "%d/%m/%Y")

    nmax = int(parser.get('DEFAULT','No_of_calibration_lists'))

    timeperiod = parser.get('DEFAULT', 'timeperiod')
    if timeperiod == "monthly":
        monthly = 1
        dischargetss = 'discharge_monthavg.tss'
        frequen = 'MS'
    else:
        monthly = 0
        dischargetss = 'discharge_daily.tss'
        frequen = 'd'

    # Multi computer as executable runs
    file_CatchmentsToProcess = os.path.join(root,parser.get('MultiComputer', 'CatchmentsToProcess'))
    listPC = os.path.join(root,parser.get('MultiComputer', 'listPC'))

    if platform == "win32":
        Run1 = parser.get('MultiComputer', 'RunCalib')
        RunCalib = Run1.split(" ")
        if len(RunCalib) > 1:
            RunCalib = Run1.split(" ")[0] + " " + os.path.join(root, Run1.split(" ")[1])
        else:
            RunCalib = os.path.join(root,Run1.split(" ")[0])

    else:
        Run1 = parser.get('MultiComputer', 'RunCalibLinux')
        RunCalib = Run1.split(" ")[0] + " " + os.path.join(root, Run1.split(" ")[1])


    ########################################################################
    #   Loop through catchments and perform calibration
    ########################################################################

    print (">> Reading Qgis3.csv file...")
    stationdata = pandas.read_csv(os.path.join(path_result,"Qgis3.csv"),sep=",",index_col=0)
    #stationdata_sorted = stationdata.sort_index(by=['CatchmentArea'],ascending=True)
    stationdata_sorted = stationdata.sort_values(by=['CatchmentArea'],ascending=True)

    CatchmentsToProcess = pandas.read_csv(file_CatchmentsToProcess,sep=",",header=None)

    for index, row in stationdata_sorted.iterrows():
        if index > 17:
           ii =1
        Series = CatchmentsToProcess.loc[:,0]
        #print 'cal_start',row['Cal_Start']
        if len(Series[Series==str(row["ID"])]) == 0: # Only process catchments whose ID is in the CatchmentsToProcess.txt file
            continue
        print ("=================== "+row['ID']+" ====================")
        path_subcatch = os.path.join(root,SubCatchmentPath,row['ID'])
        if os.path.exists(os.path.join(path_subcatch,"streamflow_simulated_best.tss")):
            print ("streamflow_simulated_best.tss already exists! Moving on...")
            continue

        # Test if other computers run something
        can_run = True
        can_run1 = False
        file = os.path.join(listPC, row['ID'] + "_list.txt")
        if os.path.isfile(file):
            with open(file, 'r') as f:
                l = f.readlines()[-1]
            tt = l.split(",")[0]
            user = l.split(",")[1][:-1]
            now = datetime.datetime.now()
            timeflag = datetime.datetime.strptime(tt, "%Y-%m-%d %H:%M")

            timeflag = timeflag + datetime.timedelta(hours=2)
            if timeflag > now:
                s = row["ID"] + " is run by " + user + " already! Moving on..."
                print(s)
                can_run = False

        if can_run:
            can_run1 = True
            # test if inflows in:
            if row['Inflow'] > 0:
                inflows = row['Inflow_ID'].split(" ")
                for subcatchment in inflows:
                    path_subcatch1 = os.path.join(SubCatchmentPath, subcatchment)
                    if not(os.path.exists(os.path.join(path_subcatch1, "streamflow_simulated_best.tss"))):
                        can_run1 = False
                        s = "streamflow_simulated_best.tss of " + subcatchment + " is calculated by someone else, but not ready! Moving on..."
                        print (s)

        if can_run1:

            print (">> Starting calibration of catchment "+row['ID']+", size "+str(row['CatchmentArea'])+" pixels...")
            inflowflag = False
            if row['Inflow'] > 0:
                inflowflag = True

            # Copy simulated streamflow from upstream catchments
            # Change inlet map by replacing the numeric ID's with 1, 2, ...
            sys.stdout.write("Upstream station(s): ")

            if inflowflag:
                cnt = 0
                header = ""
                inflows = row['Inflow_ID'].split(" ")
                inflow_tss = os.path.join(path_subcatch, "inflow", "inflow.tss")
                # inflow_tss_lastrun is for when after the optimal combination of parameters is found , when we run the full forcing period
                inflow_tss_last_run = os.path.join(path_subcatch, "inflow", "inflow_last_run.tss")


                for subcatchment in inflows:
                    cnt += 1
                    header = header + " " + subcatchment
                    # loop here till previous catchment on the list is done
                    sys.stdout.write(subcatchment+" ")
                    Qsim_tss = SubCatchmentPath + "/"+subcatchment+"/"+"streamflow_simulated_best.tss"

                    #timer = 0
                    #while not os.path.exists(Qsim_tss) and timer<=720000:
                    #    time.sleep(1)
                    #    timer+=1

                    #start and end days of normal and long run
                    Cal_S = datetime.datetime.strptime(row['Cal_Start'], '%d/%m/%Y %H:%M')
                    Cal_Realstart = datetime.datetime(Cal_S.year - 5, Cal_S.month, Cal_S.day, 0, 0).strftime('%d/%m/%Y')    # normal run
                    Cal_Realstart1 = datetime.datetime(Cal_S.year - 20, Cal_S.month, Cal_S.day, 0, 0).strftime('%d/%m/%Y')  # long run
                    Cal_End1 = datetime.datetime.strptime(row['Cal_End'], '%d/%m/%Y %H:%M').strftime('%d/%m/%Y')

                    shift_time = datetime.datetime.strptime(Cal_Realstart, "%d/%m/%Y") - datetime.datetime.strptime(Cal_Realstart, "%d/%m/%Y")
                    # difference in days between long run and normal run
                    #shift_time = datetime.datetime.strptime(row['Cal_Start'], "%d/%m/%Y %H:%M") - datetime.datetime.strptime(datetime.datetime.strftime(ForcingStart, "%d/%m/%Y %H:%M"),"%d/%m/%Y %H:%M")


                    print ('load inflow.tss')
                    #simulated_streamflow_last = pandas.read_csv(Qsim_tss, sep=",", parse_dates=True,header=None, index_col=0)
                    try:
                        simulated_streamflow_last = pandas.read_csv(Qsim_tss, sep=",", parse_dates=True, header=None,index_col=0)
                    except:
                        print ("Could not find streamflow_simulated_best.tss for upstream catchment "+subcatchment+", hence cannot run this catchment...")
                        raise Exception("Stopping...")

                    simulated_streamflow_last.columns = [ subcatchment]
                    simulated_streamflow = simulated_streamflow_last[Cal_Realstart:Cal_End1]

                    if cnt==1: # first inflow
                        inflow_last = simulated_streamflow_last.copy()
                        inflow = simulated_streamflow.copy()
                    else:
                        inflow_last = inflow_last.join(simulated_streamflow_last)
                        inflow = inflow.join(simulated_streamflow)

                inflow_last.to_csv(inflow_tss_last_run, sep=' ', header=False)
                inflow.to_csv(inflow_tss, sep=' ', header=False)

                header = "timeseries scalar  Inflows from:" + header +"\n"
                header = header + str(cnt+1) +"\ntimestep\n"
                for i in range(cnt):
                    header = header + str(i+1) + "\n"



                f = open(inflow_tss_last_run,'r+')
                content = f.read()
                content = header + content
                f.seek(0,0)
                f.write(content)
                f.close()

                f = open(inflow_tss,'r+')
                content = f.read()
                content = header + content
                f.seek(0,0)
                f.write(content)
                f.close()


                # save inflow coordinate in another format
                inflow_txt = os.path.join(path_subcatch, "inflow", "inflowloc.txt")
                inflowloc = {}
                f = open(inflow_txt, "r")
                for line in f.readlines():
                    (x, y, id) = line.split()
                    inflowloc[id]= [x,y]
                f.close()

                gaugetext =""
                for subcatchment in inflows:
                    station = stationdata.loc[stationdata["ID"] == subcatchment].iloc[0]
                    stationID = str(station['ID_number'])
                    x = inflowloc[stationID][0]
                    y = inflowloc[stationID][1]
                    gaugetext = gaugetext + " " + x + " " + y

                inflow_txt = os.path.join(path_subcatch, "inflow", "inflowloc2.txt")
                f = open(inflow_txt, "w")
                f.write(gaugetext)
                f.close()


            else:
                sys.stdout.write("none")
            sys.stdout.write("\n")
            sbc=str(row["ID"])

            cmd = RunCalib + " " + iniFile + " " + str(row["ID"])
            #cmd = RunCalib
            #cmd1 = iniFile + " " + str(row["ID"])
            print (cmd)


            #file = os.path.join(listPC, "set_" + plat.uname()[1] +".txt")
            #file = "set_" + plat.uname()[1] + ".txt"
            #f = open(file, "w")
            #f.write(cmd1)
            #f.close()


            t = os.system(cmd)
            ii =1




    print ("==================== END ====================")
