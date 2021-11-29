

""" The aim of this code is to compare observed and simulated water table with and without irrigation"""


# --------------------------------------------------------------------------
# Import Python packages

import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import pandas
import numpy as np
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
import xlrd
import xarray as xr
import os
import csv
import rasterio

# to extract parameters form settings file
import configparser  # to read the settings file
import re   # to read the settings file

import pandas as pd
import plotly.express as px
import seaborn as sns

import numpy.ma as ma

plt.rcParams['figure.figsize'] = [16, 8.0]

# --------------------------------------------------------------------------
# Import paths
path_model_outputs = 'P:/watmodel/CWATM/modelruns/burgenland_1km/CWatM_ModFlow6/output_demandV36_g'
#path_model_outputs_noirrig = 'P:/watmodel/CWATM/modelruns/burgenland_1km/CWatM_ModFlow6/output_demandV36_norrig_g'

# Model resolution
#resolution_cwatm = 1000  # m


#with rasterio.open('Seewinkel_regions_CWatMscale.tif', 'r') as src:
#    Seewinkelmask_cwatm = src.read(1).astype(np.float32)
#with rasterio.open('Seewinkel_regions_ModFlowscale.tif', 'r') as src:
#    Seewinkelmask_modflow = src.read(1).astype(np.float32)
#Seewinkelmask_cwatm = Seewinkelmask_cwatm[9:, :]
#Seewinkelmask_modflow = Seewinkelmask_modflow[9*10:, :]


# Uploading simulated maps from the netcdf files
ds = xr.open_dataset(path_model_outputs + '/' + 'modflow_watertable_monthend.nc')
GWTable_ModFlow = ds['modflow_watertable_monthend'][:].data  # Array of water levels [m] (time, nrow, ncol)
Time_ModFlow = ds['time'][:].data
Time_ModFlowV2 = (Time_ModFlow - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
Time_ModFlowV3 = np.zeros(len(Time_ModFlow))
for it in range(len(Time_ModFlow)):
    dd = datetime.utcfromtimestamp(Time_ModFlowV2[it])
    Time_ModFlowV3[it] = dd.toordinal()
ds.close()

#ds = xr.open_dataset(path_model_outputs_noirrig + '/' + 'modflow_watertable_monthend.nc')
#GWTable_ModFlow_noirrig = ds['modflow_watertable_monthend'][:].data  # Array of water levels [m] (time, nrow, ncol)
#ds.close()

ncol = np.shape(GWTable_ModFlow)[2]
nrow = np.shape(GWTable_ModFlow)[1]
res_ModFlow = 250  # m
Size = [4805000, 2800000]  # Upper lefet corner of your grid (to display maps) in regular rectangular coordinates system

Date_vector = []
for i in range(len(Time_ModFlowV3)):
    Date_vector.append(datetime.fromordinal(int(Time_ModFlowV3[i])))


with rasterio.open(r'P:\watmodel\CWATM\cwatm_input_1km_burgenland\ModFlow_inputs_new\100m\modflow_basin.tif', 'r') as src:
    basin = src.read(1).astype(np.bool)  # read in as 2-dimensional array (nrows, ncols).
basin = basin * 1  # to have 0 and 1 instead of False and True


# Uploading topography defined in ModFlow and soil thickness defined in CWATM
ds = xr.open_dataset(path_model_outputs + '/' + 'modflowtotalSoilThickness_totalend.nc')
Soil_thickness = ds['modflowtotalSoilThickness_totalend'][:].data
ds.close()

with rasterio.open(r'P:\watmodel\CWATM\cwatm_input_1km_burgenland\ModFlow_inputs_new\100m\elevation_modflow.tif', 'r') as src:
    topography = src.read(1).astype(np.float32)
with rasterio.open(r'P:\watmodel\CWATM\cwatm_input_1km_burgenland\ModFlow_inputs_new\100m\modlfow_river_percentage_drainage_canal.tif', 'r') as src:
    modflow_riverpercentage = src.read(1).astype(np.float32)

#lakeArea = Dataset('P:/watmodel/CWATM/cwatm_input_1km_burgenland/routing/lakereservoirs' + '/' + 'lakeresArea.nc', 'r').variables['lakeresArea'][:]
#lakeArea = lakeArea[9:, :]
#lakeArea_modflow = np.zeros((nrow, ncol))
# For Burgenland the ModFlow resolution is 10 times finer
#factor = 10
#for ir in range(np.shape(Seewinkelmask_cwatm)[0]):
#    for ic in range(np.shape(Seewinkelmask_cwatm)[1]):
#        if lakeArea[ir, ic] > 0:
#            lakeArea_modflow[ir * factor:(ir + 1) * factor, ic * factor:(ic + 1) * factor] = 1
#Soil_thickness = np.where(lakeArea_modflow <= 0, np.where(modflow_riverpercentage > 0, 1.5, Soil_thickness), Soil_thickness)
Topography_ModFlow = topography - Soil_thickness  # We need to add soil thickness to compare WTD


# ----------------------------------------------------------------------------------------------------------------------
# Uploading observed water table depth

path_observed_data = r'C:/Users/guillaumot/Documents/CWatM_Burgenland_1km/CWatM-ModFlow6/ObservedWaterTable/'
Number_of_observed_dates = 771  # It depends on the simulating period
n_wells = 81  # Only 81 for V2 because we removed two boreholes on the North

# Uploading coordinates, water table and topography of each borehole
x_GW = np.loadtxt(path_observed_data + 'GW_level_x_V2.txt')
y_GW = np.loadtxt(path_observed_data + 'GW_level_y_V2.txt')
z_GW = np.loadtxt(path_observed_data + 'GW_level_z_V2.txt')

# 2D array containing measured data: 382 boreholes for Bhima
Measured_watertabledepth = np.zeros((n_wells, Number_of_observed_dates))
wb = xlrd.open_workbook(path_observed_data + "piezom_ehyd_monthmean_V2.xlsx")
sh = wb.sheet_by_name(u'Tabelle1')
for il in range(np.shape(Measured_watertabledepth)[0]):
    for ic in range(np.shape(Measured_watertabledepth)[1]):
        val = sh.row_values(ic+24)[il+1]
        if isinstance(val, float):
            Measured_watertabledepth[il][ic] = z_GW[il] - val  # because here we have measured water table instead of water table depth
        else:
            Measured_watertabledepth[il][ic] = np.nan

# Uploading associated dates
Measured_dates = np.zeros(np.shape(Measured_watertabledepth)[1])
for ic in range(np.shape(Measured_watertabledepth)[1]):
    Measured_dates[ic] = datetime(1900, 1, 1).toordinal() + int(sh.row_values(ic+24)[0]) + 15

# For each date of observation, we look for the closer date of simulation
i = 0
while Measured_dates[i] < np.min(Time_ModFlowV3):
    i += 1
Measured_dates = Measured_dates[i:]
Measured_watertabledepth = Measured_watertabledepth[:, i:]
i = 0
while Measured_dates[i] < np.max(Time_ModFlowV3):
    i += 1
Measured_dates = Measured_dates[:i]
Measured_watertabledepth = Measured_watertabledepth[:, :i]
Number_of_observed_dates = len(Measured_dates)

Index_measured_date = np.zeros(len(Measured_dates))
for it in range(len(Measured_dates)):
    Index_measured_date[it] = np.argmin(np.abs(Time_ModFlowV3-Measured_dates[it]))  # index corresponding to the closer date

# List containing only boreholes in the basin and not too close from the basin limit (at least 1 cell)
x_GW_burgen = np.zeros(np.shape(Measured_watertabledepth)[0])   # Boreholes out of the basin will be removed after
y_GW_burgen = np.zeros(np.shape(Measured_watertabledepth)[0])
z_GW_burgen = np.zeros(np.shape(Measured_watertabledepth)[0])
wt_GW_burgen = np.zeros((np.shape(Measured_watertabledepth)[0], len(Measured_dates)))       # Observed water table
wtd_GW_burgen = np.zeros((np.shape(Measured_watertabledepth)[0], len(Measured_dates)))      # Observed water table depth
Sim_wtd_GW_burgen = np.zeros((np.shape(Measured_watertabledepth)[0], len(Measured_dates)))  # Simulated water table
Sim_wt_GW_burgen = np.zeros((np.shape(Measured_watertabledepth)[0], len(Measured_dates)))   # Simulated water table depth
#Sim_wtd_GW_burgen_noirrig = np.zeros((np.shape(Measured_watertabledepth)[0], len(Measured_dates)))  # Simulated water table if no irrigation
#Sim_wt_GW_burgen_noirrig = np.zeros((np.shape(Measured_watertabledepth)[0], len(Measured_dates)))   # Simulated water table depth if no irrigation
Sim_z_GW_burgen = np.zeros(np.shape(Measured_watertabledepth)[0])              # Altitude of the cell in the model (differnt from borehole information)

# Preparing data for the linear interpolation inside the grid
X = np.arange(0, ncol)
Y = np.arange(0, nrow)
X, Y = np.meshgrid(X, Y)
points = np.array((X.flatten(), Y.flatten())).T

valuesWaterTable = np.zeros((len(Measured_dates), nrow*ncol))
valuesWaterTableDepth = np.zeros((len(Measured_dates), nrow*ncol))
#valuesWaterTable_noirrig = np.zeros((len(Measured_dates), nrow*ncol))
#valuesWaterTableDepth_noirrig = np.zeros((len(Measured_dates), nrow*ncol))

for it in range(len(Measured_dates)):  # For each measured time
    PiezoMap = GWTable_ModFlow[int(Index_measured_date[it])]
    PiezoDepthMap = Topography_ModFlow - GWTable_ModFlow[int(Index_measured_date[it])]
    valuesWaterTable[it] = PiezoMap.flatten()               # Map of the simulated water table for each date of observation
    valuesWaterTableDepth[it] = PiezoDepthMap.flatten()     # Map of the simulated water table depth for each date of observation

    #PiezoMap = GWTable_ModFlow_noirrig[int(Index_measured_date[it])]
    #PiezoDepthMap = Topography_ModFlow - GWTable_ModFlow_noirrig[int(Index_measured_date[it])]
    #valuesWaterTable_noirrig[it] = PiezoMap.flatten()               # Map of the simulated water table for each date of observation
    #valuesWaterTableDepth_noirrig[it] = PiezoDepthMap.flatten()     # Map of the simulated water table depth for each date of observation

valuesTopography_ModFlow = Topography_ModFlow.flatten()  # Map of the topography defined in ModFlow

som = 0
for pz in range(len(x_GW)):  # For each monitoring borehole

    print('piezo ', pz, ' / ', len(x_GW))

    # Find the corresponding column and row where the borehole is located
    colPz = int(round((x_GW[pz] - float(Size[0])) / res_ModFlow))
    rowPz = int(round((float(Size[1]) - y_GW[pz]) / res_ModFlow))

    # If the borehole is in the model and if there is at least 50% of time data
    if colPz > 0 and rowPz > 0 and colPz < ncol and rowPz < nrow and 100 * np.count_nonzero(
            ~np.isnan(Measured_watertabledepth[pz])) / Number_of_observed_dates > 50.00:
        if basin[rowPz][colPz] == 1:  # If the borehole is in the basin

            # Interpolating the simulated value as observation points are not in the cells center
            X0 = (x_GW[pz] - float(Size[0])) / res_ModFlow
            Y0 = (float(Size[1]) - y_GW[pz]) / res_ModFlow

            close_points = points[np.sqrt((points[:, 0] - X0) ** 2 + (points[:, 1] - Y0) ** 2) < 20]  # all points at distance < 20 cell units (= 20*100m for Burgenland model)
            # If the cell is not too close from an inactive cell (= too close from the basin limits)
            #if 0 not in basin[close_points[:, 1], close_points[:, 0]]:
            if 0 not in close_points[:, 1] and nrow-1 not in close_points[:, 1] and 0 not in close_points[:, 0] and ncol-1 not in close_points[:, 0]:
                som = som + 1

                x_GW_burgen[som - 1] = x_GW[pz]
                y_GW_burgen[som - 1] = y_GW[pz]
                z_GW_burgen[som - 1] = z_GW[pz]
                wtd_GW_burgen[som - 1][:] = Measured_watertabledepth[pz]  # Each column corresponds to a specific date between April 1997 and April 2007
                wt_GW_burgen[som - 1][:] = z_GW[pz] - Measured_watertabledepth[pz]

                # As interpolation seems to long for the region, we create a new grid closer from the observation point
                points2 = points[np.sqrt((points[:, 0] - X0) ** 2 + (points[:, 1] - Y0) ** 2) < 2]

                # Interpolation
                valuesTopography_ModFlow2 = valuesTopography_ModFlow[
                    np.sqrt((points[:, 0] - X0) ** 2 + (points[:, 1] - Y0) ** 2) < 2]
                Val = griddata(points2, valuesTopography_ModFlow2, (X0, Y0), method='linear')
                Sim_z_GW_burgen[som - 1] = Val

                compt = 0
                for ii in range(len(Measured_dates)):  # For each measured time

                    valuesWaterTableDepth2 = valuesWaterTableDepth[ii][
                        np.sqrt((points[:, 0] - X0) ** 2 + (points[:, 1] - Y0) ** 2) < 2]
                    valuesWaterTable2 = valuesWaterTable[ii][
                        np.sqrt((points[:, 0] - X0) ** 2 + (points[:, 1] - Y0) ** 2) < 2]
                    Val = griddata(points2, valuesWaterTableDepth2, (X0, Y0), method='linear')
                    Sim_wtd_GW_burgen[som - 1][compt] = Val
                    Val = griddata(points2, valuesWaterTable2, (X0, Y0), method='linear')
                    Sim_wt_GW_burgen[som - 1][compt] = Val

                    #valuesWaterTableDepth2 = valuesWaterTableDepth_noirrig[ii][
                        #np.sqrt((points[:, 0] - X0) ** 2 + (points[:, 1] - Y0) ** 2) < 2]
                    #valuesWaterTable2 = valuesWaterTable_noirrig[ii][
                        #np.sqrt((points[:, 0] - X0) ** 2 + (points[:, 1] - Y0) ** 2) < 2]
                    #Val = griddata(points2, valuesWaterTableDepth2, (X0, Y0), method='linear')
                    #Sim_wtd_GW_burgen_noirrig[som - 1][compt] = Val
                    #Val = griddata(points2, valuesWaterTable2, (X0, Y0), method='linear')
                    #Sim_wt_GW_burgen_noirrig[som - 1][compt] = Val

                    compt += 1

print('Number of boreholes in the Burgenland region : ', np.shape(Measured_watertabledepth)[0])
print('Number of boreholes used for water table comparison in the selected region : ', som)

## Keep only boreholes inside the basin and not too close from the basin limits
x_GW_burgen = x_GW_burgen[:som]
y_GW_burgen = y_GW_burgen[:som]
z_GW_burgen = z_GW_burgen[:som]
wt_GW_burgen = wt_GW_burgen[:som]
wtd_GW_burgen = wtd_GW_burgen[:som]
Sim_wtd_GW_burgen = Sim_wtd_GW_burgen[:som]
Sim_wt_GW_burgen = Sim_wt_GW_burgen[:som]
#Sim_wtd_GW_burgen_noirrig = Sim_wtd_GW_burgen_noirrig[:som]
#Sim_wt_GW_burgen_noirrig = Sim_wt_GW_burgen_noirrig[:som]
Sim_z_GW_burgen = Sim_z_GW_burgen[:som]


## COMPUTING ROOT MEAN SQUARE ERROR AND CORELATION BETWEEN OBSERVATION AND SIMULATION ##
## MEAN VALUE ## -> compute a global RMS error, normalized difference, normalized RMSE and coefficient of correlation

# Time averaging of simulated and observed data
Mean_Sim_wt_GW_burgen = np.nanmean(np.where(np.isnan(wt_GW_burgen), np.nan, Sim_wt_GW_burgen), 1)
Mean_Sim_wtd_GW_burgen = np.nanmean(np.where(np.isnan(wtd_GW_burgen), np.nan, Sim_wtd_GW_burgen), 1)
#Mean_Sim_wt_GW_burgen_noirrig = np.nanmean(np.where(np.isnan(wt_GW_burgen), np.nan, Sim_wt_GW_burgen_noirrig), 1)
#Mean_Sim_wtd_GW_burgen_noirrig = np.nanmean(np.where(np.isnan(wtd_GW_burgen), np.nan, Sim_wtd_GW_burgen_noirrig), 1)
Mean_wt_GW_burgen = np.nanmean(wt_GW_burgen, 1)
Mean_wtd_GW_burgen = np.nanmean(wtd_GW_burgen, 1)

# Correlation coefficient
PearsonWaterTable = np.corrcoef(Mean_wt_GW_burgen, Mean_Sim_wt_GW_burgen)
CorrWaterTable = PearsonWaterTable[0][1]
PearsonWaterTableDepth = np.corrcoef(Mean_wtd_GW_burgen, Mean_Sim_wtd_GW_burgen)
CorrWaterTableDepth = PearsonWaterTableDepth[0][1]

#PearsonWaterTable_noirrig = np.corrcoef(Mean_wt_GW_burgen, Mean_Sim_wt_GW_burgen_noirrig)
#CorrWaterTable_noirrig = PearsonWaterTable_noirrig[0][1]
#PearsonWaterTableDepth_noirrig = np.corrcoef(Mean_wtd_GW_burgen, Mean_Sim_wtd_GW_burgen_noirrig)
#CorrWaterTableDepth_noirrig = PearsonWaterTableDepth_noirrig[0][1]

# Mean Normalized water table depth difference  # In percentage
average_Norm_diffDepth = np.nanmean(np.abs(Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen) / Mean_wtd_GW_burgen * 100)
#average_Norm_diffDepth_noirrig = np.nanmean(np.abs(Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen_noirrig) / Mean_wtd_GW_burgen * 100)
Norm_diffDepth = (Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen) / Mean_wtd_GW_burgen * 100
#Norm_diffDepth_noirrig = (Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen_noirrig) / Mean_wtd_GW_burgen * 100
# The 'average' criteria is always positive, we add - to indicate if simulated water levels are too deep
average_Norm_diffDepth = average_Norm_diffDepth * np.sign(np.nanmean(Norm_diffDepth))
#average_Norm_diffDepth_noirrig = average_Norm_diffDepth_noirrig * np.sign(np.nanmean(Norm_diffDepth_noirrig))
#print('average_Norm_diffDepth (previous version) : ', average_Norm_diffDepth)

# Removing boreholes where the difference is too high (to avoid that they count too much in the average)
average_Norm_diffDepth0 = np.abs(Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen) / Mean_wtd_GW_burgen * 100
plt.figure()
txt_95percentile = '95th percentile (on absolute error) : ' + str(np.round(np.percentile(average_Norm_diffDepth0, 95) * 10) / 10) + ' %'
plt.hist((Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen) / Mean_wtd_GW_burgen * 100, bins=50, label=txt_95percentile)
plt.grid()
plt.legend(fontsize=14)
plt.title('Distribution of the mean water table depth differences [%]', fontsize=14)
average_Norm_diffDepth = np.abs(Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen) / Mean_wtd_GW_burgen * 100
average_Norm_diffDepth[average_Norm_diffDepth0 > np.percentile(average_Norm_diffDepth0, 95)] = np.nan
average_Norm_diffDepth = np.nanmean(average_Norm_diffDepth) * np.sign(np.nanmean(Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen))
#average_Norm_diffDepth_noirrig = np.abs(Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen_noirrig) / Mean_wtd_GW_burgen * 100
#average_Norm_diffDepth_noirrig[average_Norm_diffDepth0 > np.percentile(average_Norm_diffDepth0, 95)] = np.nan
#average_Norm_diffDepth_noirrig = np.nanmean(average_Norm_diffDepth_noirrig) * np.sign(np.nanmean(Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen_noirrig))

# Mean root mean square error
RmseWaterTable = np.sqrt(np.sum((Mean_wt_GW_burgen - Mean_Sim_wt_GW_burgen)**2) / len(Mean_wt_GW_burgen))
RmseWaterTableDepth = np.sqrt(np.sum((Mean_wtd_GW_burgen - Mean_Sim_wtd_GW_burgen)**2) / len(Mean_wtd_GW_burgen))

# Mean normalized Root mean square error
RmseWaterTable_norm = RmseWaterTable/np.std(Mean_wt_GW_burgen)
RmseWaterTableDepth_norm = RmseWaterTableDepth/np.std(Mean_wtd_GW_burgen)


# Plot the basin map with scatter showing discrepancy in Water Table Depth

extent = (Size[0], Size[0]+ncol*res_ModFlow, Size[1], Size[1]-nrow*res_ModFlow)

fig4 = plt.figure()
plt.imshow(Topography_ModFlow - np.mean(GWTable_ModFlow, 0), cmap='viridis_r', alpha=0.8, extent=extent, origin='lower', vmin=0, vmax=20)
plt.gca().invert_yaxis()
cb = plt.colorbar()
#plt.title('Mean Water Table Depth Comparison at several boreholes', fontsize=16)
plt.title('Boreholes location', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.scatter(x_GW_burgen[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)],
            y_GW_burgen[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)],
            #c=Norm_diffDepth[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)],
            s=50, color='k')
#cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
#cb.set_label('Water Table Depth Difference [%]\n(<0 if too deep)', fontsize=15)
cb.set_label('Simulated water table depth [m]', fontsize=15)
cb.ax.tick_params(labelsize=14)


# Plot time-averaged results for water table comparison
fig5 = plt.figure()
# Print the perfect line 1/1
plt.plot([np.nanmin(Mean_wt_GW_burgen), np.nanmax(Mean_wt_GW_burgen)], [np.nanmin(Mean_wt_GW_burgen), np.nanmax(Mean_wt_GW_burgen)],
         linestyle='solid', linewidth=2, marker=None)
label1 = ' Pearson correlation coefficient : ' + str(round(CorrWaterTable*100)/100) + '\n RMSE [m] : ' + str(round(RmseWaterTable*100)/100) + '\n Normalized RMSE: ' + str(round(RmseWaterTable_norm*100)/100)
plt.plot(Mean_wt_GW_burgen, Mean_Sim_wt_GW_burgen, marker='o', markersize=10, color='r', linestyle='')
plt.text(np.nanmin(Mean_wt_GW_burgen), np.nanmax(Mean_wt_GW_burgen)-100, label1, fontsize=15,
         bbox=dict(facecolor='white', alpha=0.5), linespacing=2)
plt.xlabel('Measured Mean Water Table [m]', fontsize=16)
plt.ylabel('Simulated Mean Water Table [m]', fontsize=16)
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Plot time-averaged results for water table depth comparison
fig6 = plt.figure(figsize=(12, 10))
# Print the perfect line 1/1
plt.plot([-10, 0], [-10, 0], color='k', linestyle='--', linewidth=2, marker=None)
#label2 = ' Mean difference [%] : ' + str(round(average_Norm_diffDepth * 100) / 100) + '\n Pearson correlation coefficient : ' + str(round(CorrWaterTableDepth*100)/100) + '\n RMSE [m] : ' + str(round(RmseWaterTableDepth*100)/100) + '\n Normalized RMSE: ' + str(round(RmseWaterTableDepth_norm*100)/100)
#lab_meandiff1 = 'Simulation without irrigation\nMean difference = ' + str(np.round(np.abs(average_Norm_diffDepth_noirrig), 1)) + ' %'
lab_meandiff2 = 'Simulation with irrigation\nMean difference = ' + str(np.round(np.abs(average_Norm_diffDepth), 1)) + ' %'
print(' Mean difference [%] : ' + str(round(average_Norm_diffDepth*100)/100))
#plt.scatter(-1 * Mean_wtd_GW_burgen[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)],
 #           -1 * Mean_Sim_wtd_GW_burgen_noirrig[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)],
  #          s=40, color='gray', label=lab_meandiff1)
plt.scatter(-1 * Mean_wtd_GW_burgen[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)],
            -1 * Mean_Sim_wtd_GW_burgen[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)],
            #c=Norm_diffDepth[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)],
            s=30, color='r', alpha=0.8, label=lab_meandiff2)
#plt.text(np.nanmin(-Mean_wtd_GW_bhima), np.nanmax(-Mean_Sim_wtd_GW_bhima)-30, label2, fontsize=15,
 #       bbox=dict(facecolor='white', alpha=0.5), linespacing=2)
plt.xlabel('Measured time-averaged water table depth [m]', fontsize=18)
plt.ylabel('Simulated time-averaged water table depth [m]', fontsize=18)
plt.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.ylim(-10, 0)
plt.xlim(0, -10)
plt.axis('equal')
plt.ylim(-10, 0)
plt.xlim(0, -10)
#cb = plt.colorbar()
#cb.ax.tick_params(labelsize=15)
#cb.set_label('Water Table Depth Difference [%]\n(<0 if too deep)', fontsize=16)
#cb.ax.tick_params(labelsize=15)

print('Observed mean water table depth at boreholes : ', np.nanmean(Mean_wtd_GW_burgen[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)]))
print('Simulated mean water table depth at boreholes : ', np.nanmean(Mean_Sim_wtd_GW_burgen[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)]))


## COMPUTING ROOT MEAN SQUARE ERROR AND CORELATION BETWEEN OBSERVATION AND SIMULATION ##
## TRANSIENT VALUE ## -> compute nRMSE for each borehole, after subtracting the mean as the mean behavior is already compared by other criteria

## Computing normalized root mean square error on water table fluctuations for each borehole
Transient_nRMSE = np.zeros(len(wt_GW_burgen))
for ib in range(len(wt_GW_burgen)):  # For each borehole
    if np.count_nonzero(~np.isnan(wt_GW_burgen[ib]))/len(wt_GW_burgen[ib]) < 0.5:
        print('remove from the comparison')
        Transient_nRMSE[ib] = np.nan
    else:
        Transient_nRMSE[ib] = np.sqrt(np.nansum((wt_GW_burgen[ib]-np.nanmean(wt_GW_burgen[ib])-
                                                 (Sim_wt_GW_burgen[ib]-np.nanmean(Sim_wt_GW_burgen[ib])))**2)
                                      / np.count_nonzero(~np.isnan(wt_GW_burgen[ib]))) / np.nanstd(wt_GW_burgen[ib])
print('95th percentile of nRMSE on transient fluctuations : ', np.nanpercentile(Transient_nRMSE, 95))
Transient_nRMSE[Transient_nRMSE > np.nanpercentile(Transient_nRMSE, 95)] = np.nan


# Computing water table fluctuations at each borehole
wt_fluctu = np.zeros((len(wt_GW_burgen), len(Measured_dates)))
Sim_wt_fluctu = np.zeros((len(wt_GW_burgen), len(Measured_dates)))
#Sim_wt_fluctu_noirrig = np.zeros((len(wt_GW_burgen), len(Measured_dates)))
for ib in range(len(wt_GW_burgen)):  # for each borehole, remove the mean

    wt_fluctu[ib] = wt_GW_burgen[ib] - np.nanmean(wt_GW_burgen[ib])
    Sim_wt_fluctu[ib] = Sim_wt_GW_burgen[ib] - np.nanmean(np.where(np.isnan(wt_GW_burgen[ib]), np.nan, Sim_wt_GW_burgen[ib]))
    #Sim_wt_fluctu_noirrig[ib] = Sim_wt_GW_burgen_noirrig[ib] - np.nanmean(np.where(np.isnan(wt_GW_burgen[ib]), np.nan, Sim_wt_GW_burgen_noirrig[ib]))

# Mean value of the nRMSE in transient
Average_transient_nRMSE = np.sqrt(np.nansum((np.nanmean(wt_fluctu, 0)-np.nanmean(Sim_wt_fluctu, 0))**2) / np.count_nonzero(~np.isnan(np.nanmean(wt_fluctu, 0)))) / np.nanstd(np.nanmean(wt_fluctu, 0))

# Optimize curve fitting with a sinus, to compare the seasonal amplitude
from scipy import optimize
from scipy import signal

x_data = np.linspace(1, len(Measured_dates), len(Measured_dates))
periode_annuel = 12  # Because we have only two data each year in the Bhima

def sin_func(x, ampli, phase):
    ff = np.abs(ampli) * np.sin(2*np.pi/periode_annuel * x + phase)
    return ff

params_mean_wt_fluctu, params_covariance = optimize.curve_fit(sin_func, x_data, signal.detrend(np.nanmean(wt_fluctu, 0)), p0=[1, 5])
params_mean_Sim_wt_fluctu, params_covariance = optimize.curve_fit(sin_func, x_data, signal.detrend(np.nanmean(Sim_wt_fluctu, 0)), p0=[1, 5])
seasonal_amplitude_difference = -2 * (params_mean_wt_fluctu[0]-params_mean_Sim_wt_fluctu[0])
print(params_mean_wt_fluctu[0], params_mean_Sim_wt_fluctu[0])
#Average_transient_nRMSE = Average_transient_nRMSE * (seasonal_amplitude_difference/np.abs(seasonal_amplitude_difference))
# thus the critria is > 0 when the simulated fluctuations are too high

# Plot the mean storage fluctuations at boreholes,
# Plot the results from the model and from the observed data

Date_vector = []
for i in range(len(Measured_dates)):
    Date_vector.append(datetime.fromordinal(int(Measured_dates[i])))

fig11 = plt.figure()
plt.plot_date(Date_vector, np.nanmean(wt_fluctu, 0), label='Observation from boreholes', linestyle='solid',
              marker=None, linewidth=3, color='k')
#plt.plot_date(Date_vector, np.nanmean(Sim_wt_fluctu_noirrig, 0), label='Simulation without irrigation',
             # linestyle='solid', marker=None, linewidth=2, color='gray')
lab_rmse = 'Simulation with irrigation\nnRMSE = ' + str(np.round(np.abs(Average_transient_nRMSE * 100), 1)) + ' %'
plt.plot_date(Date_vector, np.nanmean(Sim_wt_fluctu, 0), label=lab_rmse,
              linestyle='solid', marker=None, linewidth=2, color='r')
plt.grid()
plt.legend(fontsize=14)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time [monthly]', fontsize=15)
plt.ylabel('Water table fluctuations [m]\n(average across all boreholes)', fontsize=16)

fig11 = plt.figure()
plt.plot_date(Date_vector, np.nanmean(wt_fluctu, 0) - np.nanmean(Mean_wtd_GW_burgen[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)]), label='Observation from boreholes', linestyle='solid',
              marker=None, linewidth=3, color='k')
#plt.plot_date(Date_vector, np.nanmean(Sim_wt_fluctu_noirrig, 0) - np.nanmean(Mean_Sim_wtd_GW_burgen_noirrig[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)]), label='Simulation without irrigation',
              #linestyle='solid', marker=None, linewidth=2, color='gray')
lab_rmse = 'Simulation with irrigation\nnRMSE = ' + str(np.round(np.abs(Average_transient_nRMSE * 100), 1)) + ' %'
plt.plot_date(Date_vector, np.nanmean(Sim_wt_fluctu, 0) - np.nanmean(Mean_Sim_wtd_GW_burgen[average_Norm_diffDepth0 <= np.percentile(average_Norm_diffDepth0, 95)]), label=lab_rmse,
              linestyle='solid', marker=None, linewidth=2, color='r')
plt.grid()
plt.legend(fontsize=14)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time [monthly]', fontsize=15)
plt.ylabel('Water table fluctuations [m]\n(average across all boreholes)', fontsize=16)

# Plot the best 5 boreholes for water table fluctuations

sorted_Transient_nRMSE = np.argsort(Transient_nRMSE)
plt.figure()
compt = 0
for i in range(som):

    compt += 1
    if compt == 5:
        plt.tight_layout()
        plt.figure()
        compt = 1

    plt.subplot(2, 2, compt)
    plt.plot_date(Date_vector, wt_GW_burgen[sorted_Transient_nRMSE[i]]-np.nanmean(wt_GW_burgen[sorted_Transient_nRMSE[i]]),
                  label='Observed water table fluctuations [m]', linestyle='solid', marker=None, linewidth=2, color='k')
    plt.plot_date(Date_vector, Sim_wt_GW_burgen[sorted_Transient_nRMSE[i]]-np.nanmean(Sim_wt_GW_burgen[sorted_Transient_nRMSE[i]]),
                  label='Simulated water table fluctuations [m]', linestyle='solid', marker=None, linewidth=2, color='r')
    plt.grid()
    plt.legend(fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Time [monthly]', fontsize=14)
    plt.ylabel('Water Table fluctuations (best boreholes)', fontsize=14)
    txt_title = 'Transient nRMSE on Water Table Fluctuations : ' + str(np.round(Transient_nRMSE[sorted_Transient_nRMSE[i]]*100)/100)
    plt.title(txt_title, fontsize=15)

plt.tight_layout()

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("CompareWTD_irrigVSnoirrig_36_g.pdf")
for fig in range(1, plt.figure().number):
    pdf.savefig(fig)
pdf.close()
plt.close('all')
