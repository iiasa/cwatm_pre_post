# 'time', 'namePFT','latitude', 'longitude', 'irr_amount'
from netCDF4 import Dataset
import netCDF4
import numpy as np
import plotly.graph_objects as go
import math

output_folder_main = r'C:\Users\smilovic\OneDrive - IIASA\projects\Kites'
output_folder_irr = r'C:\Users\smilovic\OneDrive - IIASA\projects\Kites\OUTPUT_ipot'
irrHarvest = Dataset(output_folder_irr + '/pft_harvest.pft.nc').variables['harvest'][:, :, :, :]
irrAmount = Dataset(output_folder_irr + '/pft_airrig.nc').variables['irrig'][:, :, :, :]
irrArea = Dataset(output_folder_irr + '/CFTfrac.nc').variables['CFTfrac'][:,:,:,:]
rainfedHarvest = Dataset(r'C:\Users\smilovic\OneDrive - IIASA\projects\Kites\pft_harvest.pft.nc').variables['harvest'][:, :, :, :]
cellArea = Dataset(r'C:\Users\smilovic\OneDrive - IIASA\projects\Kites/cellArea.nc').variables['cellArea_totalend'][:, :]

CROPS = [('TEMPERATE CEREALS',0),
 ('RICE',1),
 ('TROPICAL CEREALS',3),
 ('PULSES',4),
 ('TEMPERATE ROOTS',5),
 ('GROUNDNUT OIL',9),
 ('RAPESEED OIL',10)]

wp_name = 'harvest_irr_bonus'
# 'harvest_irr', 'harvest_ET', 'harvest_trans'
wp_pareto = Dataset(output_folder_main + '/wp_pareto_' + wp_name + '.nc', 'r')

for crop in CROPS[:1]:

    fig = go.Figure()
    fig_WP = go.Figure()

    BONUSHARVEST = []
    TOTALHARVEST = []
    IRRIGATION = []

    for year in range(31):

        # max harvest for each irrigation amount
        Harvest_Irrigated_bonus_YEAR = [0]*9
        Harvest_total_YEAR = [0]*9

        # each irrigation amount (volume)
        Irrigation_amount_YEAR = [0]*9

        harvest_total = [0]*9
        harvest_bonus = [0]*9
        irrigation = [0]*9

        Harvest_Irrigated_bonus_best_YEAR = 0
        Irrigation_amount_best_YEAR = 0

        for lat in range(len(wp_pareto.variables['latitude'][:])):
            for lon in range(len(wp_pareto.variables['longitude'][:])):
                if irrHarvest[year, crop[1] + 16, lat, lon] > 0 and rainfedHarvest[year, crop[1], lat, lon] > 0:

                    harvestbonus_cell = []
                    harvesttotal_cell = []
                    irrigation_cell = []
                    x = [0, 15, 30, 45, 60, 75, 90, 105]

                    wp_best = 0
                    harvest_best = 0
                    irr_best = 0

                    rainfed_harvest = rainfedHarvest[year, crop[1], lat, lon] * irrArea[year, crop[1] + 16, lat, lon] * \
                                      cellArea[lat, lon] / 1000000000000
                    for irr in x:
                        if irr == 0:
                            harvestbonus_cell.append(0)
                            harvesttotal_cell.append(rainfed_harvest)

                            harvest_total[0] += rainfed_harvest
                            harvest_bonus[0] += 0
                        else:
                            harvest_from_wp = wp_pareto.variables['wp_' + wp_name][year, crop[1], lat, lon, irr // 15] * irr *np.where(irrArea[year,crop[1] +16,lat,lon] >= 0, irrArea[year,crop[1]+16,lat,lon], 0)*cellArea[lat,lon]/1000000000000

                            harvestbonus_cell.append(harvest_from_wp)
                            harvesttotal_cell.append(rainfed_harvest + harvest_from_wp)

                            harvest_bonus[x.index(irr)] += harvest_from_wp
                            harvest_total[x.index(irr)] += rainfed_harvest + harvest_from_wp


                        irrigation_cell.append(irr*irrArea[year,crop[1] +16,lat,lon]*cellArea[lat,lon]/1000000000000)
                        #X.append(irr*irrArea[year,crop[1] +16,lat,lon]*cellArea[lat,lon]/1000000000000)

                        Harvest_Irrigated_bonus_YEAR[x.index(irr)] += harvestbonus_cell[-1]
                        Harvest_total_YEAR[x.index(irr)] += harvesttotal_cell[-1]
                        Irrigation_amount_YEAR[x.index(irr)] += irrigation_cell[-1]

                        if wp_pareto.variables['wp_' + wp_name][year, crop[1], lat, lon, irr // 15] > wp_best:
                            wp_best = wp_pareto.variables['wp_' + wp_name][year, crop[1], lat, lon, irr // 15]
                            harvest_best = harvestbonus_cell[-1]
                            irr_best = irrigation_cell[-1]

                    # We explore the crop-specific potential of supplemental irrigation
                    # only in cells where there is currently irrigation infrastructure
                    # employed for the specific crop

                    irrigation_cell.append(irrAmount[year, crop[1] + 16, lat, lon]*irrArea[year,crop[1] +16,lat,lon]*cellArea[lat,lon]/1000000000000)
                    Irrigation_amount_YEAR[-1] += irrigation_cell[-1]

                    harvestbonus_cell.append((irrHarvest[year, crop[1] + 16, lat, lon] - rainfedHarvest[year, crop[1], lat, lon])*irrArea[year,crop[1] +16,lat,lon]*cellArea[lat,lon]/1000000000000)
                    harvesttotal_cell.append(irrHarvest[year, crop[1] + 16, lat, lon]*irrArea[year,crop[1] +16,lat,lon]*cellArea[lat,lon]/1000000000000)

                    Harvest_Irrigated_bonus_YEAR[-1] += harvestbonus_cell[-1]
                    Harvest_total_YEAR[-1] += harvesttotal_cell[-1]

                    """
                    if irrigation_cell-1]>0:
                        if (irrHarvest[year, 0 + 16, lat, lon] - rainfedHarvest[year, 0, lat, lon])/irrAmount[year, 0 + 16, lat, lon] > wp_best:
                            print(str(wp_best)+ ' 1')
                            wp_best = (irrHarvest[year, 0 + 16, lat, lon] - rainfedHarvest[year, 0, lat, lon])/irrAmount[year, 0 + 16, lat, lon]
                            print(str(wp_best) +' 2')
                            harvest_best = (irrHarvest[year, 0 + 16, lat, lon] - rainfedHarvest[year, 0, lat, lon])*irrArea[year,0+16,lat,lon]*cellArea[lat,lon]/1000000000000
                            irr_best = irrAmount[year, 0 + 16, lat, lon]*irrArea[year,0+16,lat,lon]*cellArea[lat,lon]/1000000000000
                    """

                    Harvest_Irrigated_bonus_best_YEAR += harvest_best
                    Irrigation_amount_best_YEAR += irr_best

                    fig.add_trace(go.Scatter(x=irrigation_cell, y=harvestbonus_cell,
                                             #x=X, y=y,
                                             mode='lines+markers',
                                             name=str(lat) + ' ' + str(lon)))

        fig.update_layout(
            title='Maximum yield bonus and irrigation amount, year: ' + str(year + 1979),
            xaxis_title="Irrigation amount (km3)",
            yaxis_title="Maximum harvest bonus (kilotonnes C/m2)",
            legend_title="Cells")

        #fig.show()

        # Includes the absolute optimised harvest bonus and irrigitation volume
        # Use this without lines in the Figures
        # Harvest_Irrigated_bonus_YEAR.append(Harvest_Irrigated_bonus_best_YEAR)
        # Irrigation_amount_YEAR.append(Irrigation_amount_best_YEAR)

        WP_YEAR=[Harvest_Irrigated_bonus_YEAR[i + 1] / Irrigation_amount_YEAR[i + 1] for i in
                              range(len(Irrigation_amount_YEAR) - 1)]

        #totalHarvest = [Harvest_Irrigated_bonus_YEAR[0]]
        #totalHarvest.extend([Harvest_Irrigated_bonus_YEAR[0]+i for i in Harvest_Irrigated_bonus_YEAR[1:]])

        BONUSHARVEST.append(Harvest_Irrigated_bonus_YEAR)
        TOTALHARVEST.append(Harvest_total_YEAR)
        IRRIGATION.append(Irrigation_amount_YEAR)

        fig_WP.add_trace(go.Scatter(x=Irrigation_amount_YEAR[1:] , y=WP_YEAR,
                                 mode='lines+markers',
                                 name=str(year + 1979)))

    #IRR_WP = [BONUSHARVEST[j][i + 1] / IRRIGATION[j][i + 1] for i in
    #           range(len(Irrigation_amount_YEAR) - 1)] for j in range(len(IRRIGATION))

    TOTALHARVEST_norm = [[i/max(year) for i in year] for year in TOTALHARVEST]
    IRRIGATION_norm = [[i / max(year) for i in year] for year in IRRIGATION]

    fig_Harvest = go.Figure()
    fig_totalHarvest = go.Figure()
    fig_totalHarvest_norm = go.Figure()

    for i in range(len(TOTALHARVEST_norm)):
        fig_Harvest.add_trace(go.Scatter(x=IRRIGATION[i], y=BONUSHARVEST[i],
                                         mode='lines+markers',
                                         name=str(i + 1979)))

        fig_totalHarvest.add_trace(go.Scatter(x=IRRIGATION[i], y=TOTALHARVEST[i],
                                              mode='lines+markers',
                                              name=str(i + 1979)))

        fig_totalHarvest_norm.add_trace(go.Scatter(x=IRRIGATION_norm[i], y=TOTALHARVEST_norm[i],
                                                   mode='lines+markers',
                                                   name=str(i + 1979)))

    fig_totalHarvest_norm.update_layout(
        title=crop[0] + ': Maximum total harvest and irrigation volume, normalised by relative maximums, years 1979 to ' + str(
            i + 1979),
        xaxis_title="Fraction of maximum Irrigation",
        yaxis_title="Fraction of maximum Harvest",
        legend_title="Years")
    fig_totalHarvest_norm.show()

    fig_WP.update_layout(
        title=crop[0] +': Maximum irrigation water productivity and irrigation volume, years 1979 to ' + str(year + 1979),
        xaxis_title="Irrigation amount (km3)",
        yaxis_title="Irrigation water productivity",
        legend_title="Years")
    fig_WP.show()

    fig_totalHarvest.update_layout(
        title=crop[0] +': Maximum total harvest and irrigation volume, years 1979 to ' + str(year + 1979),
        xaxis_title="Irrigation amount (km3)",
        yaxis_title="Harvest total (kilotonnes C/m2)",
        legend_title="Years")
    fig_totalHarvest.show()

    fig_Harvest.update_layout(
        title=crop[0] +': Maximum bonus harvest and irrigation volume, years 1979 to ' + str(year + 1979),
        xaxis_title="Irrigation amount (km3)",
        yaxis_title="Harvest bonus (kilotonnes C/m2)",
        legend_title="Years")
    fig_Harvest.show()