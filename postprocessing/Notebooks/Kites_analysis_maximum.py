# 'time', 'namePFT','latitude', 'longitude', 'irr_amount'
from netCDF4 import Dataset
import netCDF4
import numpy as np
import plotly.graph_objects as go

output_folder_main = r'C:\Users\smilovic\OneDrive - IIASA\projects\Kites'

output_folder_irr = r'C:\Users\smilovic\OneDrive - IIASA\projects\Kites\OUTPUT_ipot'
cellArea = Dataset(output_folder_irr + '/cellArea.nc').variables['cellArea_totalend'][:, :]
irrHarvest = Dataset(output_folder_irr + '/pft_harvest.pft.nc').variables['harvest'][:, :, :, :]
rainfedHarvest = Dataset(r'C:\Users\smilovic\OneDrive - IIASA\projects\Kites\pft_harvest.pft.nc').variables['harvest'][
                 :, :, :, :]
irrArea = Dataset(output_folder_irr + '/CFTfrac.nc').variables['CFTfrac'][:,:,:,:]
irrAmount = Dataset(output_folder_irr + '/pft_airrig.nc').variables['irrig'][:, :, :, :]

for wp_name in ['harvest_irr_bonus']:  # , 'harvest_irr', 'harvest_ET', 'harvest_trans']:
    fig = go.Figure()
    wp_pareto = Dataset(output_folder_main + '/wp_pareto_' + wp_name + '.nc', 'r')
    for year in [0]:
        Harvest_irrigated_bonus = 0
        Irrigation_amount = 0
        for lat in range(len(wp_pareto.variables['latitude'][:])):
            for lon in range(len(wp_pareto.variables['longitude'][:])):
                max_y = 0
                associated_x = 0
                if irrHarvest[year, 0 + 16, lat, lon] > 0 and rainfedHarvest[year, 0, lat, lon] > 0:
                    x = [15, 30, 45, 60, 75, 90, 105]
                    y = []
                    for irr in x:
                        y.append(wp_pareto.variables['wp_' + wp_name][year, 0, lat, lon, irr // 15])
                        if y[-1]>max_y:
                            max_y = y[-1]
                            associated_x = irr

                    x.append(irrAmount[year, 0 + 16, lat, lon])
                    y.append((irrHarvest[year, 0 + 16, lat, lon] - rainfedHarvest[year, 0, lat, lon]) / irrAmount[
                        year, 0 + 16, lat, lon])

                    # If the full irrigation amount is zero
                    if x[-1]==0:
                        Harvest_irrigated_bonus += irrHarvest[year, 0 + 16, lat, lon]*irrArea[year,0+16,lat,lon]
                        max_y = 0
                        associated_x = 0
                        print('no irrig',lat, lon, irrHarvest[year, 0 + 16, lat, lon], irrArea[year,0+16,lat,lon])
                        
                    # If the full irrigation water productivity is higher than supplemental up to 105
                    elif y[-1]>max_y:
                        max_y = y[-1]
                        associated_x = irr
                        print(lat, lon, max_y, associated_x)


                    Harvest_irrigated_bonus += max_y*associated_x*irrArea[year,0+16,lat,lon]
                    Irrigation_amount += associated_x*irrArea[year,0+16,lat,lon]

                    fig.add_trace(go.Scatter(x=[i*irrArea[year,0+16,lat,lon] for i in x], y=[i*irrArea[year,0+16,lat,lon] for i in y],
                                             #x=x, y=y,
                                             mode='lines+markers',
                                             name=str(lat) + ' ' + str(lon)))

        fig.update_layout(
            title='Maximum irrigation water productivity, year: ' + str(year + 1979),
            xaxis_title="Irrigation amount (mm*Irr_frac)",
            yaxis_title="Maximum irrigation water productivity (gC/mm*Irr_frac)",
            legend_title="Cells")

        print(Irrigation_amount)
        print(Harvest_irrigated_bonus)
        fig.show()