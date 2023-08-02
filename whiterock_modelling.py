# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 11:55:43 2023

@author: liame
"""


#%% Imports

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import pandas as pd
import pytz
import datetime
import calendar
from dateutil.relativedelta import relativedelta
import psutil
import pyarrow
from nemosis import defaults
from nemosis import dynamic_data_compiler
from nemosis import static_table


import pvlib

from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, Array, SingleAxisTrackerMount

from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

import time
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%% Opening Datasets of grib files
swr1 = xr.open_dataset('swhiterock1.grib', engine = 'cfgrib', chunks = 1000)
swr2 = xr.open_dataset('swhiterock2.grib', engine = 'cfgrib', chunks = 1000)
swr3 = xr.open_dataset('swhiterock3.grib', engine = 'cfgrib', chunks = 1000)

wr1 = xr.open_dataset('whiterock1.grib', engine = 'cfgrib', chunks = 1000)
wr2 = xr.open_dataset('whiterock2.grib', engine = 'cfgrib', chunks = 1000)
wr3 = xr.open_dataset('whiterock3.grib', engine = 'cfgrib', chunks = 1000)
wr4 = xr.open_dataset('whiterock4.grib', engine = 'cfgrib', chunks = 1000)
wr5 = xr.open_dataset('whiterock5.grib', engine = 'cfgrib', chunks = 1000)
wr6 = xr.open_dataset('whiterock6.grib', engine = 'cfgrib', chunks = 1000)
wr7 = xr.open_dataset('whiterock7.grib', engine = 'cfgrib', chunks = 1000)
wr8 = xr.open_dataset('whiterock8.grib', engine = 'cfgrib', chunks = 1000)
wr9 = xr.open_dataset('whiterock9.grib', engine = 'cfgrib', chunks = 1000)

#%% Creating numpy arrays of relevent parameters
wr_u1 = wr1.u10.values[:,0,0]
wr_u2 = wr2.u10.values[:,0,0]
wr_u3 = wr3.u10.values[:,0,0]
wr_u4 = wr4.u10.values[:,0,0]
wr_u5 = wr5.u10.values[:,0,0]
wr_u6 = wr6.u10.values[:,0,0]
wr_u7 = wr7.u10.values[:,0,0]
wr_u8 = wr8.u10.values[:,0,0]
wr_u9 = wr9.u10.values[:,0,0]

wr_v1 = wr1.v10.values[:,0,0]
wr_v2 = wr2.v10.values[:,0,0]
wr_v3 = wr3.v10.values[:,0,0]
wr_v4 = wr4.v10.values[:,0,0]
wr_v5 = wr5.v10.values[:,0,0]
wr_v6 = wr6.v10.values[:,0,0]
wr_v7 = wr7.v10.values[:,0,0]
wr_v8 = wr8.v10.values[:,0,0]
wr_v9 = wr9.v10.values[:,0,0]

wr_t1 = wr1.t2m.values[:,0,0]
wr_t2 = wr2.t2m.values[:,0,0]
wr_t3 = wr3.t2m.values[:,0,0]
wr_t4 = wr4.t2m.values[:,0,0]
wr_t5 = wr5.t2m.values[:,0,0]
wr_t6 = wr6.t2m.values[:,0,0]
wr_t7 = wr7.t2m.values[:,0,0]
wr_t8 = wr8.t2m.values[:,0,0]
wr_t9 = wr9.t2m.values[:,0,0]

nans_wr_s1 = swr1.ssr.values
nans_wr_s2 = swr2.ssr.values
nans_wr_s3 = swr3.ssr.values

wr_s1 = nans_wr_s1[~np.isnan(nans_wr_s1)]
wr_s2 = nans_wr_s2[~np.isnan(nans_wr_s2)]
wr_s3 = nans_wr_s3[~np.isnan(nans_wr_s3)]

u = np.concatenate([wr_u1, wr_u2, wr_u3, wr_u4, wr_u5, wr_u6, wr_u7,
                    wr_u8, wr_u9])

v = np.concatenate([wr_v1, wr_v2, wr_v3, wr_v4, wr_v5, wr_v6, wr_v7,
                    wr_v8, wr_v9])

t = np.concatenate([wr_t1, wr_t2, wr_t3, wr_t4, wr_t5, wr_t6, wr_t7,
                    wr_t8, wr_t9]) - 273.15

s = np.concatenate([wr_s1, wr_s2, wr_s3]) / 3600


#%% Variables

# General

Australian = pytz.timezone('Australia/Sydney')
latitude = -29.76
longitude = 151.55

# Wind

wind_capacity = 175     # MW
hub_height = 89.5       # m
P_rated_wind = 2.5      # MW
v_rated = 9.3           # m/s
v_cut_in = 3.0          # m/s
v_cut_out = 22          # m/s

# Solar

surface_tilt = 24
surface_azimuth = 0
modules_per_string= 39
strings_per_inverter = 238
number_of_inverters = 8

location = Location(latitude = latitude, longitude = longitude, tz = 'Australia/Sydney',
                    altitude = 400, name = 'White Rock Solar Farm')

celltype = 'monoSi'
pdc0 = 350
v_mp = 38.7
i_mp = 9.04
v_oc = 47.0
i_sc = 9.60
alpha_sc = 0.0005 * i_sc
beta_voc = -0.0029 * v_oc
gamma_pdc = -0.39
cells_in_series = 6*12
temp_ref = 25



#%% Wind Generation
number_of_turbines = wind_capacity // P_rated_wind

# Wind Profile Calculation This only for region betwix cut in and rated wind speeds
h1 = 10
h2 = hub_height
z0 = 0.05
v1 = np.sqrt(u**2 + v**2)
v2 = v1 * (math.log((h2)/(z0)))/(math.log((h1)/(z0)))

a = P_rated_wind * 1000000 / (v_rated**3 - v_cut_in**3)
b = (v_rated**3) / (v_rated**3 - v_cut_in**3)

E_wind = np.copy(v2)
E_wind[E_wind < v_cut_in] = 0
E_wind[(E_wind > v_cut_in) & (E_wind < v_rated)] = (a * E_wind[(E_wind > v_cut_in) &(E_wind < v_rated)]**3 - b * P_rated_wind)/1000000
E_wind[(E_wind > v_rated) & (E_wind < v_cut_out)] = P_rated_wind
E_wind[E_wind > v_cut_out] = 0

E_wind *= number_of_turbines

start_date = datetime.datetime(1990, 1, 1)
end_date = datetime.datetime(2022, 12, 31, 23, 0)

dt = []

while start_date <= end_date:
    dt.append(start_date)
    start_date += datetime.timedelta(hours=1)


raw_weather = pd.DataFrame({'time': dt, 'ghi':s, 'w10': v1, 't2m':t})
raw_weather['time'] = pd.to_datetime(raw_weather['time'])
raw_weather.set_index('time', inplace=True)
raw_weather.index = raw_weather.index.tz_localize('UTC')
raw_weather.index = raw_weather.index.tz_convert(Australian)

#%% Turbine Power Curve
step = 0.5
cut_in_wind_speed = np.arange(0, v_cut_in, step)
cubic_wind_speed = np.arange(v_cut_in, v_rated, step)
rated_wind_speed = np.arange(v_rated, v_cut_out, step)
cut_out_wind_speed = np.arange(v_cut_out, v_cut_out + 5*step, step)
cubic_pc = (a * cubic_wind_speed**3 - b * P_rated_wind)/1000000
cut_in_pc = np.zeros_like(cut_in_wind_speed)
rated_pc = np.zeros_like(rated_wind_speed) + P_rated_wind
cut_out_pc = np.zeros_like(cut_out_wind_speed)

fig, ax = plt.subplots(figsize=(9,6))
ax.stem(cut_in_wind_speed, cut_in_pc, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Below Cut-In')
ax.stem(cubic_wind_speed, cubic_pc, markerfmt = 'bo', linefmt = 'b', basefmt = 'none', label = 'Cubic Model')
ax.stem(rated_wind_speed, rated_pc, markerfmt = 'go', linefmt = 'g', basefmt = 'none', label = 'Rated')
ax.stem(cut_out_wind_speed, cut_out_pc, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Above Cut-Out')
# ax.set_ylim(0,4)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Turbine Output (MW)')
ax.set_title('Modelled Power Curve for Goldwind 121 2.5 MW')
ax.legend()

# plt.savefig('White Rock Trubine Power Curve', dpi = 600)

#%% pvlib


sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
cec_inverters = pvlib.pvsystem.retrieve_sam('CECInverter')


module = sandia_modules['Silevo_Triex_U300_Black__2014_']
inverter = cec_inverters['SMA_America__SC_2500_EV_US__550V_']


temperature_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# system = PVSystem(surface_tilt = 0, surface_azimuth = 0,
#                   module_parameters = module, inverter_parameters = inverter,
#                   temperature_model_parameters = temperature_parameters,
#                   modules_per_string = 39, strings_per_inverter = 9280)

system = PVSystem(modules_per_string = modules_per_string,
                  strings_per_inverter = strings_per_inverter)

# modelchain = ModelChain(system, location)

times = pd.date_range(start = '1990-01-01 11:00:00', end = '2023-01-01 10:00:00',
                      freq = '1H', tz = location.tz)



clear_sky = location.get_clearsky(times)

# clear_sky.plot(figsize = (16,9))
# plt.show()

# modelchain.run_model(clear_sky['2020-01'])
# modelchain.results.ac.plot(figsize = (16,9))
# plt.show()

solar_position = pvlib.solarposition.get_solarposition(times, latitude, longitude)
aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solar_position['zenith'],
                           solar_position['azimuth'])

boland = pvlib.irradiance.erbs(raw_weather['ghi'], solar_position['zenith'],
                                 datetime_or_doy = times)
#erbs

dni = boland['dni']
dhi = boland['dhi']

weather_data = pd.DataFrame({'time': dt, 'ghi':s, 'dhi':dhi, 'dni':dni,
                             'wind_speed':v1, 'temp_air':t})

weather_data['time'] = pd.to_datetime(weather_data['time'])
weather_data.set_index('time', inplace=True)
weather_data.index = weather_data.index.tz_localize('UTC')
weather_data.index = weather_data.index.tz_convert(Australian)

weather_data.to_csv('pvlib_wr.csv')

poa = pvlib.irradiance.get_total_irradiance(20, 0, solar_position['zenith'],
                                            solar_position['azimuth'],
                                                 weather_data['dni'],
                                                 weather_data['ghi'],
                                                 weather_data['dhi'])
poa_data = pd.DataFrame(columns = [
    'poa_global', 'poa_direct', 'poa_diffuse','temp_air', 'wind_speed'])
poa_data['poa_global'] = poa['poa_global']
poa_data['poa_direct'] = poa['poa_direct']
poa_data['poa_diffuse'] = poa['poa_diffuse']
poa_data['temp_air'] = weather_data['temp_air']
poa_data['wind_speed'] = weather_data['wind_speed']

iam = pvlib.iam.ashrae(aoi)
effective_irradiance = poa_data['poa_direct'] * iam + poa_data['poa_diffuse']
temp_cell = pvlib.temperature.faiman(poa_data['poa_global'], poa_data['temp_air'],
                                     poa_data['wind_speed'])

I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = pvlib.ivtools.sdm.fit_cec_sam(
    celltype, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, gamma_pdc, cells_in_series)

cec_params = pvlib.pvsystem.calcparams_cec(effective_irradiance,
                              temp_cell,
                              alpha_sc,
                              a_ref,
                              I_L_ref,
                              I_o_ref,
                              R_sh_ref,
                              R_s,
                              Adjust)

mpp = pvlib.pvsystem.max_power_point(*cec_params, method = 'newton')

custom_module = {
    'pdc0': pdc0,
    'gamma_pdc': gamma_pdc,
    'temp_ref': temp_ref,
    'cells_in_series': cells_in_series,
    'alpha_sc': alpha_sc,
    'beta_voc': beta_voc,
    'a_ref': a_ref,
    'I_L_ref': I_L_ref,
    'I_o_ref': I_o_ref,
    'R_sh_ref': R_sh_ref,
    'R_s': R_s,
    'Adjust': Adjust
}

dc_scaled = system.scale_voltage_current_power(mpp)
dc_scaled.plot(figsize=(16,9))
plt.show()

# result_dc = pvlib.pvsystem.pvwatts_dc(effective_irradiance,
#                                       temp_cell,
#                                       pdc0,
#                                       gamma_pdc,
#                                       temp_ref)
# result_dc.plot(figsize=(16,9))
# plt.title('DC Power')
# plt.show()

ac_results = number_of_inverters * pvlib.inverter.sandia(
    v_dc = dc_scaled.v_mp,
    p_dc = dc_scaled.p_mp,
    inverter = inverter)


# result_ac = pvlib.inverter.pvwatts(pdc = dc_scaled.p_mp,
#                                     pdc0 = 2500000,
#                                     eta_inv_nom = 0.96,
#                                     eta_inv_ref = 0.9637)

ac_results[ac_results<0]=0
ac_results.plot(figsize=(16,9))
plt.title('AC Power')
plt.show()



# plots showing irradiance components
# plt.plot(poa_data['poa_global']['2018-01-03'], label = 'ghi')
# plt.plot(poa_data['poa_direct']['2018-01-03'], label = 'dni')
# plt.plot(poa_data['poa_diffuse']['2018-01-03'], label = 'dhi')
# plt.legend()

# plots showing irradiance components
# plt.plot(weather_data['ghi']['2018-01-01'], label = 'ghi')
# plt.plot(weather_data['dni']['2018-01-01'], label = 'dni')
# plt.plot(weather_data['dhi']['2018-01-01'], label = 'dhi')
# plt.legend()

# plots comparing ghi in 2020 from pvlib clear sky to measured ghi
# plt.plot(clear_sky['ghi']['2019-01-01'], label='pvlib Clear Sky Model GHI')
# plt.plot(raw_weather['ghi']['2019-01-01'], label='ERA5 Reanalysis GHI')
# plt.legend()

# modelchain.run_model(weather_data.loc['2019'])
# modelchain.results.ac.resample('M').sum().plot(figsize = (16, 9))
# plt.show()

# modelchain.run_model_from_poa(poa_data['2019'])
# modelchain.results.ac.resample('M').sum().plot(figsize = (16, 9))
# plt.show()

#%% Still a work in progress to get the tracking working with the system
max_angle = np.nanmax(solar_position['apparent_zenith']) + 5 
mount = SingleAxisTrackerMount(axis_tilt = 0,
                               axis_azimuth = 0,
                               max_angle = max_angle,
                               backtrack = False)
                               

orientation = mount.get_orientation(solar_zenith = solar_position['apparent_zenith'],
                                    solar_azimuth = solar_position['azimuth'])

orientation['tracker_theta'].loc['2021-12-15'].fillna(0).plot(title='Tracker Orientation')
plt.show()


array = Array(mount = mount,
              module_parameters = custom_module,
              temperature_model_parameters = temperature_parameters,
              modules_per_string = modules_per_string,
              strings = strings_per_inverter)


system_sa = PVSystem(arrays=[array], inverter_parameters=inverter)
modelchain_sa = ModelChain(system_sa, location, aoi_model='no_loss', spectral_model='no_loss')
modelchain_sa.run_model(poa_data)
modelchain_sa.results.ac.plot(figsize=(16,9))
plt.show()
#%% Normalised graph of the SOI and ONI index with threshold at y=1

#SOI data preperation

soi = pd.read_csv('soi_monthly.csv', header=None)
soi.index=pd.to_datetime(soi[0],format='%Y%m', )
soi = soi.rename(columns={1: 'ANOM'})
soi.drop(0,inplace=True,axis=1)
soi.index = soi.index.tz_localize('UTC')
soi.index = soi.index.tz_convert(Australian)
soi = soi.resample('MS').mean()
soi = soi.loc['1990-01' : '2022-12']
soi['SOI'] = np.zeros_like(len(soi))
SMA_soi = soi['ANOM'].rolling(5,5,on=soi.index,axis=0).mean() # SMA is taken to signify extended periods of SOI over 7

for index, row in soi.iterrows():
    ANOM = row['ANOM']
    if ANOM >= 21:
        soi.at[index, 'SOI'] = 'Strong'
    elif 14 <= ANOM < 21:
        soi.at[index, 'SOI'] = 'Moderate'
    elif 7 <= ANOM < 14:
        soi.at[index, 'SOI'] = 'Weak'
    elif 0 <= ANOM < 7:
        soi.at[index, 'SOI'] = 'Neutral'
    else:
        soi.at[index, 'SOI'] = 'EN'

#ONI data preperation

oni=pd.read_csv('oni_monthly.csv',header='infer')
oni.index=pd.to_datetime(dict(year=oni['YR'], month=oni['MON'], day=oni['DAY']))
oni = oni.rolling(3,3,on=oni.index,axis=0).mean() # SMA is taken as per definition of ONI
oni.index = oni.index.tz_localize('UTC')
oni.index = oni.index.tz_convert(Australian)
oni = oni.resample('MS').mean()
oni = oni.loc['1990-01' : '2022-12']
oni['ONI'] = np.zeros_like(len(oni))

for index, row in oni.iterrows():
    ANOM = row['ANOM']
    if ANOM <= -1.5:
        oni.at[index, 'ONI'] = 'Strong'
    elif -1.5 < ANOM <= -1.0:
        oni.at[index, 'ONI'] = 'Moderate'
    elif -1.0 < ANOM <= -0.5:
        oni.at[index, 'ONI'] = 'Weak'
    elif -0.5 < ANOM <= 0:
        oni.at[index, 'ONI'] = 'Neutral'
    else:
        oni.at[index, 'ONI'] = 'EN'

# Figure with both SOI and ONI index from 2001 - end of 2021

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(SMA_soi['1990-01-01':'2023-01-01']/7, color='blue', label = 'SOI')
ax.plot(-oni['ANOM']['1990-01-01':'2023-01-01']/0.5, label = '- ONI', color='red')
ax.set_ylabel('Normalised Index Magnitude')
ax.axhline(y=1, color='black', label = 'Threshold')
ax.axhline(y=0, color='black', label = 'Neutral', linestyle='dotted')
plt.title('SOI and ONI from 1990 - 2022')
# plt.grid(which = 'both', linestyle = '--', linewidth = 0.5)
plt.ylim(-5,5)
ax.legend(loc=2)
plt.minorticks_on()

# plt.savefig('SOI and ONI from 1990 - 2022', dpi=600)

#%% DataFrame for generation (wind and solar)
modelled = pd.DataFrame({'time':dt, 'E_wind':E_wind, 'E_solar': ac_results/1000000})
modelled['SOI'] = np.nan
modelled['ONI'] = np.nan
modelled.loc[modelled['SOI'] <= -1.5, 'SOI'] = 'Strong'
modelled.loc[(modelled['SOI'] > -1.5) & (modelled['SOI'] <= -1.0), 'La_Nina_SOI'] = 'Moderate'
modelled.loc[(modelled['SOI'] > -1.0) & (modelled['SOI'] <= -0.5), 'La_Nina_SOI'] = 'Weak'

modelled['time'] = pd.to_datetime(modelled['time'])
modelled.set_index('time', inplace=True)
modelled.index = modelled.index.tz_localize('UTC')
modelled.index = modelled.index.tz_convert(Australian)
#%% Deseasonalising
# Hourly Deseasonalising



# Monthly Deseasonalising
# Wind
daily_wind = modelled['E_wind'].resample('D').sum() # MWh
daily_wind = daily_wind.to_frame()
monthly_wind = daily_wind.resample('MS').sum()/1000 # GWh
monthly_wind.index = pd.to_datetime(monthly_wind.index)
monthly_wind = monthly_wind[:-1]
monthly_wind['year'] = monthly_wind.index.year

spring_wind_gen = monthly_wind[(monthly_wind.index.month >= 9) & (monthly_wind.index.month <= 11)]
winter_wind_gen = monthly_wind[(monthly_wind.index.month >= 6) & (monthly_wind.index.month <= 8)]
autumn_wind_gen = monthly_wind[(monthly_wind.index.month >= 3) & (monthly_wind.index.month <= 5)]
summer_wind_gen = monthly_wind[(monthly_wind.index.month <=2 ) | (monthly_wind.index.month == 12)]

spring_wind_avg = monthly_wind[(monthly_wind.index.month >= 9) & (monthly_wind.index.month <= 11)].mean()
winter_wind_avg = monthly_wind[(monthly_wind.index.month >= 6) & (monthly_wind.index.month <= 8)].mean()
autumn_wind_avg = monthly_wind[(monthly_wind.index.month >= 3) & (monthly_wind.index.month <= 5)].mean()
summer_wind_avg = monthly_wind[(monthly_wind.index.month <=2 ) | (monthly_wind.index.month == 12)].mean()

de_spring_wind = (spring_wind_gen - spring_wind_avg) / np.std(spring_wind_gen)
de_winter_wind = (winter_wind_gen - winter_wind_avg) / np.std(winter_wind_gen)
de_autumn_wind = (autumn_wind_gen - autumn_wind_avg) / np.std(autumn_wind_gen)
de_summer_wind = (summer_wind_gen - summer_wind_avg) / np.std(summer_wind_gen)

de_wind = pd.concat((de_spring_wind, de_winter_wind, de_autumn_wind, de_summer_wind))
de_wind = de_wind.drop(de_wind.columns[-1], axis=1)

de_wind = de_wind.sort_values('time')


# Solar
daily_solar = modelled['E_solar'].resample('D').sum() # MWh
daily_solar = daily_solar.to_frame()
monthly_solar = daily_solar.resample('MS').sum()/1000 # GWh
monthly_solar.index = pd.to_datetime(monthly_solar.index)
monthly_solar = monthly_solar[:-1]
monthly_solar['year'] = monthly_solar.index.year

spring_solar_gen = monthly_solar[(monthly_solar.index.month >= 9) & (monthly_solar.index.month <= 11)]
winter_solar_gen = monthly_solar[(monthly_solar.index.month >= 6) & (monthly_solar.index.month <= 8)]
autumn_solar_gen = monthly_solar[(monthly_solar.index.month >= 3) & (monthly_solar.index.month <= 5)]
summer_solar_gen = monthly_solar[(monthly_solar.index.month <=2 ) | (monthly_solar.index.month == 12)]

spring_solar_avg = monthly_solar[(monthly_solar.index.month >= 9) & (monthly_solar.index.month <= 11)].mean()
winter_solar_avg = monthly_solar[(monthly_solar.index.month >= 6) & (monthly_solar.index.month <= 8)].mean()
autumn_solar_avg = monthly_solar[(monthly_solar.index.month >= 3) & (monthly_solar.index.month <= 5)].mean()
summer_solar_avg = monthly_solar[(monthly_solar.index.month <=2 ) | (monthly_solar.index.month == 12)].mean()

de_spring_solar = (spring_solar_gen - spring_solar_avg) / np.std(spring_solar_gen)
de_winter_solar = (winter_solar_gen - winter_solar_avg) / np.std(winter_solar_gen)
de_autumn_solar = (autumn_solar_gen - autumn_solar_avg) / np.std(autumn_solar_gen)
de_summer_solar = (summer_solar_gen - summer_solar_avg) / np.std(summer_solar_gen)

de_solar = pd.concat((de_spring_solar, de_winter_solar, de_autumn_solar, de_summer_solar))
de_solar = de_solar.drop(de_solar.columns[-1], axis=1)

de_solar = de_solar.sort_values('time')



#%% Trying to compare generation in LN years to non LN, couldnt get this to work but unsure why index lengths are different
start_date = datetime.datetime(1990, 1, 1)
end_date = datetime.datetime(2022, 12, 31, 23, 0)

dt_monthly = pd.date_range(start_date, end_date, freq = 'MS')
dt_monthly = pd.DataFrame({'time': dt_monthly})
dt_monthly.set_index('time', inplace=True)
dt_monthly.index = dt_monthly.index.tz_localize('UTC')
dt_monthly.index = dt_monthly.index.tz_convert(Australian)

monthly_modelled = pd.DataFrame({'time': dt_monthly.index,
                                 'E_wind': monthly_wind['E_wind'],
                                 'E_solar': monthly_solar['E_solar'],
                                 'de_wind': de_wind['E_wind'],
                                 'de_solar': de_solar['E_solar'],
                                 'soi_strength': soi['SOI'],
                                 'oni_strength': oni['ONI'],
                                 'SOI': soi['ANOM'],
                                 'ONI': oni['ANOM']})

monthly_modelled['Year'] = pd.to_datetime(monthly_modelled['time']).dt.year
monthly_modelled['Month'] = pd.to_datetime(monthly_modelled['time']).dt.month
# print(f"The SOI index counts {(monthly_modelled['soi_strength'] == 'Neutral').sum().sum()} Neutral months.")
# print(f"The ONI index counts {(monthly_modelled['oni_strength'] == 'Neutral').sum().sum()} Neutral months.")

# print(f"The SOI index counts {(monthly_modelled['soi_strength'] == 'Weak').sum().sum()} Weak LN months.")
# print(f"The ONI index counts {(monthly_modelled['oni_strength'] == 'Weak').sum().sum()} Weak LN months.")

# print(f"The SOI index counts {(monthly_modelled['soi_strength'] == 'Moderate').sum().sum()} Moderate LN months.")
# print(f"The ONI index counts {(monthly_modelled['oni_strength'] == 'Moderate').sum().sum()} Moderate LN months.")

# print(f"The SOI index counts {(monthly_modelled['soi_strength'] == 'Strong').sum().sum()} Strong LN months.")
# print(f"The ONI index counts {(monthly_modelled['oni_strength'] == 'Strong').sum().sum()} Strong LN months.")




#%% Initial sorting of years based on LN Strength

strong_soi = monthly_modelled[monthly_modelled['soi_strength'] == 'Strong']['Year']
strong_oni = monthly_modelled[monthly_modelled['oni_strength'] == 'Strong']['Year']
strong_soi_years = strong_soi.unique().tolist()
strong_oni_years = strong_oni.unique().tolist()

moderate_soi = monthly_modelled[(monthly_modelled['soi_strength']=='Moderate') &
                                (~monthly_modelled['Year'].isin(strong_soi))]['Year']

moderate_oni = monthly_modelled[(monthly_modelled['oni_strength']=='Moderate') &
                                (~monthly_modelled['Year'].isin(strong_oni))]['Year']

moderate_soi_years = moderate_soi.unique().tolist()
moderate_oni_years = moderate_oni.unique().tolist()

weak_soi = monthly_modelled[(monthly_modelled['soi_strength'] == 'Weak') &
                            (~monthly_modelled['Year'].isin(strong_soi)) &
                            (~monthly_modelled['Year'].isin(moderate_soi))]['Year']


weak_oni = monthly_modelled[(monthly_modelled['oni_strength'] == 'Weak') &
                            (~monthly_modelled['Year'].isin(strong_oni)) &
                            (~monthly_modelled['Year'].isin(moderate_oni))]['Year']

weak_soi_years = weak_soi.unique().tolist()
weak_oni_years = weak_oni.unique().tolist()



nino_soi = monthly_modelled[(monthly_modelled['soi_strength'] == 'EN') &
                               (~monthly_modelled['Year'].isin(strong_soi)) &
                               (~monthly_modelled['Year'].isin(moderate_soi)) &
                               (~monthly_modelled['Year'].isin(weak_soi))]['Year']

nino_oni = monthly_modelled[(monthly_modelled['oni_strength'] == 'EN') &
                               (~monthly_modelled['Year'].isin(strong_oni)) &
                               (~monthly_modelled['Year'].isin(moderate_oni)) &
                               (~monthly_modelled['Year'].isin(weak_oni))]['Year']

nino_soi_years = nino_soi.unique().tolist()
nino_oni_years = nino_oni.unique().tolist()


neutral_soi = monthly_modelled[(monthly_modelled['soi_strength'] == 'Neutral') &
                                (~monthly_modelled['Year'].isin(strong_soi)) &
                                (~monthly_modelled['Year'].isin(moderate_soi)) &
                                (~monthly_modelled['Year'].isin(weak_soi))]['Year']

#   &
# (~monthly_modelled['Year'].isin(nino_soi))

neutral_oni = monthly_modelled[(monthly_modelled['oni_strength'] == 'Neutral') &
                                (~monthly_modelled['Year'].isin(strong_oni)) &
                                (~monthly_modelled['Year'].isin(moderate_oni)) &
                                (~monthly_modelled['Year'].isin(weak_oni))]['Year']
#  &
# (~monthly_modelled['Year'].isin(nino_oni))

neutral_soi_years = neutral_soi.unique().tolist()
neutral_oni_years = neutral_oni.unique().tolist()



#%% Second method for sorting of years based on LN Strength (this one seems to reflect the ONI years referenced in a paper)

all_years = monthly_modelled['Year'].unique()

soi_strength_counts = monthly_modelled.groupby('Year')['soi_strength'].value_counts().unstack(fill_value=0)

oni_strength_counts = monthly_modelled.groupby('Year')['oni_strength'].value_counts().unstack(fill_value=0)

strong_threshold = 1
moderate_threshold = 1
weak_neutral_threshold = 1
weak_strong_threshold = 3
neutral_en_threshold = 5

strong_soi_years = []
moderate_soi_years = []
weak_soi_years = []
neutral_soi_years = []
neutral_soi_years = []

strong_oni_years = []
moderate_oni_years = []
weak_oni_years = []
neutral_oni_years = []

for year in all_years:
    if year in soi_strength_counts.index:
        row = soi_strength_counts.loc[year]
        strong_count = row['Strong']
        moderate_count = row['Moderate']
        weak_count = row['Weak']
        neutral_count = row['Neutral']
        en_count = row['EN']

        if strong_count >= strong_threshold:
            strong_soi_years.append(year)
        elif strong_count == 0 and moderate_count >= moderate_threshold and en_count <= moderate_threshold:
            moderate_soi_years.append(year)
        elif (weak_count >= neutral_count and moderate_count == 1) or weak_count >= weak_strong_threshold:
            weak_soi_years.append(year)
        elif (weak_count >= neutral_count and en_count >= neutral_en_threshold) or neutral_count >= neutral_en_threshold:
            neutral_soi_years.append(year)

for year in all_years:
    if year in oni_strength_counts.index:
        row = oni_strength_counts.loc[year]
        strong_count = row['Strong']
        moderate_count = row['Moderate']
        weak_count = row['Weak']
        neutral_count = row['Neutral']
        en_count = row['EN']

        if strong_count >= strong_threshold:
            strong_oni_years.append(year)
        elif strong_count == 0 and moderate_count >= moderate_threshold and en_count <= moderate_threshold:
            moderate_oni_years.append(year)
        elif (weak_count >= neutral_count and moderate_count == 1) or weak_count >= weak_strong_threshold:
            weak_oni_years.append(year)
        elif (weak_count >= neutral_count and en_count >= neutral_en_threshold) or neutral_count >= neutral_en_threshold:
            neutral_oni_years.append(year)

print("SOI Strengths:")
print("Strong Years:", strong_soi_years)
print("Moderate Years:", moderate_soi_years)
print("Weak Years:", weak_soi_years)
print("Neutral Years:", neutral_soi_years)

print("ONI Strengths:")
print("Strong Years:", strong_oni_years)
print("Moderate Years:", moderate_oni_years)
print("Weak Years:", weak_oni_years)
print("Neutral Years:", neutral_oni_years)

#%%
# Was going to do something similar here to the annual plot for varying strengths, but for daily averages across the year just to have more data points.

hourly_modelled = pd.DataFrame({'time': dt,
                                'E_wind': modelled['E_wind'],
                                'E_solar': modelled['E_solar']})

soi_strength_mapping = {}
oni_strength_mapping = {}

# Assign strengths to years in the dictionary
for year in strong_soi_years:
    soi_strength_mapping[year] = 'Strong'
for year in moderate_soi_years:
    soi_strength_mapping[year] = 'Moderate'
for year in weak_soi_years:
    soi_strength_mapping[year] = 'Weak'
for year in neutral_soi_years:
    soi_strength_mapping[year] = 'Neutral'
for year in nino_soi_years:
    soi_strength_mapping[year] = 'EN'
    
for year in strong_oni_years:
    oni_strength_mapping[year] = 'Strong'
for year in moderate_oni_years:
    oni_strength_mapping[year] = 'Moderate'
for year in weak_oni_years:
    oni_strength_mapping[year] = 'Weak'
for year in moderate_oni_years:
    oni_strength_mapping[year] = 'Neutral'
for year in nino_oni_years:
    oni_strength_mapping[year] = 'EN'
    

hourly_modelled['Year'] = pd.to_datetime(hourly_modelled['time']).dt.year
hourly_modelled['Month'] = pd.to_datetime(hourly_modelled['time']).dt.month

# Convert "time" column to datetime type
hourly_modelled['time'] = pd.to_datetime(hourly_modelled['time'])

# Function to map month values to seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

# Add "Season" column to the dataframe
hourly_modelled['Season'] = hourly_modelled['time'].dt.month.map(get_season)


hourly_modelled['soi_strength'] = hourly_modelled['Year'].map(soi_strength_mapping)
hourly_modelled['oni_strength'] = hourly_modelled['Year'].map(oni_strength_mapping)


seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
strengths = ['Strong', 'Moderate', 'Weak', 'Neutral']

soi_strength = {}

# Iterate over seasons and strengths
for season in seasons:
    for strength in strengths:
        # Filter the dataframe based on season and strength criteria
        filtered_data = hourly_modelled[(hourly_modelled['Season'] == season) & 
                                        (hourly_modelled['soi_strength'] == strength)]
        # Store the filtered dataframe in the dictionary
        key = f"{season}_{strength}"
        soi_strength[key] = filtered_data

oni_strength = {}

# Iterate over seasons and strengths
for season in seasons:
    for strength in strengths:
        # Filter the dataframe based on season and strength criteria
        filtered_data = hourly_modelled[(hourly_modelled['Season'] == season) & 
                                        (hourly_modelled['oni_strength'] == strength)]
        # Store the filtered dataframe in the dictionary
        key = f"{season}_{strength}"
        oni_strength[key] = filtered_data

#%%



#%% Plot of monthly averages across varying LN strengths

# Solar SOI
avg_monthly_generation = {}

strength_levels = ['Strong', 'Moderate', 'Weak', 'Neutral']

strong_data = monthly_modelled[monthly_modelled['Year'].isin(strong_soi_years)]
strong_avg_gen = strong_data.groupby('Month')['E_solar'].mean()
avg_monthly_generation['Strong'] = strong_avg_gen

moderate_data = monthly_modelled[monthly_modelled['Year'].isin(moderate_soi_years)]
moderate_avg_gen = moderate_data.groupby('Month')['E_solar'].mean()
avg_monthly_generation['Moderate'] = moderate_avg_gen

weak_data = monthly_modelled[monthly_modelled['Year'].isin(weak_soi_years)]
weak_avg_gen = weak_data.groupby('Month')['E_solar'].mean()
avg_monthly_generation['Weak'] = weak_avg_gen

neutral_data = monthly_modelled[monthly_modelled['Year'].isin(neutral_soi_years)]
neutral_avg_gen = neutral_data.groupby('Month')['E_solar'].mean()
avg_monthly_generation['Neutral'] = neutral_avg_gen

plt.figure(figsize=(8, 6))
for strength, avg_gen in avg_monthly_generation.items():
    month_names = [calendar.month_name[month] for month in avg_gen.index]
    plt.plot(month_names, avg_gen.values, label=strength)

plt.ylabel('Average Generation Output (GWh)')
plt.title('Monthly Average Solar Generation Output for Different SOI Strength Levels')
plt.legend(loc = 'lower right')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Solar ONI
avg_monthly_generation = {}

strong_data = monthly_modelled[monthly_modelled['Year'].isin(strong_oni_years)]
strong_avg_gen = strong_data.groupby('Month')['E_solar'].mean()
avg_monthly_generation['Strong'] = strong_avg_gen

moderate_data = monthly_modelled[monthly_modelled['Year'].isin(moderate_oni_years)]
moderate_avg_gen = moderate_data.groupby('Month')['E_solar'].mean()
avg_monthly_generation['Moderate'] = moderate_avg_gen

weak_data = monthly_modelled[monthly_modelled['Year'].isin(weak_oni_years)]
weak_avg_gen = weak_data.groupby('Month')['E_solar'].mean()
avg_monthly_generation['Weak'] = weak_avg_gen

neutral_data = monthly_modelled[monthly_modelled['Year'].isin(neutral_oni_years)]
neutral_avg_gen = neutral_data.groupby('Month')['E_solar'].mean()
avg_monthly_generation['Neutral'] = neutral_avg_gen

plt.figure(figsize=(8, 6))
for strength, avg_gen in avg_monthly_generation.items():
    month_names = [calendar.month_name[month] for month in avg_gen.index]
    plt.plot(month_names, avg_gen.values, label=strength)

plt.ylabel('Average Generation Output (GWh)')
plt.title('Monthly Average Solar Generation Output for Different ONI Strength Levels')
plt.legend(loc = 'lower right')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Wind SOI
avg_monthly_generation = {}

strength_levels = ['Strong', 'Moderate', 'Weak', 'Neutral']

strong_data = monthly_modelled[monthly_modelled['Year'].isin(strong_soi_years)]
strong_avg_gen = strong_data.groupby('Month')['E_wind'].mean()
avg_monthly_generation['Strong'] = strong_avg_gen

moderate_data = monthly_modelled[monthly_modelled['Year'].isin(moderate_soi_years)]
moderate_avg_gen = moderate_data.groupby('Month')['E_wind'].mean()
avg_monthly_generation['Moderate'] = moderate_avg_gen

weak_data = monthly_modelled[monthly_modelled['Year'].isin(weak_soi_years)]
weak_avg_gen = weak_data.groupby('Month')['E_wind'].mean()
avg_monthly_generation['Weak'] = weak_avg_gen

neutral_data = monthly_modelled[monthly_modelled['Year'].isin(neutral_soi_years)]
neutral_avg_gen = neutral_data.groupby('Month')['E_wind'].mean()
avg_monthly_generation['Neutral'] = neutral_avg_gen

plt.figure(figsize=(8, 6))
for strength, avg_gen in avg_monthly_generation.items():
    month_names = [calendar.month_name[month] for month in avg_gen.index]
    plt.plot(month_names, avg_gen.values, label=strength)

plt.ylabel('Average Generation Output (GWh)')
plt.title('Monthly Average Wind Generation Output for Different SOI Strength Levels')
plt.legend(loc = 'lower right')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Solar ONI
avg_monthly_generation = {}

strong_data = monthly_modelled[monthly_modelled['Year'].isin(strong_oni_years)]
strong_avg_gen = strong_data.groupby('Month')['E_wind'].mean()
avg_monthly_generation['Strong'] = strong_avg_gen

moderate_data = monthly_modelled[monthly_modelled['Year'].isin(moderate_oni_years)]
moderate_avg_gen = moderate_data.groupby('Month')['E_wind'].mean()
avg_monthly_generation['Moderate'] = moderate_avg_gen

weak_data = monthly_modelled[monthly_modelled['Year'].isin(weak_oni_years)]
weak_avg_gen = weak_data.groupby('Month')['E_wind'].mean()
avg_monthly_generation['Weak'] = weak_avg_gen

neutral_data = monthly_modelled[monthly_modelled['Year'].isin(neutral_oni_years)]
neutral_avg_gen = neutral_data.groupby('Month')['E_wind'].mean()
avg_monthly_generation['Neutral'] = neutral_avg_gen

plt.figure(figsize=(8, 6))
for strength, avg_gen in avg_monthly_generation.items():
    month_names = [calendar.month_name[month] for month in avg_gen.index]
    plt.plot(month_names, avg_gen.values, label=strength)

plt.ylabel('Average Generation Output (GWh)')
plt.title('Monthly Average Wind Generation Output for Different ONI Strength Levels')
plt.legend(loc = 'lower right')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

#%% Plot of monthly averages in strong LN years and neutral years for both SOI and ONI
# Solar
avg_monthly_generation_soi = {}
avg_monthly_generation_oni = {}

strength_levels = ['Strong', 'Neutral']

plt.figure(figsize=(9, 6))
month_names = [calendar.month_name[month] for month in range(1, 13)]
bar_width = 0.15
opacity = 0.8
index = np.arange(len(month_names))

colors_soi = ['b', 'c']  # Colors for SOI bars
colors_oni = ['r', 'orange']  # Colors for ONI bars

for i, strength in enumerate(strength_levels):
    data_soi = monthly_modelled[monthly_modelled['Year'].isin(eval(f'{strength.lower()}_soi_years'))]
    avg_gen_soi = data_soi.groupby('Month')['E_solar'].mean()
    avg_monthly_generation_soi[strength] = avg_gen_soi

    data_oni = monthly_modelled[monthly_modelled['Year'].isin(eval(f'{strength.lower()}_oni_years'))]
    avg_gen_oni = data_oni.groupby('Month')['E_solar'].mean()
    avg_monthly_generation_oni[strength] = avg_gen_oni

    plt.bar(index + i * bar_width, avg_gen_soi.values, bar_width,
            alpha=opacity, label=f'{strength} (SOI)', align='center',
            color=colors_soi[i])

    plt.bar(index + i * bar_width + 2 * bar_width, avg_gen_oni.values, bar_width,
            alpha=opacity, label=f'{strength} (ONI)', align='center',
            color=colors_oni[i])

plt.ylabel('Average Generation Output (GWh)')
# plt.title('Monthly Average Solar Generation for Strong and Neutral La Nina Conditions (White Rock)')
plt.title('Average Monthly Solar Generation at White Rock for varying La Nina Conditions')
plt.xticks(index + bar_width/2, month_names, rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('Monthly Average Solar Generation (White Rock)', dpi=600)
plt.show()



# Wind
avg_monthly_generation_soi = {}
avg_monthly_generation_oni = {}

strength_levels = ['Strong', 'Neutral']

plt.figure(figsize=(9, 6))
month_names = [calendar.month_name[month] for month in range(1, 13)]
bar_width = 0.15
opacity = 0.8
index = np.arange(len(month_names))

colors_soi = ['b', 'c']  # Colors for SOI bars
colors_oni = ['r', 'orange']  # Colors for ONI bars

for i, strength in enumerate(strength_levels):
    data_soi = monthly_modelled[monthly_modelled['Year'].isin(eval(f'{strength.lower()}_soi_years'))]
    avg_gen_soi = data_soi.groupby('Month')['E_wind'].mean()
    avg_monthly_generation_soi[strength] = avg_gen_soi

    data_oni = monthly_modelled[monthly_modelled['Year'].isin(eval(f'{strength.lower()}_oni_years'))]
    avg_gen_oni = data_oni.groupby('Month')['E_wind'].mean()
    avg_monthly_generation_oni[strength] = avg_gen_oni

    plt.bar(index + i * bar_width, avg_gen_soi.values, bar_width,
            alpha=opacity, label=f'{strength} (SOI)', align='center',
            color=colors_soi[i])

    plt.bar(index + i * bar_width + 2 * bar_width, avg_gen_oni.values, bar_width,
            alpha=opacity, label=f'{strength} (ONI)', align='center',
            color=colors_oni[i])

plt.ylabel('Average Generation Output (GWh)')
# plt.title('Monthly Average Wind Generation for Strong and Neutral La Nina Conditions (White Rock)')
plt.title('Average Monthly Solar Generation at White Rock for varying La Nina Conditions')

plt.xticks(index + bar_width/2, month_names, rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('Monthly Average Wind Generation (White Rock).png', dpi=600)
plt.show()




#%% Sum of solar and wind and varying LN strengths




#%% Analysis (LN Index vs Deseasonalised Generation)
x_soi = soi['1990-01-01':'2022-12-01']['ANOM']
x_oni = - oni['1990-01-01':'2022-12-01']['ANOM']
norm_x_soi = SMA_soi['1990-01-01':'2022-12-01']/7
norm_x_oni = -oni['ANOM']['1990-01-01':'2022-12-01']/0.5

# Wind
plt.figure(figsize=(12,9))
plt.scatter(norm_x_soi, de_wind, label = 'Wind SOI')
plt.scatter(norm_x_oni, de_wind, label = 'Wind ONI')
plt.xlim(0,4)
plt.ylim(-4,4)
plt.title('Wind')
plt.legend()
plt.show()

# Solar
plt.figure(figsize=(12,9))
plt.scatter(norm_x_soi, de_solar, label = 'Solar SOI')
plt.scatter(norm_x_oni, de_solar, label = 'Solar ONI')
plt.xlim(0,4)
plt.ylim(-4,4)
plt.title('Solar')
plt.legend()
plt.show()



#%%
# Strong LN
x_soi_strong = monthly_modelled[monthly_modelled['Year'].isin(strong_soi_years)]['SOI']
x_soi_moderate = monthly_modelled[monthly_modelled['Year'].isin(moderate_soi_years)]['SOI']
x_soi_weak = monthly_modelled[monthly_modelled['Year'].isin(weak_soi_years)]['SOI']
x_soi_neutral = monthly_modelled[monthly_modelled['Year'].isin(neutral_soi_years)]['SOI']

y_solar_strong_soi = monthly_modelled[monthly_modelled['Year'].isin(strong_soi_years)]['de_solar']
y_wind_strong_soi = monthly_modelled[monthly_modelled['Year'].isin(strong_soi_years)]['de_wind']
y_solar_moderate_soi = monthly_modelled[monthly_modelled['Year'].isin(moderate_soi_years)]['de_solar']
y_wind_moderate_soi = monthly_modelled[monthly_modelled['Year'].isin(moderate_soi_years)]['de_wind']
y_solar_weak_soi = monthly_modelled[monthly_modelled['Year'].isin(weak_soi_years)]['de_solar']
y_wind_weak_soi = monthly_modelled[monthly_modelled['Year'].isin(weak_soi_years)]['de_wind']
y_solar_neutral_soi = monthly_modelled[monthly_modelled['Year'].isin(neutral_soi_years)]['de_solar']
y_wind_neutral_soi = monthly_modelled[monthly_modelled['Year'].isin(neutral_soi_years)]['de_wind']

x_oni_strong = - monthly_modelled[monthly_modelled['Year'].isin(strong_oni_years)]['ONI']
x_oni_moderate = - monthly_modelled[monthly_modelled['Year'].isin(moderate_oni_years)]['ONI']
x_oni_weak = - monthly_modelled[monthly_modelled['Year'].isin(weak_oni_years)]['ONI']
x_oni_neutral = - monthly_modelled[monthly_modelled['Year'].isin(neutral_oni_years)]['ONI']

y_solar_strong_oni = monthly_modelled[monthly_modelled['Year'].isin(strong_oni_years)]['de_solar']
y_wind_strong_oni = monthly_modelled[monthly_modelled['Year'].isin(strong_oni_years)]['de_wind']
y_solar_moderate_oni = monthly_modelled[monthly_modelled['Year'].isin(moderate_oni_years)]['de_solar']
y_wind_moderate_oni = monthly_modelled[monthly_modelled['Year'].isin(moderate_oni_years)]['de_wind']
y_solar_weak_oni = monthly_modelled[monthly_modelled['Year'].isin(weak_oni_years)]['de_solar']
y_wind_weak_oni = monthly_modelled[monthly_modelled['Year'].isin(weak_oni_years)]['de_wind']
y_solar_neutral_oni = monthly_modelled[monthly_modelled['Year'].isin(neutral_oni_years)]['de_solar']
y_wind_neutral_oni = monthly_modelled[monthly_modelled['Year'].isin(neutral_oni_years)]['de_wind']

# Solar
plt.figure(figsize=(8,6))
plt.scatter(x_soi_strong, y_solar_strong_soi, label = 'Strong LN')
plt.scatter(x_soi_moderate, y_solar_moderate_soi, label = 'Moderate LN')
plt.scatter(x_soi_weak, y_solar_weak_soi, label = 'Weak LN')
plt.scatter(x_soi_neutral, y_solar_neutral_soi, label = 'Neutral')
plt.xlim(-30,30)
plt.ylim(-4,4)
plt.xlabel('SOI index')
plt.ylabel('Generation Anomoly')
plt.legend()
plt.title('Solar Generation During LN Years - SOI')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(x_oni_strong, y_solar_strong_oni, label = 'Strong LN')
plt.scatter(x_oni_moderate, y_solar_moderate_oni, label = 'Moderate LN')
plt.scatter(x_oni_weak, y_solar_weak_oni, label = 'Weak LN')
plt.scatter(x_oni_neutral, y_solar_neutral_oni, label = 'Neutral')
plt.xlim(-2,2)
plt.ylim(-4,4)
plt.xlabel('ONI index')
plt.ylabel('Generation Anomoly')
plt.legend()
plt.title('Solar Generation During LN Years - ONI')
plt.show()

# Wind
plt.figure(figsize=(8,6))
plt.scatter(x_soi_strong, y_wind_strong_soi, label = 'Strong LN')
plt.scatter(x_soi_moderate, y_wind_moderate_soi, label = 'Moderate LN')
plt.scatter(x_soi_weak, y_wind_weak_soi, label = 'Weak LN')
plt.scatter(x_soi_neutral, y_wind_neutral_soi, label = 'Neutral')
plt.xlim(-30,30)
plt.ylim(-4,4)
plt.xlabel('SOI index')
plt.ylabel('Generation Anomoly')
plt.legend()
plt.title('Wind Generation During LN Years - SOI')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(x_oni_strong, y_wind_strong_oni, label = 'Strong LN')
plt.scatter(x_oni_moderate, y_wind_moderate_oni, label = 'Moderate LN')
plt.scatter(x_oni_weak, y_wind_weak_oni, label = 'Weak LN')
plt.scatter(x_oni_neutral, y_wind_neutral_oni, label = 'Neutral')
plt.xlim(-2,2)
plt.ylim(-4,4)
plt.xlabel('ONI index')
plt.ylabel('Generation Anomoly')
plt.legend()
plt.title('Wind Generation During LN Years - ONI')
plt.show()

# Strong LN vs Neutral
plt.figure(figsize=(8,6))
plt.scatter(x_soi_strong, y_solar_strong_soi, label = 'Strong LN')
plt.scatter(x_soi_neutral, y_solar_neutral_soi, label = 'Neutral')
plt.xlim(-30,30)
plt.ylim(-4,4)
plt.xlabel('SOI index')
plt.ylabel('Generation Anomoly')
plt.legend()
plt.title('Solar Generation During Strong LN Relative to Neutral - SOI')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(x_oni_strong, y_solar_strong_oni, label = 'Strong LN')
plt.scatter(x_oni_neutral, y_solar_neutral_oni, label = 'Neutral')
plt.xlim(-2,2)
plt.ylim(-4,4)
plt.xlabel('ONI index')
plt.ylabel('Generation Anomoly')
plt.legend()
plt.title('Solar Generation During Strong LN Relative to Neutral - ONI')
plt.show()

# SOI vs ONI
plt.figure(figsize=(8,6))
plt.scatter(x_soi_strong/7, y_solar_strong_soi, label = 'Strong LN - SOI - Solar')
plt.scatter(x_oni_strong/0.5, y_solar_strong_oni, label = 'Strong LN - ONI - Solar')
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.xlabel('Normalised SOI and ONI index')
plt.ylabel('Generation Anomoly')
plt.legend()
plt.title('SOI vs ONI - Solar')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(x_soi_strong/7, y_wind_strong_soi, label = 'Strong LN - SOI - Wind')
plt.scatter(x_oni_strong/0.5, y_wind_strong_oni, label = 'Strong LN - ONI - Wind')
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.xlabel('Normalised SOI and ONI index')
plt.ylabel('Generation Anomoly')
plt.legend()
plt.title('SOI vs ONI - Wind')
plt.show()

# # x_soi_strong = soi['ANOM'].loc[strong_soi_years]
# x_soi_strong = soi['ANOM'][soi.index.isin([strong_soi_years])]
# x_oni = - oni['1990-01-01':'2022-12-01']['ANOM']
# norm_x_soi = SMA_soi['1990-01-01':'2022-12-01']/7
# norm_x_oni = -oni['ANOM']['1990-01-01':'2022-12-01']/0.5

# # Wind
# plt.figure(figsize=(12,9))
# plt.scatter(norm_x_soi, de_wind, label = 'Wind SOI')
# plt.scatter(norm_x_oni, de_wind, label = 'Wind ONI')
# plt.xlim(0,4)
# plt.ylim(-4,4)
# plt.title('Wind')
# plt.legend()
# plt.show()

# # Solar
# plt.figure(figsize=(12,9))
# plt.scatter(norm_x_soi, de_solar, label = 'Solar SOI')
# plt.scatter(norm_x_oni, de_solar, label = 'Solar ONI')
# plt.xlim(0,4)
# plt.ylim(-4,4)
# plt.title('Solar')
# plt.legend()
# plt.show()

#%% 
# Strong LN vs Neutral
plt.figure(figsize=(9,7))
plt.scatter(x_soi_strong/7, y_solar_strong_soi, label = 'Strong LN (SOI)', color='b', marker='o')
plt.scatter(x_soi_neutral/7, y_solar_neutral_soi, label = 'Neutral (SOI)', color='b', marker='x')
plt.scatter(x_oni_strong/0.5, y_solar_strong_oni, label = 'Strong LN (ONI)', color='r', marker='o')
plt.scatter(x_oni_neutral/0.5, y_solar_neutral_oni, label = 'Neutral (ONI)', color='r', marker='x')
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.axhline(0, color='black')
plt.xlabel('Normalised Indicator Value')
plt.ylabel('Generation Anomoly')
plt.legend()
plt.title('Solar Generation During Strong LN Relative to Neutral')
plt.show()

plt.figure(figsize=(9,7))
plt.scatter(x_soi_strong/7, y_wind_strong_soi, label = 'Strong LN (SOI)', color='b', marker='o')
plt.scatter(x_soi_neutral/7, y_wind_neutral_soi, label = 'Neutral (SOI)', color='b', marker='x')
plt.scatter(x_oni_strong/0.5, y_wind_strong_oni, label = 'Strong LN (ONI)', color='r', marker='o')
plt.scatter(x_oni_neutral/0.5, y_wind_neutral_oni, label = 'Neutral (ONI)', color='r', marker='x')
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.axhline(0, color='black')
plt.xlabel('Normalised Indicator Value')
plt.ylabel('Wind Generation Anomoly')
plt.legend()
plt.title('Wind Generation During Strong LN Relative to Neutral')
plt.show()

#%% Plots of solar anomoly against SOI and ONI index

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Solar Generation - SOI
axes[0].scatter(x_soi_strong, y_solar_strong_soi, label='Strong LN (SOI)', color='steelblue', marker='o')
axes[0].scatter(x_soi_neutral, y_solar_neutral_soi, label='Neutral (SOI)', color='orange', marker='o')
axes[0].set_xlim(-7, 28)
axes[0].set_ylim(-5, 5)
axes[0].set_ylabel('Solar Generation Anomaly')
axes[0].set_xlabel('Southern Oscillation Index')
axes[0].set_title('SOI Classification for La Nina Years')
axes[0].legend()

# Solar Generation - ONI
axes[1].scatter(x_oni_strong, y_solar_strong_oni, label='Strong LN (ONI)', color='steelblue', marker='o')
axes[1].scatter(x_oni_neutral, y_solar_neutral_oni, label='Neutral (ONI)', color='orange', marker='o')
axes[1].set_xlim(-0.5, 2)
axes[1].set_ylim(-5, 5)
axes[1].set_xlabel('Oceanic Nino Index')
axes[1].set_title('ONI Classification for La Nina Years')
axes[1].legend()

fig.suptitle('Solar Generation Anomolies During Strong La Nina and Neutral Years', fontsize=14, fontweight='bold')
plt.legend()

plt.tight_layout()
# plt.savefig('Solar Anomoly vs Index for strong and neutral years (seminar)', dpi=600)
plt.show()

from scipy.stats import linregress

# Linear regression for Solar Generation - SOI
slope_solar_soi, intercept_solar_soi, r_value_solar_soi, p_value_solar_soi, std_err_solar_soi = linregress(x_soi_strong, y_solar_strong_soi)

# Linear regression for Solar Generation - ONI
slope_solar_oni, intercept_solar_oni, r_value_solar_oni, p_value_solar_oni, std_err_solar_oni = linregress(x_oni_strong, y_solar_strong_oni)

# Print the regression parameters
print("Solar Generation - SOI")
print("Slope:", slope_solar_soi)
print("Intercept:", intercept_solar_soi)
print("Correlation Coefficient:", r_value_solar_soi)
print("P-value:", p_value_solar_soi)
print("Standard Error:", std_err_solar_soi)
print()

print("Solar Generation - ONI")
print("Slope:", slope_solar_oni)
print("Intercept:", intercept_solar_oni)
print("Correlation Coefficient:", r_value_solar_oni)
print("P-value:", p_value_solar_oni)
print("Standard Error:", std_err_solar_oni)
print()



# # Create subplots
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# # Solar Generation - SOI
# axes[0].scatter(x_soi_strong/7, y_solar_strong_soi, label='Strong LN (SOI)', color='steelblue', marker='o')
# axes[0].scatter(x_soi_neutral/7, y_solar_neutral_soi, label='Neutral (SOI)', color='orange', marker='o')
# axes[0].set_xlim(-5, 5)
# axes[0].set_ylim(-5, 5)
# axes[0].set_ylabel('Solar Generation Anomaly')
# axes[0].set_xlabel('Southern Oscillation Index')
# axes[0].set_title('SOI Classification for La Nina Years')
# axes[0].legend()

# # Solar Generation - ONI
# axes[1].scatter(x_oni_strong/0.5, y_solar_strong_oni, label='Strong LN (ONI)', color='steelblue', marker='o')
# axes[1].scatter(x_oni_neutral/0.5, y_solar_neutral_oni, label='Neutral (ONI)', color='orange', marker='o')
# axes[1].set_xlim(-5, 5)
# axes[1].set_ylim(-5, 5)
# axes[1].set_xlabel('Oceanic Nino Index')
# axes[1].set_title('ONI Classification for La Nina Years')

# fig.suptitle('Solar Generation Anomolies During Strong La Nina and Neutral Years', fontsize=14, fontweight='bold')
# plt.legend()

# plt.tight_layout()
# plt.savefig('Solar Anomoly vs Index for strong and neutral years (seminar)', dpi=600)
# plt.show()


# from scipy.stats import linregress

# # Linear regression for Solar Generation - SOI
# slope_solar_soi, intercept_solar_soi, r_value_solar_soi, p_value_solar_soi, std_err_solar_soi = linregress(x_soi_strong/7, y_solar_strong_soi)

# # Linear regression for Solar Generation - ONI
# slope_solar_oni, intercept_solar_oni, r_value_solar_oni, p_value_solar_oni, std_err_solar_oni = linregress(x_oni_strong/0.5, y_solar_strong_oni)

# # Print the regression parameters
# print("Solar Generation - SOI")
# print("Slope:", slope_solar_soi)
# print("Intercept:", intercept_solar_soi)
# print("Correlation Coefficient:", r_value_solar_soi)
# print("P-value:", p_value_solar_soi)
# print("Standard Error:", std_err_solar_soi)
# print()

# print("Solar Generation - ONI")
# print("Slope:", slope_solar_oni)
# print("Intercept:", intercept_solar_oni)
# print("Correlation Coefficient:", r_value_solar_oni)
# print("P-value:", p_value_solar_oni)
# print("Standard Error:", std_err_solar_oni)
# print()




#%% Seeing if I can find the average dirunal cycle of solar duirng varying strength LN
aaa = monthly_modelled[monthly_modelled['Year'].isin(neutral_soi_years)]



#%% Theil-Sen Regression (wind)
cc = np.corrcoef(de_wind['E_wind'], x_soi)[0,1]
estimators = [
    ("OLS", LinearRegression()),
    ("Theil-Sen", TheilSenRegressor(random_state=42)),
    ("RANSAC", RANSACRegressor(random_state=42)),
]
colors = {"OLS": "turquoise", "Theil-Sen": "gold", "RANSAC": "lightgreen"}
lw = 2

#%% PCA
#%% Composite Anlysis


# SOI PCA (wind)
scaler_soi_wind = StandardScaler()
scaled_wind = scaler_soi_wind.fit_transform(np.column_stack((de_wind['E_wind'], x_soi)))

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_wind)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Wind Energy Generation vs. SOI')
plt.show()

# ONI PCA (wind)
scaler_oni_wind = StandardScaler()
scaled_wind = scaler_oni_wind.fit_transform(np.column_stack((de_wind['E_wind'], x_oni)))

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_wind)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Wind Energy Generation vs. ONI')
plt.show()

# SOI PCA (solar)
scaler_soi_solar = StandardScaler()
scaled_solar = scaler_soi_solar.fit_transform(np.column_stack((de_solar['E_solar'], x_soi)))

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_solar)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Solar Energy Generation vs. SOI')
plt.show()

# ONI PCA (solar)
scaler_oni_solar = StandardScaler()
scaled_solar = scaler_oni_solar.fit_transform(np.column_stack((de_solar['E_solar'], x_oni)))

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_solar)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Solar Energy Generation vs. ONI')
plt.show()


#%% NEMOSIS
# Wind
# print(defaults.dynamic_tables)
# print(defaults.static_tables)

site = 'Kennedy Energy Park'
technology = 'Wind'
start_time = '2022/12/02 00:00:00'
end_time = '2022/12/03 00:00:00'
table_1 = 'Generators and Scheduled Loads'
table_2 = 'DISPATCH_UNIT_SCADA'
table_3 = 'BIDPEROFFER_D'
raw_data_cache = 'C:/Users/liame/OneDrive/Uni/2023/T2 2023/Thesis B/GUI'

site_info = static_table(table_1, raw_data_cache)
duid_wind = site_info.loc[site_info['Station Name'] == site, 'DUID'].values[1]
print(defaults.table_columns['DISPATCH_UNIT_SCADA'])

wind_gen_data = dynamic_data_compiler(start_time, end_time, table_2, raw_data_cache,
                                 keep_csv=False, filter_cols = ['DUID'],
                                 filter_values = ([duid_wind],))

curtailment = dynamic_data_compiler(start_time, end_time, table_3, raw_data_cache,
                                 keep_csv=False, filter_cols = ['DUID'],
                                 filter_values = ([duid_wind],))

plt.figure(figsize=(16,9))
plt.plot(wind_gen_data['SETTLEMENTDATE'], wind_gen_data['SCADAVALUE'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%b %H:%M'))
plt.xticks()
plt.title('Wind')

# Solar
# print(defaults.dynamic_tables)
# print(defaults.static_tables)

technology = 'Solar'

duid_solar = site_info.loc[site_info['Station Name'] == site, 'DUID'].values[0]

solar_gen_data = dynamic_data_compiler(start_time, end_time, table_2, raw_data_cache,
                                  keep_csv=False, filter_cols = ['DUID'],
                                  filter_values = ([duid_solar],))


plt.figure(figsize=(16,9))
plt.plot(solar_gen_data['SETTLEMENTDATE'], solar_gen_data['SCADAVALUE'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%b %H:%M'))
plt.title('Solar')
plt.show()

plt.figure(figsize=(16,9))
plt.plot(wind_gen_data['SETTLEMENTDATE'], wind_gen_data['SCADAVALUE'])
plt.plot(solar_gen_data['SETTLEMENTDATE'], solar_gen_data['SCADAVALUE'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%b %H:%M'))
plt.title('Wind and Solar')
plt.show()

#%% Solar modelled vs NEMOSIS
plt.figure(figsize=(16,9))
plt.plot(solar_gen_data['SETTLEMENTDATE'], solar_gen_data['SCADAVALUE'])


#%% Solar modelled vs NEMOSIS
plt.figure(figsize=(16,9))
plt.plot(modelled['E_solar'].loc['2022-12-02'])

#%% Power Curve
step = 0.5
cut_in_wind_speed = np.arange(0, v_cut_in, step)
cubic_wind_speed = np.arange(v_cut_in, v_rated, step)
rated_wind_speed = np.arange(v_rated, v_cut_out, step)
cut_out_wind_speed = np.arange(v_cut_out, v_cut_out + 5*step, step)
cubic_pc = (a * cubic_wind_speed**3 - b * P_rated_wind)/1000000
cut_in_pc = np.zeros_like(cut_in_wind_speed)
rated_pc = np.zeros_like(rated_wind_speed) + P_rated_wind
cut_out_pc = np.zeros_like(cut_out_wind_speed)

fig, ax = plt.subplots(figsize=(9,6))
ax.stem(cut_in_wind_speed, cut_in_pc, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Below Cut-In')
ax.stem(cubic_wind_speed, cubic_pc, markerfmt = 'bo', linefmt = 'b', basefmt = 'none', label = 'Cubic Model')
ax.stem(rated_wind_speed, rated_pc, markerfmt = 'go', linefmt = 'g', basefmt = 'none', label = 'Rated')
ax.stem(cut_out_wind_speed, cut_out_pc, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Above Cut-Out')
ax.set_ylim(0,4)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Turbine Output (MW)')
ax.set_title('Modelled Power Curve for Vestas V136 3.6MW')
ax.legend()

# plt.savefig('KEP Power Curve', dpi = 600)

#%%
plt.figure(figsize=(16,9))

plt.plot(modelled['E_solar'].loc['2022-12-02'])


