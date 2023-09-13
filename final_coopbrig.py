# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:55:36 2023

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
scoopbrig1 = xr.open_dataset('scoopbrig1.grib', engine = 'cfgrib', chunks = 1000)
scoopbrig2 = xr.open_dataset('scoopbrig1.grib', engine = 'cfgrib', chunks = 1000)
scoopbrig3 = xr.open_dataset('scoopbrig3.grib', engine = 'cfgrib', chunks = 1000)

coopbrig1 = xr.open_dataset('coopbrig1.grib', engine = 'cfgrib', chunks = 1000)
coopbrig2 = xr.open_dataset('coopbrig2.grib', engine = 'cfgrib', chunks = 1000)
coopbrig3 = xr.open_dataset('coopbrig3.grib', engine = 'cfgrib', chunks = 1000)
coopbrig4 = xr.open_dataset('coopbrig4.grib', engine = 'cfgrib', chunks = 1000)
coopbrig5 = xr.open_dataset('coopbrig5.grib', engine = 'cfgrib', chunks = 1000)
coopbrig6 = xr.open_dataset('coopbrig6.grib', engine = 'cfgrib', chunks = 1000)
coopbrig7 = xr.open_dataset('coopbrig7.grib', engine = 'cfgrib', chunks = 1000)
coopbrig8 = xr.open_dataset('coopbrig8.grib', engine = 'cfgrib', chunks = 1000)
coopbrig9 = xr.open_dataset('coopbrig9.grib', engine = 'cfgrib', chunks = 1000)


#%% Creating numpy arrays of relevent parameters
coopbrig_u1 = coopbrig1.u10.values[:,0,0]
coopbrig_u2 = coopbrig2.u10.values[:,0,0]
coopbrig_u3 = coopbrig3.u10.values[:,0,0]
coopbrig_u4 = coopbrig4.u10.values[:,0,0]
coopbrig_u5 = coopbrig5.u10.values[:,0,0]
coopbrig_u6 = coopbrig6.u10.values[:,0,0]
coopbrig_u7 = coopbrig7.u10.values[:,0,0]
coopbrig_u8 = coopbrig8.u10.values[:,0,0]
coopbrig_u9 = coopbrig9.u10.values[:,0,0]

coopbrig_v1 = coopbrig1.v10.values[:,0,0]
coopbrig_v2 = coopbrig2.v10.values[:,0,0]
coopbrig_v3 = coopbrig3.v10.values[:,0,0]
coopbrig_v4 = coopbrig4.v10.values[:,0,0]
coopbrig_v5 = coopbrig5.v10.values[:,0,0]
coopbrig_v6 = coopbrig6.v10.values[:,0,0]
coopbrig_v7 = coopbrig7.v10.values[:,0,0]
coopbrig_v8 = coopbrig8.v10.values[:,0,0]
coopbrig_v9 = coopbrig9.v10.values[:,0,0]

coopbrig_t1 = coopbrig1.t2m.values[:,0,0]
coopbrig_t2 = coopbrig2.t2m.values[:,0,0]
coopbrig_t3 = coopbrig3.t2m.values[:,0,0]
coopbrig_t4 = coopbrig4.t2m.values[:,0,0]
coopbrig_t5 = coopbrig5.t2m.values[:,0,0]
coopbrig_t6 = coopbrig6.t2m.values[:,0,0]
coopbrig_t7 = coopbrig7.t2m.values[:,0,0]
coopbrig_t8 = coopbrig8.t2m.values[:,0,0]
coopbrig_t9 = coopbrig9.t2m.values[:,0,0]

nans_coopbrig_s1 = scoopbrig1.ssr.values
nans_coopbrig_s2 = scoopbrig2.ssr.values
nans_coopbrig_s3 = scoopbrig3.ssr.values

coopbrig_s1 = nans_coopbrig_s1[~np.isnan(nans_coopbrig_s1)]
coopbrig_s2 = nans_coopbrig_s2[~np.isnan(nans_coopbrig_s2)]
coopbrig_s3 = nans_coopbrig_s3[~np.isnan(nans_coopbrig_s3)]

u = np.concatenate([coopbrig_u1, coopbrig_u2, coopbrig_u3, coopbrig_u4, coopbrig_u5, coopbrig_u6, coopbrig_u7,
                    coopbrig_u8, coopbrig_u9])

v = np.concatenate([coopbrig_v1, coopbrig_v2, coopbrig_v3, coopbrig_v4, coopbrig_v5, coopbrig_v6, coopbrig_v7,
                    coopbrig_v8, coopbrig_v9])

t = np.concatenate([coopbrig_t1, coopbrig_t2, coopbrig_t3, coopbrig_t4, coopbrig_t5, coopbrig_t6, coopbrig_t7,
                    coopbrig_t8, coopbrig_t9]) - 273.15

s = np.concatenate([coopbrig_s1, coopbrig_s2, coopbrig_s3]) / 3600


#%% Variables

# General

Australian = pytz.timezone('Australia/Queensland')
latitude = -27.71
longitude = 151.55

# Wind
# (GE3.6-137 3.6 MW)
wind_capacity_GE137 = 330      # MW
hub_height_GE137 = 110          # m
P_rated_wind_GE137 = 3.63       # MW
v_rated_GE137 = 12           # m/s
v_cut_in_GE137 = 3.0           # m/s
v_cut_out_GE137 = 25           # m/s

# (GE3.8-130 1.5 MW)
wind_capacity_GE130 = 122.5      # MW
hub_height_GE130 = 115           # m
P_rated_wind_GE130 = 3.83      # MW
v_rated_GE130 = 12.5            # m/s
v_cut_in_GE130 = 3.5            # m/s
v_cut_out_GE130 = 27            # m/s

# Solar
surface_tilt = 30
surface_azimuth = 0
modules_per_string= 30
strings_per_inverter = 250
number_of_inverters = 14

location = Location(latitude = latitude, longitude = longitude, tz = Australian,
                    altitude = 400, name = 'Brigalow Solar Farm')

celltype = 'monoSi'
pdc0 = 370
v_mp = 37.3
i_mp = 8.72
v_oc = 46.3
i_sc = 9.24
alpha_sc = 0.0005 * i_sc
beta_voc = -0.0032 * v_oc
gamma_pdc = -0.37
cells_in_series = 72
temp_ref = 25

#%% Wind Generation
number_of_turbines_GE137 = wind_capacity_GE137 // P_rated_wind_GE137
number_of_turbines_GE130 = wind_capacity_GE130 // P_rated_wind_GE130


# Wind Profile Calculation This only for region betwix cut in and rated wind speeds
# GE137
h1 = 10
h2_GE137 = hub_height_GE137
z0 = 0.05
v1 = np.sqrt(u**2 + v**2)
v2_GE137 = v1 * (math.log((h2_GE137)/(z0)))/(math.log((h1)/(z0)))

a_GE137 = P_rated_wind_GE137 * 1000000 / (v_rated_GE137**3 - v_cut_in_GE137**3)
b_GE137 = (v_rated_GE137**3) / (v_rated_GE137**3 - v_cut_in_GE137**3)

E_wind_GE137 = np.copy(v2_GE137)
E_wind_GE137[E_wind_GE137 < v_cut_in_GE137] = 0
E_wind_GE137[(E_wind_GE137 > v_cut_in_GE137) & (E_wind_GE137 < v_rated_GE137)] = (a_GE137 * E_wind_GE137[(E_wind_GE137 > v_cut_in_GE137) & (E_wind_GE137 < v_rated_GE137)]**3 - b_GE137 * P_rated_wind_GE137)/1000000
E_wind_GE137[(E_wind_GE137 > v_rated_GE137) & (E_wind_GE137 < v_cut_out_GE137)] = P_rated_wind_GE137
E_wind_GE137[E_wind_GE137 > v_cut_out_GE137] = 0

E_wind_GE137 *= number_of_turbines_GE137

# GE130
h2_GE130 = hub_height_GE130
v2_GE130 = v1 * (math.log((h2_GE130)/(z0)))/(math.log((h1)/(z0)))

a_GE130 = P_rated_wind_GE130 * 1000000 / (v_rated_GE130**3 - v_cut_in_GE130**3)
b_GE130 = (v_rated_GE130**3) / (v_rated_GE130**3 - v_cut_in_GE130**3)

E_wind_GE130 = np.copy(v2_GE137)
E_wind_GE130[E_wind_GE130 < v_cut_in_GE130] = 0
E_wind_GE130[(E_wind_GE130 > v_cut_in_GE130) & (E_wind_GE130 < v_rated_GE130)] = (a_GE130 * E_wind_GE130[(E_wind_GE130 > v_cut_in_GE130) & (E_wind_GE130 < v_rated_GE130)]**3 - b_GE130 * P_rated_wind_GE130)/1000000
E_wind_GE130[(E_wind_GE130 > v_rated_GE130) & (E_wind_GE130 < v_cut_out_GE130)] = P_rated_wind_GE130
E_wind_GE130[E_wind_GE130 > v_cut_out_GE130] = 0

E_wind_GE130 *= number_of_turbines_GE130
E_wind = E_wind_GE137 + E_wind_GE130

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
# GE137
step = 0.5
cut_in_wind_speed_GE137 = np.arange(0, v_cut_in_GE137, step)
cubic_wind_speed_GE137 = np.arange(v_cut_in_GE137, v_rated_GE137, step)
rated_wind_speed_GE137 = np.arange(v_rated_GE137, v_cut_out_GE137, step)
cut_out_wind_speed_GE137 = np.arange(v_cut_out_GE137, v_cut_out_GE137 + 5*step, step)

cubic_pc_GE137 = (a_GE137 * cubic_wind_speed_GE137**3 - b_GE137 * P_rated_wind_GE137)/1000000
cut_in_pc_GE137 = np.zeros_like(cut_in_wind_speed_GE137)
rated_pc_GE137 = np.zeros_like(rated_wind_speed_GE137) + P_rated_wind_GE137
cut_out_pc_GE137 = np.zeros_like(cut_out_wind_speed_GE137)

fig, ax = plt.subplots(figsize=(9,6))
ax.stem(cut_in_wind_speed_GE137, cut_in_pc_GE137, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Below Cut-In')
ax.stem(cubic_wind_speed_GE137, cubic_pc_GE137, markerfmt = 'bo', linefmt = 'b', basefmt = 'none', label = 'Cubic Model')
ax.stem(rated_wind_speed_GE137, rated_pc_GE137, markerfmt = 'go', linefmt = 'g', basefmt = 'none', label = 'Rated')
ax.stem(cut_out_wind_speed_GE137, cut_out_pc_GE137, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Above Cut-Out')
# ax.set_ylim(0,4)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Turbine Output (MW)')
ax.set_title('Modelled Power Curve for General Electric GE137 3.6 MW (Coopers Gap)')
ax.legend()
plt.savefig('Coopers Gap Turbine Power Curve (3_6 MW)', dpi = 600)

# GE130
step = 0.5
cut_in_wind_speed_GE130 = np.arange(0, v_cut_in_GE130, step)
cubic_wind_speed_GE130 = np.arange(v_cut_in_GE130, v_rated_GE130, step)
rated_wind_speed_GE130 = np.arange(v_rated_GE130, v_cut_out_GE130, step)
cut_out_wind_speed_GE130 = np.arange(v_cut_out_GE130, v_cut_out_GE130 + 5*step, step)

cubic_pc_GE130 = (a_GE130 * cubic_wind_speed_GE130**3 - b_GE130 * P_rated_wind_GE130)/1000000
cut_in_pc_GE130 = np.zeros_like(cut_in_wind_speed_GE130)
rated_pc_GE130 = np.zeros_like(rated_wind_speed_GE130) + P_rated_wind_GE130
cut_out_pc_GE130 = np.zeros_like(cut_out_wind_speed_GE130)

fig, ax = plt.subplots(figsize=(9,6))
ax.stem(cut_in_wind_speed_GE130, cut_in_pc_GE130, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Below Cut-In')
ax.stem(cubic_wind_speed_GE130, cubic_pc_GE130, markerfmt = 'bo', linefmt = 'b', basefmt = 'none', label = 'Cubic Model')
ax.stem(rated_wind_speed_GE130, rated_pc_GE130, markerfmt = 'go', linefmt = 'g', basefmt = 'none', label = 'Rated')
ax.stem(cut_out_wind_speed_GE130, cut_out_pc_GE130, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Above Cut-Out')
# ax.set_ylim(0,4)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Turbine Output (MW)')
ax.set_title('Modelled Power Curve for General Electric GE130 3.8 MW (Coopers Gap)')
ax.legend()

plt.savefig('Coopers Gap Turbine Power Curve (3_8 MW)', dpi = 600)

#%% pvlib


sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
cec_inverters = pvlib.pvsystem.retrieve_sam('CECInverter')


module = sandia_modules['Silevo_Triex_U300_Black__2014_']
inverter = cec_inverters['SMA_America__SC_2500_EV_US__550V_']


temperature_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']


system = PVSystem(modules_per_string = modules_per_string,
                  strings_per_inverter = strings_per_inverter)

# modelchain = ModelChain(system, location)

times = pd.date_range(start = '1990-01-01 11:00:00', end = '2023-01-01 09:00:00',
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
#boland

dni = boland['dni']
dhi = boland['dhi']

weather_data = pd.DataFrame({'time': dt, 'ghi':s, 'dhi':dhi, 'dni':dni,
                             'wind_speed':v1, 'temp_air':t})

weather_data['time'] = pd.to_datetime(weather_data['time'])
weather_data.set_index('time', inplace=True)
weather_data.index = weather_data.index.tz_localize('UTC')
weather_data.index = weather_data.index.tz_convert(Australian)

weather_data.to_csv('pvlib_agnew.csv')

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
#%% Wind speed histrogram at 10m
plt.figure(figsize=(9,6))
plt.hist(v1, bins=31, edgecolor='black')
plt.xlabel('Wind Speed at 10m (m/s)')
plt.ylabel('Frequency')
plt.title('Histogram of 10m Wind Speed at Coopers Gap Wind Farm')
plt.xlim(0,15)
plt.savefig('10m WindSpeed Histogram Coopers Gap', dpi = 600)

#%% Wind speed histrogram at 110m
plt.figure(figsize=(9,6))
plt.hist(v2_GE137, bins=31, edgecolor='black')
plt.xlabel('Wind Speed at 110m (m/s)')
plt.ylabel('Frequency')
plt.title('Histogram of Wind Speed at Hub Height for GE 3.6-137 (Coopers Gap Wind Farm)')
plt.xlim(0,15)
# plt.savefig('110m WindSpeed Histogram Coopers Gap', dpi = 600)

#%% Wind speed histrogram at 115m
plt.figure(figsize=(9,6))
plt.hist(v2_GE130, bins=31, edgecolor='black')
plt.xlabel('Wind Speed at 115m (m/s)')
plt.ylabel('Frequency')
plt.title('Histogram of Wind Speed at Hub Height for GE 3.8-130 (Coopers Gap Wind Farm)')
plt.xlim(0,15)
# plt.savefig('10m WindSpeed Histogram Coopers Gap', dpi = 600)


#%% GHI  histrogram (non-zero values)
plt.figure(figsize=(9,6))
plt.hist(weather_data['ghi'], bins=30, edgecolor='black')
plt.xlim(34,1000)
plt.ylim(0,20000)
plt.xlabel('Global Horizontal Irradiance (W/$m^2$)')
plt.ylabel('Frequency')
plt.title('Histogram of GHI at Brigalow Solar Farm')
plt.savefig('GHI Histogram Brigalow', dpi = 600)
plt.show()

#%% DHI  histrogram (non-zero values)
plt.figure(figsize=(9,6))
plt.hist(weather_data['dhi'], bins=30, edgecolor='black')
plt.xlim(17,1000)
plt.ylim(0,10000)
plt.xlabel('Diffuse Horizontal Irradiance (W/$m^2$)')
plt.ylabel('Frequency')
plt.title('Histogram of DHI at Brigalow Solar Farm')
plt.savefig('DHI Histogram Brigalow', dpi = 600)
plt.show()

#%% DNI  histrogram (non-zero values)
plt.figure(figsize=(9,6))
plt.hist(weather_data['dni'], bins=30, edgecolor='black')
plt.xlim(62,1500)
plt.ylim(0,20000)
plt.xlabel('Direct Normal Irradiance (W/$m^2$)')
plt.ylabel('Frequency')
plt.title('Histogram of DNI at Brigalow Solar Farm')
plt.savefig('DNI Histogram Brigalow', dpi = 600)
plt.show()


#%% GHI, DNI, DHI Plot

hours_day = pd.date_range(start='2000-01-01', periods=24, freq='h')
summer = (weather_data['ghi'].loc['2017-01-09']).values
autumn = (weather_data['ghi'].loc['2019-03-31']).values
winter = (weather_data['ghi'].loc['2005-07-31']).values
spring = (weather_data['ghi'].loc['2003-10-28']).values

diurnal_data = {
    'hours_day': pd.date_range(start='2000-01-01', periods=24, freq='h'),
    'summer': summer,
    'autumn': autumn,
    'winter': winter,
    'spring': spring
}
diurnal_weather_data = pd.DataFrame(diurnal_data)
diurnal_weather_data.set_index('hours_day', inplace=True)

summer_avg_idx = np.argmax(summer)
autumn_avg_idx = np.argmax(autumn)
winter_avg_idx = np.argmax(winter)
spring_avg_idx = np.argmax(spring)

midday_index = 12
summer_shift = midday_index - summer_avg_idx + 1
autumn_shift = midday_index - autumn_avg_idx
winter_shift = midday_index - winter_avg_idx + 1
spring_shift = midday_index - spring_avg_idx

shifted_summer = np.roll(summer, summer_shift)
shifted_autumn = np.roll(autumn, autumn_shift)
shifted_winter = np.roll(winter, winter_shift)
shifted_spring = np.roll(spring, spring_shift)

plt.figure(figsize=(10, 6))
plt.plot(diurnal_weather_data.index, shifted_summer, label='Summer')
plt.plot(diurnal_weather_data.index, shifted_spring, label='Spring')
plt.plot(diurnal_weather_data.index, shifted_autumn, label='Autumn')
plt.plot(diurnal_weather_data.index, shifted_winter, label='Winter')

plt.xlabel('Time')
plt.ylabel('Global Horizontal Irradiance (W/$m^2$)')
plt.title('Diurnal Cycle of GHI at Merredin Solar Farm')
plt.legend()

hours_formatter = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(hours_formatter)

# plt.savefig('Diurnal Cycle GHI - Merredin', dpi=600)

#%% Annual GHI in neutral and LN year
ghi_ln = weather_data['ghi'].loc['2022'].resample('m').mean().values
ghi_neutral = weather_data['ghi'].loc['2009'].resample('m').mean().values
ghi_en = weather_data['ghi'].loc['2010'].resample('m').mean().values

plt.figure(figsize=(10,6))
plt.plot(ghi_ln, label='La Nina')
plt.plot(ghi_neutral, label='Neutral')
plt.plot(ghi_en, label='El Nino')

plt.xlabel('Month')
plt.ylabel('Global Horizontal Irradiance (W/$m^2$)')
plt.title('Monthly mean GHI across varying ENSO conditions')
plt.legend(loc='lower right')

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
ax.plot(-oni['ANOM']['1990-01-01':'2023-01-01']/0.5, label =  r'$\mathbf{-}$ ONI', color='red')
ax.plot(SMA_soi['1990-01-01':'2023-01-01']/7, color='blue', label = 'SOI')
ax.set_ylabel('Normalised Index Magnitude')
ax.axhline(y=1, color='black', label = 'Threshold')
ax.axhline(y=0, color='black', label = 'Neutral', linestyle='dotted')
plt.title('SOI and ONI from 1990 - 2022')
# plt.grid(which = 'both', linestyle = '--', linewidth = 0.5)
plt.ylim(-5,5)
ax.legend(loc=2)
plt.minorticks_on()

plt.savefig('SOI and ONI from 1990 - 2022', dpi=600)

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


neutral_oni = monthly_modelled[(monthly_modelled['oni_strength'] == 'Neutral') &
                                (~monthly_modelled['Year'].isin(strong_oni)) &
                                (~monthly_modelled['Year'].isin(moderate_oni)) &
                                (~monthly_modelled['Year'].isin(weak_oni))]['Year']


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

#%% Was going to do something similar here to the annual plot for varying strengths, but for daily averages across the year just to have more data points.

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
plt.title('Average Monthly Solar Generation at Brigalow Solar Farm for varying La Nina Conditions')
plt.xticks(index + bar_width/2, month_names, rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('Monthly Average Solar Generation (Brigalow Solar Farm)', dpi=600)
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
plt.title('Average Monthly Wind Generation at Coopers Gap Wind Farm for varying La Nina Conditions')

plt.xticks(index + bar_width/2, month_names, rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('Monthly Average Wind Generation (Coopers Gap Wind Farm).png', dpi=600)
plt.show()




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
axes[1].scatter(x_oni_strong, y_solar_strong_oni, label='Strong LN (ONI)', color='blue', marker='o')
axes[1].scatter(x_oni_neutral, y_solar_neutral_oni, label='Neutral (ONI)', color='green', marker='o')
axes[1].set_xlim(-0.5, 2)
axes[1].set_ylim(-5, 5)
axes[1].set_xlabel('Oceanic Nino Index')
axes[1].set_title('ONI Classification for La Nina Years')
axes[1].legend()

fig.suptitle('Solar Generation Anomolies During Strong La Nina and Neutral Years (Agnew Gold Mine)', fontsize=14, fontweight='bold')
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




#%% Principal Component Analysis
date_range_soi = pd.date_range('1990-05-01', end_date, freq='M')
date_range_oni = pd.date_range(start_date, end_date, freq='M')

pca_soi = SMA_soi.dropna()
pca_oni = - x_oni # defintion for x_oni is for comparison to soi so take negative to actually reflect actual oni anomolies


#%% SOI PCA (wind)
scaler_soi_wind = StandardScaler()
scaled_wind = scaler_soi_wind.fit_transform(np.column_stack((de_wind['E_wind'].loc['1990-05':'2022-12'], pca_soi)))

pca = PCA(n_components=2)
principal_components_swind = pca.fit_transform(scaled_wind)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components_swind[:, 0], principal_components_swind[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Wind Energy Generation vs. SOI')
plt.show()

pca_swind = pd.DataFrame({'Date':date_range_soi})
pca_swind['SOI'] = pca_soi
pca_swind['pca1'] = principal_components_swind[:,0]
pca_swind['pca2'] = principal_components_swind[:,1]

correlation_coefficient = np.corrcoef(pca_soi, pca_swind['pca1'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_soi, pca_soi, label='SOI ANOM', color='blue')
ax1.set_ylabel('SOI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-40,40)

ax2 = ax1.twinx()

ax2.plot(date_range_soi, pca_swind['pca1'], label='PCA 1', color='green')
ax2.set_ylabel('PC 1', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-4,4)

# ax1.legend()
# ax2.legend()
plt.text(0.05, 0.95, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
# plt.text(0.05, 0.95, f'Correlation: {explained_variance_ratio:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
plt.title('SOI vs Wind PC1 (Coopers Gap Wind Farm)')
plt.savefig('Coopers Gap SOI PCA 1 (Wind).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)

# PC2
scaler_soi_wind = StandardScaler()
scaled_wind = scaler_soi_wind.fit_transform(np.column_stack((de_wind['E_wind'].loc['1990-05':'2022-12'], pca_soi)))

pca = PCA(n_components=2)
principal_components_swind = pca.fit_transform(scaled_wind)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components_swind[:, 0], principal_components_swind[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Wind Energy Generation vs. SOI')
plt.show()

pca_swind = pd.DataFrame({'Date':date_range_soi})
pca_swind['SOI'] = pca_soi
pca_swind['pca1'] = principal_components_swind[:,0]
pca_swind['pca2'] = principal_components_swind[:,1]

correlation_coefficient = np.corrcoef(pca_soi, pca_swind['pca2'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_soi, pca_soi, label='SOI ANOM', color='blue')
ax1.set_ylabel('SOI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-40,40)

ax2 = ax1.twinx()

ax2.plot(date_range_soi, pca_swind['pca2'], label='PCA 2', color='green')
ax2.set_ylabel('PC 2', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-4,4)

# ax1.legend()
# ax2.legend()
plt.text(0.05, 0.95, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
# plt.text(0.05, 0.95, f'Correlation: {explained_variance_ratio:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
plt.title('SOI vs Wind PC2 (Coopers Gap Wind Farm)')
plt.savefig('Coopers Gap SOI PCA 2 (Wind).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)
#%% ONI PCA (wind)

scaler_oni_wind = StandardScaler()
scaled_wind = scaler_oni_wind.fit_transform(np.column_stack((de_wind['E_wind'], pca_oni)))

pca = PCA(n_components=2)
principal_components_owind = pca.fit_transform(scaled_wind)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components_owind[:, 0], principal_components_owind[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Wind Energy Generation vs. ONI')
plt.show()

pca_owind = pd.DataFrame({'Date':date_range_oni})
pca_owind['ONI'] = pca_oni
pca_owind['pca1'] = principal_components_owind[:,0]
pca_owind['pca2'] = principal_components_owind[:,1]

correlation_coefficient = np.corrcoef(pca_oni, pca_owind['pca1'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_oni, pca_oni, label='ONI ANOM', color='blue')
ax1.set_ylabel('ONI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-4,4)

ax2 = ax1.twinx()

ax2.plot(date_range_oni, pca_owind['pca1'], label='PCA 1', color='green')
ax2.set_ylabel('PC 1', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-4,4)

# ax1.legend()
# ax2.legend()
plt.text(0.05, 0.95, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
plt.title('ONI vs Wind PC1 (Coopers Gap Wind Farm)')
plt.savefig('Coopers Gap ONI PCA 1 (Wind).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)

# PC2

scaler_oni_wind = StandardScaler()
scaled_wind = scaler_oni_wind.fit_transform(np.column_stack((de_wind['E_wind'], pca_oni)))

pca = PCA(n_components=2)
principal_components_owind = pca.fit_transform(scaled_wind)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components_owind[:, 0], principal_components_owind[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Wind Energy Generation vs. ONI')
plt.show()

pca_owind = pd.DataFrame({'Date':date_range_oni})
pca_owind['ONI'] = pca_oni
pca_owind['pca1'] = principal_components_owind[:,0]
pca_owind['pca2'] = principal_components_owind[:,1]

correlation_coefficient = np.corrcoef(pca_oni, pca_owind['pca2'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_oni, pca_oni, label='ONI ANOM', color='blue')
ax1.set_ylabel('ONI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-4,4)

ax2 = ax1.twinx()

ax2.plot(date_range_oni, pca_owind['pca2'], label='PCA 2', color='green')
ax2.set_ylabel('PC 2', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-4,4)

# ax1.legend()
# ax2.legend()
plt.text(0.05, 0.95, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
plt.title('ONI vs Wind PC2 (Coopers Gap Wind Farm)')
plt.savefig('Coopers Gap ONI PCA 2 (Wind).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)



#%% SOI PCA (solar)
scaler_soi_solar = StandardScaler()
scaled_solar = scaler_soi_solar.fit_transform(np.column_stack((de_solar['E_solar'].loc['1990-05':'2022-12'], pca_soi)))

pca = PCA(n_components=2)
principal_components_ssolar = pca.fit_transform(scaled_solar)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components_ssolar[:, 0], principal_components_ssolar[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Solar Energy Generation vs. SOI')
plt.show()

pca_ssolar = pd.DataFrame({'Date':date_range_soi})
pca_ssolar['SOI'] = pca_soi
pca_ssolar['pca1'] = principal_components_ssolar[:,0]
pca_ssolar['pca2'] = principal_components_ssolar[:,1]

correlation_coefficient = np.corrcoef(pca_soi, pca_ssolar['pca1'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_soi, pca_soi, label='SOI ANOM', color='blue')
ax1.set_ylabel('SOI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-30,30)

ax2 = ax1.twinx()

ax2.plot(date_range_soi, pca_ssolar['pca1'], label='PCA 1', color='green')
ax2.set_ylabel('PC 1', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-3,3)

# ax1.legend()
# ax2.legend()
plt.text(0.05, 0.95, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
plt.title('SOI vs PC1 (Brigalow Solar Farm)')
plt.savefig('Brigalow SOI PCA 1 (Solar).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)

# PC2
scaler_soi_solar = StandardScaler()
scaled_solar = scaler_soi_solar.fit_transform(np.column_stack((de_solar['E_solar'].loc['1990-05':'2022-12'], pca_soi)))

pca = PCA(n_components=2)
principal_components_ssolar = pca.fit_transform(scaled_solar)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components_ssolar[:, 0], principal_components_ssolar[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Solar Energy Generation vs. SOI')
plt.show()

pca_ssolar = pd.DataFrame({'Date':date_range_soi})
pca_ssolar['SOI'] = pca_soi
pca_ssolar['pca1'] = principal_components_ssolar[:,0]
pca_ssolar['pca2'] = principal_components_ssolar[:,1]

correlation_coefficient = np.corrcoef(pca_soi, pca_ssolar['pca2'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_soi, pca_soi, label='SOI ANOM', color='blue')
ax1.set_ylabel('SOI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-30,30)

ax2 = ax1.twinx()

ax2.plot(date_range_soi, pca_ssolar['pca2'], label='PCA 2', color='green')
ax2.set_ylabel('PC 2', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-3,3)

# ax1.legend()
# ax2.legend()
plt.text(0.05, 0.95, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
plt.title('SOI vs PC2 (Brigalow Solar Farm)')
plt.savefig('Brigalow SOI PCA 2 (Solar).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)

#%% ONI PCA (solar)

scaler_oni_solar = StandardScaler()
scaled_solar = scaler_oni_solar.fit_transform(np.column_stack((de_solar['E_solar'], pca_oni)))

pca = PCA(n_components=2)
principal_components_osolar = pca.fit_transform(scaled_solar)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components_osolar[:, 0], principal_components_osolar[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Solar Energy Generation vs. ONI')
plt.show()

pca_osolar = pd.DataFrame({'Date':date_range_oni})
pca_osolar['ONI'] = pca_oni
pca_osolar['pca1'] = principal_components_osolar[:,0]
pca_osolar['pca2'] = principal_components_osolar[:,1]

correlation_coefficient = np.corrcoef(pca_oni, pca_osolar['pca1'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_oni, pca_oni, label='ONI ANOM', color='blue')
ax1.set_ylabel('ONI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-4,4)

ax2 = ax1.twinx()

ax2.plot(date_range_oni, pca_osolar['pca1'], label='PCA 1', color='green')
ax2.set_ylabel('PC 1', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-4,4)

# ax1.legend()
# ax2.legend()
plt.text(0.05, 0.95, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
plt.title('ONI ANOM vs PC1 (Brigalow Solar Farm)')
plt.savefig('Brigalow ONI PCA 1 (Solar).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)

# PC2

scaler_oni_solar = StandardScaler()
scaled_solar = scaler_oni_solar.fit_transform(np.column_stack((de_solar['E_solar'], pca_oni)))

pca = PCA(n_components=2)
principal_components_osolar = pca.fit_transform(scaled_solar)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(12,9))
plt.scatter(principal_components_osolar[:, 0], principal_components_osolar[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Solar Energy Generation vs. ONI')
plt.show()

pca_osolar = pd.DataFrame({'Date':date_range_oni})
pca_osolar['ONI'] = pca_oni
pca_osolar['pca1'] = principal_components_osolar[:,0]
pca_osolar['pca2'] = principal_components_osolar[:,1]

correlation_coefficient = np.corrcoef(pca_oni, pca_osolar['pca2'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_oni, pca_oni, label='ONI ANOM', color='blue')
ax1.set_ylabel('ONI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-4,4)

ax2 = ax1.twinx()

ax2.plot(date_range_oni, pca_osolar['pca2'], label='PCA 2', color='green')
ax2.set_ylabel('PC 2', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-4,4)

# ax1.legend()
# ax2.legend()
plt.text(0.05, 0.95, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
plt.title('ONI ANOM vs PC2 (Brigalow Solar Farm)')
plt.savefig('Brigalow ONI PCA 2 (Solar).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)

#%%
plt.figure(figsize=(12,9))
plt.plot(-oni['ANOM']['2008-01-01':'2011-12-01']/0.5, label =  r'$\mathbf{-}$ ONI', color='red')
plt.plot(SMA_soi['2008-01-01':'2011-12-31']/7, color='blue', label = 'SOI')
plt.plot(de_wind['2008-01':'2011-12'], color='green')
plt.axhline(0, color='black')
plt.legend()
plt.ylim(-4,4)

#%% NEMOSIS
# Wind
# print(defaults.dynamic_tables)
# print(defaults.static_tables)

site = 'Coopers Gap Wind Farm'
technology = 'Wind'
start_time = '2022/12/02 00:00:00'
end_time = '2022/12/03 00:00:00'
table_1 = 'Generators and Scheduled Loads'
table_2 = 'DISPATCH_UNIT_SCADA'
table_3 = 'BIDPEROFFER_D'
raw_data_cache = 'C:/Users/liame/OneDrive/Uni/2023/T2 2023/Thesis B/GUI'

site_info = static_table(table_1, raw_data_cache)
duid_wind = site_info.loc[site_info['Station Name'] == site, 'DUID'].values[0]
print(defaults.table_columns['DISPATCH_UNIT_SCADA'])

wind_gen_data = dynamic_data_compiler(start_time, end_time, table_2, raw_data_cache,
                                 keep_csv=False, filter_cols = ['DUID'],
                                 filter_values = ([duid_wind],))

curtailment = dynamic_data_compiler(start_time, end_time, table_3, raw_data_cache,
                                 keep_csv=False, filter_cols = ['DUID'],
                                 filter_values = ([duid_wind],))

plt.figure(figsize=(16,9))
plt.plot(wind_gen_data['SETTLEMENTDATE'], wind_gen_data['SCADAVALUE'])
plt.plot(modelled['E_wind'].loc['2022-12-02'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%b %H:%M'))
plt.xticks()
plt.title('Wind')

# Solar
# print(defaults.dynamic_tables)
# print(defaults.static_tables)

technology = 'Solar'

duid_solar = site_info.loc[site_info['Station Name'] == site, 'DUID'].values[0]




wind_df = pd.DataFrame(wind_gen_data)
wind_gen = wind_df[wind_df['SETTLEMENTDATE'].dt.minute == 0]
wind_gen.reset_index(drop = True, inplace = True)


plt.figure(figsize=(16,9))
plt.plot(wind_gen['SETTLEMENTDATE'], wind_gen['SCADAVALUE'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%b %H:%M'))
plt.plot(modelled['E_wind'].loc['2022-12-02'])
plt.title('Wind and Solar')
plt.show()


plt.figure(figsize=(12,8))
plt.hist(modelled['E_wind'].loc['2022-12-02'])
plt.hist(wind_gen['SCADAVALUE'])
plt.show()



#%%

plt.figure(figsize=(12,8))
plt.hist(modelled['E_wind'].loc['2022-12-02'])
plt.hist(wind_gen_data['SCADAVALUE'])
plt.show()

#%% Solar modelled vs NEMOSIS
plt.figure(figsize=(16,9))
plt.plot(solar_gen_data['SETTLEMENTDATE'], solar_gen_data['SCADAVALUE'])


#%% Solar modelled vs NEMOSIS
plt.figure(figsize=(16,9))
plt.plot(modelled['E_solar'].loc['2022-12-02'])



#%%
plt.figure(figsize=(16,9))

plt.plot(modelled['E_solar'].loc['2022-12-02'])


