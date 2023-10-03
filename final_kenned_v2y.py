# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:52:58 2023

@author: liame
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:39:11 2023

@author: liame
"""

mmm

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

from scipy.stats import weibull_min


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
skep1 = xr.open_dataset('skep1.grib', engine = 'cfgrib', chunks = 1000)
skep2 = xr.open_dataset('skep1.grib', engine = 'cfgrib', chunks = 1000)
skep3 = xr.open_dataset('skep3.grib', engine = 'cfgrib', chunks = 1000)

kep1 = xr.open_dataset('kep1.grib', engine = 'cfgrib', chunks = 1000)
kep2 = xr.open_dataset('kep2.grib', engine = 'cfgrib', chunks = 1000)
kep3 = xr.open_dataset('kep3.grib', engine = 'cfgrib', chunks = 1000)
kep4 = xr.open_dataset('kep4.grib', engine = 'cfgrib', chunks = 1000)
kep5 = xr.open_dataset('kep5.grib', engine = 'cfgrib', chunks = 1000)
kep6 = xr.open_dataset('kep6.grib', engine = 'cfgrib', chunks = 1000)
kep7 = xr.open_dataset('kep7.grib', engine = 'cfgrib', chunks = 1000)
kep8 = xr.open_dataset('kep8.grib', engine = 'cfgrib', chunks = 1000)
kep9 = xr.open_dataset('kep9.grib', engine = 'cfgrib', chunks = 1000)

vkep1 = xr.open_dataset('vkep1.grib', engine = 'cfgrib', chunks = 1000)
vkep2 = xr.open_dataset('vkep2.grib', engine = 'cfgrib', chunks = 1000)
vkep3 = xr.open_dataset('vkep3.grib', engine = 'cfgrib', chunks = 1000)

#%% Creating numpy arrays of relevent parameters
kep_u1 = kep1.u10.values[:,0,0]
kep_u2 = kep2.u10.values[:,0,0]
kep_u3 = kep3.u10.values[:,0,0]
kep_u4 = kep4.u10.values[:,0,0]
kep_u5 = kep5.u10.values[:,0,0]
kep_u6 = kep6.u10.values[:,0,0]
kep_u7 = kep7.u10.values[:,0,0]
kep_u8 = kep8.u10.values[:,0,0]
kep_u9 = kep9.u10.values[:,0,0]

kep_v1 = vkep1.v10.values[:,0,0]
kep_v2 = vkep2.v10.values[:,0,0]
kep_v3 = vkep3.v10.values[:,0,0]

kep_t1 = kep1.t2m.values[:,0,0]
kep_t2 = kep2.t2m.values[:,0,0]
kep_t3 = kep3.t2m.values[:,0,0]
kep_t4 = kep4.t2m.values[:,0,0]
kep_t5 = kep5.t2m.values[:,0,0]
kep_t6 = kep6.t2m.values[:,0,0]
kep_t7 = kep7.t2m.values[:,0,0]
kep_t8 = kep8.t2m.values[:,0,0]
kep_t9 = kep9.t2m.values[:,0,0]

nans_kep_s1 = skep1.ssr.values
nans_kep_s2 = skep2.ssr.values
nans_kep_s3 = skep3.ssr.values

kep_s1 = nans_kep_s1[~np.isnan(nans_kep_s1)]
kep_s2 = nans_kep_s2[~np.isnan(nans_kep_s2)]
kep_s3 = nans_kep_s3[~np.isnan(nans_kep_s3)]

u = np.concatenate([kep_u1, kep_u2, kep_u3, kep_u4, kep_u5, kep_u6, kep_u7,
                    kep_u8, kep_u9])

v = np.concatenate([kep_v1, kep_v2, kep_v3])

t = np.concatenate([kep_t1, kep_t2, kep_t3, kep_t4, kep_t5, kep_t6, kep_t7,
                    kep_t8, kep_t9]) - 273.15

s = np.concatenate([kep_s1, kep_s2, kep_s3]) / 3600


#%% Variables

# General

Australian = pytz.timezone('Australia/Queensland')
latitude = -20.75
longitude = 144.5

# Wind

wind_capacity = 43.2     # MW
hub_height = 132       # m
P_rated_wind = 3.6      # MW
v_rated = 13           # m/s
v_cut_in = 3.0         # m/s
v_cut_out = 22.5        # m/s

# Solar

surface_tilt = 20
surface_azimuth = 0
modules_per_string= 39
strings_per_inverter = 238
number_of_inverters = 6

location = Location(latitude = latitude, longitude = longitude, tz = 'Australia/Queensland',
                    altitude = 400, name = 'Kennedy Energy Park')

celltype = 'monoSi'
pdc0 = 345
v_mp = 38.9
i_mp = 8.87
v_oc = 47.3
i_sc = 9.85
alpha_sc = 0.00048 * i_sc
beta_voc = -0.0029 * v_oc
gamma_pdc = -0.37
cells_in_series = 12*12
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
ax.set_title('Modelled Power Curve for Vestas V136 3.6 MW')
ax.legend()

# plt.savefig('KEP Turbine Power Curve', dpi = 600)

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

weather_data.to_csv('pvlib_kep.csv')

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


#%% 10m Wind Speed Profile
shape, loc, scale = weibull_min.fit(v1, floc=0)
x_weibull_wind = np.linspace(0, max(v1), 1000)  # Create a range of wind speeds
pdf = weibull_min.pdf(x_weibull_wind, shape, loc, scale) # probability density function

plt.figure(figsize=(9,6))
plt.hist(v1, bins=50, density=True, alpha=0.5, label='Wind Speed Data')
plt.plot(x_weibull_wind, pdf, 'r-', lw=2, label='Weibull Fit')
plt.title('Kennedy Energy Park - 10m Wind Speed Weibull Distribution')
plt.xlabel('Wind Speed (m/s)')
plt.xticks(np.arange(0, max(v1)+1,2))
plt.ylabel('Probability Density')
plt.legend()
plt.text(0.7, 0.1, f'Shape Parameter (k): {shape:.2f}', fontsize=10, transform=plt.gca().transAxes)
plt.text(0.7, 0.05, fr'Scale Parameter ($\lambda$): {scale:.2f}', fontsize=10, transform=plt.gca().transAxes)
# plt.savefig('KEP 10m Wind Speed Weibull Dist.png', dpi=600)

plt.show()
print(shape, loc, scale)

#%% Hub Height Wind Speed Profile
shape, loc, scale = weibull_min.fit(v2, floc=0)
x_weibull_wind = np.linspace(0, max(v2), 1000)  # Create a range of wind speeds
pdf = weibull_min.pdf(x_weibull_wind, shape, loc, scale) # probability density function

plt.figure(figsize=(9,6))
plt.hist(v2, bins=50, density=True, alpha=0.5, label='Wind Speed Data')
plt.plot(x_weibull_wind, pdf, 'r-', lw=2, label='Weibull Fit')
plt.title('Kennedy Energy Park - Wind Speed at Hub Height Weibull Distribution')
plt.xlabel('Wind Speed (m/s)')
plt.xticks(np.arange(0, max(v2)+1,2))
plt.ylabel('Probability Density')
plt.legend()
plt.text(0.7, 0.1, f'Shape Parameter (k): {shape:.2f}', fontsize=10, transform=plt.gca().transAxes)
plt.text(0.7, 0.05, fr'Scale Parameter ($\lambda$): {scale:.2f}', fontsize=10, transform=plt.gca().transAxes)
# plt.savefig('KEP hub height Wind Speed Weibull Dist', dpi=600)
plt.show()

# print(round(np.sum(v2>v_rated)/len(v2)*100, 2),'%')

#%% GHI Profile
filtered_ghi = s[s>0]
shape, loc, scale = weibull_min.fit(filtered_ghi, floc=0)
x_weibull_ghi = np.linspace(0, max(filtered_ghi), 1000)  # Create a range of wind speeds
pdf = weibull_min.pdf(x_weibull_ghi, shape, loc, scale) # probability density function

plt.figure(figsize=(9,6))
plt.hist(filtered_ghi, bins=50, density=True, alpha=0.5, label='GHI Data')
plt.plot(x_weibull_ghi, pdf, 'r-', lw=2, label='Weibull Fit')
plt.xlabel('Global Horizontal Irradiance (W/$m^2$)')
plt.ylabel('Probability Density')
plt.legend()
plt.text(0.7, 0.1, f'Shape Parameter (k): {shape:.2f}', fontsize=10, transform=plt.gca().transAxes)
plt.text(0.7, 0.05, fr'Scale Parameter ($\lambda$): {scale:.2f}', fontsize=10, transform=plt.gca().transAxes)
plt.show()

#%% Average GHI for each hour across seasons

spring_hourly_solar_ghi = raw_weather[(raw_weather.index.month >= 9) & (raw_weather.index.month <= 11)]
winter_hourly_solar_ghi = raw_weather[(raw_weather.index.month >= 6) & (raw_weather.index.month <= 8)]
autumn_hourly_solar_ghi = raw_weather[(raw_weather.index.month >= 3) & (raw_weather.index.month <= 5)]
summer_hourly_solar_ghi = raw_weather[(raw_weather.index.month <=2 ) | (raw_weather.index.month == 12)]

hours = list(range(0,24,3))

def calculate_hourly_averages(data):
    data['hour'] = data.index.hour
    hourly_averages = data.groupby(['hour']).agg({'ghi': 'mean'})
    return hourly_averages

def format_hour(hour):
    return f"{hour:02d}:00"

spring_hourly_averages = calculate_hourly_averages(spring_hourly_solar_ghi.copy())
winter_hourly_averages = calculate_hourly_averages(winter_hourly_solar_ghi.copy())
autumn_hourly_averages = calculate_hourly_averages(autumn_hourly_solar_ghi.copy())
summer_hourly_averages = calculate_hourly_averages(summer_hourly_solar_ghi.copy())

plt.figure(figsize=(12,9))
# plt.plot(spring_hourly_averages, label = 'Spring')
plt.plot(summer_hourly_averages, label = 'Summer')
plt.plot(autumn_hourly_averages, label = 'Autumn')
plt.plot(winter_hourly_averages, label = 'Winter')
plt.ylabel('Global Horizontal Irradiance (W/$m^2$)')
plt.axvline(12)
plt.xlabel('Time of Day')

plt.title('Average GHI at Kennedy Energy Park')
plt.xticks(hours, [format_hour(hour) for hour in hours])
plt.legend()
# plt.savefig('KEP average GHI (summer, autumn, winter)', dpi=600)


#%% GHI, DNI, DHI Plot

hours_day = pd.date_range(start='2000-01-01', periods=24, freq='h')
summer = (weather_data['ghi'].loc['2017-01-27']).values
autumn = (weather_data['ghi'].loc['2019-04-13']).values
winter = (weather_data['ghi'].loc['2005-07-22']).values
spring = (weather_data['ghi'].loc['2003-10-29']).values

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
summer_shift = midday_index - summer_avg_idx
autumn_shift = midday_index - autumn_avg_idx
winter_shift = midday_index - winter_avg_idx
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
plt.title('Diurnal Cycle of GHI at Kennedy Energy Park')
plt.legend()

hours_formatter = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(hours_formatter)

# plt.savefig('KEP Diurnal Cycle GHI', dpi=600)

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


#%% Deseasonalising Raw Weather
# Daily Deseasonalising
# Wind Speed

daily_wind_speed = raw_weather['w10'].resample('D').mean() # MWh
daily_wind_speed = daily_wind_speed.to_frame()

jan_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 1)]
feb_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 2)]
mar_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 3)]
apr_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 4)]
may_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 5)]
jun_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 6)]
jul_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 7)]
aug_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 8)]
sep_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 9)]
oct_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 10)]
nov_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 11)]
dec_wind_speed = daily_wind_speed[(daily_wind_speed.index.month == 12)]

jan_wind_speed_avg = jan_wind_speed[(jan_wind_speed.index.month == 1)].mean()
feb_wind_speed_avg = feb_wind_speed[(feb_wind_speed.index.month == 2)].mean()
mar_wind_speed_avg = mar_wind_speed[(mar_wind_speed.index.month == 3)].mean()
apr_wind_speed_avg = apr_wind_speed[(apr_wind_speed.index.month == 4)].mean()
may_wind_speed_avg = may_wind_speed[(may_wind_speed.index.month == 5)].mean()
jun_wind_speed_avg = jun_wind_speed[(jun_wind_speed.index.month == 6)].mean()
jul_wind_speed_avg = jul_wind_speed[(jul_wind_speed.index.month == 7)].mean()
aug_wind_speed_avg = aug_wind_speed[(aug_wind_speed.index.month == 8)].mean()
sep_wind_speed_avg = sep_wind_speed[(sep_wind_speed.index.month == 9)].mean()
oct_wind_speed_avg = oct_wind_speed[(oct_wind_speed.index.month == 10)].mean()
nov_wind_speed_avg = nov_wind_speed[(nov_wind_speed.index.month == 11)].mean()
dec_wind_speed_avg = dec_wind_speed[(dec_wind_speed.index.month == 12)].mean()

de_jan_wind_speed = (jan_wind_speed - jan_wind_speed_avg) / np.std(jan_wind_speed)
de_feb_wind_speed = (feb_wind_speed - feb_wind_speed_avg) / np.std(feb_wind_speed)
de_mar_wind_speed = (mar_wind_speed - mar_wind_speed_avg) / np.std(mar_wind_speed)
de_apr_wind_speed = (apr_wind_speed - apr_wind_speed_avg) / np.std(apr_wind_speed)
de_may_wind_speed = (may_wind_speed - may_wind_speed_avg) / np.std(may_wind_speed)
de_jun_wind_speed = (jun_wind_speed - jun_wind_speed_avg) / np.std(jun_wind_speed)
de_jul_wind_speed = (jul_wind_speed - jul_wind_speed_avg) / np.std(jul_wind_speed)
de_aug_wind_speed = (aug_wind_speed - aug_wind_speed_avg) / np.std(aug_wind_speed)
de_sep_wind_speed = (sep_wind_speed - sep_wind_speed_avg) / np.std(sep_wind_speed)
de_oct_wind_speed = (oct_wind_speed - oct_wind_speed_avg) / np.std(oct_wind_speed)
de_nov_wind_speed = (nov_wind_speed - nov_wind_speed_avg) / np.std(nov_wind_speed)
de_dec_wind_speed = (dec_wind_speed - dec_wind_speed_avg) / np.std(dec_wind_speed)

de_daily_wind_speed = pd.concat((de_jan_wind_speed, de_feb_wind_speed,
                                 de_mar_wind_speed, de_apr_wind_speed,
                                 de_may_wind_speed, de_jun_wind_speed,
                                 de_jul_wind_speed, de_aug_wind_speed,
                                 de_sep_wind_speed, de_oct_wind_speed,
                                 de_nov_wind_speed, de_dec_wind_speed))

de_daily_wind_speed = de_daily_wind_speed.sort_values('time')


# GHI
daily_ghi = raw_weather['ghi'].resample('D').sum() # MWh
daily_ghi = daily_ghi.to_frame()


jan_ghi = daily_ghi[(daily_ghi.index.month == 1)]
feb_ghi = daily_ghi[(daily_ghi.index.month == 2)]
mar_ghi = daily_ghi[(daily_ghi.index.month == 3)]
apr_ghi = daily_ghi[(daily_ghi.index.month == 4)]
may_ghi = daily_ghi[(daily_ghi.index.month == 5)]
jun_ghi = daily_ghi[(daily_ghi.index.month == 6)]
jul_ghi = daily_ghi[(daily_ghi.index.month == 7)]
aug_ghi = daily_ghi[(daily_ghi.index.month == 8)]
sep_ghi = daily_ghi[(daily_ghi.index.month == 9)]
oct_ghi = daily_ghi[(daily_ghi.index.month == 10)]
nov_ghi = daily_ghi[(daily_ghi.index.month == 11)]
dec_ghi = daily_ghi[(daily_ghi.index.month == 12)]

jan_ghi_avg = jan_ghi[(jan_ghi.index.month == 1)].mean()
feb_ghi_avg = feb_ghi[(feb_ghi.index.month == 2)].mean()
mar_ghi_avg = mar_ghi[(mar_ghi.index.month == 3)].mean()
apr_ghi_avg = apr_ghi[(apr_ghi.index.month == 4)].mean()
may_ghi_avg = may_ghi[(may_ghi.index.month == 5)].mean()
jun_ghi_avg = jun_ghi[(jun_ghi.index.month == 6)].mean()
jul_ghi_avg = jul_ghi[(jul_ghi.index.month == 7)].mean()
aug_ghi_avg = aug_ghi[(aug_ghi.index.month == 8)].mean()
sep_ghi_avg = sep_ghi[(sep_ghi.index.month == 9)].mean()
oct_ghi_avg = oct_ghi[(oct_ghi.index.month == 10)].mean()
nov_ghi_avg = nov_ghi[(nov_ghi.index.month == 11)].mean()
dec_ghi_avg = dec_ghi[(dec_ghi.index.month == 12)].mean()

de_jan_ghi = (jan_ghi - jan_ghi_avg) / np.std(jan_ghi)
de_feb_ghi = (feb_ghi - feb_ghi_avg) / np.std(feb_ghi)
de_mar_ghi = (mar_ghi - mar_ghi_avg) / np.std(mar_ghi)
de_apr_ghi = (apr_ghi - apr_ghi_avg) / np.std(apr_ghi)
de_may_ghi = (may_ghi - may_ghi_avg) / np.std(may_ghi)
de_jun_ghi = (jun_ghi - jun_ghi_avg) / np.std(jun_ghi)
de_jul_ghi = (jul_ghi - jul_ghi_avg) / np.std(jul_ghi)
de_aug_ghi = (aug_ghi - aug_ghi_avg) / np.std(aug_ghi)
de_sep_ghi = (sep_ghi - sep_ghi_avg) / np.std(sep_ghi)
de_oct_ghi = (oct_ghi - oct_ghi_avg) / np.std(oct_ghi)
de_nov_ghi = (nov_ghi - nov_ghi_avg) / np.std(nov_ghi)
de_dec_ghi = (dec_ghi - dec_ghi_avg) / np.std(dec_ghi)

de_daily_ghi = pd.concat((de_jan_ghi, de_feb_ghi, de_mar_ghi, de_apr_ghi,
                          de_may_ghi, de_jun_ghi, de_jul_ghi, de_aug_ghi,
                          de_sep_ghi, de_oct_ghi, de_nov_ghi, de_dec_ghi))

de_daily_ghi = de_daily_ghi.sort_values('time')


# 2-meter Temperature
daily_t2m = raw_weather['t2m'].resample('D').mean() # MWh
daily_t2m = daily_t2m.to_frame()

jan_t2m = daily_t2m[(daily_t2m.index.month == 1)]
feb_t2m = daily_t2m[(daily_t2m.index.month == 2)]
mar_t2m = daily_t2m[(daily_t2m.index.month == 3)]
apr_t2m = daily_t2m[(daily_t2m.index.month == 4)]
may_t2m = daily_t2m[(daily_t2m.index.month == 5)]
jun_t2m = daily_t2m[(daily_t2m.index.month == 6)]
jul_t2m = daily_t2m[(daily_t2m.index.month == 7)]
aug_t2m = daily_t2m[(daily_t2m.index.month == 8)]
sep_t2m = daily_t2m[(daily_t2m.index.month == 9)]
oct_t2m = daily_t2m[(daily_t2m.index.month == 10)]
nov_t2m = daily_t2m[(daily_t2m.index.month == 11)]
dec_t2m = daily_t2m[(daily_t2m.index.month == 12)]

jan_t2m_avg = jan_t2m[(jan_t2m.index.month == 1)].mean()
feb_t2m_avg = feb_t2m[(feb_t2m.index.month == 2)].mean()
mar_t2m_avg = mar_t2m[(mar_t2m.index.month == 3)].mean()
apr_t2m_avg = apr_t2m[(apr_t2m.index.month == 4)].mean()
may_t2m_avg = may_t2m[(may_t2m.index.month == 5)].mean()
jun_t2m_avg = jun_t2m[(jun_t2m.index.month == 6)].mean()
jul_t2m_avg = jul_t2m[(jul_t2m.index.month == 7)].mean()
aug_t2m_avg = aug_t2m[(aug_t2m.index.month == 8)].mean()
sep_t2m_avg = sep_t2m[(sep_t2m.index.month == 9)].mean()
oct_t2m_avg = oct_t2m[(oct_t2m.index.month == 10)].mean()
nov_t2m_avg = nov_t2m[(nov_t2m.index.month == 11)].mean()
dec_t2m_avg = dec_t2m[(dec_t2m.index.month == 12)].mean()

de_jan_t2m = (jan_t2m - jan_t2m_avg) / np.std(jan_t2m)
de_feb_t2m = (feb_t2m - feb_t2m_avg) / np.std(feb_t2m)
de_mar_t2m = (mar_t2m - mar_t2m_avg) / np.std(mar_t2m)
de_apr_t2m = (apr_t2m - apr_t2m_avg) / np.std(apr_t2m)
de_may_t2m = (may_t2m - may_t2m_avg) / np.std(may_t2m)
de_jun_t2m = (jun_t2m - jun_t2m_avg) / np.std(jun_t2m)
de_jul_t2m = (jul_t2m - jul_t2m_avg) / np.std(jul_t2m)
de_aug_t2m = (aug_t2m - aug_t2m_avg) / np.std(aug_t2m)
de_sep_t2m = (sep_t2m - sep_t2m_avg) / np.std(sep_t2m)
de_oct_t2m = (oct_t2m - oct_t2m_avg) / np.std(oct_t2m)
de_nov_t2m = (nov_t2m - nov_t2m_avg) / np.std(nov_t2m)
de_dec_t2m = (dec_t2m - dec_t2m_avg) / np.std(dec_t2m)

de_daily_t2m = pd.concat((de_jan_t2m, de_feb_t2m, de_mar_t2m, de_apr_t2m,
                          de_may_t2m, de_jun_t2m, de_jul_t2m, de_aug_t2m,
                          de_sep_t2m, de_oct_t2m, de_nov_t2m, de_dec_t2m))

de_daily_t2m = de_daily_t2m.sort_values('time')


# Monthly Deseasonalising
# Wind Speed

monthly_wind_speed = daily_wind_speed.resample('MS').sum()/1000 # GWh
monthly_wind_speed.index = pd.to_datetime(monthly_wind_speed.index)
monthly_wind_speed = monthly_wind_speed[:-1]
monthly_wind_speed['year'] = monthly_wind_speed.index.year

spring_wind_speed = monthly_wind_speed[(monthly_wind_speed.index.month >= 9) & (monthly_wind_speed.index.month <= 11)]
winter_wind_speed = monthly_wind_speed[(monthly_wind_speed.index.month >= 6) & (monthly_wind_speed.index.month <= 8)]
autumn_wind_speed = monthly_wind_speed[(monthly_wind_speed.index.month >= 3) & (monthly_wind_speed.index.month <= 5)]
summer_wind_speed = monthly_wind_speed[(monthly_wind_speed.index.month <=2 ) | (monthly_wind_speed.index.month == 12)]

spring_wind_speed_avg = monthly_wind_speed[(monthly_wind_speed.index.month >= 9) & (monthly_wind_speed.index.month <= 11)].mean()
winter_wind_speed_avg = monthly_wind_speed[(monthly_wind_speed.index.month >= 6) & (monthly_wind_speed.index.month <= 8)].mean()
autumn_wind_speed_avg = monthly_wind_speed[(monthly_wind_speed.index.month >= 3) & (monthly_wind_speed.index.month <= 5)].mean()
summer_wind_speed_avg = monthly_wind_speed[(monthly_wind_speed.index.month <=2 ) | (monthly_wind_speed.index.month == 12)].mean()

de_spring_wind_speed = (spring_wind_speed - spring_wind_speed_avg) / np.std(spring_wind_speed)
de_winter_wind_speed = (winter_wind_speed - winter_wind_speed_avg) / np.std(winter_wind_speed)
de_autumn_wind_speed = (autumn_wind_speed - autumn_wind_speed_avg) / np.std(autumn_wind_speed)
de_summer_wind_speed = (summer_wind_speed - summer_wind_speed_avg) / np.std(summer_wind_speed)

de_wind_speed = pd.concat((de_spring_wind_speed, de_winter_wind_speed, de_autumn_wind_speed, de_summer_wind_speed))
de_wind_speed = de_wind_speed.drop(de_wind_speed.columns[-1], axis=1)

de_wind_speed = de_wind_speed.sort_values('time')


# GHI

monthly_ghi = daily_ghi.resample('MS').sum()/1000 # GWh
monthly_ghi.index = pd.to_datetime(monthly_ghi.index)
monthly_ghi = monthly_ghi[:-1]
monthly_ghi['year'] = monthly_ghi.index.year

spring_solar_ghi = monthly_ghi[(monthly_ghi.index.month >= 9) & (monthly_ghi.index.month <= 11)]
winter_solar_ghi = monthly_ghi[(monthly_ghi.index.month >= 6) & (monthly_ghi.index.month <= 8)]
autumn_solar_ghi = monthly_ghi[(monthly_ghi.index.month >= 3) & (monthly_ghi.index.month <= 5)]
summer_solar_ghi = monthly_ghi[(monthly_ghi.index.month <=2 ) | (monthly_ghi.index.month == 12)]

spring_solar_ghi_avg = monthly_ghi[(monthly_ghi.index.month >= 9) & (monthly_ghi.index.month <= 11)].mean()
winter_solar_ghi_avg = monthly_ghi[(monthly_ghi.index.month >= 6) & (monthly_ghi.index.month <= 8)].mean()
autumn_solar_ghi_avg = monthly_ghi[(monthly_ghi.index.month >= 3) & (monthly_ghi.index.month <= 5)].mean()
summer_solar_ghi_avg = monthly_ghi[(monthly_ghi.index.month <=2 ) | (monthly_ghi.index.month == 12)].mean()

de_spring_solar_ghi = (spring_solar_ghi - spring_solar_ghi_avg) / np.std(spring_solar_ghi)
de_winter_solar_ghi = (winter_solar_ghi - winter_solar_ghi_avg) / np.std(winter_solar_ghi)
de_autumn_solar_ghi = (autumn_solar_ghi - autumn_solar_ghi_avg) / np.std(autumn_solar_ghi)
de_summer_solar_ghi = (summer_solar_ghi - summer_solar_ghi_avg) / np.std(summer_solar_ghi)

de_solar_ghi = pd.concat((de_spring_solar_ghi, de_winter_solar_ghi, de_autumn_solar_ghi, de_summer_solar_ghi))
de_solar_ghi = de_solar_ghi.drop(de_solar_ghi.columns[-1], axis=1)

de_solar_ghi = de_solar_ghi.sort_values('time')


# Temperature
monthly_t2m = daily_t2m.resample('MS').sum()/1000 # GWh
monthly_t2m.index = pd.to_datetime(monthly_t2m.index)
monthly_t2m = monthly_t2m[:-1]
monthly_t2m['year'] = monthly_t2m.index.year

spring_t2m = monthly_t2m[(monthly_t2m.index.month >= 9) & (monthly_t2m.index.month <= 11)]
winter_t2m = monthly_t2m[(monthly_t2m.index.month >= 6) & (monthly_t2m.index.month <= 8)]
autumn_t2m = monthly_t2m[(monthly_t2m.index.month >= 3) & (monthly_t2m.index.month <= 5)]
summer_t2m = monthly_t2m[(monthly_t2m.index.month <=2 ) | (monthly_t2m.index.month == 12)]

spring_t2m_avg = monthly_t2m[(monthly_t2m.index.month >= 9) & (monthly_t2m.index.month <= 11)].mean()
winter_t2m_avg = monthly_t2m[(monthly_t2m.index.month >= 6) & (monthly_t2m.index.month <= 8)].mean()
autumn_t2m_avg = monthly_t2m[(monthly_t2m.index.month >= 3) & (monthly_t2m.index.month <= 5)].mean()
summer_t2m_avg = monthly_t2m[(monthly_t2m.index.month <=2 ) | (monthly_t2m.index.month == 12)].mean()

de_spring_t2m = (spring_t2m - spring_t2m_avg) / np.std(spring_t2m)
de_winter_t2m = (winter_t2m - winter_t2m_avg) / np.std(winter_t2m)
de_autumn_t2m = (autumn_t2m - autumn_t2m_avg) / np.std(autumn_t2m)
de_summer_t2m = (summer_t2m - summer_t2m_avg) / np.std(summer_t2m)

de_t2m = pd.concat((de_spring_t2m, de_winter_t2m, de_autumn_t2m, de_summer_t2m))
de_t2m = de_t2m.drop(de_t2m.columns[-1], axis=1)

de_t2m = de_t2m.sort_values('time')

#%% Deseasonalising Generation
# Daily Deseasonalising

# Wind
daily_wind = modelled['E_wind'].resample('D').sum() # MWh
daily_wind = daily_wind.to_frame()

jan_wind = daily_wind[(daily_wind.index.month == 1)]
feb_wind = daily_wind[(daily_wind.index.month == 2)]
mar_wind = daily_wind[(daily_wind.index.month == 3)]
apr_wind = daily_wind[(daily_wind.index.month == 4)]
may_wind = daily_wind[(daily_wind.index.month == 5)]
jun_wind = daily_wind[(daily_wind.index.month == 6)]
jul_wind = daily_wind[(daily_wind.index.month == 7)]
aug_wind = daily_wind[(daily_wind.index.month == 8)]
sep_wind = daily_wind[(daily_wind.index.month == 9)]
oct_wind = daily_wind[(daily_wind.index.month == 10)]
nov_wind = daily_wind[(daily_wind.index.month == 11)]
dec_wind = daily_wind[(daily_wind.index.month == 12)]

jan_wind_avg = jan_wind[(jan_wind.index.month == 1)].mean()
feb_wind_avg = feb_wind[(feb_wind.index.month == 2)].mean()
mar_wind_avg = mar_wind[(mar_wind.index.month == 3)].mean()
apr_wind_avg = apr_wind[(apr_wind.index.month == 4)].mean()
may_wind_avg = may_wind[(may_wind.index.month == 5)].mean()
jun_wind_avg = jun_wind[(jun_wind.index.month == 6)].mean()
jul_wind_avg = jul_wind[(jul_wind.index.month == 7)].mean()
aug_wind_avg = aug_wind[(aug_wind.index.month == 8)].mean()
sep_wind_avg = sep_wind[(sep_wind.index.month == 9)].mean()
oct_wind_avg = oct_wind[(oct_wind.index.month == 10)].mean()
nov_wind_avg = nov_wind[(nov_wind.index.month == 11)].mean()
dec_wind_avg = dec_wind[(dec_wind.index.month == 12)].mean()

de_jan_wind = (jan_wind - jan_wind_avg) / np.std(jan_wind)
de_feb_wind = (feb_wind - feb_wind_avg) / np.std(feb_wind)
de_mar_wind = (mar_wind - mar_wind_avg) / np.std(mar_wind)
de_apr_wind = (apr_wind - apr_wind_avg) / np.std(apr_wind)
de_may_wind = (may_wind - may_wind_avg) / np.std(may_wind)
de_jun_wind = (jun_wind - jun_wind_avg) / np.std(jun_wind)
de_jul_wind = (jul_wind - jul_wind_avg) / np.std(jul_wind)
de_aug_wind = (aug_wind - aug_wind_avg) / np.std(aug_wind)
de_sep_wind = (sep_wind - sep_wind_avg) / np.std(sep_wind)
de_oct_wind = (oct_wind - oct_wind_avg) / np.std(oct_wind)
de_nov_wind = (nov_wind - nov_wind_avg) / np.std(nov_wind)
de_dec_wind = (dec_wind - dec_wind_avg) / np.std(dec_wind)

de_daily_wind = pd.concat((de_jan_wind, de_feb_wind, de_mar_wind, de_apr_wind,
                          de_may_wind, de_jun_wind, de_jul_wind, de_aug_wind,
                          de_sep_wind, de_oct_wind, de_nov_wind, de_dec_wind))

de_daily_wind = de_daily_wind.sort_values('time')


# Solar
daily_solar = modelled['E_solar'].resample('D').sum() # MWh
daily_solar = daily_solar.to_frame()

jan_solar = daily_solar[(daily_solar.index.month == 1)]
feb_solar = daily_solar[(daily_solar.index.month == 2)]
mar_solar = daily_solar[(daily_solar.index.month == 3)]
apr_solar = daily_solar[(daily_solar.index.month == 4)]
may_solar = daily_solar[(daily_solar.index.month == 5)]
jun_solar = daily_solar[(daily_solar.index.month == 6)]
jul_solar = daily_solar[(daily_solar.index.month == 7)]
aug_solar = daily_solar[(daily_solar.index.month == 8)]
sep_solar = daily_solar[(daily_solar.index.month == 9)]
oct_solar = daily_solar[(daily_solar.index.month == 10)]
nov_solar = daily_solar[(daily_solar.index.month == 11)]
dec_solar = daily_solar[(daily_solar.index.month == 12)]

jan_solar_avg = jan_solar[(jan_solar.index.month == 1)].mean()
feb_solar_avg = feb_solar[(feb_solar.index.month == 2)].mean()
mar_solar_avg = mar_solar[(mar_solar.index.month == 3)].mean()
apr_solar_avg = apr_solar[(apr_solar.index.month == 4)].mean()
may_solar_avg = may_solar[(may_solar.index.month == 5)].mean()
jun_solar_avg = jun_solar[(jun_solar.index.month == 6)].mean()
jul_solar_avg = jul_solar[(jul_solar.index.month == 7)].mean()
aug_solar_avg = aug_solar[(aug_solar.index.month == 8)].mean()
sep_solar_avg = sep_solar[(sep_solar.index.month == 9)].mean()
oct_solar_avg = oct_solar[(oct_solar.index.month == 10)].mean()
nov_solar_avg = nov_solar[(nov_solar.index.month == 11)].mean()
dec_solar_avg = dec_solar[(dec_solar.index.month == 12)].mean()

de_jan_solar = (jan_solar - jan_solar_avg) / np.std(jan_solar)
de_feb_solar = (feb_solar - feb_solar_avg) / np.std(feb_solar)
de_mar_solar = (mar_solar - mar_solar_avg) / np.std(mar_solar)
de_apr_solar = (apr_solar - apr_solar_avg) / np.std(apr_solar)
de_may_solar = (may_solar - may_solar_avg) / np.std(may_solar)
de_jun_solar = (jun_solar - jun_solar_avg) / np.std(jun_solar)
de_jul_solar = (jul_solar - jul_solar_avg) / np.std(jul_solar)
de_aug_solar = (aug_solar - aug_solar_avg) / np.std(aug_solar)
de_sep_solar = (sep_solar - sep_solar_avg) / np.std(sep_solar)
de_oct_solar = (oct_solar - oct_solar_avg) / np.std(oct_solar)
de_nov_solar = (nov_solar - nov_solar_avg) / np.std(nov_solar)
de_dec_solar = (dec_solar - dec_solar_avg) / np.std(dec_solar)

de_daily_solar = pd.concat((de_jan_solar, de_feb_solar, de_mar_solar, de_apr_solar,
                          de_may_solar, de_jun_solar, de_jul_solar, de_aug_solar,
                          de_sep_solar, de_oct_solar, de_nov_solar, de_dec_solar))

de_daily_solar = de_daily_solar.sort_values('time')


# Monthly Deseasonalising
# Wind
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
plt.title('Average Monthly Solar Generation at Kennedy Energy Park for varying La Nina Conditions')
plt.xticks(index + bar_width/2, month_names, rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
# plt.savefig('Monthly Average Solar Generation (Kennedy Energy Park)', dpi=600)
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
plt.title('Average Monthly Wind Generation at Kennedy Energy Park for varying La Nina Conditions')

plt.xticks(index + bar_width/2, month_names, rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
# plt.savefig('Monthly Average Wind Generation (Kennedy Energy Park).png', dpi=600)
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


#%%

date_range_soi = pd.date_range('1990-05-01', end_date, freq='M')
date_range_oni = pd.date_range(start_date, end_date, freq='M')

pca_soi = SMA_soi.dropna()
pca_oni = - x_oni # defintion for x_oni is for comparison to soi so take negative to actually reflect actual oni anomolies


#%% Principal Component Analysis v3
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');


#%% Principal Component Analysis (Wind) v2

pca_modelled_wind = pd.concat([de_wind['E_wind'].loc['1990-05':'2022-12'],
                               de_wind_speed['w10'].loc['1990-05':'2022-12'],
                               de_solar_ghi['ghi'].loc['1990-05':'2022-12'],
                               de_t2m['t2m'].loc['1990-05':'2022-12']],
                               axis=1)


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

columns_wind = ['E_wind', 'w10', 'ghi', 't2m']

wind_scaler = pca_modelled_wind[columns_wind]

scaler = StandardScaler()
scaler.fit(wind_scaler)
scaled_data = scaler.transform(wind_scaler)

#View in Frame
wind_scaled = pd.DataFrame(scaled_data, columns=columns_wind)
wind_scaled = wind_scaled.round(2)
wind_scaled

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
pca.fit(scaled_data)
reduced = pca.transform(scaled_data)
print('Original Dimensions: ',scaled_data.shape)
print('Reduced Dimensions: ',reduced.shape)

import seaborn as sn
dfScree = pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':['PC1','PC2','PC3']})
sn.barplot(x='PC',y="var",data=dfScree, color="c").set_title('Fig 2. Component Variance');


x_pca=pca.transform(scaled_data)

import seaborn as sn
df_pc = pd.DataFrame(data = x_pca, columns = ['pc1','pc2','pc3'])


df_col = pd.concat([df_pc,wind_scaled], axis=1)
df_col
covMatrix = pd.DataFrame.cov(df_col)
sn.set(rc={'figure.figsize':(14,6)})
sn.heatmap(covMatrix, annot=True, fmt='g')
plt.figure(figsize=(28,18))
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(x_pca[:,0],x_pca[:,1])
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_frame_on(False)
ax1.grid(True)
ax1.set_title('PC1 vs PC2')

ax2.scatter(x_pca[:,0],x_pca[:,2])
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC3")
ax2.set_title('PC1 vs PC3')
ax2.grid(True)
ax2.set_frame_on(False)
fig.show()

import plotly.express as px
features = ['E_wind', 'w10', 'ghi', 't2m']
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig = px.scatter(x_pca, x=0, y=1)
for i, feature in enumerate(features):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0]*4.5,
        y1=loadings[i, 1]*4.5
    )
    fig.add_annotation(
        x=loadings[i, 0]*5.5,
        y=loadings[i, 1]*5.5,
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )
plt.savefig('idkv2.png')
fig.show()

sn.pairplot(df_col, kind="reg")
plt.show()

from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
ax = plt.axes(projection='3d')
xline=x_pca[:,0]
yline=x_pca[:,1]
zline=x_pca[:,2]
ax.scatter3D(xline, yline, zline,c=zline,cmap='BrBG_r')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
# plt.savefig('3D Test.png',dpi=600)
plt.show()



#%% PCA plot WIND

correlation_coefficient = np.corrcoef(pca_soi, df_pc['pc3'])[0,1]

fig, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(date_range_soi, pca_soi, label='SOI ANOM', color='blue')
ax1.set_ylabel('SOI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-30,30)

ax2 = ax1.twinx()
ax2.plot(date_range_soi, df_pc['pc3'], label='PCA 3', color='green')
ax2.set_ylabel('PC 3', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-4,4)

# ax1.legend()
# ax2.legend()
plt.text(0.4, 0.08, f'Correlation Coefficient: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
# plt.text(0.4, 0.04, f'Explained Variance in PC3: {explained_variance_ratio[2]:.4f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.title('KEP - SOI vs Wind PC3')
plt.savefig('KEP pc3.png', dpi=600)
fig.show()

print("Correlation Coefficient:", correlation_coefficient)
# print("Explained Variance:", explained_variance_ratio)

#%% Cumulative Variance vs components WIND
pca = PCA().fit(scaled_data)
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(1, 5, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')
plt.xlabel('Fig 1 Number of Components')
plt.xticks(np.arange(0, 4, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
ax.grid(axis='x')
plt.savefig('KEP - cumulative explained variance plot.png', dpi = 600)
plt.show()

#%% Principal Component Analysis (Solar) v2

pca_modelled_solar = pd.concat([de_solar['E_solar'].loc['1990-05':'2022-12'],
                               de_wind_speed['w10'].loc['1990-05':'2022-12'],
                               de_solar_ghi['ghi'].loc['1990-05':'2022-12'],
                               de_t2m['t2m'].loc['1990-05':'2022-12']],
                               axis=1)


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

columns_solar = ['E_solar', 'w10', 'ghi', 't2m']

solar_scaler = pca_modelled_solar[columns_solar]

scaler = StandardScaler()
scaler.fit(solar_scaler)
scaled_data = scaler.transform(solar_scaler)

#View in Frame
solar_scaled = pd.DataFrame(scaled_data, columns=columns_solar)
solar_scaled = solar_scaled.round(2)
solar_scaled

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
pca.fit(scaled_data)
reduced = pca.transform(scaled_data)
print('Original Dimensions: ',scaled_data.shape)
print('Reduced Dimensions: ',reduced.shape)

import seaborn as sn
dfScree = pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':['PC1','PC2','PC3']})
sn.barplot(x='PC',y="var",data=dfScree, color="c").set_title('Fig 2. Component Variance');


x_pca=pca.transform(scaled_data)

import seaborn as sn
df_pc = pd.DataFrame(data = x_pca, columns = ['pc1','pc2','pc3'])


df_col = pd.concat([df_pc,solar_scaled], axis=1)
df_col
covMatrix = pd.DataFrame.cov(df_col)
sn.set(rc={'figure.figsize':(14,6)})
sn.heatmap(covMatrix, annot=True, fmt='g')
plt.figure(figsize=(28,18))
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(x_pca[:,0],x_pca[:,1])
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_frame_on(False)
ax1.grid(True)
ax1.set_title('PC1 vs PC2')

ax2.scatter(x_pca[:,0],x_pca[:,2])
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC3")
ax2.set_title('PC1 vs PC3')
ax2.grid(True)
ax2.set_frame_on(False)
fig.show()

import plotly.express as px
features = ['E_solar', 'w10', 'ghi', 't2m']
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig = px.scatter(x_pca, x=0, y=1)
for i, feature in enumerate(features):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0]*4.5,
        y1=loadings[i, 1]*4.5
    )
    fig.add_annotation(
        x=loadings[i, 0]*5.5,
        y=loadings[i, 1]*5.5,
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )
plt.savefig('idkv3.png')
fig.show()

sn.pairplot(df_col, kind="reg")
plt.show()

from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
ax = plt.axes(projection='3d')
xline=x_pca[:,0]
yline=x_pca[:,1]
zline=x_pca[:,2]
ax.scatter3D(xline, yline, zline,c=zline,cmap='BrBG_r')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
# plt.savefig('3D Test.png',dpi=600)
plt.show()



#%% PCA plot SOLAR

correlation_coefficient = np.corrcoef(pca_soi, df_pc['pc3'])[0,1]

fig, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(date_range_soi, pca_soi, label='SOI ANOM', color='blue')
ax1.set_ylabel('SOI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-30,30)

ax2 = ax1.twinx()
ax2.plot(date_range_soi, df_pc['pc3'], label='PCA 3', color='green')
ax2.set_ylabel('PC 3', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-4,4)

# ax1.legend()
# ax2.legend()
plt.text(0.4, 0.08, f'Correlation Coefficient: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
# plt.text(0.4, 0.04, f'Explained Variance in PC3: {explained_variance_ratio[2]:.4f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.title('KEP - SOI vs Solar PC3')
plt.savefig('KEP solar pc3.png', dpi=600)
fig.show()

print("Correlation Coefficient:", correlation_coefficient)
# print("Explained Variance:", explained_variance_ratio)

#%% cumulative explained variance plot SOLAR
pca = PCA().fit(scaled_data)
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(1, 5, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')
plt.xlabel('Fig 1 Number of Components')
plt.xticks(np.arange(0, 4, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
ax.grid(axis='x')
plt.savefig('KEP solar - cumulative explained variance plot.png', dpi = 600)
plt.show()

#%% Principal Component Analysis (Solar) v2 ONI

pca_modelled_solar = pd.concat([de_solar['E_solar'],
                               de_wind_speed['w10'],
                               de_solar_ghi['ghi'],
                               de_t2m['t2m']],
                               axis=1)


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

columns_solar = ['E_solar', 'w10', 'ghi', 't2m']

solar_scaler = pca_modelled_solar[columns_solar]

scaler = StandardScaler()
scaler.fit(solar_scaler)
scaled_data = scaler.transform(solar_scaler)

#View in Frame
solar_scaled = pd.DataFrame(scaled_data, columns=columns_solar)
solar_scaled = solar_scaled.round(2)
solar_scaled

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
pca.fit(scaled_data)
reduced = pca.transform(scaled_data)
print('Original Dimensions: ',scaled_data.shape)
print('Reduced Dimensions: ',reduced.shape)

import seaborn as sn
dfScree = pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':['PC1','PC2','PC3']})
sn.barplot(x='PC',y="var",data=dfScree, color="c").set_title('Fig 2. Component Variance');


x_pca=pca.transform(scaled_data)

import seaborn as sn
df_pc = pd.DataFrame(data = x_pca, columns = ['pc1','pc2','pc3'])


df_col = pd.concat([df_pc,solar_scaled], axis=1)
df_col
covMatrix = pd.DataFrame.cov(df_col)
sn.set(rc={'figure.figsize':(14,6)})
sn.heatmap(covMatrix, annot=True, fmt='g')
plt.figure(figsize=(28,18))
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(x_pca[:,0],x_pca[:,1])
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_frame_on(False)
ax1.grid(True)
ax1.set_title('PC1 vs PC2')

ax2.scatter(x_pca[:,0],x_pca[:,2])
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC3")
ax2.set_title('PC1 vs PC3')
ax2.grid(True)
ax2.set_frame_on(False)
fig.show()

import plotly.express as px
features = ['E_solar', 'w10', 'ghi', 't2m']
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig = px.scatter(x_pca, x=0, y=1)
for i, feature in enumerate(features):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0]*4.5,
        y1=loadings[i, 1]*4.5
    )
    fig.add_annotation(
        x=loadings[i, 0]*5.5,
        y=loadings[i, 1]*5.5,
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )
plt.savefig('idkv3.png')
fig.show()

sn.pairplot(df_col, kind="reg")
plt.show()

from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
ax = plt.axes(projection='3d')
xline=x_pca[:,0]
yline=x_pca[:,1]
zline=x_pca[:,2]
ax.scatter3D(xline, yline, zline,c=zline,cmap='BrBG_r')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
# plt.savefig('3D Test.png',dpi=600)
plt.show()



#%% PCA plot SOLAR

correlation_coefficient = np.corrcoef(pca_oni, df_pc['pc3'])[0,1]

fig, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(date_range_oni, pca_oni, label='ONI ANOM', color='blue')
ax1.set_ylabel('ONI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-4,4)

ax2 = ax1.twinx()
ax2.plot(date_range_oni, df_pc['pc3'], label='PCA 3', color='green')
ax2.set_ylabel('PC 3', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-4,4)

# ax1.legend()
# ax2.legend()
plt.text(0.4, 0.08, f'Correlation Coefficient: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
# plt.text(0.4, 0.04, f'Explained Variance in PC3: {explained_variance_ratio[2]:.4f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.title('KEP - ONI vs Solar PC3')
plt.savefig('KEP solar ONI pc3.png', dpi=600)
fig.show()

print("Correlation Coefficient:", correlation_coefficient)
# print("Explained Variance:", explained_variance_ratio)

#%% cumulative explained variance plot SOLAR
pca = PCA().fit(scaled_data)
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(1, 5, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')
plt.xlabel('Fig 1 Number of Components')
plt.xticks(np.arange(0, 4, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
ax.grid(axis='x')
plt.savefig('KEP solar - cumulative explained variance plot.png', dpi = 600)
plt.show()







#%% Principal Component Analysis (Solar)
pca_modelled_solar = pd.concat([de_solar['E_solar'], de_wind_speed['w10'],
                        de_solar_ghi['ghi'], de_t2m['t2m']], axis=1)

columns_solar = ['E_solar', 'w10', 'ghi', 't2m']
solar_scaler = pca_modelled_solar[columns_solar]

scaler = StandardScaler()
scaler.fit(solar_scaler)
scaled_data = scaler.transform(solar_scaler)

#View in Frame
solar_scaled = pd.DataFrame(scaled_data, columns=columns_solar)
solar_scaled = solar_scaled.round(2)
solar_scaled

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
pca.fit(scaled_data)
reduced = pca.transform(scaled_data)
print('Original Dimensions: ',scaled_data.shape)
print('Reduced Dimensions: ',reduced.shape)





#%% Principal Component Analysis
date_range_soi = pd.date_range('1990-05-01', end_date, freq='M')
date_range_oni = pd.date_range(start_date, end_date, freq='M')

pca_soi = SMA_soi.dropna()
pca_oni = - x_oni # defintion for x_oni is for comparison to soi so take negative to actually reflect actual oni anomolies


#%% SOI PCA (wind)
scaler_soi_wind = StandardScaler()
scaled_wind = scaler_soi_wind.fit_transform(np.column_stack((de_wind['E_wind'].loc['1990-05':'2022-12'],
                                                             de_wind_speed['w10'].loc['1990-05':'2022-12'],
                                                             de_solar_ghi['ghi'].loc['1990-05':'2022-12'],
                                                             de_t2m['t2m'].loc['1990-05':'2022-12']
                                                             )))

pca = PCA(n_components=4)
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
pca_swind['pca3'] = principal_components_swind[:,2]
pca_swind['pca4'] = principal_components_swind[:,3]
# pca_swind['pca5'] = principal_components_swind[:,4]

PC_values = np.arange(pca.n_components_) + 1
plt.figure(figsize=(12,9))
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

correlation_coefficient = np.corrcoef(pca_soi, pca_swind['pca4'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(date_range_soi, pca_soi, label='SOI ANOM', color='blue')
ax1.set_ylabel('SOI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-30,30)

ax2 = ax1.twinx()
ax2.plot(date_range_soi, pca_swind['pca4'], label='PCA 3', color='green')
ax2.set_ylabel('PC 3', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-3,3)

# ax1.legend()
# ax2.legend()
plt.text(0.4, 0.08, f'Correlation Coefficient: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.text(0.4, 0.04, f'Explained Variance in PC3: {explained_variance_ratio[2]:.4f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.title('SOI vs Wind PC3')
# plt.savefig('KEP SOI PCA 3 (Wind).png', dpi=600)
plt.show()

print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)



#%% ONI PCA (wind)

scaler_oni_wind = StandardScaler()
scaled_wind = scaler_oni_wind.fit_transform(np.column_stack((de_wind['E_wind'],
                                                             de_wind_speed['w10'],
                                                             de_solar_ghi['ghi'],
                                                             de_t2m['t2m'],
                                                             pca_oni)))

pca = PCA(n_components=5)
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
pca_owind['pca3'] = principal_components_owind[:,2]
pca_owind['pca4'] = principal_components_owind[:,3]
pca_owind['pca5'] = principal_components_owind[:,4]

PC_values = np.arange(pca.n_components_) + 1
plt.figure(figsize=(12,9))
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

correlation_coefficient = np.corrcoef(pca_oni, pca_owind['pca3'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_oni, pca_oni, label='ONI ANOM', color='blue')
ax1.set_ylabel('ONI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-3,3)

ax2 = ax1.twinx()

ax2.plot(date_range_oni, pca_owind['pca3'], label='PCA 3', color='green')
ax2.set_ylabel('PC 3', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-3,3)

# ax1.legend()
# ax2.legend()
plt.text(0.4, 0.08, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.text(0.4, 0.04, f'Explained Variance in PC3: {explained_variance_ratio[2]:.4f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.title('ONI vs Wind PC3')
# plt.savefig('KEP ONI PCA 3 (Wind).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)



#%% SOI PCA (solar)
scaler_soi_solar = StandardScaler()
scaled_solar = scaler_soi_solar.fit_transform(np.column_stack((de_solar['E_solar'].loc['1990-05':'2022-12'],
                                                               de_wind_speed['w10'].loc['1990-05':'2022-12'],
                                                               de_solar_ghi['ghi'].loc['1990-05':'2022-12'],
                                                               de_t2m['t2m'].loc['1990-05':'2022-12'],
                                                               pca_soi)))


pca = PCA(n_components=5)
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
pca_ssolar['pca3'] = principal_components_ssolar[:,2]
pca_ssolar['pca4'] = principal_components_ssolar[:,3]
pca_ssolar['pca5'] = principal_components_ssolar[:,4]


PC_values = np.arange(pca.n_components_) + 1
plt.figure(figsize=(12,9))
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

correlation_coefficient = np.corrcoef(pca_soi, -pca_ssolar['pca2'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_soi, pca_soi, label='SOI ANOM', color='blue')
ax1.set_ylabel('SOI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-30,30)

ax2 = ax1.twinx()

ax2.plot(date_range_soi, - pca_ssolar['pca2'], label='PCA 2', color='green')
ax2.set_ylabel('PC 2', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-3,3)

# ax1.legend()
# ax2.legend()
plt.text(0.4, 0.08, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.text(0.4, 0.04, f'Explained Variance in PC2: {explained_variance_ratio[1]:.4f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.title('SOI vs Negative Solar PC2')
# plt.savefig('KEP SOI PCA 2 (neg Solar).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)


#%% ONI PCA (solar)

scaler_oni_solar = StandardScaler()
scaled_solar = scaler_oni_solar.fit_transform(np.column_stack((de_solar['E_solar'],
                                                               de_wind_speed['w10'],
                                                               de_solar_ghi['ghi'],
                                                               de_t2m['t2m'],
                                                               pca_oni)))

pca = PCA(n_components=5)
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
pca_osolar['pca3'] = principal_components_osolar[:,2]
pca_osolar['pca4'] = principal_components_osolar[:,3]
pca_osolar['pca5'] = principal_components_osolar[:,4]

PC_values = np.arange(pca.n_components_) + 1
plt.figure(figsize=(12,9))
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

correlation_coefficient = np.corrcoef(pca_oni, pca_osolar['pca2'])[0, 1]

fig, ax1 = plt.subplots(figsize=(10,8))

ax1.plot(date_range_oni, pca_oni, label='ONI ANOM', color='blue')
ax1.set_ylabel('ONI ANOM', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(-3,3)

ax2 = ax1.twinx()

ax2.plot(date_range_oni, pca_osolar['pca2'], label='PCA 2', color='green')
ax2.set_ylabel('PC 2', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-3,3)

# ax1.legend()
# ax2.legend()
plt.text(0.4, 0.08, f'Correlation: {correlation_coefficient:.3f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.text(0.4, 0.04, f'Explained Variance in PC2: {explained_variance_ratio[1]:.4f}', transform=ax1.transAxes, fontsize=11, verticalalignment='top')
plt.title('ONI vs Solar PC2')
# plt.savefig('KEP ONI PCA 2 (Solar).png', dpi=600)
plt.show()



print("Correlation Coefficient:", correlation_coefficient)
print("Explained Variance:", explained_variance_ratio)


#%% NEMOSIS
# Solar
# print(defaults.dynamic_tables)
# print(defaults.static_tables)
import pandas as pd
import numpy as np
from datetime import timedelta
from nemosis import static_table, dynamic_data_compiler
import plotly.express as px

raw_data_cache = 'C:/Users/liame/OneDrive/Uni/2023/T2 2023/Thesis B/GUI'

scada_data = dynamic_data_compiler(start_time='2022/12/01 00:00:00',
                                   end_time='2022/12/07 00:00:00',
                                   table_name='DISPATCH_UNIT_SCADA',
                                   raw_data_location=raw_data_cache)

scada_data = scada_data[scada_data['DUID'] == 'KEPSF1']
# scada_data.set_index('SETTLEMENTDATE', inplace=False)
# hourly_scada = scada_data['SCADAVALUE'].resample('1H').mean()
scada_data.head()

# Plot the results.
plt.plot(scada_data['SETTLEMENTDATE'], scada_data['SCADAVALUE'])
plt.plot(modelled['E_solar'].loc['2022-12-01':'2022-12-07'])

#%%
# Wind
# print(defaults.dynamic_tables)
# print(defaults.static_tables)
import pandas as pd
import numpy as np
from datetime import timedelta
from nemosis import static_table, dynamic_data_compiler
import plotly.express as px

raw_data_cache = 'C:/Users/liame/OneDrive/Uni/2023/T2 2023/Thesis B/GUI'

scada_data = dynamic_data_compiler(start_time='2022/05/01 00:00:00',
                                   end_time='2022/05/07 00:00:00',
                                   table_name='DISPATCH_UNIT_SCADA',
                                   raw_data_location=raw_data_cache)

scada_data = scada_data[scada_data['DUID'] == 'KEPWF1']
nem_gen = pd.DataFrame({
    'time': pd.date_range(start='2022-05-01 00:00:05', end='2022-05-07 00:00:00', freq='5T'),
    'SCADAVALUE': scada_data['SCADAVALUE']})

nem_gen.set_index('time', inplace=True)
hourly_scada = nem_gen.resample('1H').mean()
nem_gen.head()
hourly_scada.head()

# Plot the results.
plt.plot(hourly_scada.index, hourly_scada['SCADAVALUE']/wind_capacity)
plt.plot(modelled['2022-05-01':'2022-05-06'].index,
         modelled['E_wind'].loc['2022-05-01':'2022-05-06']/wind_capacity)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%b'))
plt.xticks()
plt.title('CF for KEP in Winter 2022')

#%%

site_info = static_table(table_1, raw_data_cache)
duid_wind = site_info.loc[site_info['Station Name'] == site, 'DUID'].values[1]
print(defaults.table_columns['DISPATCH_UNIT_SCADA'])

wind_gen_data = dynamic_data_compiler(start_time, end_time, table_2, raw_data_cache,
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

solar_df = pd.DataFrame(solar_gen_data)
solar_gen = solar_df[solar_df['SETTLEMENTDATE'].dt.minute == 0]
solar_gen.reset_index(drop = True, inplace = True)

wind_df = pd.DataFrame(wind_gen_data)
wind_gen = wind_df[wind_df['SETTLEMENTDATE'].dt.minute == 0]
wind_gen.reset_index(drop = True, inplace = True)

plt.figure(figsize=(16,9))
plt.plot(solar_gen['SCADAVALUE'])
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%b %H:%M'))
plt.title('Solar')
plt.show()

plt.figure(figsize=(16,9))
plt.plot(wind_gen['SCADAVALUE'])
plt.plot(solar_gen['SCADAVALUE'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%b %H:%M'))
plt.title('Wind and Solar')
plt.show()

plt.figure(figsize=(12,8))
plt.hist(modelled['E_solar'].loc['2022-12-02'])
plt.hist(solar_gen['SCADAVALUE'])
plt.show()

plt.figure(figsize=(12,8))
plt.hist(modelled['E_wind'].loc['2022-12-02'])
plt.hist(wind_gen['SCADAVALUE'])
plt.show()



#%%


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

plt.figure(figsize=(12,8))
plt.hist(modelled['E_solar'].loc['2022-12-02'])
plt.hist(solar_gen_data['SCADAVALUE'])
plt.show()

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

#%%



