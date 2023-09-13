# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:37:51 2023

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

mtemer1 = xr.open_dataset('mtemer1.grib', engine = 'cfgrib', chunks = 1000)
mtemer2 = xr.open_dataset('mtemer2.grib', engine = 'cfgrib', chunks = 1000)
mtemer3 = xr.open_dataset('mtemer3.grib', engine = 'cfgrib', chunks = 1000)
mtemer4 = xr.open_dataset('mtemer4.grib', engine = 'cfgrib', chunks = 1000)
mtemer5 = xr.open_dataset('mtemer5.grib', engine = 'cfgrib', chunks = 1000)
mtemer6 = xr.open_dataset('mtemer6.grib', engine = 'cfgrib', chunks = 1000)
mtemer7 = xr.open_dataset('mtemer7.grib', engine = 'cfgrib', chunks = 1000)
mtemer8 = xr.open_dataset('mtemer8.grib', engine = 'cfgrib', chunks = 1000)
mtemer9 = xr.open_dataset('mtemer9.grib', engine = 'cfgrib', chunks = 1000)

#%% Creating numpy arrays of relevent parameters
mtemer_u1 = mtemer1.u10.values[:,0,0]
mtemer_u2 = mtemer2.u10.values[:,0,0]
mtemer_u3 = mtemer3.u10.values[:,0,0]
mtemer_u4 = mtemer4.u10.values[:,0,0]
mtemer_u5 = mtemer5.u10.values[:,0,0]
mtemer_u6 = mtemer6.u10.values[:,0,0]
mtemer_u7 = mtemer7.u10.values[:,0,0]
mtemer_u8 = mtemer8.u10.values[:,0,0]
mtemer_u9 = mtemer9.u10.values[:,0,0]

mtemer_v1 = mtemer1.v10.values[:,0,0]
mtemer_v2 = mtemer2.v10.values[:,0,0]
mtemer_v3 = mtemer3.v10.values[:,0,0]
mtemer_v4 = mtemer4.v10.values[:,0,0]
mtemer_v5 = mtemer5.v10.values[:,0,0]
mtemer_v6 = mtemer6.v10.values[:,0,0]
mtemer_v7 = mtemer7.v10.values[:,0,0]
mtemer_v8 = mtemer8.v10.values[:,0,0]
mtemer_v9 = mtemer9.v10.values[:,0,0]

# mtemer_t1 = mtemer1.t2m.values[:,0,0]
# mtemer_t2 = mtemer2.t2m.values[:,0,0]
# mtemer_t3 = mtemer3.t2m.values[:,0,0]
# mtemer_t4 = mtemer4.t2m.values[:,0,0]
# mtemer_t5 = mtemer5.t2m.values[:,0,0]
# mtemer_t6 = mtemer6.t2m.values[:,0,0]
# mtemer_t7 = mtemer7.t2m.values[:,0,0]
# mtemer_t8 = mtemer8.t2m.values[:,0,0]
# mtemer_t9 = mtemer9.t2m.values[:,0,0]


u = np.concatenate([mtemer_u1, mtemer_u2, mtemer_u3, mtemer_u4, mtemer_u5, mtemer_u6, mtemer_u7,
                    mtemer_u8, mtemer_u9])

v = np.concatenate([mtemer_v1, mtemer_v2, mtemer_v3, mtemer_v4, mtemer_v5, mtemer_v6, mtemer_v7,
                    mtemer_v8, mtemer_v9])

# t = np.concatenate([mtemer_t1, mtemer_t2, mtemer_t3, mtemer_t4, mtemer_t5, mtemer_t6, mtemer_t7,
#                     mtemer_t8, mtemer_t9]) - 273.15



#%% Variables

# General

Australian = pytz.timezone('Australia/Queensland')
latitude = -17.17
longitude = 145.38

# Wind
# (Vestas V117 3.45MW)
wind_capacity_V117 = 127.7    # MW
hub_height_V117 = 90          # m
P_rated_wind_V117 = 3.45      # MW
v_rated_V117 = 12.5           # m/s
v_cut_in_V117 = 3             # m/s
v_cut_out_V117 = 25           # m/s

# (Vestas V117 3.3MW)
wind_capacity_V112 = 52.8     # MW
hub_height_V112 = 84          # m
P_rated_wind_V112 = 3.3       # MW
v_rated_V112 = 13             # m/s
v_cut_in_V112 = 3             # m/s
v_cut_out_V112 = 25           # m/s


#%% Wind Generation
number_of_turbines_V117 = wind_capacity_V117 // P_rated_wind_V117
number_of_turbines_V112 = wind_capacity_V112 // P_rated_wind_V112


# Wind Profile Calculation This only for region betwix cut in and rated wind speeds
# V117
h1 = 10
h2_V117 = hub_height_V117
z0 = 0.05
v1 = np.sqrt(u**2 + v**2)
v2_V117 = v1 * (math.log((h2_V117)/(z0)))/(math.log((h1)/(z0)))

a_V117 = P_rated_wind_V117 * 1000000 / (v_rated_V117**3 - v_cut_in_V117**3)
b_V117 = (v_rated_V117**3) / (v_rated_V117**3 - v_cut_in_V117**3)

E_wind_V117 = np.copy(v2_V117)
E_wind_V117[E_wind_V117 < v_cut_in_V117] = 0
E_wind_V117[(E_wind_V117 > v_cut_in_V117) & (E_wind_V117 < v_rated_V117)] = (a_V117 * E_wind_V117[(E_wind_V117 > v_cut_in_V117) & (E_wind_V117 < v_rated_V117)]**3 - b_V117 * P_rated_wind_V117)/1000000
E_wind_V117[(E_wind_V117 > v_rated_V117) & (E_wind_V117 < v_cut_out_V117)] = P_rated_wind_V117
E_wind_V117[E_wind_V117 > v_cut_out_V117] = 0

E_wind_V117 *= number_of_turbines_V117

# V112
h2_V112 = hub_height_V112
v2_V112 = v1 * (math.log((h2_V112)/(z0)))/(math.log((h1)/(z0)))

a_V112 = P_rated_wind_V112 * 1000000 / (v_rated_V112**3 - v_cut_in_V112**3)
b_V112 = (v_rated_V112**3) / (v_rated_V112**3 - v_cut_in_V112**3)

E_wind_V112 = np.copy(v2_V117)
E_wind_V112[E_wind_V112 < v_cut_in_V112] = 0
E_wind_V112[(E_wind_V112 > v_cut_in_V112) & (E_wind_V112 < v_rated_V112)] = (a_V112 * E_wind_V112[(E_wind_V112 > v_cut_in_V112) & (E_wind_V112 < v_rated_V112)]**3 - b_V112 * P_rated_wind_V112)/1000000
E_wind_V112[(E_wind_V112 > v_rated_V112) & (E_wind_V112 < v_cut_out_V112)] = P_rated_wind_V112
E_wind_V112[E_wind_V112 > v_cut_out_V112] = 0

E_wind_V112 *= number_of_turbines_V112
E_wind = E_wind_V117 + E_wind_V112

start_date = datetime.datetime(1990, 1, 1)
end_date = datetime.datetime(2022, 12, 31, 23, 0)

dt = []

while start_date <= end_date:
    dt.append(start_date)
    start_date += datetime.timedelta(hours=1)


raw_weather = pd.DataFrame({'time': dt, 'w10': v1})
raw_weather['time'] = pd.to_datetime(raw_weather['time'])
raw_weather.set_index('time', inplace=True)
raw_weather.index = raw_weather.index.tz_localize('UTC')
raw_weather.index = raw_weather.index.tz_convert(Australian)

#%% Turbine Power Curve
# V117
step = 0.5
cut_in_wind_speed_V117 = np.arange(0, v_cut_in_V117, step)
cubic_wind_speed_V117 = np.arange(v_cut_in_V117, v_rated_V117, step)
rated_wind_speed_V117 = np.arange(v_rated_V117, v_cut_out_V117, step)
cut_out_wind_speed_V117 = np.arange(v_cut_out_V117, v_cut_out_V117 + 5*step, step)

cubic_pc_V117 = (a_V117 * cubic_wind_speed_V117**3 - b_V117 * P_rated_wind_V117)/1000000
cut_in_pc_V117 = np.zeros_like(cut_in_wind_speed_V117)
rated_pc_V117 = np.zeros_like(rated_wind_speed_V117) + P_rated_wind_V117
cut_out_pc_V117 = np.zeros_like(cut_out_wind_speed_V117)

fig, ax = plt.subplots(figsize=(9,6))
ax.stem(cut_in_wind_speed_V117, cut_in_pc_V117, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Below Cut-In')
ax.stem(cubic_wind_speed_V117, cubic_pc_V117, markerfmt = 'bo', linefmt = 'b', basefmt = 'none', label = 'Cubic Model')
ax.stem(rated_wind_speed_V117, rated_pc_V117, markerfmt = 'go', linefmt = 'g', basefmt = 'none', label = 'Rated')
ax.stem(cut_out_wind_speed_V117, cut_out_pc_V117, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Above Cut-Out')
# ax.set_ylim(0,4)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Turbine Output (MW)')
ax.set_title('Modelled Power Curve for Vestas V117 3.45 MW (Mt Emerald)')
ax.legend()
plt.savefig('Mt Emerald Turbine Power Curve (3_45 MW)', dpi = 600)

# V112
step = 0.5
cut_in_wind_speed_V112 = np.arange(0, v_cut_in_V112, step)
cubic_wind_speed_V112 = np.arange(v_cut_in_V112, v_rated_V112, step)
rated_wind_speed_V112 = np.arange(v_rated_V112, v_cut_out_V112, step)
cut_out_wind_speed_V112 = np.arange(v_cut_out_V112, v_cut_out_V112 + 5*step, step)

cubic_pc_V112 = (a_V112 * cubic_wind_speed_V112**3 - b_V112 * P_rated_wind_V112)/1000000
cut_in_pc_V112 = np.zeros_like(cut_in_wind_speed_V112)
rated_pc_V112 = np.zeros_like(rated_wind_speed_V112) + P_rated_wind_V112
cut_out_pc_V112 = np.zeros_like(cut_out_wind_speed_V112)

fig, ax = plt.subplots(figsize=(9,6))
ax.stem(cut_in_wind_speed_V112, cut_in_pc_V112, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Below Cut-In')
ax.stem(cubic_wind_speed_V112, cubic_pc_V112, markerfmt = 'bo', linefmt = 'b', basefmt = 'none', label = 'Cubic Model')
ax.stem(rated_wind_speed_V112, rated_pc_V112, markerfmt = 'go', linefmt = 'g', basefmt = 'none', label = 'Rated')
ax.stem(cut_out_wind_speed_V112, cut_out_pc_V112, markerfmt = 'ro', linefmt = 'r', basefmt = 'none', label = 'Above Cut-Out')
# ax.set_ylim(0,4)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Turbine Output (MW)')
ax.set_title('Modelled Power Curve for Vestas V112 3.3 MW (Mt Emerald)')
ax.legend()

plt.savefig('Mt Emerald Turbine Power Curve (3_3 MW)', dpi = 600)


#%% Wind speed histrogram at 10m
plt.figure(figsize=(9,6))
plt.hist(v1, bins=31, edgecolor='black')
plt.xlabel('Wind Speed at 10m (m/s)')
plt.ylabel('Frequency')
plt.title('Histogram of 10m Wind Speed at Mt Emerald Wind Farm')
plt.xlim(0,15)
plt.savefig('10m WindSpeed Histogram Mt Emerald', dpi = 600)

#%% Wind speed histrogram at 110m
plt.figure(figsize=(9,6))
plt.hist(v2_V117, bins=31, edgecolor='black')
plt.xlabel('Wind Speed at 90m (m/s)')
plt.ylabel('Frequency')
plt.title('Histogram of Wind Speed at Hub Height for V117 3.45 MW (Mt Emerald Wind Farm)')
plt.xlim(0,15)
plt.savefig('90m WindSpeed Histogram Mt Emerald', dpi = 600)

#%% Wind speed histrogram at 115m
plt.figure(figsize=(9,6))
plt.hist(v2_V112, bins=31, edgecolor='black')
plt.xlabel('Wind Speed at 84m (m/s)')
plt.ylabel('Frequency')
plt.title('Histogram of Wind Speed at Hub Height for V112 3.3 MW (Mt Emerald Wind Farm)')
plt.xlim(0,15)
plt.savefig('84m WindSpeed Histogram Mt Emerald', dpi = 600)



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
modelled = pd.DataFrame({'time':dt, 'E_wind':E_wind})
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
                                 'de_wind': de_wind['E_wind'],
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
                                })

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
plt.title('Average Monthly Wind Generation at Mount Emerald Wind Farm for varying La Nina Conditions')

plt.xticks(index + bar_width/2, month_names, rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('Monthly Average Wind Generation (Mount Emerald Wind Farm).png', dpi=600)
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




#%%
# Strong LN
x_soi_strong = monthly_modelled[monthly_modelled['Year'].isin(strong_soi_years)]['SOI']
x_soi_moderate = monthly_modelled[monthly_modelled['Year'].isin(moderate_soi_years)]['SOI']
x_soi_weak = monthly_modelled[monthly_modelled['Year'].isin(weak_soi_years)]['SOI']
x_soi_neutral = monthly_modelled[monthly_modelled['Year'].isin(neutral_soi_years)]['SOI']

# y_solar_strong_soi = monthly_modelled[monthly_modelled['Year'].isin(strong_soi_years)]['de_solar']
y_wind_strong_soi = monthly_modelled[monthly_modelled['Year'].isin(strong_soi_years)]['de_wind']
# y_solar_moderate_soi = monthly_modelled[monthly_modelled['Year'].isin(moderate_soi_years)]['de_solar']
y_wind_moderate_soi = monthly_modelled[monthly_modelled['Year'].isin(moderate_soi_years)]['de_wind']
# y_solar_weak_soi = monthly_modelled[monthly_modelled['Year'].isin(weak_soi_years)]['de_solar']
y_wind_weak_soi = monthly_modelled[monthly_modelled['Year'].isin(weak_soi_years)]['de_wind']
# y_solar_neutral_soi = monthly_modelled[monthly_modelled['Year'].isin(neutral_soi_years)]['de_solar']
y_wind_neutral_soi = monthly_modelled[monthly_modelled['Year'].isin(neutral_soi_years)]['de_wind']

x_oni_strong = - monthly_modelled[monthly_modelled['Year'].isin(strong_oni_years)]['ONI']
x_oni_moderate = - monthly_modelled[monthly_modelled['Year'].isin(moderate_oni_years)]['ONI']
x_oni_weak = - monthly_modelled[monthly_modelled['Year'].isin(weak_oni_years)]['ONI']
x_oni_neutral = - monthly_modelled[monthly_modelled['Year'].isin(neutral_oni_years)]['ONI']

# y_solar_strong_oni = monthly_modelled[monthly_modelled['Year'].isin(strong_oni_years)]['de_solar']
y_wind_strong_oni = monthly_modelled[monthly_modelled['Year'].isin(strong_oni_years)]['de_wind']
# y_solar_moderate_oni = monthly_modelled[monthly_modelled['Year'].isin(moderate_oni_years)]['de_solar']
y_wind_moderate_oni = monthly_modelled[monthly_modelled['Year'].isin(moderate_oni_years)]['de_wind']
# y_solar_weak_oni = monthly_modelled[monthly_modelled['Year'].isin(weak_oni_years)]['de_solar']
y_wind_weak_oni = monthly_modelled[monthly_modelled['Year'].isin(weak_oni_years)]['de_wind']
# y_solar_neutral_oni = monthly_modelled[monthly_modelled['Year'].isin(neutral_oni_years)]['de_solar']
y_wind_neutral_oni = monthly_modelled[monthly_modelled['Year'].isin(neutral_oni_years)]['de_wind']



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

# SOI vs ONI


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
plt.title('SOI vs Wind PC1 (Mt Emerald Wind Farm)')
plt.savefig('Mt Emerald SOI PCA 1 (Wind).png', dpi=600)
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
plt.title('SOI vs Wind PC2 (Mt Emerald Wind Farm)')
plt.savefig('Mt Emerald SOI PCA 2 (Wind).png', dpi=600)
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
plt.title('ONI vs Wind PC1 (Mt Emerald Wind Farm)')
plt.savefig('Mt Emerald ONI PCA 1 (Wind).png', dpi=600)
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
plt.title('ONI vs Wind PC2 (Mt Emerald Wind Farm)')
plt.savefig('Mt Emerald ONI PCA 2 (Wind).png', dpi=600)
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

site = 'Mount Emerald Wind Farm'
technology = 'Wind'
start_time = '2022/12/02 00:00:00'
end_time = '2022/12/03 00:00:00'
table_1 = 'Generators and Scheduled Loads'
table_2 = 'DISPATCH_UNIT_SCADA'
table_3 = 'BIDPEROFFER_D'
raw_data_cache = 'C:/Users/liame/OneDrive/Uni/2023/T2 2023/Thesis B/GUI'

site_info = static_table(table_1, raw_data_cache)
duid_wind = 'MEWF1'
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


wind_df = pd.DataFrame(wind_gen_data)
wind_gen = wind_df[wind_df['SETTLEMENTDATE'].dt.minute == 0]
wind_gen.reset_index(drop = True, inplace = True)



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


