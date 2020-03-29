#!/usr/bin/env python3

"""This file define common functions and parameters for
   collecting and processing weather data.
"""

from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt
import pickle
import os.path
import json
import sys

# API KEYS for "darksky" web
API_KEY_Austin = '4778c8e06b15f3691b07cbccda46b8fe'
API_KEY_Llano = 'c499aaf8c23a151dafffb5135240df81'
API_KEY_SanAntonio = 'd2480b9c724e6685640f49b7db6d6f37'
API_KEY_Brenham = 'b898ce9e4ad35c66c2d2dae30f31c722'
API_KEY_Temple = 'd42bc3137b5e37d84e89d6a695d6b740'

# longitude and latitude for cities
# Austin is the city to predict weather
# rest of cities are surrounding cities
LOCATION_Austin = '30.2711,-97.7437'
LOCATION_Llano = '30.7102,-98.6841'
LOCATION_SanAntonio = '29.5444,-98.4147'
LOCATION_Brenham ='30.1768,-96.3965'
LOCATION_Temple = '31.0982,-97.3428'

BASE_URL = "https://api.darksky.net/forecast/{}/{},{}T00:00:00?exclude=currently,flag"
#https://api.darksky.net/forecast/4778c8e06b15f3691b07cbccda46b8fe/30.2711,-97.7437,2016-05-16T00:00:00?exclude=currently,hourly,flags
#https://api.darksky.net/forecast/4778c8e06b15f3691b07cbccda46b8fe/30.2711,-97.7437,2016-05-16T00:00:00?exclude=currently,flags


# https://darksky.net/dev/docs#time-machine-request

# weather features
# _A: Austin, _L: Llano, _S: SanAntonio, _B: Brenham, _T: Temple
# n_*: night features at 10pm
features = ["date", "daytime", 
        "icon_A", "precipProbability_A", "precipIntensity_A", "humidity_A", "cloudCover_A", "visibility_A", "temperatureMin_A", "temperatureMax_A", "windSpeed_A", "windBearing_A", "pressure_A", "dewPoint_A", 
        "n_icon_A", "n_precipProbability_A", "n_precipIntensity_A", "n_humidity_A", "n_cloudCover_A", "n_visibility_A", "n_temperature_A", "n_windSpeed_A", "n_windBearing_A", "n_pressure_A", "n_dewPoint_A",

        "icon_L", "precipProbability_L", "precipIntensity_L", "humidity_L", "cloudCover_L", "visibility_L", "temperatureMin_L", "temperatureMax_L", "windSpeed_L", "windBearing_L", "pressure_L", "dewPoint_L", 
        "n_icon_L", "n_precipProbability_L", "n_precipIntensity_L", "n_humidity_L", "n_cloudCover_L", "n_visibility_L", "n_temperature_L", "n_windSpeed_L", "n_windBearing_L", "n_pressure_L", "n_dewPoint_L",

        "icon_S", "precipProbability_S", "precipIntensity_S", "humidity_S", "cloudCover_S", "visibility_S", "temperatureMin_S", "temperatureMax_S", "windSpeed_S", "windBearing_S", "pressure_S", "dewPoint_S", 
        "n_icon_S", "n_precipProbability_S", "n_precipIntensity_S", "n_humidity_S", "n_cloudCover_S", "n_visibility_S", "n_temperature_S", "n_windSpeed_S", "n_windBearing_S", "n_pressure_S", "n_dewPoint_S",

        "icon_B", "precipProbability_B", "precipIntensity_B", "humidity_B", "cloudCover_B", "visibility_B", "temperatureMin_B", "temperatureMax_B", "windSpeed_B", "windBearing_B", "pressure_B", "dewPoint_B", 
        "n_icon_B", "n_precipProbability_B", "n_precipIntensity_B", "n_humidity_B", "n_cloudCover_B", "n_visibility_B", "n_temperature_B", "n_windSpeed_B", "n_windBearing_B", "n_pressure_B", "n_dewPoint_B",

        "icon_T", "precipProbability_T", "precipIntensity_T", "humidity_T", "cloudCover_T", "visibility_T", "temperatureMin_T", "temperatureMax_T", "windSpeed_T", "windBearing_T", "pressure_T", "dewPoint_T", 
        "n_icon_T", "n_precipProbability_T", "n_precipIntensity_T", "n_humidity_T", "n_cloudCover_T", "n_visibility_T", "n_temperature_T", "n_windSpeed_T", "n_windBearing_T", "n_pressure_T", "n_dewPoint_T"
]

icons = {"clear-day":0, "clear-night":0, "partly-cloudy-day":1, "partly-cloudy-night":1, "cloudy":2, "rain":3, "wind":4, "fog":5, "sleet":6, "snow":7}

# namedtuple class for features
DailySummary = namedtuple("DailySummary", features)

# max previous days 
max_N = 3

half_his_days = 10
days_a_year = 365

# skip first  3years + one day in leap year + half_history_days + max_N
# these days have no historic average data
# his_years is num of years to calculate average temperature
his_years = 3
start_train_day = days_a_year * his_years + 1 + half_his_days + max_N

"""
response.json format:
{
    "timezone": "America/Chicago",
    "daily": {
        "data": [
            {
                "time": 1563339600,
                "icon": "partly-cloudy-day"
            }
        ]
    }
}
"""

def get_daily_and_ten_data(response, last_daily_data, last_ten_pm_data, date_format, city):
    """get daily weather data and 10pm weather data from web response"""
    daily_data = response.json()["daily"]["data"][0]
    ten_pm_data = response.json()["hourly"]["data"][-2]

    # fix missing data
    missing_icon = ["pressure", "precipProbability", "precipIntensity", "humidity", "cloudCover", "visibility", "windSpeed", "dewPoint"]
    for i in missing_icon:
        if i not in daily_data:
            print(f'### ERROR: {i} is missing on {city} {date_format}')
            daily_data[i] = last_daily_data[i]
        if i not in ten_pm_data:
            print(f'### ERROR: {i} is missing at ten on {city} {date_format}')
            nine_pm_data = response.json()["hourly"]["data"][-3]
            if i not in nine_pm_data:
                ten_pm_data[i] = last_ten_pm_data[i]
            else:
                ten_pm_data[i] = 0.5 * (last_ten_pm_data[i] + nine_pm_data[i])

    # correct icon
    if "icon" not in daily_data:
        print(f'### ERROR: icon is missing on {city} {date_format}')
        daily_data["icon"] = response.json()["hourly"]["icon"]
    # windBearing could be missing if windSpeed is 0
    if "windBearing" not in daily_data:
        daily_data["windBearing"] = 0
    if "windBearing" not in ten_pm_data:
        ten_pm_data["windBearing"] = 0

    return daily_data, ten_pm_data, daily_data, ten_pm_data

def collect_weather_data(url, target_date, days):
    """request weahter data from web and add them into historical_data_df"""
    historical_data = []

    last_daily_data_A = {}
    last_daily_data_L = {}
    last_daily_data_S = {}
    last_daily_data_B = {}
    last_daily_data_T = {}

    last_ten_pm_data_A = {}
    last_ten_pm_data_L = {}
    last_ten_pm_data_S = {}
    last_ten_pm_data_B = {}
    last_ten_pm_data_T = {}

    # num of times failed to get response from web 
    failed_times = 0

    for i in range(days):
        date_format = target_date.strftime('%Y-%m-%d')
        print(f'{date_format}')

        request_Austin = BASE_URL.format(API_KEY_Austin, LOCATION_Austin, date_format)
        request_Llano = BASE_URL.format(API_KEY_Llano, LOCATION_Llano, date_format)
        request_SanAntonio= BASE_URL.format(API_KEY_SanAntonio, LOCATION_SanAntonio, date_format)
        request_Brenham = BASE_URL.format(API_KEY_Brenham, LOCATION_Brenham, date_format)
        request_Temple = BASE_URL.format(API_KEY_Temple, LOCATION_Temple, date_format)

        response_A = requests.get(request_Austin)
        response_L = requests.get(request_Llano)
        response_S = requests.get(request_SanAntonio)
        response_B = requests.get(request_Brenham)
        response_T = requests.get(request_Temple)

        if response_A.status_code == 200 and response_L.status_code == 200 and response_S.status_code == 200 and response_B.status_code == 200 and response_T.status_code == 200:
            # last_daily_data and last_ten_pm_data is from last loop (last day)
            daily_data_A, ten_pm_data_A, last_daily_data_A, last_ten_pm_data_A = get_daily_and_ten_data(response_A, last_daily_data_A, last_ten_pm_data_A, date_format, "A")
            daily_data_L, ten_pm_data_L, last_daily_data_L, last_ten_pm_data_L = get_daily_and_ten_data(response_L, last_daily_data_L, last_ten_pm_data_L, date_format, "L")
            daily_data_S, ten_pm_data_S, last_daily_data_S, last_ten_pm_data_S = get_daily_and_ten_data(response_S, last_daily_data_S, last_ten_pm_data_S, date_format, "S")
            daily_data_B, ten_pm_data_B, last_daily_data_B, last_ten_pm_data_B = get_daily_and_ten_data(response_B, last_daily_data_B, last_ten_pm_data_B, date_format, "B")
            daily_data_T, ten_pm_data_T, last_daily_data_T, last_ten_pm_data_T = get_daily_and_ten_data(response_T, last_daily_data_T, last_ten_pm_data_T, date_format, "T")
            # create new DailySummary namedtuple and add it into historical_data
            historical_data.append(DailySummary(
                date = date_format,
                daytime = round((daily_data_A['sunsetTime'] - daily_data_A['sunriseTime']) / 3600, 2),

                icon_A = icons[daily_data_A['icon']],
                precipProbability_A = daily_data_A['precipProbability'],
                precipIntensity_A = daily_data_A['precipIntensity'],
                humidity_A = daily_data_A['humidity'],
                cloudCover_A = daily_data_A['cloudCover'],
                visibility_A = daily_data_A['visibility'],
                temperatureMin_A = daily_data_A['temperatureMin'],
                temperatureMax_A = daily_data_A['temperatureMax'],
                windSpeed_A = daily_data_A['windSpeed'],
                windBearing_A = daily_data_A['windBearing'],
                pressure_A = daily_data_A['pressure'],
                dewPoint_A = daily_data_A['dewPoint'],
                n_icon_A = icons[ten_pm_data_A['icon']],
                n_precipProbability_A = ten_pm_data_A['precipProbability'],
                n_precipIntensity_A = ten_pm_data_A['precipIntensity'],
                n_humidity_A = ten_pm_data_A['humidity'],
                n_cloudCover_A = ten_pm_data_A['cloudCover'],
                n_visibility_A = ten_pm_data_A['visibility'],
                n_temperature_A = ten_pm_data_A['temperature'],
                n_windSpeed_A = ten_pm_data_A['windSpeed'],
                n_windBearing_A = ten_pm_data_A['windBearing'],
                n_pressure_A = ten_pm_data_A['pressure'],
                n_dewPoint_A = ten_pm_data_A['dewPoint'],

                icon_L = icons[daily_data_L['icon']],
                precipProbability_L = daily_data_L['precipProbability'],
                precipIntensity_L = daily_data_L['precipIntensity'],
                humidity_L = daily_data_L['humidity'],
                cloudCover_L = daily_data_L['cloudCover'],
                visibility_L = daily_data_L['visibility'],
                temperatureMin_L = daily_data_L['temperatureMin'],
                temperatureMax_L = daily_data_L['temperatureMax'],
                windSpeed_L = daily_data_L['windSpeed'],
                windBearing_L = daily_data_L['windBearing'],
                pressure_L = daily_data_L['pressure'],
                dewPoint_L = daily_data_L['dewPoint'],
                n_icon_L = icons[ten_pm_data_L['icon']],
                n_precipProbability_L = ten_pm_data_L['precipProbability'],
                n_precipIntensity_L = ten_pm_data_L['precipIntensity'],
                n_humidity_L = ten_pm_data_L['humidity'],
                n_cloudCover_L = ten_pm_data_L['cloudCover'],
                n_visibility_L = ten_pm_data_L['visibility'],
                n_temperature_L = ten_pm_data_L['temperature'],
                n_windSpeed_L = ten_pm_data_L['windSpeed'],
                n_windBearing_L = ten_pm_data_L['windBearing'],
                n_pressure_L = ten_pm_data_L['pressure'],
                n_dewPoint_L = ten_pm_data_L['dewPoint'],


                icon_S = icons[daily_data_S['icon']],
                precipProbability_S = daily_data_S['precipProbability'],
                precipIntensity_S = daily_data_S['precipIntensity'],
                humidity_S = daily_data_S['humidity'],
                cloudCover_S = daily_data_S['cloudCover'],
                visibility_S = daily_data_S['visibility'],
                temperatureMin_S = daily_data_S['temperatureMin'],
                temperatureMax_S = daily_data_S['temperatureMax'],
                windSpeed_S = daily_data_S['windSpeed'],
                windBearing_S = daily_data_S['windBearing'],
                pressure_S = daily_data_S['pressure'],
                dewPoint_S = daily_data_S['dewPoint'],
                n_icon_S = icons[ten_pm_data_S['icon']],
                n_precipProbability_S = ten_pm_data_S['precipProbability'],
                n_precipIntensity_S = ten_pm_data_S['precipIntensity'],
                n_humidity_S = ten_pm_data_S['humidity'],
                n_cloudCover_S = ten_pm_data_S['cloudCover'],
                n_visibility_S = ten_pm_data_S['visibility'],
                n_temperature_S = ten_pm_data_S['temperature'],
                n_windSpeed_S = ten_pm_data_S['windSpeed'],
                n_windBearing_S = ten_pm_data_S['windBearing'],
                n_pressure_S = ten_pm_data_S['pressure'],
                n_dewPoint_S = ten_pm_data_S['dewPoint'],
                

                icon_B = icons[daily_data_B['icon']],
                precipProbability_B = daily_data_B['precipProbability'],
                precipIntensity_B = daily_data_B['precipIntensity'],
                humidity_B = daily_data_B['humidity'],
                cloudCover_B = daily_data_B['cloudCover'],
                visibility_B = daily_data_B['visibility'],
                temperatureMin_B = daily_data_B['temperatureMin'],
                temperatureMax_B = daily_data_B['temperatureMax'],
                windSpeed_B = daily_data_B['windSpeed'],
                windBearing_B = daily_data_B['windBearing'],
                pressure_B = daily_data_B['pressure'],
                dewPoint_B = daily_data_B['dewPoint'],
                n_icon_B = icons[ten_pm_data_B['icon']],
                n_precipProbability_B = ten_pm_data_B['precipProbability'],
                n_precipIntensity_B = ten_pm_data_B['precipIntensity'],
                n_humidity_B = ten_pm_data_B['humidity'],
                n_cloudCover_B = ten_pm_data_B['cloudCover'],
                n_visibility_B = ten_pm_data_B['visibility'],
                n_temperature_B = ten_pm_data_B['temperature'],
                n_windSpeed_B = ten_pm_data_B['windSpeed'],
                n_windBearing_B = ten_pm_data_B['windBearing'],
                n_pressure_B = ten_pm_data_B['pressure'],
                n_dewPoint_B = ten_pm_data_B['dewPoint'],


                icon_T = icons[daily_data_T['icon']],
                precipProbability_T = daily_data_T['precipProbability'],
                precipIntensity_T = daily_data_T['precipIntensity'],
                humidity_T = daily_data_T['humidity'],
                cloudCover_T = daily_data_T['cloudCover'],
                visibility_T = daily_data_T['visibility'],
                temperatureMin_T = daily_data_T['temperatureMin'],
                temperatureMax_T = daily_data_T['temperatureMax'],
                windSpeed_T = daily_data_T['windSpeed'],
                windBearing_T = daily_data_T['windBearing'],
                pressure_T = daily_data_T['pressure'],
                dewPoint_T = daily_data_T['dewPoint'],
                n_icon_T = icons[ten_pm_data_T['icon']],
                n_precipProbability_T = ten_pm_data_T['precipProbability'],
                n_precipIntensity_T = ten_pm_data_T['precipIntensity'],
                n_humidity_T = ten_pm_data_T['humidity'],
                n_cloudCover_T = ten_pm_data_T['cloudCover'],
                n_visibility_T = ten_pm_data_T['visibility'],
                n_temperature_T = ten_pm_data_T['temperature'],
                n_windSpeed_T = ten_pm_data_T['windSpeed'],
                n_windBearing_T = ten_pm_data_T['windBearing'],
                n_pressure_T = ten_pm_data_T['pressure'],
                n_dewPoint_T = ten_pm_data_T['dewPoint']

            ))
            failed_times = 0
            target_date += timedelta(days=1)
        else:
            print("### failed to connect Dark Sky web")
            print("### response status_code of A, L, S, B, T:")
            print(response_A.status_code, response_L.status_code, response_S.status_code, response_B.status_code, response_T.status_code)
            # exit program when failure times is more than 20
            if failed_times > 20:
                print("### failed more 20 imes")
                sys.exit(1)
            else:
                failed_times += 1
            i -= 1
        time.sleep(6)

        # convert list of namedtuple to DataFrame
        historical_data_df = pd.DataFrame(historical_data)

    return historical_data_df

def add_prior_nth_day_feature(df, n, feature):
    """shift the column of a feature by n rows and add this new column to df"""
    assert n <= df.shape[0], "df row number is less than n"

    # shift n days
    # first n days are NaN
    prior_nth_day_feature_col = df[feature].shift(periods = n)

    # fill first n days with data of first day
    for i in range(n):
        prior_nth_day_feature_col[i] = df[feature][0]
    
    # add new_feature to df
    new_feature = f'{feature}_{n}'
    df[new_feature] = prior_nth_day_feature_col

def get_days_a_year(year):
    """return days of year: 365 or 366"""
    if year % his_years == 2:
        return days_a_year + 1
    else:
        return days_a_year

def avg_temp(temp_col, day):
    """get average temperature of 20 days near 'day' """
    temp_sum = 0
    for i in range(day - half_his_days, day + half_his_days):
        temp_sum += temp_col[i]
    return temp_sum / (2 * half_his_days)

def get_his_avg_temp(temp_col, day):
    """get average temperature of 20 days near 'day' in past his_years"""
    avg_temp_sum = 0
    day = day - get_days_a_year(years) # 3 years (his_years), 365, 365(730), 366(1096) (365.25, 730.5, 1095.75)
    while years < his_years:
        avg_temp_sum += avg_temp(temp_col, day)
        years += 1
        day = day - get_days_a_year(years)
    return avg_temp_sum / his_years

def add_his_avg_temp(df):
    """add columns of average temperature of 20 days of last his_years(three years)"""
    days_num = df.shape[0]
    temp_min_col = df["temperatureMin_A"].copy() # deep copy
    temp_max_col = df["temperatureMax_A"].copy() # deep copy

    for i in range(start_train_day, days_num): # if days_num is less than start_train_day, no for-loop
        temp_min_col[i] = get_his_avg_temp(temp_min_col, i)
        temp_max_col[i] = get_his_avg_temp(temp_max_col, i)

    df["history_avg_min"] = temp_min_col
    df["history_avg_max"] = temp_max_col

def get_4_to_7_avg_min_max_temp(temp_col, day):
    """get average/min/max temperature of last 4 to 7 days"""
    temp_avg = 0
    temp_min = 200
    temp_max = -200
    for i in range(day - 7, day - 3):
        temp_avg += temp_col[i]
        if temp_col[i] < temp_min:
            temp_min = temp_col[i]
        if temp_col[i] > temp_max:
            temp_max = temp_col[i]
    temp_avg = temp_avg / 4
    return temp_avg, temp_min, temp_max


def add_4_to_7_avg_temp(df):
    """add columns of average/min/max temperature of last 4 to 7 days"""
    days_num = df.shape[0]
    temp_min_col_avg = df["temperatureMin_A"].copy() # deep copy
    temp_min_col_min = df["temperatureMin_A"].copy() # deep copy
    temp_min_col_max = df["temperatureMin_A"].copy() # deep copy

    temp_max_col_avg = df["temperatureMax_A"].copy() # deep copy
    temp_max_col_min = df["temperatureMax_A"].copy() # deep copy
    temp_max_col_max = df["temperatureMax_A"].copy() # deep copy

    # if days_num is less than start_train_day, no for-loop
    for i in range(7, days_num): 
        temp_min_col_avg[i], temp_min_col_min[i], temp_min_col_max[i] = get_4_to_7_avg_min_max_temp(df["temperatureMin_A"], i)
        temp_max_col_avg[i], temp_max_col_min[i], temp_max_col_max[i] = get_4_to_7_avg_min_max_temp(df["temperatureMax_A"], i)

    df["past_4_to_7_min_avg"] = temp_min_col_avg
    df["past_4_to_7_min_min"] = temp_min_col_min
    df["past_4_to_7_min_max"] = temp_min_col_max

    df["past_4_to_7_max_avg"] = temp_max_col_avg
    df["past_4_to_7_max_min"] = temp_max_col_min
    df["past_4_to_7_max_max"] = temp_max_col_max


def process_date(df, max_N):
    """change date format to day number + years
       2009/1/1 is start date
       2009/1/30 is 29 days + 0 years
    """
    rowNum = df.shape[0]
    first_day = df['date'][0]
    first_day = datetime.strptime(first_day, '%Y-%m-%d')
    first_day = first_day + timedelta(days=max_N)
    
    four_year_days = days_a_year * 4 + 1
    four_year_count = -1 # because day_diff % four_year_days = 0 at 0
    days_li = []
    years_li = []
    for i in range(max_N):
        days_li.append(0)
        years_li.append(0)

    for i in range(max_N, rowNum):
        current_day = df['date'][i]
        current_day = datetime.strptime(current_day, '%Y-%m-%d')
        day_diff = abs((current_day - first_day).days)
        if day_diff % four_year_days == 0:
            four_year_count += 1
        day_diff -= four_year_count
        years = int(day_diff / days_a_year)
        days = day_diff - (days_a_year * years)
        years_li.append(years)
        days_li.append(days)

    df['days'] = days_li
    df['years'] = years_li

def get_start_day_first_day_last_day(df, origin_first_day):
    """get start day to collect historical weather data
       start day is last day + 1 of previous historical data.
       if the start day is less than origin_first_day, we need to clean df.
    """
    clean_df = 0
    first_day_str = df['date'][0]
    df_len = len(df)
    last_day_str = df['date'][df_len - 1]
    first_day = datetime.strptime(first_day_str, '%Y-%m-%d')
    last_day = datetime.strptime(last_day_str, '%Y-%m-%d')
    start_date = last_day + timedelta(days=1)
    if origin_first_day > start_date:
        start_date = origin_first_day
        clean_df = 1

    # str[:10] is first 10 letter, it is just for renaming historical_data.pkl
    return start_date, first_day_str[:10], last_day_str[:10], clean_df 
    
