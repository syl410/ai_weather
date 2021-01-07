#!/usr/bin/env python3

"""This file define common functions and parameters for
   collecting and processing weather data.
"""

from datetime import datetime, timedelta
import time
from collections import namedtuple
from constant import *
import pandas as pd
import requests
import pickle
import os
import os.path
import json
import sys
import random
import pytz

start_train_day = 1 + MAX_N

# namedtuple class for features
DailySummary = namedtuple("DailySummary", FEATURES)

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

def collect_data_func(collect_times, year, month, day, file_output, force_restart, isWeb):
    """request weather data from 'darksky' website , 
       collect useful feature data and load them in pkl file

    Args:
        collect_times: num of times to collect weather info from web
        isWeb: False if it is for training and True if it is for web app
    """
    start_date = datetime(year, month, day)
    start_date_bk = start_date
    
    now = datetime.now(pytz.timezone('America/Chicago'))
    
    # historical weather data
    historical_data = pd.DataFrame()
    
    # if file_output exists
    # get first_day and last_day from collect historical weather data
    # if start_date is after last_day, df will be cleaned and re-generated
    # otherwise start_date is the last_day + 1
    # if start_date is before first_day, program will still start from last_day + 1,
    # unless force_restart = True
    if os.path.isfile(file_output):
        with open(file_output, 'rb') as f_origin:
            historical_data = pickle.load(f_origin) # historical_data is DataFrame
    
        df = pd.DataFrame(historical_data, columns=FEATURES)
        start_date, first_day, last_day, clean_df = get_start_day_first_day_last_day(df, start_date)
        if clean_df == 1 or force_restart: # ignore previous historical_data
            historical_data = pd.DataFrame()
        if force_restart:
            start_date = start_date_bk
        if not isWeb: # for web app, it doesn't need to backup
            os.system(f'mv {file_output} {file_output}_{first_day}__{last_day}')
    
    # check if 
    # 1, start_date is later than today
    # 2, end_day is later than today
    today = datetime(now.year, now.month, now.day)
    end_day = start_date + timedelta(days=(collect_times - 1))
    if start_date > today:
        print(f'start_date {start_date} is later than today {today}')
    else:
        if end_day > today:
            collect_times = (today - start_date).days + 1 # +1 means if start_date is today, we still need to collect
            print("WARNING end_day is later than today")
            print("collect_times has been corrected")
        # for web app, one more day will be collected as dummy data to use date and daytime easily
        if isWeb:
            collect_times += 1
        # collect new weather data and add it into historical_data
        historical_data = historical_data.append(collect_weather_data(BASE_URL, start_date, collect_times), ignore_index=True)
    
    data_length = len(historical_data)
    if data_length >= 2:
        print("### last two days:")
        print((historical_data[-2:][["date", "temperatureMin_A", "temperatureMax_A", "icon4_A", "icon5_A", "icon6_A", "icon7_A"]]).to_string())
    print(f'{data_length} historical data have been collected')
    
    # with: no need to close
    # dump to pkl file
    with open(file_output, 'wb') as f1:
        pickle.dump(historical_data, f1)
    # write a txt file
    txt_output = file_output.replace("pkl", "") + "txt"
    with open(txt_output, 'w') as f1_txt:
        f1_txt.write(historical_data.to_string())

def load_and_process_data_func(file_input, file_output):
    """load weather data from pkl and process the data for NN trainning
    
    Args:
        file_input: '*data/historical_data.pkl'
        file_output: '*data/processed_data.pkl'
    """
    # load pickle file of weather data
    with open(file_input, 'rb') as pickle_f_in:
        historical_data = pickle.load(pickle_f_in)
    
    # transfer to DataFrame
    df = pd.DataFrame(historical_data, columns=FEATURES)
    
    # change date to days and years(year is extra column added) 
    process_date(df)
    # add new columns - prior data
    
    # for each feature, shift the column of a feature by n rows and add this new column to df
    for each_feature in FEATURES:
        if each_feature != 'date' and each_feature != 'visibility':
            for n in range(1, MAX_N + 1):
                add_prior_nth_day_feature(df, n, each_feature)
    
    add_4_to_7_avg_temp(df)
    
    # remove first MAX_N rows
    df = df.iloc[MAX_N:]
    df.info()
    
    # check if output exists
    # if yes, change file name
    # if os.path.exists(file_output):
    #    now = datetime.now()
    #    today_str = str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.time())
    #    os.system(f'mv {file_output} {file_output}_{today_str}')
    
    # dump to a pkl file
    with open(file_output, 'wb') as pickle_f_out:
        pickle.dump(df, pickle_f_out)
    # # write a txt file
    txt_output = file_output.replace("pkl", "") + "txt"
    with open(txt_output, 'w') as f1_txt:
        f1_txt.write(df.to_string())

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
        try:
            daily_data["icon"] = response.json()["hourly"]["icon"]
        except:
            print(f'### BIG ERROR: icon is missing on {city} {date_format}')
            daily_data["icon"] = last_daily_data["icon"]
    if "icon" not in ten_pm_data:
        print(f'### ERROR: icon is missing at ten on {city} {date_format}')
        nine_pm_data = response.json()["hourly"]["data"][-3]
        if "icon" not in nine_pm_data:
            ten_pm_data["icon"] = last_ten_pm_data["icon"]
        else:
            ten_pm_data["icon"] = nine_pm_data["icon"]
   
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

        request_Austin = BASE_URL.format(API_KEY_AUSTN, LOCATION_AUSTN, date_format)
        request_Llano = BASE_URL.format(API_KEY_LLANO, LOCATION_LLANO, date_format)
        request_SanAntonio= BASE_URL.format(API_KEY_SAN_ANTONIO, LOCATION_SAN_ANTONIO, date_format)
        request_Brenham = BASE_URL.format(API_KEY_BRENHAM, LOCATION_BRENHAM, date_format)
        request_Temple = BASE_URL.format(API_KEY_TEMPLE, LOCATION_TEMPLE, date_format)

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

                icon0_A = ICONS[daily_data_A['icon']][0],
                icon1_A = ICONS[daily_data_A['icon']][1],
                icon2_A = ICONS[daily_data_A['icon']][2],
                icon3_A = ICONS[daily_data_A['icon']][3],
                icon4_A = ICONS[daily_data_A['icon']][4],
                icon5_A = ICONS[daily_data_A['icon']][5],
                icon6_A = ICONS[daily_data_A['icon']][6],
                icon7_A = ICONS[daily_data_A['icon']][7],
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
                n_icon0_A = ICONS[ten_pm_data_A['icon']][0],
                n_icon1_A = ICONS[ten_pm_data_A['icon']][1],
                n_icon2_A = ICONS[ten_pm_data_A['icon']][2],
                n_icon3_A = ICONS[ten_pm_data_A['icon']][3],
                n_icon4_A = ICONS[ten_pm_data_A['icon']][4],
                n_icon5_A = ICONS[ten_pm_data_A['icon']][5],
                n_icon6_A = ICONS[ten_pm_data_A['icon']][6],
                n_icon7_A = ICONS[ten_pm_data_A['icon']][7],
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

                icon0_L = ICONS[daily_data_L['icon']][0],
                icon1_L = ICONS[daily_data_L['icon']][1],
                icon2_L = ICONS[daily_data_L['icon']][2],
                icon3_L = ICONS[daily_data_L['icon']][3],
                icon4_L = ICONS[daily_data_L['icon']][4],
                icon5_L = ICONS[daily_data_L['icon']][5],
                icon6_L = ICONS[daily_data_L['icon']][6],
                icon7_L = ICONS[daily_data_L['icon']][7],
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
                n_icon0_L = ICONS[ten_pm_data_L['icon']][0],
                n_icon1_L = ICONS[ten_pm_data_L['icon']][1],
                n_icon2_L = ICONS[ten_pm_data_L['icon']][2],
                n_icon3_L = ICONS[ten_pm_data_L['icon']][3],
                n_icon4_L = ICONS[ten_pm_data_L['icon']][4],
                n_icon5_L = ICONS[ten_pm_data_L['icon']][5],
                n_icon6_L = ICONS[ten_pm_data_L['icon']][6],
                n_icon7_L = ICONS[ten_pm_data_L['icon']][7],
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

                icon0_S = ICONS[daily_data_S['icon']][0],
                icon1_S = ICONS[daily_data_S['icon']][1],
                icon2_S = ICONS[daily_data_S['icon']][2],
                icon3_S = ICONS[daily_data_S['icon']][3],
                icon4_S = ICONS[daily_data_S['icon']][4],
                icon5_S = ICONS[daily_data_S['icon']][5],
                icon6_S = ICONS[daily_data_S['icon']][6],
                icon7_S = ICONS[daily_data_S['icon']][7],
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
                n_icon0_S = ICONS[ten_pm_data_S['icon']][0],
                n_icon1_S = ICONS[ten_pm_data_S['icon']][1],
                n_icon2_S = ICONS[ten_pm_data_S['icon']][2],
                n_icon3_S = ICONS[ten_pm_data_S['icon']][3],
                n_icon4_S = ICONS[ten_pm_data_S['icon']][4],
                n_icon5_S = ICONS[ten_pm_data_S['icon']][5],
                n_icon6_S = ICONS[ten_pm_data_S['icon']][6],
                n_icon7_S = ICONS[ten_pm_data_S['icon']][7],
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
                
                icon0_B = ICONS[daily_data_B['icon']][0],
                icon1_B = ICONS[daily_data_B['icon']][1],
                icon2_B = ICONS[daily_data_B['icon']][2],
                icon3_B = ICONS[daily_data_B['icon']][3],
                icon4_B = ICONS[daily_data_B['icon']][4],
                icon5_B = ICONS[daily_data_B['icon']][5],
                icon6_B = ICONS[daily_data_B['icon']][6],
                icon7_B = ICONS[daily_data_B['icon']][7],
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
                n_icon0_B = ICONS[ten_pm_data_B['icon']][0],
                n_icon1_B = ICONS[ten_pm_data_B['icon']][1],
                n_icon2_B = ICONS[ten_pm_data_B['icon']][2],
                n_icon3_B = ICONS[ten_pm_data_B['icon']][3],
                n_icon4_B = ICONS[ten_pm_data_B['icon']][4],
                n_icon5_B = ICONS[ten_pm_data_B['icon']][5],
                n_icon6_B = ICONS[ten_pm_data_B['icon']][6],
                n_icon7_B = ICONS[ten_pm_data_B['icon']][7],
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


                icon0_T = ICONS[daily_data_T['icon']][0],
                icon1_T = ICONS[daily_data_T['icon']][1],
                icon2_T = ICONS[daily_data_T['icon']][2],
                icon3_T = ICONS[daily_data_T['icon']][3],
                icon4_T = ICONS[daily_data_T['icon']][4],
                icon5_T = ICONS[daily_data_T['icon']][5],
                icon6_T = ICONS[daily_data_T['icon']][6],
                icon7_T = ICONS[daily_data_T['icon']][7],
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
                n_icon0_T = ICONS[ten_pm_data_T['icon']][0],
                n_icon1_T = ICONS[ten_pm_data_T['icon']][1],
                n_icon2_T = ICONS[ten_pm_data_T['icon']][2],
                n_icon3_T = ICONS[ten_pm_data_T['icon']][3],
                n_icon4_T = ICONS[ten_pm_data_T['icon']][4],
                n_icon5_T = ICONS[ten_pm_data_T['icon']][5],
                n_icon6_T = ICONS[ten_pm_data_T['icon']][6],
                n_icon7_T = ICONS[ten_pm_data_T['icon']][7],
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

def avg_temp(temp_col, day):
    """get average temperature of 20 days near 'day' """
    temp_sum = 0
    for i in range(day - half_his_days, day + half_his_days):
        temp_sum += temp_col[i]
    return temp_sum / (2 * half_his_days)

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


def process_date(df):
    """change date format to day number
       2009/1/1 is 1
       2009/2/28 is 59
    """
    rowNum = df.shape[0]
    
    days_li = []
    for i in range(rowNum):
        current_day = df['date'][i] # 2020-10-20 / 2020-10-03??
        date_arr = current_day.split("-")
        month = int(date_arr[1])
        day = int(date_arr[2])
        days_li.append(MONTH_DAYS[month - 1] + day)

    df['days'] = days_li

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
