#!/usr/bin/env python3

"""request weather data from 'darksky' website , 
   collect useful feature data and load them in pkl file
"""
import pandas as pd
import numpy as np
import pickle
import os
import random
import os.path
import time
from datetime import datetime, timedelta

from neural_network_model import *
from weather_module import *
from constant import *

print("start running web_collect_process_predict.py")
"""collect data"""
collect_times = 2
year = 2020
month = 12
day = 10
history_data_file = 'machine_learning_for_weather_forecast/prediction_data/historical_data.pkl'
# it will force to collect data from start_date each time. Check carefully each time!
force_restart = False 

# collect_data_func(collect_times, year, month, day)
collect_data_func(collect_times, year, month, day, history_data_file, force_restart, True)

"""load and process"""
process_data_output = 'machine_learning_for_weather_forecast/prediction_data/processed_data.pkl'
load_and_process_data_func(history_data_file, process_data_output)
# remove data of last day for it was collected as dummy data to use date and daytime easily
his_df = pd.DataFrame()
with open(history_data_file, 'rb') as his_file: # read history_data
    his_df = pickle.load(his_file)
    his_df = his_df[:-1] # remove last day
with open(history_data_file, 'wb') as his_file: # override history_data
    pickle.dump(his_df, his_file)

"""predict"""
# load pickle file of processed weather data
process_data_input = 'machine_learning_for_weather_forecast/prediction_data/processed_data.pkl'
with open(process_data_input, 'rb') as pickle_f_in:
    df = pickle.load(pickle_f_in)

# load pickle file of weather data for training
train_data_input = 'machine_learning_for_weather_forecast/data/processed_data.pkl'
with open(train_data_input, 'rb') as train_pickle_f_in:
    train_df = pickle.load(train_pickle_f_in)

# X_icon is numpy.ndarray: (1, number features)
X_icon = df[X_ICON_FEATURES].to_numpy()[-1 :]
X_icon_train = train_df[X_ICON_FEATURES].to_numpy()

# X_tempMax/X_tempMin is numpy.ndarray: (number of data set, number of features)
X_tempMax       = df[X_TEMPMAX_FEATURES].to_numpy()[-1 :]
X_tempMax_train = train_df[X_TEMPMAX_FEATURES].to_numpy()
X_tempMin       = df[X_TEMPMIN_FEATURES].to_numpy()[-1 :]
X_tempMin_train = train_df[X_TEMPMIN_FEATURES].to_numpy()

hidden_units_icon = [20,20]
output_num_icon = ICON_NUM
learning_rate_icon = 0.02
lamda_icon = 0.08
scaling_factor_icon = 1
# read icon theta and bias
with open("machine_learning_for_weather_forecast/prediction_data/icon_theta.pkl", 'rb') as icon_theta_f:
    icon_theta = pickle.load(icon_theta_f)
with open("machine_learning_for_weather_forecast/prediction_data/icon_bias.pkl", 'rb') as icon_bias_f:
    icon_bias = pickle.load(icon_bias_f)
# initialize icon regressor
regressor_icon = NN_regressor(X_icon_train, hidden_units_icon, output_num_icon, learning_rate_icon, lamda_icon, scaling_factor_icon, "classification")
# predict result
Y_icon_predict = regressor_icon.predict(X_icon, icon_theta, icon_bias)
Y_icon_predict_processed = process_Y_predict(Y_icon_predict)
for i in range(ICON_NUM):
    if Y_icon_predict_processed[0][ICON_NUM - 1 - i] == 1:
        weather_icon = WEATHER_LIST[i]
        weather = weather_icon.replace('-', ' ')
        break

hidden_units_tempMax = [30,30]
output_num_tempMax = 1
learning_rate_tempMax = 0.002
lamda_tempMax = 10
scaling_factor_tempMax = 1
# read tempMax theta and bias
with open("machine_learning_for_weather_forecast/prediction_data/max_temperature_theta.pkl", 'rb') as tempMax_theta_f:
    tempMax_theta = pickle.load(tempMax_theta_f)
with open("machine_learning_for_weather_forecast/prediction_data/max_temperature_bias.pkl", 'rb') as tempMax_bias_f:
    tempMax_bias = pickle.load(tempMax_bias_f)
# initialize max temperature regressor
regressor_tempMax = NN_regressor(X_tempMax_train, hidden_units_tempMax, output_num_tempMax, learning_rate_tempMax, lamda_tempMax, scaling_factor_tempMax, "regression")
# predict result
Y_tempMax_predict = regressor_tempMax.predict(X_tempMax, tempMax_theta, tempMax_bias)
tempMax = round(float(Y_tempMax_predict[0]))

hidden_units_tempMin = [30,30]
output_num_tempMin = 1
learning_rate_tempMin = 0.004
lamda_tempMin = 13
scaling_factor_tempMin = 1
# read tempMin theta and bias
with open("machine_learning_for_weather_forecast/prediction_data/min_temperature_theta.pkl", 'rb') as tempMin_theta_f:
    tempMin_theta = pickle.load(tempMin_theta_f)
with open("machine_learning_for_weather_forecast/prediction_data/min_temperature_bias.pkl", 'rb') as tempMin_bias_f:
    tempMin_bias = pickle.load(tempMin_bias_f)
# initialize min temperature regressor
regressor_tempMin = NN_regressor(X_tempMin_train, hidden_units_tempMin, output_num_tempMin, learning_rate_tempMin, lamda_tempMin, scaling_factor_tempMin, "regression")
# predict result
Y_tempMin_predict = regressor_tempMin.predict(X_tempMin, tempMin_theta, tempMin_bias)
tempMin = round(float(Y_tempMin_predict[0]))


# print date
today = datetime.today()
tomorrow = today + timedelta(days=1)
weekday = tomorrow.strftime('%A')

with open("machine_learning_for_weather_forecast/new_forecast.json", 'w') as json_f:
    json_f.write("{\n")
    json_f.write(f'\t"what_day" : "{weekday}",\n')
    json_f.write(f'\t"forecast_date" : "{tomorrow.month}/{tomorrow.day}",\n')
    json_f.write(f'\t"updated_date" : "{today.month}/{today.day}",\n')
    json_f.write(f'\t"weather" : "{weather}",\n')
    json_f.write(f'\t"weather_icon" : "{weather_icon}",\n')
    json_f.write(f'\t"max_temperature" : "{tempMax}",\n')
    json_f.write(f'\t"min_temperature" : "{tempMin}"\n')
    json_f.write("}\n")
