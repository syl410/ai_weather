#!/usr/bin/python3

"""There are some incorrect data from 'darksky' website by checking other weather websites. 
   This python script is to correct those data using the data from other weather websites.
"""

from weather_module import *
import random
import os

# load pickle file of weather data
file_input = 'data/historical_data.pkl'
file_input_bk = 'data/historical_data.pkl_bk'
file_output = 'data/historical_data.pkl'
file_output_txt = 'data/historical_data.txt'
file_output_txt_bk = 'data/historical_data.txt.bk'

with open(file_input, 'rb') as pickle_in:
    historical_data = pickle.load(pickle_in)

os.system(f'mv {file_input} {file_input_bk}')

# transfer to DataFrame
df = pd.DataFrame(historical_data, columns=FEATURES).set_index('date')
with open(file_output_txt_bk, 'w') as txt_bk_out:
    txt_bk_out.write(df.to_string())

# fix incorrect data from darksky web
df.at['2012-12-17', 'temperatureMin_A'] = 44
df.at['2012-12-18', 'temperatureMin_A'] = 40
df.at['2013-04-02', 'temperatureMin_A'] = 55
df.at['2013-07-14', 'temperatureMax_A'] = 96
df.at['2013-07-16', 'temperatureMax_A'] = 92
df.at['2013-07-17', 'temperatureMax_A'] = 90
df.at['2013-08-14', 'temperatureMax_A'] = 105
df.at['2014-02-25', 'temperatureMax_A'] = 69
df.at['2015-01-19', 'temperatureMin_A'] = 40
df.at['2016-05-10', 'temperatureMin_A'] = 67
df.at['2017-02-07', 'temperatureMin_A'] = 56
df.at['2017-02-08', 'temperatureMin_A'] = 51
df.at['2017-02-19', 'temperatureMin_A'] = 61
df.at['2017-02-19', 'n_temperature_A'] = 65
df.at['2017-08-23', 'temperatureMax_A'] = 98
df.at['2018-04-07', 'temperatureMax_A'] = 64
df.at['2018-08-12', 'temperatureMax_A'] = 91
df.at['2018-09-02', 'temperatureMax_A'] = 97
df.at['2018-09-03', 'temperatureMax_A'] = 95
df.at['2018-09-04', 'temperatureMax_A'] = 91
df.at['2018-09-05', 'temperatureMax_A'] = 94
df.at['2018-09-06', 'temperatureMax_A'] = 95
df.at['2018-10-15', 'temperatureMax_A'] = 70
df.at['2018-12-26', 'temperatureMin_A'] = 53
df.at['2018-12-26', 'n_temperature_A'] = 60
df.at['2019-02-27', 'temperatureMin_A'] = 52
df.at['2019-02-27', 'n_temperature_A'] = 57
df.at['2019-09-09', 'temperatureMax_A'] = 98
df.at['2019-09-10', 'temperatureMax_A'] = 98
df.at['2019-09-11', 'temperatureMax_A'] = 96


df.reset_index(inplace=True)

with open(file_output_txt, 'w') as txt_out:
    txt_out.write(df.to_string())

# check if output exists
with open(file_output, 'wb') as pickle_out:
    pickle.dump(df, pickle_out)
