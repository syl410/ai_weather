#!/usr/bin/env python3

"""load weather data from pkl and process the data for NN trainning
"""

from weather_module import *
import random


# load pickle file of weather data
file_input = 'data/historical_data.pkl'

with open(file_input, 'rb') as pickle_f_in:
    historical_data = pickle.load(pickle_f_in)

# transfer to DataFrame
df = pd.DataFrame(historical_data, columns=features)

# change date to days and years(year is extra column added) 
process_date(df, max_N)
# add new columns - prior data

# for each feature, shift the column of a feature by n rows and add this new column to df
for each_feature in features:
    if each_feature != 'date' and each_feature != 'visibility':
        for n in range(1, max_N + 1):
            add_prior_nth_day_feature(df, n, each_feature)

add_his_avg_temp(df)
add_4_to_7_avg_temp(df)

# remove first max_N rows
df = df.iloc[max_N:]
df.info()

# check if output exists
# if yes, change file name
file_output = 'data/processed_data.pkl'
if os.path.exists(file_output):
    now = datetime.now()
    today_str = str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.time())
    os.system(f'mv {file_output} {file_output}_{today_str}')

# print whole dataframe
# print(df.to_string())

with open(file_output, 'wb') as pickle_f_out:
    pickle.dump(df, pickle_f_out)
