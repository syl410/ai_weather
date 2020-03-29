#!/usr/bin/env python3

"""request weather data from 'darksky' website , 
   collect useful feature data and load them in pkl file
"""

from weather_module import *
import os.path
import os
import time

collect_times = 490 # num of times to collect weather info from web
start_date = datetime(2009, 10, 1)
start_date_bk = start_date
# check carefully each time!
# it will force to collect data from start_date each time
force_restart = False 

now = datetime.now()

file_output = 'data/historical_data.pkl'

# historical weather data
historical_data = []

# if file_output exists
# get first_day and last_day from collect historical weather data
# if start_date is after last_day, df will be cleaned and re-generated
# otherwise start_date is the last_day + 1
# if start_date is before first_day, program will still start from last_day + 1,
# unless force_restart = True
if os.path.isfile(file_output):
    with open(file_output, 'rb') as f_origin:
        historical_data = pickle.load(f_origin) # historical_data is DataFrame

    df = pd.DataFrame(historical_data, columns=features)
    start_date, first_day, last_day, clean_df = get_start_day_first_day_last_day(df, start_date)
    if clean_df == 1 or force_restart: # ignore previous historical_data
        historical_data = []
    if force_restart:
        start_date = start_date_bk
    os.system(f'mv {file_output} {file_output}_{first_day}__{last_day}')

# check if 
# 1, start_date is later than today
# 2, end_day is later than today
today = datetime(now.year, now.month, now.day)
end_day = start_date + timedelta(days=collect_times)
if start_date >= today:
    print("ERROR start_date is later than today")
    sys.exit(1)
else:
    if end_day > today:
        collect_times = (today - start_date).days
        print("WARNING end_day is later than today")
        print("collect_times has been corrected")
    # collect new weather data and add it into historical_data
    historical_data = historical_data.append(collect_weather_data(BASE_URL, start_date, collect_times), ignore_index=True)

data_length = len(historical_data)
print(historical_data)
print(f'{data_length} historical data have been collected')

# with: no need to close
with open(file_output, 'wb') as f1:
    pickle.dump(historical_data, f1)
