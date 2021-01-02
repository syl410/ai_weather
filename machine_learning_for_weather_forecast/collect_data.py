#!/usr/bin/env python3

"""request weather data from 'darksky' website , 
   collect useful feature data and load them in pkl file
"""

from weather_module import *
import os.path
import os
import time

collect_times = 360
year = 2010
month = 10
day = 2
file_output = 'data/historical_data.pkl'
# it will force to collect data from start_date each time. Check carefully each time!
force_restart = False 

# collect_data_func(collect_times, year, month, day, isWeb)
collect_data_func(collect_times, year, month, day, file_output, force_restart, False)
