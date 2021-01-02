#!/usr/bin/env python3

"""load weather data from pkl and process the data for NN trainning
"""

from weather_module import *

history_data_input = 'data/historical_data.pkl'
process_data_output = 'data/processed_data.pkl'
load_and_process_data_func(history_data_input, process_data_output)
