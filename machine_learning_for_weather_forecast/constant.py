#!/usr/bin/env python3

"""list constant variables"""

# API KEYS for "darksky" web
API_KEY_AUSTN = '4778c8e06b15f3691b07cbccda46b8fe'
API_KEY_LLANO = 'c499aaf8c23a151dafffb5135240df81'
API_KEY_SAN_ANTONIO = 'd2480b9c724e6685640f49b7db6d6f37'
API_KEY_BRENHAM = 'b898ce9e4ad35c66c2d2dae30f31c722'
API_KEY_TEMPLE = 'd42bc3137b5e37d84e89d6a695d6b740'

# longitude and latitude for cities
# Austin is the city to predict weather
# rest of cities are surrounding cities
LOCATION_AUSTN = '30.2711,-97.7437'
LOCATION_LLANO = '30.7102,-98.6841'
LOCATION_SAN_ANTONIO = '29.5444,-98.4147'
LOCATION_BRENHAM ='30.1768,-96.3965'
LOCATION_TEMPLE = '31.0982,-97.3428'

BASE_URL = "https://api.darksky.net/forecast/{}/{},{}T00:00:00?exclude=currently,flag"
#https://api.darksky.net/forecast/4778c8e06b15f3691b07cbccda46b8fe/30.2711,-97.7437,2016-05-16T00:00:00?exclude=currently,hourly,flags
#https://api.darksky.net/forecast/4778c8e06b15f3691b07cbccda46b8fe/30.2711,-97.7437,2016-05-16T00:00:00?exclude=currently,flags

# https://darksky.net/dev/docs#time-machine-request

# weather features
# _A: Austin, _L: Llano, _S: SanAntonio, _B: Brenham, _T: Temple
# n_*: night features at 10pm
FEATURES = ["date", "daytime", 
        "icon0_A", "icon1_A", "icon2_A", "icon3_A", "icon4_A", "icon5_A", "icon6_A", "icon7_A",
        "precipProbability_A", "precipIntensity_A", "humidity_A", "cloudCover_A", "visibility_A", "temperatureMin_A", "temperatureMax_A", "windSpeed_A", "windBearing_A", "pressure_A", "dewPoint_A", 
        "n_icon0_A", "n_icon1_A", "n_icon2_A", "n_icon3_A", "n_icon4_A", "n_icon5_A", "n_icon6_A", "n_icon7_A",
        "n_precipProbability_A", "n_precipIntensity_A", "n_humidity_A", "n_cloudCover_A", "n_visibility_A", "n_temperature_A", "n_windSpeed_A", "n_windBearing_A", "n_pressure_A", "n_dewPoint_A",

        "icon0_L", "icon1_L", "icon2_L", "icon3_L", "icon4_L", "icon5_L", "icon6_L", "icon7_L",
        "precipProbability_L", "precipIntensity_L", "humidity_L", "cloudCover_L", "visibility_L", "temperatureMin_L", "temperatureMax_L", "windSpeed_L", "windBearing_L", "pressure_L", "dewPoint_L", 
        "n_icon0_L", "n_icon1_L", "n_icon2_L", "n_icon3_L", "n_icon4_L", "n_icon5_L", "n_icon6_L", "n_icon7_L",
        "n_precipProbability_L", "n_precipIntensity_L", "n_humidity_L", "n_cloudCover_L", "n_visibility_L", "n_temperature_L", "n_windSpeed_L", "n_windBearing_L", "n_pressure_L", "n_dewPoint_L",

        "icon0_S", "icon1_S", "icon2_S", "icon3_S", "icon4_S", "icon5_S", "icon6_S", "icon7_S",
        "precipProbability_S", "precipIntensity_S", "humidity_S", "cloudCover_S", "visibility_S", "temperatureMin_S", "temperatureMax_S", "windSpeed_S", "windBearing_S", "pressure_S", "dewPoint_S", 
        "n_icon0_S", "n_icon1_S", "n_icon2_S", "n_icon3_S", "n_icon4_S", "n_icon5_S", "n_icon6_S", "n_icon7_S",
        "n_precipProbability_S", "n_precipIntensity_S", "n_humidity_S", "n_cloudCover_S", "n_visibility_S", "n_temperature_S", "n_windSpeed_S", "n_windBearing_S", "n_pressure_S", "n_dewPoint_S",

        "icon0_B", "icon1_B", "icon2_B", "icon3_B", "icon4_B", "icon5_B", "icon6_B", "icon7_B",
        "precipProbability_B", "precipIntensity_B", "humidity_B", "cloudCover_B", "visibility_B", "temperatureMin_B", "temperatureMax_B", "windSpeed_B", "windBearing_B", "pressure_B", "dewPoint_B", 
        "n_icon0_B", "n_icon1_B", "n_icon2_B", "n_icon3_B", "n_icon4_B", "n_icon5_B", "n_icon6_B", "n_icon7_B",
        "n_precipProbability_B", "n_precipIntensity_B", "n_humidity_B", "n_cloudCover_B", "n_visibility_B", "n_temperature_B", "n_windSpeed_B", "n_windBearing_B", "n_pressure_B", "n_dewPoint_B",

        "icon0_T", "icon1_T", "icon2_T", "icon3_T", "icon4_T", "icon5_T", "icon6_T", "icon7_T",
        "precipProbability_T", "precipIntensity_T", "humidity_T", "cloudCover_T", "visibility_T", "temperatureMin_T", "temperatureMax_T", "windSpeed_T", "windBearing_T", "pressure_T", "dewPoint_T", 
        "n_icon0_T", "n_icon1_T", "n_icon2_T", "n_icon3_T", "n_icon4_T", "n_icon5_T", "n_icon6_T", "n_icon7_T",
        "n_precipProbability_T", "n_precipIntensity_T", "n_humidity_T", "n_cloudCover_T", "n_visibility_T", "n_temperature_T", "n_windSpeed_T", "n_windBearing_T", "n_pressure_T", "n_dewPoint_T"
]

# max previous days 
MAX_N = 3
DAYS_A_YEAR = 365

# days to the month
MONTH_DAYS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

# weather one hot mapping
ICONS = {   'clear-day':            [0, 0, 0, 0, 0, 0, 0, 1], 'clear-night': [0, 0, 0, 0, 0, 0, 0, 1], 
            'partly-cloudy-day':    [0, 0, 0, 0, 0, 0, 1, 0], 'partly-cloudy-night': [0, 0, 0, 0, 0, 0, 1, 0], 
            'cloudy':               [0, 0, 0, 0, 0, 1, 0, 0], 
            'rain':                 [0, 0, 0, 0, 1, 0, 0, 0], 
            'wind':                 [0, 0, 0, 1, 0, 0, 0, 0], 
            'fog':                  [0, 0, 1, 0, 0, 0, 0, 0], 
            'sleet':                [0, 1, 0, 0, 0, 0, 0, 0], 
            'snow':                 [1, 0, 0, 0, 0, 0, 0, 0]
        }

# weather list
WEATHER_LIST = [ 'clear-day', 
                'partly-cloudy-day', 
                'cloudy', 
                'rain', 
                'wind', 
                'fog', 
                'sleet', 
                'snow']

# number of icons, like: rain, cloud, sunny, ...
ICON_NUM = 8

# icon feature
X_ICON_FEATURES = [
        'days', 'daytime',
        'icon0_A_1', 'icon1_A_1', 'icon2_A_1', 'icon3_A_1', 'icon4_A_1', 'icon5_A_1', 'icon6_A_1', 'icon7_A_1', 
        'precipProbability_A_1', 'humidity_A_1', 'cloudCover_A_1', 'temperatureMin_A_1', 'temperatureMax_A_1', 'pressure_A_1', 'windSpeed_A_1', 'windBearing_A_1',
        'icon0_A_2', 'icon1_A_2', 'icon2_A_2', 'icon3_A_2', 'icon4_A_2', 'icon5_A_2', 'icon6_A_2', 'icon7_A_2', 
        'precipProbability_A_2', 'humidity_A_2', 'cloudCover_A_2', 'temperatureMin_A_2', 'temperatureMax_A_2', 'pressure_A_2', 'windSpeed_A_2', 'windBearing_A_2',
        'icon0_A_3', 'icon1_A_3', 'icon2_A_3', 'icon3_A_3', 'icon4_A_3', 'icon5_A_3', 'icon6_A_3', 'icon7_A_3', 
        'precipProbability_A_3', 'humidity_A_3', 'cloudCover_A_3', 'temperatureMin_A_3', 'temperatureMax_A_3', 'pressure_A_3', 'windSpeed_A_3', 'windBearing_A_3',

        'precipProbability_L_1', 'humidity_L_1', 'cloudCover_L_1', 'temperatureMin_L_1', 'temperatureMax_L_1', 
        'precipProbability_S_1', 'humidity_S_1', 'cloudCover_S_1', 'temperatureMin_S_1', 'temperatureMax_S_1', 
        'precipProbability_B_1', 'humidity_B_1', 'cloudCover_B_1', 'temperatureMin_B_1', 'temperatureMax_B_1', 
        'precipProbability_T_1', 'humidity_T_1', 'cloudCover_T_1', 'temperatureMin_T_1', 'temperatureMax_T_1', 
        
        'n_icon0_A_1', 'n_icon1_A_1', 'n_icon2_A_1', 'n_icon3_A_1', 'n_icon4_A_1', 'n_icon5_A_1', 'n_icon6_A_1', 'n_icon7_A_1',
        'n_humidity_A_1', 'n_cloudCover_A_1', 'n_temperature_A_1', 'n_windSpeed_A_1', 'n_pressure_A_1', 'n_windBearing_A_1',
        'n_humidity_L_1', 'n_cloudCover_L_1', 'n_temperature_L_1', 'n_windSpeed_L_1',  
        'n_humidity_S_1', 'n_cloudCover_S_1', 'n_temperature_S_1', 'n_windSpeed_S_1',  
        'n_humidity_B_1', 'n_cloudCover_B_1', 'n_temperature_B_1', 'n_windSpeed_B_1',  
        'n_humidity_T_1', 'n_cloudCover_T_1', 'n_temperature_T_1', 'n_windSpeed_T_1',
        'n_icon0_A_2', 'n_icon1_A_2', 'n_icon2_A_2', 'n_icon3_A_2', 'n_icon4_A_2', 'n_icon5_A_2', 'n_icon6_A_2', 'n_icon7_A_2'
        ]
# output feature for icon/weather prediction
Y_ICON_FEATURE = ['icon0_A', 'icon1_A', 'icon2_A', 'icon3_A', 'icon4_A', 'icon5_A', 'icon6_A', 'icon7_A']


# input features for predicting max temperature
X_TEMPMAX_FEATURES = [
        'days', 'daytime',
        'past_4_to_7_min_avg', #'past_4_to_7_min_min', 'past_4_to_7_min_max',
        'past_4_to_7_max_avg', 'past_4_to_7_max_min', 'past_4_to_7_max_max',

        'precipIntensity_A_1', 'humidity_A_1', 'cloudCover_A_1', 'temperatureMin_A_1', 'temperatureMax_A_1', 'pressure_A_1', 'windSpeed_A_1', 'windBearing_A_1',
        'precipIntensity_A_2', 'humidity_A_2', 'cloudCover_A_2', 'temperatureMin_A_2', 'temperatureMax_A_2', 'pressure_A_2', 'windSpeed_A_2', 'windBearing_A_2',
        'precipIntensity_A_3', 'humidity_A_3', 'cloudCover_A_3', 'temperatureMin_A_3', 'temperatureMax_A_3', 'pressure_A_3', 'windSpeed_A_3', 'windBearing_A_3',

        'precipIntensity_L_1', 'humidity_L_1', 'cloudCover_L_1', 'temperatureMin_L_1', 'temperatureMax_L_1', #'windSpeed_L_1', 'pressure_L_1', 'windBearing_L_1',
        'precipIntensity_S_1', 'humidity_S_1', 'cloudCover_S_1', 'temperatureMin_S_1', 'temperatureMax_S_1', #'windSpeed_S_1', 'pressure_S_1', 'windBearing_S_1',
        'precipIntensity_B_1', 'humidity_B_1', 'cloudCover_B_1', 'temperatureMin_B_1', 'temperatureMax_B_1', #'windSpeed_B_1', 'pressure_B_1', 'windBearing_B_1',
        'precipIntensity_T_1', 'humidity_T_1', 'cloudCover_T_1', 'temperatureMin_T_1', 'temperatureMax_T_1', #'windSpeed_T_1', 'pressure_T_1', 'windBearing_T_1',
        
        'n_humidity_A_1', 'n_cloudCover_A_1', 'n_temperature_A_1', 'n_windSpeed_A_1', 'n_pressure_A_1', 'n_windBearing_A_1', 
        'n_humidity_L_1', 'n_cloudCover_L_1', 'n_temperature_L_1', 'n_windSpeed_L_1', #'n_pressure_L_1', # 'n_windBearing_L_1', 
        'n_humidity_S_1', 'n_cloudCover_S_1', 'n_temperature_S_1', 'n_windSpeed_S_1', #'n_pressure_S_1', # 'n_windBearing_S_1', 
        'n_humidity_B_1', 'n_cloudCover_B_1', 'n_temperature_B_1', 'n_windSpeed_B_1', #'n_pressure_B_1', # 'n_windBearing_B_1', 
        'n_humidity_T_1', 'n_cloudCover_T_1', 'n_temperature_T_1', 'n_windSpeed_T_1', #'n_pressure_T_1', # 'n_windBearing_T_1', 
        'n_temperature_A_2'
        ]
# output feature for max temperature prediction
Y_TEMPMAX_FEATURE = 'temperatureMax_A'

# input features for predicting min temperature
X_TEMPMIN_FEATURES = [
        'days', 'daytime',
        'past_4_to_7_min_avg', 'past_4_to_7_min_min', 'past_4_to_7_min_max',
        'past_4_to_7_max_avg', #'past_4_to_7_max_min', 'past_4_to_7_max_max',

        'precipIntensity_A_1', 'humidity_A_1', 'cloudCover_A_1', 'temperatureMin_A_1', 'temperatureMax_A_1', 'pressure_A_1', 'windSpeed_A_1', 'windBearing_A_1',
        'precipIntensity_A_2', 'humidity_A_2', 'cloudCover_A_2', 'temperatureMin_A_2', 'temperatureMax_A_2', 'pressure_A_2', 'windSpeed_A_2', 'windBearing_A_2',
        'precipIntensity_A_3', 'humidity_A_3', 'cloudCover_A_3', 'temperatureMin_A_3', 'temperatureMax_A_3', 'pressure_A_3', 'windSpeed_A_3', 'windBearing_A_3',

        'precipIntensity_L_1', 'humidity_L_1', 'cloudCover_L_1', 'temperatureMin_L_1', 'temperatureMax_L_1', #'windSpeed_L_1', 'pressure_L_1', 'windBearing_L_1',
        'precipIntensity_S_1', 'humidity_S_1', 'cloudCover_S_1', 'temperatureMin_S_1', 'temperatureMax_S_1', #'windSpeed_S_1', 'pressure_S_1', 'windBearing_S_1',
        'precipIntensity_B_1', 'humidity_B_1', 'cloudCover_B_1', 'temperatureMin_B_1', 'temperatureMax_B_1', #'windSpeed_B_1', 'pressure_B_1', 'windBearing_B_1',
        'precipIntensity_T_1', 'humidity_T_1', 'cloudCover_T_1', 'temperatureMin_T_1', 'temperatureMax_T_1', #'windSpeed_T_1', 'pressure_T_1', 'windBearing_T_1',
        
        'n_humidity_A_1', 'n_cloudCover_A_1', 'n_temperature_A_1', 'n_windSpeed_A_1', 'n_pressure_A_1', 'n_windBearing_A_1', 
        'n_humidity_L_1', 'n_cloudCover_L_1', 'n_temperature_L_1', 'n_windSpeed_L_1', #'n_pressure_L_1', # 'n_windBearing_L_1', 
        'n_humidity_S_1', 'n_cloudCover_S_1', 'n_temperature_S_1', 'n_windSpeed_S_1', #'n_pressure_S_1', # 'n_windBearing_S_1', 
        'n_humidity_B_1', 'n_cloudCover_B_1', 'n_temperature_B_1', 'n_windSpeed_B_1', #'n_pressure_B_1', # 'n_windBearing_B_1', 
        'n_humidity_T_1', 'n_cloudCover_T_1', 'n_temperature_T_1', 'n_windSpeed_T_1', #'n_pressure_T_1', # 'n_windBearing_T_1', 
        'n_temperature_A_2'
        ]
# output feature for min temperature prediction
Y_TEMPMIN_FEATURE = 'temperatureMin_A'

