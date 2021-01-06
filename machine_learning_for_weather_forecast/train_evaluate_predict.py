#!/usr/bin/env python3

"""use NN api to train historical weather; evaluate NN models; predict weather"""

import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split
import pickle
import statistics
import matplotlib.pyplot as plt
import os
import random

from neural_network_model import *
from weather_module import *
from constant import *

# select weather/max temperature/min temperature to train
icon_loop_run = False
temp_max_loop_run = False
temp_min_loop_run = True
# if it is final test, test dataset will be used for evaluating prediction results
final_test = False
# if true, dump theta and bias for every day prediction
save_weight = False

# load pickle file of processed weather data
file_input = 'data/processed_data.pkl'
with open(file_input, 'rb') as pickle_f_in:
    df = pickle.load(pickle_f_in)

# X_ICON_FEATURES: icon feature
# Y_ICON_FEATURE: output feature for icon/weather prediction

# X_TEMPMAX_FEATURES: input features for predicting max temperature
# Y_TEMPMAX_FEATURE: output feature for max temperature prediction

# X_TEMPMIN_FEATURES: input features for predicting min temperature
# Y_TEMPMIN_FEATURE output feature for min temperature prediction

# X_icon is numpy.ndarray: (number of data set, number features)
X_icon = df[X_ICON_FEATURES].to_numpy()[start_train_day :]
# Y_icon is numpy.ndarray: (number of data set, icon_num), a Y_icon is like [[0,1,0..0], ...]
Y_icon = df[Y_ICON_FEATURE].to_numpy()[start_train_day :]
# Y_icon = num_to_one_hot(Y_icon_origin)
Y_icon_num = Y_icon.shape[1]

# X_tempMax/X_tempMin is numpy.ndarray: (number of data set, number of features)
X_tempMax = df[X_TEMPMAX_FEATURES].to_numpy()[start_train_day :]
X_tempMin = df[X_TEMPMIN_FEATURES].to_numpy()[start_train_day :]
# Y_tempMax/Y_tempMin is numpy.ndarray: (number of data set,), it is like a list
Y_tempMax = df[Y_TEMPMAX_FEATURE].to_numpy()[start_train_day :]
Y_tempMin = df[Y_TEMPMIN_FEATURE].to_numpy()[start_train_day :]
Y_tempMaxMin_num = 1
# reshape: 1xn -> nx1
Y_tempMax.shape = len(Y_tempMax), Y_tempMaxMin_num
Y_tempMin.shape = len(Y_tempMin), Y_tempMaxMin_num

# split data to: 1, train (70%), 2, validation (15%), 3, test (15%)
# X/Y_*_test_val: temporary set for both validation data and test data
print(X_icon.shape)
print(Y_icon.shape)
X_icon_train, X_icon_test_val, Y_icon_train, Y_icon_test_val = train_test_split(X_icon, Y_icon, test_size=0.3, random_state=15)
X_icon_test, X_icon_val, Y_icon_test, Y_icon_val = train_test_split(X_icon_test_val, Y_icon_test_val, test_size=0.5, random_state=15)

X_tempMax_train, X_tempMax_test_val, Y_tempMax_train, Y_tempMax_test_val = train_test_split(X_tempMax, Y_tempMax, test_size=0.3, random_state=16)
X_tempMax_test, X_tempMax_val, Y_tempMax_test, Y_tempMax_val = train_test_split(X_tempMax_test_val, Y_tempMax_test_val, test_size=0.5, random_state=16)

X_tempMin_train, X_tempMin_test_val, Y_tempMin_train, Y_tempMin_test_val = train_test_split(X_tempMin, Y_tempMin, test_size=0.3, random_state=17)
X_tempMin_test, X_tempMin_val, Y_tempMin_test, Y_tempMin_val = train_test_split(X_tempMin_test_val, Y_tempMin_test_val, test_size=0.5, random_state=17)

if final_test:
    X_icon_val = X_icon_test
    Y_icon_val = Y_icon_test
    X_tempMax_val = X_tempMax_test
    Y_tempMax_val = Y_tempMax_test
    X_tempMin_val = X_tempMin_test
    Y_tempMin_val = Y_tempMin_test

# list size of trainning, validation and testing dataset
print("Training instances - temp   {}".format(X_tempMax_train.shape[0]))
print("Validation instances - temp {}".format(X_tempMax_val.shape[0]))
print("Testing instances - temp    {}".format(X_tempMax_test.shape[0]))

print("Training features - icon    {}".format(X_icon_train.shape[1]))
print("Validation features - icon  {}".format(X_icon_val.shape[1]))
print("Testing features - icon     {}".format(X_icon_test.shape[1]))

print("Training features - temp    {}".format(X_tempMin_train.shape[1]))
print("Validation features - temp  {}".format(X_tempMin_val.shape[1]))
print("Testing features - temp     {}".format(X_tempMin_test.shape[1]))

def train_evaluate_predict_func(X, hidden_units, output_num, learning_rate, lamda, scaling_factor,
        X_train, Y_train, shuffle, steps, batch_num, X_val, Y_val, X_test, Y_test, 
        iteration, regression_or_classification, category, print_loss, print_CPU):
    """train NN, evaluate result and predict
       X is all x data, output_num is num of output
       iteration*steps = total num of partial/full batchs. Different iterations could have different batch num.
    """

    start_time = time.time()
    regressor = NN_regressor(X, hidden_units, output_num, learning_rate, lamda, scaling_factor, regression_or_classification)

    error_cost = []
    # train and evaluate
    iter_70 = int(0.7 * iteration) # 70%-90%  - half iteration
    iter_90 = int(0.9 * iteration) # 90%-100% - full iteration

    for i in range(iter_70):
        regressor.train(X_train, Y_train, shuffle, steps, batch_num)
        if print_loss == True:
            loss = regressor.evaluate(X_val, Y_val)
            print('Loss function value: ', str(round(loss, 3)))
            error_cost.append(loss)
    for i in range(iter_70, iter_90):
        regressor.train(X_train, Y_train, shuffle, steps, 2)
        if print_loss == True:
            if i == iter_70:
                print('*** start half batch')
            loss = regressor.evaluate(X_val, Y_val)
            print('Loss function value: ', str(round(loss, 3)))
            error_cost.append(loss)
    for i in range(iter_90, iteration):
        regressor.train(X_train, Y_train, shuffle, steps, 1)
        if print_loss == True:
            if i == iter_90:
                print('*** start full batch')
            loss = regressor.evaluate(X_val, Y_val)
            print('Loss function value: ', str(round(loss, 3)))
            error_cost.append(loss)
    
    time1 = time.time() - start_time
    if print_CPU == True:
        print("CPU time is " + str(time1))
    if print_loss == True:
        print("")
        #plt.plot(error_cost)
        #plt.show()
    
    # prediction
    val_num = len(Y_val)
    Y_predict = regressor.predict(X_val, None, None)
    prediction_str = ""
    std_dev = 0

    if regression_or_classification == "classification":
        Y_predict_processed = process_Y_predict(Y_predict)
        correct_num = 0
        for i in range(val_num):
            if np.array_equal(Y_predict_processed[i], Y_val[i]):
                correct_num += 1
        prediction_str = ", icon_prediction_rate: " +  str(round(correct_num / val_num, 2))
    else:
        variance = "Exp_Var: " + str(round(explained_variance_score(Y_val, Y_predict), 2)) # explained variance 
        mean = "Mean_Abs_Err: " + str(round(mean_absolute_error(Y_val, Y_predict), 2))
        median = "Median_Abs_Err: " + str(round(median_absolute_error(Y_val, Y_predict), 2))
        prediction_str = ", " + variance + ", " + mean + ", " + median
        if mean_absolute_error(Y_val, Y_predict) > 100:
            prediction_str = ", ERROR"
        # standard deviation
        abs_diff = []
        for i in range(val_num):
            abs_diff.append(abs(Y_val[i][0] - Y_predict[i][0]))
        std_dev = round(statistics.stdev(abs_diff), 2)

    if save_weight:
        regressor.save_weight(category)
        
    print(  "hidden_units: " +  str(hidden_units).replace(" ", "") +
            ", lr: " +  str(learning_rate) +
            ", lamda: " +  str(lamda) +
            ", scaling_factor: " +  str(scaling_factor) +
            ", iteration: " +  str(iteration) +
            ", steps: " +  str(steps) +
            ", batch_num: " +  str(batch_num) +
            prediction_str + 
            ", stdev: " + str(std_dev))
    """ debug
    print("### X_val")
    print(X_val[:,[0, 1]])
    print("### Y_val")
    print(Y_val)
    print("### Y_predict")
    print(Y_predict)
    """

########### adjustable number ############
icon_hidden_units_li = [[30, 30]]
icon_learning_rate_li = [0.2]
icon_lamda_li = [0.3]
icon_scaling_factor_li = [1]
icon_iteration_li = [200]

icon_steps_li = [200]
icon_batch_num_li = [5]
icon_shuffle = True
icon_regression_or_classification = "classification"
icon_print_loss = False

temp_hidden_units_li = [[30, 30]] 

tempMax_learning_rate_li = [0.005]
tempMin_learning_rate_li = [0.001]

tempMax_lamda_li = [28]
tempMin_lamda_li = [12]

temp_scaling_factor_li = [1]

tempMax_iteration_li = [80]
tempMin_iteration_li = [100]

tempMax_steps_li = [200]
tempMin_steps_li = [250]

tempMax_batch_num_li = [100]
tempMin_batch_num_li = [70]

temp_shuffle = True
temp_regression_or_classification = "regression"
temp_print_loss = False

temp_max_seq_run = False
temp_min_seq_run = False

def train_loop( hidden_units_li, 
                learning_rate_li, 
                lamda_li, 
                scaling_factor_li, 
                iteration_li,
                steps_li,
                batch_num_li,
                X,
                output_num,
                X_train,
                Y_train,
                X_val,
                Y_val,
                X_test,
                Y_test,
                shuffle,
                regression_or_classification,
                category,
                print_loss,
                print_CPU):
    for hidden_units in hidden_units_li:
        for learning_rate in learning_rate_li:
            for lamda in lamda_li:
                for scaling_factor in scaling_factor_li:
                    for iteration in iteration_li:
                        for steps in steps_li:
                            for batch_num in batch_num_li:
                                train_evaluate_predict_func(
                                        X = X, 
                                        hidden_units = hidden_units,
                                        output_num = output_num,
                                        learning_rate = learning_rate,
                                        lamda = lamda,
                                        scaling_factor = scaling_factor,
                                        X_train = X_train,
                                        Y_train = Y_train,
                                        shuffle = shuffle,
                                        steps = steps,
                                        batch_num = batch_num,
                                        X_val = X_val,
                                        Y_val = Y_val,
                                        X_test = X_test,
                                        Y_test = Y_test,
                                        iteration = iteration,
                                        regression_or_classification = regression_or_classification,
                                        category = category,
                                        print_loss = print_loss,
                                        print_CPU = print_CPU)

if icon_loop_run == True:
    train_loop( icon_hidden_units_li, 
                icon_learning_rate_li, 
                icon_lamda_li, 
                icon_scaling_factor_li, 
                icon_iteration_li,
                icon_steps_li,
                icon_batch_num_li,
                X = X_icon,
                output_num = Y_icon_num,
                X_train = X_icon_train,
                Y_train = Y_icon_train,
                X_val = X_icon_val,
                Y_val = Y_icon_val,
                X_test = X_icon_test,
                Y_test = Y_icon_test,
                shuffle = icon_shuffle,
                regression_or_classification = icon_regression_or_classification,
                category = "icon",
                print_loss = icon_print_loss,
                print_CPU = False)

if temp_max_loop_run == True:
    train_loop( temp_hidden_units_li, 
                tempMax_learning_rate_li, 
                tempMax_lamda_li, 
                temp_scaling_factor_li, 
                tempMax_iteration_li,
                tempMax_steps_li,
                tempMax_batch_num_li,
                X = X_tempMax,
                output_num = Y_tempMaxMin_num,
                X_train = X_tempMax_train,
                Y_train = Y_tempMax_train,
                X_val = X_tempMax_val,
                Y_val = Y_tempMax_val,
                X_test = X_tempMax_test,
                Y_test = Y_tempMax_test,
                shuffle = temp_shuffle,
                regression_or_classification = temp_regression_or_classification,
                category = "max_temperature",
                print_loss = temp_print_loss,
                print_CPU = False)

if temp_min_loop_run == True:
    train_loop( temp_hidden_units_li, 
                tempMin_learning_rate_li, 
                tempMin_lamda_li, 
                temp_scaling_factor_li, 
                tempMin_iteration_li,
                tempMin_steps_li,
                tempMin_batch_num_li,
                X = X_tempMin,
                output_num = Y_tempMaxMin_num,
                X_train = X_tempMin_train,
                Y_train = Y_tempMin_train,
                X_val = X_tempMin_val,
                Y_val = Y_tempMin_val,
                X_test = X_tempMin_test,
                Y_test = Y_tempMin_test,
                shuffle = temp_shuffle,
                regression_or_classification = temp_regression_or_classification,
                category = "min_temperature",
                print_loss = temp_print_loss,
                print_CPU = False)
