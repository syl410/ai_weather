#!/usr/bin/env python3

"""Neural network api"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from sklearn.utils import shuffle

icon_num = 8

def num_to_one_hot(Y_origin):
    """create the one-hot encoded vector array"""
    one_hot_Y = np.zeros((Y_origin.shape[0], icon_num))
    for i in range(Y_origin.shape[0]):
        one_hot_Y[i, Y_origin[i]] = 1
    return one_hot_Y

def process_Y_predict(Y_predict):
    """process Y_predict result for icon prediction;
    Y_predict is (m, n), m is number of data set, n is icon_num;
    The function will find max element and set the corresponding index in one hot list to 1"""
    one_hot_Y = np.zeros((Y_predict.shape[0], icon_num))
    for i in range(Y_predict.shape[0]):
        a_list = Y_predict[i].tolist()
        one_hot_Y[i, a_list.index(max(a_list))] = 1
    return one_hot_Y

class NN_regressor:
    """Neural Network class to train, evaluate, and predict"""

    def __init__(self, X, hidden_units, output_num, learning_rate, lamda, scaling_factor, regression_or_classification):
        self.feature_num = X.shape[1]
        self.hidden_units = hidden_units.copy()
        self.output_num = output_num
        self.lr_init = learning_rate
        self.lamda = lamda
        self.scaling_factor = scaling_factor
        self.regression_or_classification = regression_or_classification
        
        self.get_X_mean_and_range(X)
        self.create_theta_bias(self.hidden_units, self.feature_num, self.output_num)

    def get_X_mean_and_range(self, X):
        """get mean and range of each cols for feature scaling"""
        row = X.shape[0] # num of dataset
        col = self.feature_num # feature number
        sum_min_max = np.zeros((col, 3)) # sum(avg), min, max
        self.X_mean_and_range = np.zeros((col, 2)) # mean, range

        # accumulate sum, get min and max of each feature
        for j in range(col):
            sum_min_max[j][1] = value = X[0][j]
            sum_min_max[j][2] = value = X[0][j]
            for i in range(row):
                value = X[i][j]
                sum_min_max[j][0] += value
                if value < sum_min_max[j][1]:
                    sum_min_max[j][1] = value
                if value > sum_min_max[j][2]:
                    sum_min_max[j][2] = value

        # calculate mean and range
        for j in range(col):
            # mean
            self.X_mean_and_range[j][0] = sum_min_max[j][0] / row
            # range
            self.X_mean_and_range[j][1] = sum_min_max[j][2] - sum_min_max[j][1]

    def create_theta_bias(self, hidden_units, feature_num, output_num):
        """create theta list which contains theta_num theta arrays
           create bias list which contains theta_num bias vector
        """
        self.hidden_layers = len(hidden_units)
        self.theta_num = self.hidden_layers + 1
        self.layer_num = self.theta_num + 1
        if self.hidden_layers < 1:
            print("ERROR, hidden layers is less than 1")
            sys.exit(1)
        self.theta = []
        self.dJ_dtheta = [] # d(J)/d(theta)
        self.bias = []
        self.dJ_db = [] # d(J)/d(theta of bias), bias is always 1. b is actuall each theta
        # nn is neural network list
        # nn: [input layer unit num, hidden layer1 unit num, ... , output layer unit num]
        self.nn = hidden_units.copy() # clone!
        self.nn.insert(0, feature_num)
        self.nn.append(output_num)
        # initialize theta, dJ_dtheta, bias(theta), dJ_db with random value in [-1, 1]
        for i in range(self.theta_num):
            # attention, theta t is left x right (feature_num x hidden_nodes, not hidden_nodes x feature_num)
            # it is for avoiding m for-loop
            np.random.seed(i)
            t = np.random.rand(self.nn[i], self.nn[i + 1])
            t = (t - 0.5) * 2
            # b is weight for bias
            # b is 1 x hidden_nodes or 1 x output_num
            np.random.seed(i)
            b = np.random.rand(self.nn[i + 1])
            b = (b - 0.5) * 2

            self.theta.append(t)
            self.dJ_dtheta.append(t)
            self.bias.append(b)
            self.dJ_db.append(b)

    def sigmoid(self, x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        """sigmoid derivative"""
        return self.sigmoid(x) *(1 - self.sigmoid (x))

    def tanh(self, x):
        """tanh function"""
        return np.tanh(x)

    def tanh_der(self, x):
        """tanh derivative"""
        return 1.0 - np.square(np.tanh(x))

    def softmax(self, A):
        """softmax funtion: A and expA are vector"""
        expA = np.exp(A)
        # axis = 0 means along the column and axis = 1 means working along the row.
        return expA / expA.sum(axis=1, keepdims=True)

    def get_batch_size_and_split_x_y(self, X, Y, batch_num):
        """split total X/Y to batch_num dataset [dataset1, ...]
           If x_size size is 8 and batch_num is 3, first two data will be ignore.
           This way can make sure all batches have equal batch_size
        """
        x_size = len(X)
        batch_x = []
        batch_y = []
        batch_size = int(x_size / batch_num)
        i = x_size % batch_num
        times = 0
        while times < batch_num:
            batch_x.append(X[i : i + batch_size])
            batch_y.append(Y[i : i + batch_size])
            i += batch_size
            times += 1

        return batch_size, batch_x, batch_y

    def create_empty_nna_nnz(self, batch_size):
        """nn_a is a1, a2, ... (layer_num x m x unit_num of each layer)"""
        nn_a = []
        nn_z = []
        for i in range(self.layer_num):
            zero_arr = np.zeros((batch_size, self.nn[i]))
            nn_a.append(zero_arr)
            zero_arr = np.zeros((batch_size, self.nn[i])) # new
            nn_z.append(zero_arr)
        return nn_a, nn_z

    def classification_forward_propagation(self, x, nn_a, nn_z):
        """update nn_a, nn_z with forward propagation"""
        nn_a[0] = x
        # m a1, nn_a[0] is m x feature_num
        # use m for loop will be too slow, use matrix so CPU can do parallel computing
        # l is layer
        # np.dot: dot product for two vectors / matrix multiplication for arrays
        # z is array: each row is one z set 
        for l in range(self.theta_num):
            nn_z[l + 1] = np.dot(nn_a[l], self.theta[l]) + self.bias[l]
            if l != self.theta_num - 1:
                nn_a[l + 1] = self.sigmoid(nn_z[l + 1])
            else:
                nn_a[l + 1] = self.softmax(nn_z[l + 1])

    def regression_forward_propagation(self, x, nn_a, nn_z):
        """update nn_a, nn_z with forward propagation"""
        nn_a[0] = x
        for l in range(self.theta_num):
            nn_z[l + 1] = np.dot(nn_a[l], self.theta[l]) + self.bias[l]
            if l != self.theta_num - 1:
                nn_a[l + 1] = self.tanh(nn_z[l + 1])
            else:
                nn_a[l + 1] = nn_z[l + 1]

    def classification_back_propagation(self, x, y, nn_a, nn_z, batch_size):
        """update sefl.dJ_dtheta, self.dJ_db"""
        # last delta means the pre_delta of last loop
        last_delta = []
        # pre_delta means l + 1 layer delta
        pre_delta = []
        for l in range(self.theta_num - 1, -1, -1):
            if l == self.theta_num - 1:
                pre_delta = (nn_a[l + 1] - y) / batch_size # formula is the same as sigmoid
            else:
                pre_delta = np.dot(last_delta, self.theta[l + 1].T) * self.sigmoid_der(nn_z[l + 1])
            dJ_dtheta = np.dot(nn_a[l].T, pre_delta) + (self.lamda / batch_size) * self.theta[l]
            dJ_db = pre_delta.sum(axis = 0)
            self.dJ_dtheta[l] = dJ_dtheta
            self.dJ_db[l] = dJ_db
            last_delta = pre_delta

    def regression_back_propagation(self, x, y, nn_a, nn_z, batch_size):
        """update sefl.dJ_dtheta, self.dJ_db"""
        last_delta = []
        pre_delta = []
        for l in range(self.theta_num - 1, -1, -1):
            if l == self.theta_num - 1:
                pre_delta = (nn_a[l + 1] - y) / batch_size # formula is the same as sigmoid
            else:
                pre_delta = np.dot(last_delta, self.theta[l + 1].T) * self.tanh_der(nn_z[l + 1])
            dJ_dtheta = np.dot(nn_a[l].T, pre_delta) + (self.lamda / batch_size) * self.theta[l]
            dJ_db = pre_delta.sum(axis = 0)
            self.dJ_dtheta[l] = dJ_dtheta
            self.dJ_db[l] = dJ_db
            last_delta = pre_delta

    def feature_scaling(self, X_origin):
        """scale feature value to around [-0.5, 0.5]"""
        row = X_origin.shape[0]
        col = self.feature_num
        X = np.zeros((row, col))
        for j in range(col):
            X[:, j] = (X_origin[:, j] - self.X_mean_and_range[j][0]) / self.X_mean_and_range[j][1] * self.scaling_factor
        return X

    def train(self, X_origin, Y, shuffle, steps, batch_num):
        """The function is to train NN using X_origin and update the theta (weight).
           batch_num: if batch_num is 2, X size is 1000, batch is 500
        """
        # if theta connot converge, exit train.
        if self.theta[-1][0][0] > 100000 or self.theta[-1][0][0] < -100000:
            return

        X = self.feature_scaling(X_origin)
        # shuffle X and Y together
        if shuffle == True:
            randomize = np.arange(len(X))
            np.random.seed(1)
            np.random.shuffle(randomize)
            X = X[randomize]
            Y = Y[randomize]

        batch_size, batch_x, batch_y = self.get_batch_size_and_split_x_y(X, Y, batch_num)
        lr = self.lr_init
        nn_a, nn_z = self.create_empty_nna_nnz(batch_size)

        # train steps times
        for i in range(steps):
            x = batch_x[i % batch_num]
            y = batch_y[i % batch_num]

            if self.regression_or_classification == "classification":
                ### Forward Propagation
                self.classification_forward_propagation(x, nn_a, nn_z)
                ### Back Propagation
                # update sefl.dJ_dtheta, self.dJ_db
                self.classification_back_propagation(x, y, nn_a, nn_z, batch_size)
            else:
                self.regression_forward_propagation(x, nn_a, nn_z)
                self.regression_back_propagation(x, y, nn_a, nn_z, batch_size)

            ### Update Weights
            for j in range(self.theta_num):
                self.theta[j] -= np.multiply(lr, self.dJ_dtheta[j]) # cannot use *
                self.bias[j] -= np.multiply(lr, self.dJ_db[j])

            if self.theta[-1][0][0] > 100000 or self.theta[-1][0][0] < -100000:
                break

    def theta_square(self):
        """square of theta"""
        square_sum = 0
        for i in range(self.theta_num):
            square_sum += np.sum(np.square(self.theta[i]))
        return square_sum

    def evaluate(self, X_origin, Y):
        """using VALIDATION dataset to calculate loss (cost function) for evaluating models"""
        X = self.feature_scaling(X_origin)
        X_len = len(X)
        nn_a, nn_z = self.create_empty_nna_nnz(X_len)
        square_sum = self.theta_square()
        if self.regression_or_classification == "classification":
            self.classification_forward_propagation(X, nn_a, nn_z)
            loss = (np.sum(-Y * np.log(nn_a[self.layer_num - 1])) + self.lamda * square_sum / 2) / X_len
        else:
            self.regression_forward_propagation(X, nn_a, nn_z)
            loss = (np.sum(np.square(nn_a[self.layer_num - 1] - Y)) + self.lamda * square_sum) / (2 * X_len)
        return loss

    def predict(self, X_origin):
        """using TEST dataset as input to do prediction"""
        X = self.feature_scaling(X_origin)
        nn_a, nn_z = self.create_empty_nna_nnz(len(X))
        if self.regression_or_classification == "classification":
            self.classification_forward_propagation(X, nn_a, nn_z) # [0.1, 0,89, ...]
        else:
            self.regression_forward_propagation(X, nn_a, nn_z)
        return nn_a[self.layer_num - 1]
