import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def import_train_data(train_file='train_2008.csv', ):
    data = np.genfromtxt(train_file, delimiter=',', dtype=str)[1:]
    data_x = data[:, list(range(0, 382))]
    data_y = data[:, [382]]
    data_y = [i[0] for i in data_y]

    return data_x, data_y

def import_test_data(test_file='test_2008.csv'):
    data = np.genfromtxt(test_file, delimiter=',', dtype=str)[1:]
    data_x = data[:, list(range(0, 382))]

    return data_x

"""
def normalize_data(train_data_x, train_data_y, test_data_x):
    num_cols = len(train_data_x[0])

    for n in num_cols:
        test[:, n]

    mean_x1 = np.mean(x1, axis=0)
    std_x1 = np.std(x1, axis=0)
    mean_x2 = np.mean(x2, axis=0)
    std_x2 = np.std(x2, axis=0)
    x1 = (x1 - mean_x1) / std_x1
    xt1 = (xt - mean_x1) / std_x1
    x2 = (x2 - mean_x2) / std_x2
    xt2 = (xt - mean_x2) / std_x2
"""
# normalize minus mean divided by standard deviation
#
