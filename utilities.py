import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from io import BytesIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

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

#def normalize_data(train_data_x, train_data_y, test_data_x):


# normalize minus mean divided by standard deviation
