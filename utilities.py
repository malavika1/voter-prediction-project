import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def import_train_data(train_file='train_2008.csv', one_d_array=True):
    data = np.genfromtxt(train_file, delimiter=',', dtype=float)[1:]
    data_x = data[:, list(range(3, 382))]
    data_y = data[:, [382]]

    if one_d_array:
        data_y = [i[0] for i in data_y]

    return data_x, data_y

def import_test_data(test_file='test_2008.csv'):
    data = np.genfromtxt(test_file, delimiter=',', dtype=float)[1:]
    data_x = data[:, list(range(3, 382))]

    return data_x

def normalize_data(train_x, test_x):
    """Function normalizes both the training x and testing x vectors
    """
    num_cols = len(train_x[0])

    mean_train_x = np.mean(train_x, axis=0)
    std_train_x = np.std(train_x, axis=0)

    train_x = (train_x - mean_train_x) / std_train_x
    test_x = (test_x - mean_train_x) / std_train_x

    return train_x, test_x

def import_data(train_file = 'train_2008.csv', test_file='test_2008.csv', one_d_array=True):
    train_x, train_y = import_train_data(train_file, one_d_array)
    test_x = import_test_data(test_file)

    #train_x, test_x = normalize_data(train_x, test_x)

    return train_x, train_y, test_x

def write_output_file(output, file_name='output.csv'):
    output_id = np.array(list((range(0, len(output)))))
    columns = ['id', 'PES1']
    df = pd.DataFrame({'id': output_id, 'PES1': output.astype(int)}, columns=columns)
    df.to_csv(file_name, index=False)

input = np.array([0.0, 1.0])
write_output_file(input)
