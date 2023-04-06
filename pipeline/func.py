import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM


def index_change(table, col_name, col_target_name):
    table = table.set_index(col_name)
    table.index = pd.to_datetime(table.index)
    for col in table.columns:
        scaler = MinMaxScaler()
        table[col] = scaler.fit_transform(table)
    x = table[col_target_name].values.tolist()
    return x

# dataframe, 바꿀 index column, pride column 인자로 받아서
# index 바꾸고 datetime으로 바꾸고 pride column을 minmax scaling한 후
# minmax scaling한 값들을 list로 return


def train_test(data, data_cut, step):
    train, test = data[:data_cut], data[data_cut:]
    train = np.append(train, np.repeat(train[-1], step))
    test = np.append(test, np.repeat(test[-1], step))
    return train, test

# train, test dataset으로 나눌 data와 나눌 개수 인자로 받아서
# 자르고 마지막 데이터를 4번 반복추가


def convert_to_matrix(data, step):
    x, y = [], []
    for i in range(len(data) - step):
        d = i + step
        x.append(data[i:d])
        y.append(data[d])
    return np.array(x), np.array(y)


def reshape(train_x, test_x):
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    return train_x, test_x


def create_model(loss_name, opt_name, metrics_name):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(1, 4)))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss=loss_name, optimizer=opt_name, metrics=[metrics_name])
    return model


def model_fit(model, train_x, train_y, epochs_count):
    model.fit(train_x, train_y, epochs=epochs_count)
