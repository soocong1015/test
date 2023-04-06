from pipeline.load import data_loader
from pipeline.func import *

import pandas as pd


def main():

    table = data_loader('../archive (1)/BrentOilPrices.csv')

    x = index_change(table, 'Date', 'Price')

    train, test = train_test(x, 650, 4)

    train_x, train_y = convert_to_matrix(train, 4)
    test_x, test_y = convert_to_matrix(test, 4)

    train_x, test_x = reshape(train_x, test_x)

    model = create_model('mean_squared_error', 'adam', 'acc')

    model_fit(model, train_x, train_y, 10)


if __name__ == '__main__':
    print('start')
    main()
    print('~~~end~~~')
