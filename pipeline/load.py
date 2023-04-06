import pandas as pd


def data_loader(path):
    table = pd.read_csv(path)
    return table
