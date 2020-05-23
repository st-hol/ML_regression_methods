import numpy as np
from utils.constants import num_vars, drop_subset
from utils.constants import z_cols


def remove_q_marks(cars):
    cars = cars.replace('?', np.nan)
    return cars


def numerize(cars):
    for i in num_vars:
        cars[i] = cars[i].astype('float64')
    return cars


def drop_nan_rows(cars):
    cars = cars.dropna(subset=drop_subset)
    print(cars.isnull().sum())
    return cars


def normalize_cars(cars):
    cars = cars[z_cols] # z
    normalized_cars = (cars - cars.mean()) / (cars.std())
    return normalized_cars


def prep_data_fully(cars):
    c = remove_q_marks(cars)
    c = numerize(c)
    c = drop_nan_rows(c)
    return normalize_cars(c)
