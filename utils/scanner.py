import pandas as pd
from utils.constants import cols
import utils.utililty as ut


def read_csv():
    cars = pd.read_csv("data/features.data", names=cols)
    return cars


def ask_if_should_be_printed_all_predicted():
    inp = input(
        "\n\n!!! Please enter '1' if you want to print all predicted values or '0' if you want to see only result >>> \n ")
    printPredicted = ut.to_num(inp) == 1
    return printPredicted