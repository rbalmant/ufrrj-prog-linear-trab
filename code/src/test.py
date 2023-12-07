from os import path
import pandas as pd
from train.lpda import lpda, simple_predict
from data_utils import get_data_folder

def main():
    train = pd.read_csv(path.join(get_data_folder(), 'heart_disease', 'heart.csv'))

    a, b, X_test = lpda(train, 'target')

    print(a)
    print(b)

    for i in X_test:
        simple_predict(a, b, i)