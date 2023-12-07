from os import path
import pandas as pd
from train.lpda import lpda, simple_predict
from data_utils import get_data_folder
from dataset import diabetes

def main():
    train = pd.read_csv(path.join(get_data_folder(), diabetes.folder, diabetes.file))

    a, b, X_test = lpda(train, diabetes.y_column)

    print(a)
    print(b)

    for i in X_test:
        simple_predict(a, b, i)

main()