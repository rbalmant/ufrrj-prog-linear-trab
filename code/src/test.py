from os import path
import pandas as pd
from train.lpda import lpda, simple_predict, predict
from data_utils import get_data_folder
from dataset import heart, diabetes, tree

def main():
    train = pd.read_csv(path.join(get_data_folder(), heart.folder, heart.file))

    a, b, X_test, y_test = lpda(train, heart.y_column)

    # Solution to problem
    print(a)
    print(b)

    # Calculate some metrics based on test data
    precision, accuracy, f1 = predict(a, b, X_test, y_test)
    print(f"Precision: {str(precision)}")
    print(f"Accuracy: {str(accuracy)}")
    print(f"F1 Score: {str(f1)}")

main()