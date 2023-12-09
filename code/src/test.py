from os import path
import pandas as pd
from train.lpda import lpda, simple_predict, predict
from data_utils import get_data_folder
from dataset import heart, meta_heart, diabetes, meta_diabetes, diabetes2, meta_diabetes2

def main(dataset: pd.DataFrame, y_column: str) -> (float, float, float):
    a, b, X_test, y_test = lpda(dataset, y_column)

    # Solution to problem
    print(a)
    print(b)

    # Calculate some metrics based on test data
    precision, accuracy, f1 = predict(a, b, X_test, y_test)
    print(f"Precision: {str(precision)}")
    print(f"Accuracy: {str(accuracy)}")
    print(f"F1 Score: {str(f1)}")

    return (precision, accuracy, f1)

main(diabetes2, meta_diabetes2.y_column)