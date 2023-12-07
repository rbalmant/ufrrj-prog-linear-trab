import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(data_train: pd.DataFrame, class_col_str: str, data_test: pd.DataFrame = None, stratify: bool = True, test_size: float = 0.25):
    y = data_train[class_col_str] # classification column
    labels = np.unique(y) # list of classes
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    aux = []
    for label in labels:
        X_class = data_train[y == label]
        y_class = y[y == label]
        if (data_test is None or not len(data_test) > 0):
            # If test dataset is NOT provided, split training set into train and test sets.
            X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=test_size)
        else:
            # Otherwise, use provided test set.
            X_class_train = data_train
            y_class_train = y
            X_class_test = data_test
            y_class_test = data_test[y == label]
        
        category = {
            'X_train': X_class_train,
            'X_test': X_class_test,
            'y_train': y_class_train,
            'y_test': y_class_test
        }
        
        X_train.append(X_class_train)
        X_test.append(X_class_test)
        y_train.append(y_class_train)
        y_test.append(y_class_test)
        aux.append(category)
    #return X_train, X_test, y_train, y_test, labels, aux
    return X_train, X_test, y_test, aux