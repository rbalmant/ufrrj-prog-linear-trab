import logging
from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_data_folder():
    return path.join(__file__, '..', '..', 'data')

def split_data(data_train: pd.DataFrame, class_col_str: str, data_test: pd.DataFrame = None, test_size: float = 0.25) -> dict:
    logging.debug(">> (data_utils) split_data(data_train, class_col_str, data_test, stratify, test_size)")
    np.random.seed(42)
    
    y = data_train[class_col_str] # classification column
    labels = np.unique(y) # list of classes
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    aux = []
    for label in labels:
        X_class = data_train[y == label]
        y_class = y[y == label]
        #print(X_class.shape)
        #print(y_class.shape)
        if (data_test is None or not len(data_test) > 0):
            # If test dataset is NOT provided, split training set into train and test sets.
            X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=test_size, random_state = 42)
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
        
        X_train = pd.concat([X_train, X_class_train], ignore_index=True)
        X_test = pd.concat([X_test, X_class_test], ignore_index=True)
        y_train = pd.concat([y_train, y_class_train], ignore_index=True)
        y_test = pd.concat([y_test, y_class_test], ignore_index=True)

        aux.append(category)

    # Get partition from method split_data instead of doing additional calculation
    X1 = aux[0]['X_train']
    X2 = aux[1]['X_train']

    X1 = X1.to_numpy()[:, :-1]
    X2 = X2.to_numpy()[:, :-1]
    X_test = X_test.to_numpy()[:, :-1]

    # Please note the last column gets removed due being the classification column, since data is partitioned into sets per category there is not reason for this column.

    logging.debug("<< (data_utils) split_data(data_train, class_col_str, data_test, stratify, test_size)")
    return X1, X2, X_test