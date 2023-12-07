import logging
import numpy as np
import pandas as pd

from gzip import gzip
from scipy.optimize import linprog

from data_utils import split_data

def lpda(data_train, y_column, data_test = None):
    logging.debug(">> lpda(data group)")

    X_train, X_test, y_test, aux = split_data(data_train, y_column, data_test, True, 0.4)

    # Get partition from method split_data instead of doing additional calculation
    X1 = aux[0]['X_train']
    X2 = aux[1]['X_train']

    p = X_train.shape[1]
    
    n1 = len(X1)
    n2 = len(X2)
    n = n1 + n2

    m = np.repeat(1/n1, n1)
    w = np.repeat(1/n2, n2)

    obj_f_vars = np.repeat(0, p+1)

    # Objective function
    f = np.concatenate((m, w, obj_f_vars))

    #print(f)

    # A_ub
    u1 = np.eye(n1) * -1
    v1 = np.zeros((n1, n2))

    buf = np.empty_like(X1)
    ab1 = np.c_[np.multiply(float(-1), X1, buf), np.repeat(1, n1)]

    B1 = np.c_[u1, v1, ab1]

    u2 = u1
    v2 = v1
    ab2 = np.zeros((n1, p+1))

    B2 = np.c_[u2, v2, ab2]

    u3 = np.zeros((n2, n1))
    v3 = np.eye(n2) * -1
    ab3 = np.c_[X2, np.repeat(-1, n2)]

    B3 = np.c_[u3, v3, ab3]

    u4 = u3
    v4 = v3
    ab4 = np.zeros((n2, p+1))

    B4 = np.c_[u4, v4, ab4]

    A = np.concatenate((B1, B2, B3, B4))

    s1 = np.repeat(-1, n1)
    s2 = np.repeat(0, n1)
    s3 = np.repeat(-1, n2)
    s4 = np.repeat(0, n2)

    b = np.concatenate((s1, s2, s3, s4))

    nvar = n + (p+1)

    lower = [-np.inf for i in range(nvar)]
    upper = [np.inf for i in range(nvar)]
    bounds = list(zip(lower, upper))

    #print(bounds)

    # Attempt to solve the LP problem
    result = None
    try:
        result = linprog(f, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    except Exception as e:
        print(f"An error occurred during LP optimization: {e}")
        return None
    if result is None or not result.success:
        print("LP optimization did not converge.")
        return None
    logging.debug("<< lpda(X, y)")
    return result, X_test, y_test

def lpda_save_to_file(file_path, result, X_test, y_test):
    model = {
        result,
        X_test,
        y_test
    }

    # Very basic compression
    with gzip.open(file_path, 'wb') as f:
        f.write(model)

def lpda_load_from_file(file_path):
    model = {}

    with gzip.open(file_path, 'rb') as f:
        model = f.read()
        
    return model

def predict(model, test):
    logging.debug(">> predict(model, test)")
        
    # Hyperplane equation
    coefficients = model.x[len(model.x) - test.shape[0]:]
    #print(coefficients)

    # Binary classification
    predict = np.dot(test, coefficients)
    print(predict)
        
    logging.debug("<< predict(model, test)")
    return predict
