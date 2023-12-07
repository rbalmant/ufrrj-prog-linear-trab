import logging
import numpy as np
import pandas as pd
from os import path

import gzip
import pickle
from pulp import *
from sklearn.metrics import precision_score, accuracy_score, f1_score

from data_utils import split_data, get_model_folder

def lpda(data_train: pd.DataFrame, y_column: str, data_test: pd.DataFrame = None) -> (list, float, np.array, np.array):
    logging.debug(">> (lpda) lpda(data group)")

    X1, X2, X_test, y_test = split_data(data_train, y_column, data_test, 0.2)

    # Now we can define and solve our LP problem using Pulp library with a solver of our choice
    prob = LpProblem("Heart_Disease_LPDA", LpMinimize)
    a = []
    b = []
    u = []
    v = []

    n1 = len(X1)
    n2 = len(X2)
    n = n1 + n2

    # We will take mi = 1/n1 and wi = 1/n2, so features are of equal importance.
    m = 1/n1
    w = 1/n2
    
    # Define decision variable a & b (hyperplane H)

    # For a actually we have a vector of decision variable
    for i in range(len(X1[0])):
        ai = LpVariable("a"+ str(i), 0, None, LpContinuous)
        a.append(ai)
    
    # On the other hand, b is scalar
    b = LpVariable("b", 0, None, LpContinuous)

    # Define decision variables ui and vj
    for i in range(n1):
        ui = LpVariable("u" + str(i), 0, None, LpContinuous)
        u.append(ui)

    for j in range(n2):
        vj = LpVariable("v"+ str(j), 0, None, LpContinuous)
        v.append(vj)


    u_sum = LpAffineExpression([(ui, m) for ui in u])
    v_sum = LpAffineExpression([(vj, w) for vj in v])

    # Objective function:
    prob += lpSum([u_sum, v_sum])

    # Constraints
    for i in range(n1):
        prob += u[i] >= - lpDot(a, X1[i]) + b + 1, "u_const_"+str(i)
        prob += u[i] >= 0, "u_const_non_negativity_"+str(i)

    for j in range(n2):
        prob += v[j] >= lpDot(a, X2[j]) - b + 1, "v_const_"+str(j)
        prob += v[j] >= 0, "v_const_non_negativity_"+str(j)

    prob.writeLP("lpProblem.lp")

    # Solve using HiGHS solver (please see https://highs.dev/)
    #prob.solve(HiGHS_CMD())
    prob.solve()

    # The status of the solution is printed to the screen if logging is INFO level
    logging.info("LP problem status: ", LpStatus[prob.status])

    # Fetch values of a & b decision variables
    a = []
    b = None
    for v in prob.variables():
        #print(v.name, "=", v.varValue)
        if v.name == 'b':
            b = v.varValue
        if v.name.startswith('a'):
            a.insert(int(v.name.replace('a', '')), v.varValue)

    return a, b, X_test, y_test

def lpda_save_to_file(file_name, a, b):
    model = {
       'h_a': a,
       'h_b': b
    }

    bytes = pickle.dumps(model)
    file_path = path.join(get_model_folder(), file_name)

    # Very basic compression
    with gzip.open(file_path, 'wb') as f:
        f.write(bytes)

def lpda_load_from_file(file_name) -> (np.array, float):
    model = {}
    file_path = path.join(get_model_folder(), file_name)

    with gzip.open(file_path, 'rb') as f:
        model = f.read()
        
    model = pickle.loads(model)

    return model['h_a'], model['h_b']

def simple_predict(a, b, test) -> int:
    logging.debug(">> (lpda) predict(a, b, test)")

    # "Above" hyperplane line -- blue colour in article
    if np.dot(test, a) - b >= 0:
        logging.debug("<< (lpda) predict(a, b, test)")
        return 0
    # "Below" hyperplane line -- red colour in article
    else:
        logging.debug("<< (lpda) predict(a, b, test)")
        return 1
    
def calculate_scores(y_test, predictions) -> (float, float, float):
    logging.debug(">> (lpda) calculate_scores(y_test, predictions)")
    precision = precision_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    logging.debug("<< (lpda) calculate_scores(y_test, predictions)")
    return precision, accuracy, f1


def predict(a, b, test_group, y_test) -> (float, float, float):
    logging.debug(">> (lpda) predict(a, b, test_group, y_test)")
    predictions = []
    for point in zip(test_group, y_test):
        prediction = None

        if np.dot(point[0], a) - b >= 0:
            prediction = 0
        else:
            prediction = 1
            
        predictions.append(prediction)
    
    logging.debug("<< (lpda) predict(a, b, test_group, y_test)")
    return calculate_scores(y_test, predictions)

        

