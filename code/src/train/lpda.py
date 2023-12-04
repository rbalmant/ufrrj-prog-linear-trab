import logging
import numpy as np
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split

class LPDA:
    def lpda(self, X, y):
        logging.debug(">> " + __class__.__name__ + ".lpda(X, y)")
        p = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
        X_train = [X_train[i] for i in range(len(X_train)) if not y_train[i]]
        Y_train = [X_train[i] for i in range(len(X_train)) if y_train[i]]

        X_test = [X_test[i] for i in range(len(X_test)) if not y_test[i]]
        Y_test = [X_test[i] for i in range(len(X_test)) if y_test[i]]

        n1 = len(X_train)
        n2 = len(Y_train)
        n = n1 + n2

        #print(n1)
        #print(n2)

        #print(X_train)

        m = np.repeat(1/n1, n1)
        w = np.repeat(1/n2, n2)

        obj_f_vars = np.repeat(0, p+1)

        # Objective function
        f = np.concatenate((m, w, obj_f_vars))

        #print(f)

        # A_ub
        u1 = np.eye(n1) * -1
        v1 = np.zeros((n1, n2))

        buf = np.empty_like(X_train)
        ab1 = np.c_[np.multiply(float(-1), X_train, buf), np.repeat(1, n1)]

        B1 = np.c_[u1, v1, ab1]

        u2 = u1
        v2 = v1
        ab2 = np.zeros((n1, p+1))

        B2 = np.c_[u2, v2, ab2]

        u3 = np.zeros((n2, n1))
        v3 = np.eye(n2) * -1
        ab3 = np.c_[Y_train, np.repeat(-1, n2)]

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
        logging.debug("<< " + __class__.__name__ + ".lpda(X, y)")
        return result, X_test, Y_test
    
    def predict(self, model):
        logging.debug(">> " + __class__.__name__ + ".predict(model)")
        # TODO
        logging.debug("<< " + __class__.__name__ + ".predict(model)")
        return None