import numpy as np
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from train.lpda import LPDA

# Generate synthetic data for illustration
np.random.seed(42)
X = np.random.rand(100, 2)  # Replace with your NHANES features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary labels based on a simple condition

# LPDA Implementation
lpda = LPDA()
model, X_test, Y_test = lpda.lpda(X, y)

print(model)
