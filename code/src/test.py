from os import path
import pandas as pd
import numpy as np
from train.lpda import lpda, simple_predict, predict
from data_utils import get_data_folder
from dataset import heart, meta_heart, diabetes, meta_diabetes

class ModelResult:
    solution: dict
    test: dict
    scores: dict

    def __str__(self):
        np.set_printoptions(linewidth=np.inf)
        return f"Solution: \n" \
                    f"\ta = {str(self.solution['a'])}\n" \
                    f"\tb = {str(self.solution['b'])}\n" \
                f"\n\n" \
                f"Test set: \n" \
                    f"\tX_test = {str(self.test['X_test'])}\n" \
                    f"\ty_test = {str(self.test['y_test'])}\n" \
                f"\n\n" \
                f"Scores: \n" \
                    f"\tPrecision: {str(self.scores['precision'])}\n" \
                    f"\tAccuracy: {str(self.scores['precision'])}\n" \
                    f"\tF1 score: {str(self.scores['f1'])}\n" \
                

def main(dataset: pd.DataFrame, y_column: str) -> ModelResult:
    a, b, X_test, y_test = lpda(dataset, y_column)

    solution = {
        'a': a,
        'b': b
    }

    test = {
        'X_test': X_test,
        'y_test': y_test
    }

    # Calculate some metrics based on test data
    precision, accuracy, f1 = predict(a, b, X_test, y_test)

    scores = {
        'precision': precision,
        'accuracy': accuracy,
        'f1': f1
    }

    model_result = ModelResult()
    model_result.solution = solution
    model_result.test = test
    model_result.scores = scores

    return model_result

model_result = main(heart, meta_heart.y_column)
print(model_result)