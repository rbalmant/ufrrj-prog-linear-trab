import numpy as np
from train.lpda import lpda_load_from_file
class DataSet:
    folder: str
    file: str
    y_column: str
    solution: (np.array, float)


datasets_raw = [
    {
        "folder": "heart_disease",
        "file": "heart.csv",
        "y_column": "target" 
    },
    {
        "folder": "diabetes",
        "file": "diabetes_binary_5050split_health_indicators_BRFSS2021.csv",
        "y_column": "Diabetes_binary"
    },
    {
        "folder": "diabetes2",
        "file": "diabetes_binary_health_indicators_BRFSS2021.csv",
        "y_column": "Diabetes_binary"
    }
]



heart: DataSet = DataSet()
heart.folder = datasets_raw[0]['folder']
heart.file = datasets_raw[0]['file']
heart.y_column = datasets_raw[0]['y_column']
heart.solution = lpda_load_from_file("model.dat")

diabetes: DataSet = DataSet()
diabetes.folder = datasets_raw[1]['folder']
diabetes.file = datasets_raw[1]['file']
diabetes.y_column = datasets_raw[1]['y_column']