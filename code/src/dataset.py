import numpy as np
import pandas as pd
from os import path
from train.lpda import lpda_load_from_file
from data_utils import get_data_folder
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

meta_heart: DataSet = DataSet()
meta_heart.folder = datasets_raw[0]['folder']
meta_heart.file = datasets_raw[0]['file']
meta_heart.y_column = datasets_raw[0]['y_column']
meta_heart.solution = lpda_load_from_file("model.dat")

heart = pd.read_csv(path.join(get_data_folder(), meta_heart.folder, meta_heart.file))

meta_diabetes: DataSet = DataSet()
meta_diabetes.folder = datasets_raw[1]['folder']
meta_diabetes.file = datasets_raw[1]['file']
meta_diabetes.y_column = datasets_raw[1]['y_column']

diabetes = pd.read_csv(path.join(get_data_folder(), meta_diabetes.folder, meta_diabetes.file))

meta_diabetes2: DataSet = DataSet()
meta_diabetes2.folder = datasets_raw[2]['folder']
meta_diabetes2.file = datasets_raw[2]['file']
meta_diabetes2.y_column = datasets_raw[2]['y_column']

diabetes2 = pd.read_csv(path.join(get_data_folder(), meta_diabetes2.folder, meta_diabetes2.file))