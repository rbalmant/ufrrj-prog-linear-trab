from os import path
import pandas as pd
from train.lpda import lpda, simple_predict
from data_utils import get_data_folder

datasets = [
    {
        "folder": "heart_disease",
        "file": "heart.csv",
        "y_column": "target" 
    },
    {
        "folder": "fake_job_postings",
        "file": "fake_job_postings.csv",
        "y_column": "fraudulent"
    }
]

def main():
    train = pd.read_csv(path.join(get_data_folder(), datasets[1]['folder'], datasets[1]['file']))

    a, b, X_test = lpda(train, datasets[1]['y_column'])

    print(a)
    print(b)

    for i in X_test:
        simple_predict(a, b, i)

main()